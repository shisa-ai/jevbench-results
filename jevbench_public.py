"""Run JevBench's public tiers against shisa-ai/shisa-de-1.

This repository is the record of one run: the 231 published JevBench v1.2
decisions answered by DE-1 through the letter-slot readout in `readout/`, and
scored with jevbench's own runner and `score_task`. This script reproduces that
run against any vLLM completions endpoint serving the same checkpoint.

jevbench itself (the frozen task records, the serial runner, the scoring, and
the published comparison artifact) is the pinned checkout in `vendor/jevbench`.
`run-jevbench.sh` fetches it at commit ee677f01f177; set `JEVBENCH_DIR` to use
an existing checkout instead.

The public files are the only task records JevBench publishes: easy 48,
standard (file `original`) 72, hard 111. The judge tier (146 items) and the
held-out halves are not published, so this covers 231 of the 534 v1.2
decisions and no JevBench Score is computed.

Usage:
  python jevbench_public.py run \\
      --base-url http://127.0.0.1:8021 --model shisa-ai/shisa-de-1 \\
      --tokenizer shisa-ai/shisa-de-1 --out-dir runs/de-1-public
  python jevbench_public.py report --run-dir results/de-1-public
  python jevbench_public.py timing --run-dir runs/de-1-public \\
      --raw-dir runs/de-1-public/scratch

`run-jevbench.sh` wraps all three against a served endpoint.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from readout.vllm_scoring import VLLMScoring
from readout.format import Case, Question

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_JEVBENCH = REPO_ROOT / "vendor" / "jevbench"

# jevbench tier -> public dataset file (see datasets/manifest.json)
TIER_FILES = {"easy": "easy.jsonl", "standard": "original.jsonl", "hard": "hard.jsonl"}
QID = "decision"
ADAPTER_NAME = "vllm_scoring_letter_slots"
COST_BASIS = "local_gpu_no_provider_tariff"
READOUT = "restricted softmax over the option-letter slots of one vLLM completion"
PUBLISHED_REFERENCE = ("jev-1.13.0", "jeff")  # rows shown beside DE-1 in the per-family table

_JEV: SimpleNamespace | None = None
DecisionResult = None  # bound by load_jevbench(); the runner builds these


class JevBenchUnavailable(RuntimeError):
    """The vendored jevbench checkout is missing."""


def jevbench_dir() -> Path:
    path = Path(os.environ.get("JEVBENCH_DIR", DEFAULT_JEVBENCH))
    if not (path / "jevbench" / "cli.py").exists():
        raise JevBenchUnavailable(
            f"jevbench checkout not found at {path}; run ./run-jevbench.sh to fetch it, "
            "or set JEVBENCH_DIR to an existing checkout"
        )
    return path


def load_jevbench() -> SimpleNamespace:
    """Import jevbench from the pinned checkout, once per process."""
    global _JEV, DecisionResult
    if _JEV is None:
        path = jevbench_dir()
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        from jevbench.adapters import base as _base
        from jevbench.budget import Ledger
        from jevbench.composite_v12 import tvd
        from jevbench.metrics import percentile
        from jevbench.runner import Runner
        from jevbench.summarize import summarize
        from jevbench.tasks import dataset_hash, load_jsonl

        DecisionResult = _base.DecisionResult
        _JEV = SimpleNamespace(
            path=path,
            Ledger=Ledger,
            Runner=Runner,
            summarize=summarize,
            dataset_hash=dataset_hash,
            load_jsonl=load_jsonl,
            tvd=tvd,
            percentile=percentile,
        )
    return _JEV


def jevbench_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def task_to_case(task: Any) -> Case:
    """One jevbench task as one case in this harness's format.

    Choice options are lettered in the record's own `labels` order and score
    levels follow the record's criteria order, so the readout sees the same
    option order the published runs saw. Nothing from `expected` or the
    provenance rationale reaches the model.
    """
    qtype = task.question["type"]
    criteria = task.question.get("criteria")
    options = levels = None
    if qtype == "choice":
        if not isinstance(criteria, dict):
            raise ValueError(f"{task.id}: choice criteria is not a map")
        if set(criteria) != set(task.labels):
            raise ValueError(f"{task.id}: choice criteria keys differ from labels")
        options = {str(label): str(criteria[label]) for label in task.labels}
    elif qtype == "score":
        if not isinstance(criteria, (list, tuple)):
            raise ValueError(f"{task.id}: score criteria is not a list")
        levels = [str(level) for level in criteria]
        if [str(i) for i in range(len(levels))] != [str(label) for label in task.labels]:
            raise ValueError(f"{task.id}: score levels differ from labels")
    question = Question(
        qid=QID, type=qtype, instructions=str(task.question["instructions"]),
        options=options, levels=levels,
    )
    question.validate()
    return Case(
        id=task.id,
        suite="jevbench-public",
        split="test",
        language="en",
        group=task.group or task.id,
        source=f"fstandhartinger/jevbench public {qtype} item ({task.family})",
        state=task.state,
        questions={QID: question},
        gold={QID: task.expected},
    )


class _CapturingScoring(VLLMScoring):
    """VLLMScoring that keeps the last rendered prompt as raw evidence."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.last_prompt: str | None = None

    def _prompt(self, case: Case, question_text: str, options: list[tuple[str, str]]) -> str:
        prompt = super()._prompt(case, question_text, options)
        self.last_prompt = prompt
        return prompt


class JevBenchScoringAdapter:
    """jevbench adapter: this repository's readout, jevbench's runner and scoring.

    `probs` are the model's own next-token distribution restricted to the
    answer slots, which is what jevbench calls a native distribution; no
    probabilities are verbalized or repaired. The route has no tariff, so
    `reserve_estimate` returns None and every record carries cost_basis
    `local_gpu_no_provider_tariff` with a null cost.
    """

    name = ADAPTER_NAME
    cost_basis = COST_BASIS
    price_input_per_m = None
    price_output_per_m = None

    def __init__(self, backend: _CapturingScoring) -> None:
        self.backend = backend

    def reserve_estimate(self, task: Any) -> None:
        return None

    def run(self, task: Any) -> Any:
        out = self.backend.evaluate(task_to_case(task))
        status = out.get("status")
        meta = ((out.get("raw") or {}).get("per_question") or {}).get(QID) or {}
        tokens = out.get("tokens") or {}
        result = DecisionResult(
            adapter=self.name,
            ok=status == "ok",
            probs_source="native",
            model=self.backend.model,
            latency_s=round(float((out.get("latency_ms") or {}).get("total") or 0.0) / 1000.0, 6),
            usage={
                "input_tokens": tokens.get("input") or 0,
                "output_tokens": meta.get("n_requests") or 0,
            },
            raw=out,
            request_body={
                "prompt": self.backend.last_prompt,
                "readout": READOUT,
                "base_url": self.backend.base_url,
                "model": self.backend.model,
            },
        )
        if status != "ok":
            result.error = json.dumps(out.get("error") or {"kind": status})[:300]
            return result
        answer = (out.get("answers") or {}).get(QID) or {}
        qtype = task.question["type"]
        if qtype == "noul":
            p_yes = float(answer["noul"])
            result.probs = {"yes": p_yes, "no": 1.0 - p_yes}
        else:
            result.probs = {str(k): float(v) for k, v in answer["probabilities"].items()}
        return result


def scoring_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer_source": args.tokenizer or args.model,
        "timeout_s": args.timeout_s,
        "scaffold": args.scaffold,
    }


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clear_previous_run(out_dir: Path, scratch: Path) -> list[str]:
    """Remove a previous run from a directory that `--force` is about to reuse.

    Overwriting means the directory holds the new run and nothing else. Leaving
    a tier the new run does not cover behind would put stale results under a
    fresh manifest, and the report would read them as this run's.
    """
    removed: list[str] = []
    for path in [
        *sorted(out_dir.glob("*.jsonl")),
        *sorted(out_dir.glob("*.summary.json")),
        out_dir / "manifest.json",
        out_dir / "timing.json",
    ]:
        if path.exists():
            path.unlink()
            removed.append(path.name)
    for tier in TIER_FILES:
        tier_raw = scratch / tier
        if tier_raw.exists():
            shutil.rmtree(tier_raw)
            removed.append(f"{scratch.name}/{tier}/")
        ledger = scratch / f"{tier}.ledger.jsonl"
        if ledger.exists():
            ledger.unlink()
            removed.append(f"{scratch.name}/{tier}.ledger.jsonl")
    return removed


def run_task_ids(run_dir: Path) -> set[str]:
    """The task ids this run's own records cover."""
    ids: set[str] = set()
    for tier in TIER_FILES:
        path = run_dir / f"{tier}.jsonl"
        if path.exists():
            ids.update(record["task_id"] for record in read_records(path))
    return ids


def stale_tier_files(run_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Tier files present in a run directory that its manifest does not cover."""
    covered = set(manifest.get("tiers") or {})
    if not covered:
        return []
    return [
        f"{tier}.jsonl" for tier in TIER_FILES
        if tier not in covered and (run_dir / f"{tier}.jsonl").exists()
    ]


def server_provenance(base_url: str) -> dict[str, Any]:
    """What the endpoint reports about itself: build and limits.

    The serving flags are not visible over the API, so a run records the build
    and the limits it can see. A request that fails is recorded rather than
    raised: the run itself can still proceed.
    """
    base = base_url.rstrip("/")
    out: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(f"{base}/version", timeout=5) as resp:
            out["version"] = json.loads(resp.read().decode("utf-8")).get("version")
    except Exception as exc:
        out["version"] = None
        out.setdefault("unavailable", {})["/version"] = f"{type(exc).__name__}: {exc}"
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as resp:
            data = (json.loads(resp.read().decode("utf-8")).get("data") or [{}])[0]
        out["served_model_id"] = data.get("id")
        out["max_model_len"] = data.get("max_model_len")
    except Exception as exc:
        out.setdefault("unavailable", {})["/v1/models"] = f"{type(exc).__name__}: {exc}"
    return out


def client_provenance() -> dict[str, Any]:
    """Client versions that decide how a prompt is rendered."""
    out: dict[str, Any] = {"python": sys.version.split()[0]}
    for name in ("transformers", "httpx"):
        try:
            from importlib.metadata import version

            out[name] = version(name)
        except Exception:
            out[name] = None
    return out


def _raw_readout(raw_dir: Path | None, tier: str) -> dict[str, Any] | None:
    """Server-side request time and readout request counts, from the raw responses.

    Those files live in the run's scratch directory outside the repository, so
    this block is null when they are gone. It separates the vLLM request from
    the client-side render and parse that the record's own latency includes, and
    it counts the requests the readout made. A decision costs one top-logprobs
    request plus one full-prompt fallback request per option letter outside the
    returned top-20: those fallbacks are inside the decision's own latency but
    outside the timer around the first request, and the records' input tokens
    omit their prompt tokens.
    """
    if raw_dir is None:
        return None
    tier_raw = raw_dir / tier / "raw"
    if not tier_raw.is_dir():
        return None
    request_values: list[float] = []
    non_request_values: list[float] = []
    by_missing: dict[int, list[float]] = {}
    decisions = 0
    items_with_fallback = 0
    requests_total = 0
    fallback_requests = 0
    missing_total = 0
    tokens_in_records = 0
    tokens_all_requests = 0
    for path in sorted(tier_raw.glob("*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        response = body.get("response") or {}
        latency = response.get("latency_ms") or {}
        per_question = (response.get("raw") or {}).get("per_question") or {}
        if not per_question:
            continue
        decisions += 1
        decision_fallbacks = 0
        for question in per_question.values():
            missing = len(question.get("missing_from_top") or [])
            prompt_tokens = question.get("input_tokens") or 0
            requests_total += question.get("n_requests") or 1
            decision_fallbacks += missing
            missing_total += missing
            tokens_in_records += prompt_tokens
            # Each fallback re-sends the prompt plus the one-token letter slot,
            # which the boundary check pins as exactly one token.
            tokens_all_requests += prompt_tokens * (1 + missing) + missing
        fallback_requests += decision_fallbacks
        items_with_fallback += 1 if decision_fallbacks else 0
        request_ms = latency.get("request")
        total_ms = latency.get("total")
        if isinstance(request_ms, (int, float)):
            request_values.append(float(request_ms))
            if isinstance(total_ms, (int, float)):
                outside = float(total_ms) - float(request_ms)
                non_request_values.append(outside)
                by_missing.setdefault(decision_fallbacks, []).append(outside)
    if not decisions:
        return None
    jev = load_jevbench()
    block: dict[str, Any] = {
        "n_decisions": decisions,
        "mean_requests_per_decision": round(requests_total / decisions, 2),
        "items_with_fallback": items_with_fallback,
        "fallback_requests": fallback_requests,
        "mean_missing_letters": round(missing_total / decisions, 2),
        "input_tokens_per_decision_in_records": round(tokens_in_records / decisions, 2),
        "input_tokens_per_decision_all_requests": round(tokens_all_requests / decisions, 2),
        "source": "latency_ms and raw.per_question in the run's raw responses (not committed)",
    }
    if request_values:
        block["server_request_p50_ms"] = round(jev.percentile(request_values, 0.5), 1)
        block["server_request_p95_ms"] = round(jev.percentile(request_values, 0.95), 1)
    if non_request_values:
        block["non_request_p50_ms"] = round(jev.percentile(non_request_values, 0.5), 1)
    if by_missing:
        # Time outside the first request, grouped by how many option letters
        # were missing from the top-20, which is what the fallback requests cost.
        block["non_request_ms_mean_by_missing_letters"] = {
            str(count): round(statistics.fmean(values), 1)
            for count, values in sorted(by_missing.items())
        }
    return block


def timing_block(run_dir: Path, raw_dir: Path | None = None) -> dict[str, Any]:
    """Per-tier and per-run timing in milliseconds.

    `wall_ms` is first to last record timestamp, which includes every request
    the readout made for a decision; `sum_latency_ms` adds the per-decision
    latencies the runner recorded. Percentiles use jevbench's own percentile
    function so they match the tier summaries. `readout` carries the server-side
    request time and the request counts recovered from the raw responses.
    """
    jev = load_jevbench()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    tiers: dict[str, Any] = {}
    total_latency_ms = 0.0
    for tier in TIER_FILES:
        records_path = run_dir / f"{tier}.jsonl"
        if not records_path.exists():
            continue
        records = read_records(records_path)
        if not records:
            continue
        latencies = [r["latency_s"] * 1000 for r in records]
        stamps = [r["ts"] for r in records]
        total_latency_ms += sum(latencies)
        tiers[tier] = {
            "n": len(records),
            "wall_ms": round((max(stamps) - min(stamps)) * 1000, 1),
            "sum_latency_ms": round(sum(latencies), 1),
            "mean_ms": round(statistics.fmean(latencies), 1),
            "min_ms": round(min(latencies), 1),
            "p50_ms": round(jev.percentile(latencies, 0.5), 1),
            "p95_ms": round(jev.percentile(latencies, 0.95), 1),
            "max_ms": round(max(latencies), 1),
            "first_ms": round(latencies[0], 1),
            "readout": _raw_readout(raw_dir, tier),
        }
    run: dict[str, Any] = {"sum_latency_ms": round(total_latency_ms, 1)}
    if manifest.get("started_utc") and manifest.get("finished_utc"):
        started = dt.datetime.fromisoformat(manifest["started_utc"])
        finished = dt.datetime.fromisoformat(manifest["finished_utc"])
        run["total_wall_ms"] = round((finished - started).total_seconds() * 1000, 1)
    else:
        run["total_wall_ms"] = None
    run["warmup_ms"] = None if manifest.get("warmup_s") is None else round(manifest["warmup_s"] * 1000, 1)
    return {
        "run_label": manifest.get("run_label") or run_dir.name,
        "run_dir": str(run_dir),
        "raw_dir": None if raw_dir is None else str(raw_dir),
        "tiers": tiers,
        "run": run,
    }


def cmd_timing(args: argparse.Namespace) -> int:
    if args.raw_dir and len(args.raw_dir) != len(args.run_dir):
        print("--raw-dir must be given once per --run-dir, in the same order", file=sys.stderr)
        return 2
    raw_dirs = args.raw_dir or [None] * len(args.run_dir)
    print("| run | tier | n | wall ms | mean ms | p50 ms | p95 ms | max ms | first ms "
          "| server request p50 ms | req/decision | fallbacks |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run_dir, raw_dir in zip(args.run_dir, raw_dirs):
        block = timing_block(Path(run_dir).resolve(), Path(raw_dir).resolve() if raw_dir else None)
        target = Path(run_dir).resolve() / "timing.json"
        target.write_text(json.dumps(block, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for tier in TIER_FILES:
            entry = block["tiers"].get(tier)
            if entry is None:
                continue
            readout = entry.get("readout") or {}
            server_cell = (
                "" if readout.get("server_request_p50_ms") is None
                else f"{readout['server_request_p50_ms']:.1f}"
            )
            requests_cell = (
                "" if readout.get("mean_requests_per_decision") is None
                else f"{readout['mean_requests_per_decision']:.2f}"
            )
            fallback_cell = (
                "" if readout.get("items_with_fallback") is None
                else f"{readout['items_with_fallback']}/{readout['n_decisions']}"
            )
            print(
                f"| {block['run_label']} | {tier} | {entry['n']} | {entry['wall_ms']:.0f} "
                f"| {entry['mean_ms']:.1f} | {entry['p50_ms']:.1f} | {entry['p95_ms']:.1f} "
                f"| {entry['max_ms']:.0f} | {entry['first_ms']:.0f} | {server_cell} "
                f"| {requests_cell} | {fallback_cell} |"
            )
            if readout:
                print(
                    f"  {tier}: {readout['mean_requests_per_decision']:.2f} requests/decision, "
                    f"{readout['mean_missing_letters']:.2f} missing letters, "
                    f"{readout['items_with_fallback']}/{readout['n_decisions']} items with a fallback, "
                    f"{readout['non_request_p50_ms']:.1f} ms p50 outside the first request, "
                    f"input tokens/decision {readout['input_tokens_per_decision_in_records']:.1f} recorded "
                    f"against {readout['input_tokens_per_decision_all_requests']:.1f} across all requests"
                )
        print(f"\n{block['run_label']}: total wall {block['run']['total_wall_ms']} ms, "
              f"sum of decision latencies {block['run']['sum_latency_ms']} ms, "
              f"warmup {block['run']['warmup_ms']} ms -> {target}\n")
    return 0


def warmup(backend: _CapturingScoring) -> float:
    """Discarded request plus tokenizer load, so item 1 is not a cold start.

    Without it the first measured decision carries the server's graph capture
    and the client's tokenizer load, which is a property of the run rather
    than of the model.
    """
    backend._ensure_tokenizer()
    started = time.perf_counter()
    body = {"model": backend.model, "prompt": "warmup", "max_tokens": 1, "temperature": 0.0}
    response = backend.client.post("/v1/completions", content=json.dumps(body))
    response.raise_for_status()
    return round(time.perf_counter() - started, 3)


def cmd_run(args: argparse.Namespace) -> int:
    jev = load_jevbench()
    tiers = [tier.strip() for tier in args.tiers.split(",") if tier.strip()]
    unknown = [tier for tier in tiers if tier not in TIER_FILES]
    if unknown:
        print(f"unknown tier(s): {unknown}; known: {sorted(TIER_FILES)}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir).resolve()
    scratch = Path(args.scratch_dir).resolve() if args.scratch_dir else out_dir / "scratch"
    holds_results = out_dir.exists() and any(out_dir.glob("*.jsonl"))
    if holds_results and not args.force:
        print(f"{out_dir} already holds results; pass --force to overwrite", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    if holds_results:
        cleared = clear_previous_run(out_dir, scratch)
        print(f"[jevbench] --force: cleared {len(cleared)} previous artifact(s): "
              f"{', '.join(cleared) if cleared else 'none'}", flush=True)

    backend = _CapturingScoring(scoring_config(args))
    adapter = JevBenchScoringAdapter(backend)
    commit = jevbench_commit(jev.path)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "run_label": args.run_label,
        "adapter": ADAPTER_NAME,
        "readout": READOUT,
        "backend_config": backend.config,
        "endpoint": args.base_url,
        "model": args.model,
        "tokenizer": backend.tokenizer_source,
        "scaffold": args.scaffold,
        "jevbench_commit": commit,
        "jevbench_protocol": json.loads((jev.path / "datasets" / "manifest.json").read_text())["protocol"],
        "cost_basis": COST_BASIS,
        "started_utc": started,
        "server": server_provenance(args.base_url),
        "client": client_provenance(),
        "scratch_dir": str(scratch),
        "raw_dir": str(scratch),
        "tiers": {},
    }
    if args.warmup:
        manifest["warmup_s"] = warmup(backend)
        print(f"[jevbench] warmup {manifest['warmup_s']}s", flush=True)

    for tier in tiers:
        tasks_path = jev.path / "datasets" / "public" / TIER_FILES[tier]
        tasks = jev.load_jsonl(str(tasks_path))
        if args.limit:
            tasks = tasks[: args.limit]
        results_path = out_dir / f"{tier}.jsonl"
        if results_path.exists():
            results_path.unlink()
        tier_raw = scratch / tier / "raw"
        if tier_raw.exists() and args.force:
            # The runner creates raw files with mode "x"; a re-run needs a fresh directory.
            shutil.rmtree(tier_raw)
        ledger = jev.Ledger(str(scratch / f"{tier}.ledger.jsonl"), cap_usd=args.cap_usd)
        runner = jev.Runner(
            adapter, ledger, raw_dir=str(scratch / tier / "raw"), default_reserve_usd=0.0
        )
        print(f"[jevbench] {tier}: {len(tasks)} tasks from {tasks_path.name}", flush=True)
        records = runner.run_all(tasks, results_path=str(results_path))
        summary = jev.summarize(tasks, records, ledger.charged)
        entry = {
            "dataset": tasks_path.name,
            "dataset_hash": jev.dataset_hash(tasks),
            "n_planned": len(tasks),
            "n_attempted": len(records),
            "n_failed": sum(1 for r in records if not r["ok"]),
            "n_correct": summary["n_correct"],
            "accuracy": summary["n_correct"] / len(tasks) if tasks else None,
            "accuracy_over_valid": summary["accuracy"],
            "brier_mean": summary["brier_mean"],
            "ece": (summary["ece"] or {}).get("ece"),
            "latency_p50_s": summary["latency"]["p50_s"],
            "latency_p95_s": summary["latency"]["p95_s"],
            "mean_input_tokens": (
                statistics.mean(r["usage"].get("input_tokens") or 0 for r in records if r["ok"])
                if any(r["ok"] for r in records) else None
            ),
            "per_family": {
                fam: {"correct": v["n_correct"], "n": v["n_scorable"]}
                for fam, v in summary["per_family"].items()
            },
            "summary": summary,
        }
        if tier == "hard":
            by_id = {task.id: task for task in tasks}
            tvds = [
                jev.tvd(r["probs"], by_id[r["task_id"]].provenance["gold_probs"], by_id[r["task_id"]].labels)
                for r in records
                if by_id[r["task_id"]].provenance.get("gold_probs") and r.get("probs")
            ]
            entry["gold_probability_items"] = len(tvds)
            entry["mean_tvd_to_gold"] = statistics.mean(tvds) if tvds else None
        (out_dir / f"{tier}.summary.json").write_text(
            json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest["tiers"][tier] = {
            key: entry[key] for key in
            ("dataset", "dataset_hash", "n_planned", "n_attempted", "n_failed", "accuracy")
        }
        print(f"[jevbench] {tier}: accuracy {entry['accuracy']:.4f} "
              f"({entry['n_correct']}/{entry['n_planned']}), failed {entry['n_failed']}", flush=True)

    manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[jevbench] wrote {out_dir}")
    return 0


def tier_entries(run_dir: Path) -> dict[str, dict[str, Any]]:
    entries = {}
    for tier in TIER_FILES:
        path = run_dir / f"{tier}.summary.json"
        if path.exists():
            entries[tier] = json.loads(path.read_text(encoding="utf-8"))
    return entries


def published_public_items(jev: SimpleNamespace, task_ids: set[str]) -> list[dict[str, Any]]:
    """Published systems restricted to the same public items this run used."""
    per_task = json.loads(
        (jev.path / "results" / "v1.2" / "jevbench-v1.2-per-task.json").read_text(encoding="utf-8")
    )
    official = json.loads(
        (jev.path / "results" / "v1.2" / "jevbench-v1.2-results.json").read_text(encoding="utf-8")
    )
    tier_of = {t["id"]: t["tier"] for t in per_task["tasks"]}
    ranked = sorted(
        (row for row in official["systems"] if row.get("ranked")),
        key=lambda row: -row["jevbench_score"],
    )
    rank = {row["key"]: i + 1 for i, row in enumerate(ranked)}
    official_by_key = {row["key"]: row for row in official["systems"]}

    rows = []
    for key, system in per_task["systems"].items():
        counts = {tier: [0, 0] for tier in TIER_FILES}
        for tid, outcome in (system.get("public_tasks") or {}).items():
            if tid not in task_ids:
                continue
            tier = tier_of[tid]
            counts[tier][1] += 1
            counts[tier][0] += outcome[0] == "c"
        if not any(n for _c, n in counts.values()):
            continue
        row = official_by_key.get(key) or {}
        rows.append({
            "key": key,
            "display": system["display"],
            "partial": bool(system.get("partial")),
            "accuracy": {
                tier: (c / n if n else None) for tier, (c, n) in counts.items()
            },
            "counts": counts,
            "official_hard": (row.get("tiers") or {}).get("hard"),
            "official_score": row.get("jevbench_score"),
            "official_rank": rank.get(key),
        })
    return sorted(rows, key=lambda row: (-(row["accuracy"]["hard"] or 0), row["display"]))


def _pct(value: float | None, nd: int = 1) -> str:
    return "" if value is None else f"{100 * value:.{nd}f}%"


def _num(value: float | None, nd: int = 3) -> str:
    return "" if value is None else f"{value:.{nd}f}"


def _ms(seconds: float | None) -> str:
    return "" if seconds is None else f"{1000 * seconds:.0f}"


def cmd_report(args: argparse.Namespace) -> int:
    jev = load_jevbench()
    runs = []
    for run_dir in args.run_dir:
        path = Path(run_dir).resolve()
        entries = tier_entries(path)
        if not entries:
            print(f"no tier summaries under {path}", file=sys.stderr)
            return 2
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        stale = stale_tier_files(path, manifest)
        if stale:
            print(f"warning: {path} holds {', '.join(stale)} that manifest.json does not cover; "
                  f"an earlier run left them there and they are reported below as this run's",
                  file=sys.stderr)
        readout: dict[str, dict[str, Any]] = {}
        timing_path = path / "timing.json"
        if timing_path.exists():
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            readout = {
                tier: entry["readout"] for tier, entry in timing.get("tiers", {}).items()
                if entry.get("readout")
            }
        runs.append({
            "dir": path,
            "label": manifest.get("run_label") or path.name,
            "manifest": manifest,
            "entries": entries,
            "readout": readout,
            "record_ids": run_task_ids(path),
        })
    commit = runs[0]["manifest"].get("jevbench_commit", "unknown")

    print(f"## JevBench public tiers (jevbench @ {commit[:7]}, {len(runs)} run(s))\n")
    print("| run | tier | items | attempted | failed | correct | accuracy | Brier | ECE "
          "| p50 ms | p95 ms | input tokens/decision | input tokens/decision, all requests |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        for tier, entry in run["entries"].items():
            all_requests = (run["readout"].get(tier) or {}).get("input_tokens_per_decision_all_requests")
            print(
                f"| {run['label']} | {tier} | {entry['n_planned']} | {entry['n_attempted']} "
                f"| {entry['n_failed']} | {entry['n_correct']} | {_pct(entry['accuracy'], 2)} "
                f"| {_num(entry['brier_mean'])} | {_num(entry['ece'])} "
                f"| {_ms(entry['latency_p50_s'])} | {_ms(entry['latency_p95_s'])} "
                f"| {_num(entry.get('mean_input_tokens'), 0)} "
                f"| {_num(all_requests, 0)} |"
            )
    print()
    for run in runs:
        hard = run["entries"].get("hard") or {}
        if hard.get("mean_tvd_to_gold") is not None:
            print(
                f"- {run['label']}: hard-tier fidelity to the {hard['gold_probability_items']} "
                f"gold distributions, mean TVD {_num(hard['mean_tvd_to_gold'])}, "
                f"fidelity {_num(100 * (1 - hard['mean_tvd_to_gold']), 1)}."
            )

    print("\n### Timing (ms)\n")
    print("| run | tier | n | wall | mean | p50 | p95 | max | first | server request p50 "
          "| req/decision | fallbacks |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        timing_path = run["dir"] / "timing.json"
        if not timing_path.exists():
            print(f"| {run['label']} | (no timing.json; run the timing subcommand) "
                  "| | | | | | | | | | |")
            continue
        block = json.loads(timing_path.read_text(encoding="utf-8"))
        for tier in TIER_FILES:
            entry = block["tiers"].get(tier)
            if entry is None:
                continue
            readout = entry.get("readout") or {}
            server_cell = (
                "" if readout.get("server_request_p50_ms") is None
                else f"{readout['server_request_p50_ms']:.1f}"
            )
            requests_cell = (
                "" if readout.get("mean_requests_per_decision") is None
                else f"{readout['mean_requests_per_decision']:.2f}"
            )
            fallback_cell = (
                "" if readout.get("items_with_fallback") is None
                else f"{readout['items_with_fallback']}/{readout['n_decisions']}"
            )
            print(
                f"| {block['run_label']} | {tier} | {entry['n']} | {entry['wall_ms']:.0f} "
                f"| {entry['mean_ms']:.1f} | {entry['p50_ms']:.1f} | {entry['p95_ms']:.1f} "
                f"| {entry['max_ms']:.0f} | {entry['first_ms']:.0f} | {server_cell} "
                f"| {requests_cell} | {fallback_cell} |"
            )
    print()
    for run in runs:
        timing_path = run["dir"] / "timing.json"
        if not timing_path.exists():
            continue
        block = json.loads(timing_path.read_text(encoding="utf-8"))
        wall = block["run"]["total_wall_ms"]
        print(
            f"- {block['run_label']}: total wall {_num(wall, 0)} ms including warmup and "
            f"tokenizer load, sum of decision latencies {_num(block['run']['sum_latency_ms'], 0)} ms, "
            f"warmup {_num(block['run']['warmup_ms'], 0)} ms."
        )
        for tier in TIER_FILES:
            entry = block["tiers"].get(tier) or {}
            readout = entry.get("readout") or {}
            if not readout.get("fallback_requests"):
                continue
            print(
                f"- {block['run_label']} {tier}: the readout needed "
                f"{_num(readout['mean_requests_per_decision'], 2)} requests per decision "
                f"({readout['items_with_fallback']}/{readout['n_decisions']} items fell back, "
                f"{_num(readout['mean_missing_letters'], 2)} letters missing from the top-20), "
                f"spending {_num(readout['non_request_p50_ms'], 1)} ms p50 outside the first "
                f"request; input tokens per decision were "
                f"{_num(readout['input_tokens_per_decision_in_records'], 1)} in the records "
                f"against {_num(readout['input_tokens_per_decision_all_requests'], 1)} "
                f"across every request."
            )

    task_ids: set[str] = set()
    for run in runs:
        task_ids |= run["record_ids"]
    published_total = sum(
        len(jev.load_jsonl(str(jev.path / "datasets" / "public" / TIER_FILES[tier])))
        for tier in TIER_FILES
    )
    if not task_ids:
        # No records to read (summaries only): fall back to the whole public half.
        for tier in TIER_FILES:
            task_ids.update(
                task.id for task in jev.load_jsonl(str(jev.path / "datasets" / "public" / TIER_FILES[tier]))
            )
    rows = published_public_items(jev, task_ids)
    print(f"\n### Published systems on the same {len(task_ids)} public items\n")
    if len(task_ids) < published_total:
        print(f"This run covers {len(task_ids)} of the {published_total} published items, so every "
              f"published row below is restricted to those items and is not a full-benchmark "
              f"score.\n")
    print("| system | easy | standard | hard | official hard (220) | official JevBench Score |")
    print("|---|---:|---:|---:|---:|---:|")
    for run in runs:
        accuracy = {tier: entry["accuracy"] for tier, entry in run["entries"].items()}
        print(
            f"| **{run['label']}** (this run) | {_pct(accuracy.get('easy'), 2)} "
            f"| {_pct(accuracy.get('standard'), 2)} | {_pct(accuracy.get('hard'), 2)} "
            f"| not run (held out) | not computed (judge tier unpublished) |"
        )
    for row in rows:
        acc = row["accuracy"]
        score = "" if row["official_score"] is None else _num(row["official_score"], 1)
        if row["official_rank"]:
            score += f" (#{row['official_rank']})"
        if row["partial"]:
            score = f"{score} partial".strip()
        print(
            f"| {row['display']} | {_pct(acc['easy'], 2)} | {_pct(acc['standard'], 2)} "
            f"| {_pct(acc['hard'], 2)} | {_pct(row['official_hard'], 1)} | {score} |"
        )

    print("\n### Hard tier by family (correct / n)\n")
    families = sorted({f for run in runs for f in (run["entries"].get("hard") or {}).get("per_family", {})})
    if families:
        columns = [run["label"] for run in runs]
        columns += [row["display"] for row in rows if row["key"] in PUBLISHED_REFERENCE]
        print("| family | " + " | ".join(columns) + " |")
        print("|---|" + "---:|" * len(columns))
        per_task = json.loads(
            (jev.path / "results" / "v1.2" / "jevbench-v1.2-per-task.json").read_text(encoding="utf-8")
        )
        topic_of = {t["id"]: t["topic"] for t in per_task["tasks"]}
        for family in families:
            cells = []
            for run in runs:
                counts = ((run["entries"].get("hard") or {}).get("per_family") or {}).get(family)
                cells.append(f"{counts['correct']}/{counts['n']}" if counts else "")
            for key in PUBLISHED_REFERENCE:
                system = per_task["systems"].get(key) or {}
                c = n = 0
                for tid, outcome in (system.get("public_tasks") or {}).items():
                    if topic_of.get(tid) != family:
                        continue
                    c += outcome[0] == "c"
                    n += 1
                cells.append(f"{c}/{n}" if n else "")
            print(f"| {family} | " + " | ".join(cells) + " |")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jevbench_public", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the public tiers against a vLLM scoring endpoint")
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--tokenizer", default=None, help="defaults to --model")
    run.add_argument("--scaffold", default="systemone_json")
    run.add_argument("--timeout-s", dest="timeout_s", type=float, default=180.0)
    run.add_argument("--out-dir", dest="out_dir", required=True,
                     help="run output; keep it outside the repository, the runner writes raw responses")
    run.add_argument("--scratch-dir", dest="scratch_dir", default=None,
                     help="raw responses and ledger; defaults to <out-dir>/scratch")
    run.add_argument("--tiers", default="easy,standard,hard")
    run.add_argument("--limit", type=int, default=None, help="smoke test only; breaks comparability")
    run.add_argument("--cap-usd", dest="cap_usd", type=float, default=15.0)
    run.add_argument("--run-label", dest="run_label", default=None)
    run.add_argument("--no-warmup", dest="warmup", action="store_false",
                     help="keep the cold start inside the first measured decision")
    run.add_argument("--force", action="store_true")
    run.set_defaults(fn=cmd_run)

    report = sub.add_parser("report", help="print the markdown tables for completed runs")
    report.add_argument("--run-dir", dest="run_dir", action="append", required=True,
                        help="repeat for each run to compare; the first supplies the jevbench pin")
    report.set_defaults(fn=cmd_report)

    timing = sub.add_parser("timing", help="write per-run timing.json in milliseconds")
    timing.add_argument("--run-dir", dest="run_dir", action="append", required=True)
    timing.add_argument("--raw-dir", dest="raw_dir", action="append", default=None,
                        help="the run's scratch directory, to add the server-side request split; "
                             "give one per --run-dir in the same order")
    timing.set_defaults(fn=cmd_timing)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except JevBenchUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
