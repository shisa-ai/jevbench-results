"""Case loading, validation, and result-row helpers.

A case is one JSON object per line in a suite's cases.jsonl. See evals/README.md
for the field contract. This module is standard-library only so the same code
runs in the jev (3.14) and jev-gpu (3.12) environments.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

QUESTION_TYPES = ("noul", "choice", "score")


class CaseError(ValueError):
    pass


@dataclass
class Question:
    qid: str
    type: str
    instructions: str
    options: dict[str, str] | None = None  # choice: option id -> description
    levels: list[str] | None = None  # score: level descriptions, lowest first

    def validate(self) -> None:
        if self.type not in QUESTION_TYPES:
            raise CaseError(f"question {self.qid!r}: unknown type {self.type!r}")
        if not self.instructions:
            raise CaseError(f"question {self.qid!r}: instructions must be non-empty")
        if self.type == "choice":
            if not self.options:
                raise CaseError(f"choice question {self.qid!r}: options required")
            if not 2 <= len(self.options) <= 255:
                raise CaseError(
                    f"choice question {self.qid!r}: 2-255 options required, got {len(self.options or {})}"
                )
        if self.type == "score":
            if not self.levels:
                raise CaseError(f"score question {self.qid!r}: levels required")
            if len(self.levels) < 2:
                raise CaseError(f"score question {self.qid!r}: at least two levels required")


@dataclass
class Case:
    id: str
    suite: str
    split: str
    language: str
    group: str
    source: str
    state: Any
    questions: dict[str, Question]
    gold: dict[str, Any]
    acceptable: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    upstream: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, obj: dict[str, Any], suite_name: str) -> "Case":
        for key in ("id", "split", "language", "group", "state", "questions", "gold"):
            if key not in obj:
                raise CaseError(f"case missing required field {key!r}")
        questions: dict[str, Question] = {}
        for qid, q in obj["questions"].items():
            q = dict(q)
            if q.get("type") == "choice" and "options" not in q and "criteria" in q:
                # TypeSafe names the option map "criteria"; accept both spellings.
                q["options"] = q.pop("criteria")
            if q.get("type") == "score" and "levels" not in q and "legend" in q:
                q["levels"] = q.pop("legend")
            question = Question(qid=qid, **{
                k: v for k, v in q.items()
                if k in ("qid", "type", "instructions", "options", "levels")
            })
            question.qid = qid
            question.validate()
            questions[qid] = question
        gold = dict(obj["gold"])
        for qid in gold:
            if qid not in questions:
                raise CaseError(f"case {obj['id']!r}: gold for unknown question {qid!r}")
        return cls(
            id=obj["id"],
            suite=obj.get("suite", suite_name),
            split=obj["split"],
            language=obj["language"],
            group=obj["group"],
            source=obj.get("source", "unknown"),
            state=obj["state"],
            questions=questions,
            gold=gold,
            acceptable=dict(obj.get("acceptable", {})),
            notes=obj.get("notes", ""),
            upstream=dict(obj.get("upstream", {})),
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "suite": self.suite,
            "split": self.split,
            "language": self.language,
            "group": self.group,
            "source": self.source,
            "state": self.state,
            "questions": {},
            "gold": self.gold,
        }
        for qid, q in self.questions.items():
            jq: dict[str, Any] = {"type": q.type, "instructions": q.instructions}
            if q.options is not None:
                jq["options"] = q.options
            if q.levels is not None:
                jq["levels"] = q.levels
            out["questions"][qid] = jq
        if self.acceptable:
            out["acceptable"] = self.acceptable
        if self.notes:
            out["notes"] = self.notes
        if self.upstream:
            out["upstream"] = self.upstream
        return out

    def state_text(self) -> str:
        """The state rendered as plain text for prompt-based backends."""
        if isinstance(self.state, str):
            return self.state
        return json.dumps(self.state, ensure_ascii=False, indent=1)


def load_suite(suite_dir: str | Path) -> tuple[str, list[Case]]:
    """Load all cases from a suite directory containing cases.jsonl."""
    suite_dir = Path(suite_dir)
    cases_path = suite_dir / "cases.jsonl"
    if not cases_path.exists():
        raise FileNotFoundError(f"no cases.jsonl under {suite_dir}")
    suite_name = suite_dir.name
    cases: list[Case] = []
    seen: set[str] = set()
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        obj = json.loads(line)
        case = Case.from_json(obj, suite_name)
        if case.id in seen:
            raise CaseError(f"duplicate case id {case.id!r} in {cases_path}")
        seen.add(case.id)
        cases.append(case)
    return suite_name, cases


def temper_binary(p: float, t: float) -> float:
    """Binary temperature scaling: pull an over/under-confident noul toward one-half."""
    if t <= 0:
        return p
    a, b = max(p, 1e-12) ** (1.0 / t), max(1.0 - p, 1e-12) ** (1.0 / t)
    return a / (a + b)


def temper_distribution(probs: dict[str, float], t: float) -> dict[str, float]:
    """Multiclass temperature scaling: p_i**(1/T) renormalized; argmax preserved."""
    if t <= 0:
        return probs
    powered = {k: max(v, 1e-12) ** (1.0 / t) for k, v in probs.items()}
    total = sum(powered.values()) or 1.0
    return {k: v / total for k, v in powered.items()}


def apply_temperature(out: dict, t: float) -> dict:
    """Apply temperature T to a backend evaluate() output dict, in place.

    Noul probabilities get binary scaling; choice/score distributions are
    renormalized after scaling; confidence is recomputed from the tempered
    distribution and confidence_kind records the applied temperature. Used by
    the runner when --calibration T=<value> is passed, so gated probabilities
    come from calibrated distributions (task #12).
    """
    for qid, ans in (out.get("answers") or {}).items():
        if "noul" in ans:
            ans["noul"] = temper_binary(float(ans["noul"]), t)
        elif "probabilities" in ans:
            tempered = temper_distribution(
                {k: float(v) for k, v in ans["probabilities"].items()}, t
            )
            ans["probabilities"] = tempered
            if "choice" in ans:
                ans["choice"] = max(tempered, key=tempered.get)
            if "score" in ans:
                try:
                    ans["score"] = int(max(tempered, key=tempered.get))
                except (TypeError, ValueError):
                    pass
    if out.get("confidence"):
        for qid, ans in (out.get("answers") or {}).items():
            if "probabilities" in ans and qid in out["confidence"]:
                out["confidence"][qid] = max(ans["probabilities"].values())
            elif "noul" in ans and qid in out["confidence"]:
                out["confidence"][qid] = max(ans["noul"], 1.0 - ans["noul"])
        out["confidence_kind"] = f"{out.get('confidence_kind', 'top_prob')}+T={t}"
    return out


def config_hash(config: dict[str, Any]) -> str:
    """Stable hash of a backend configuration, used for run identity."""
    blob = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def new_result_row(
    case: Case,
    backend: str,
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run_id,
        "case_id": case.id,
        "suite": case.suite,
        "split": case.split,
        "language": case.language,
        "group": case.group,
        "backend": backend,
        "backend_config": config,
        "status": "pending",
        "answers": {},
        "confidence": {},
        "confidence_kind": "none",
        "raw": None,
        "latency_ms": {},
        "tokens": None,
        "error": None,
    }
    return row
