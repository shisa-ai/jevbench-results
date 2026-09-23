# JevBench v1.2 public items: shisa-ai/shisa-de-1

[Shisa DE-1](https://huggingface.co/shisa-ai/shisa-de-1) (Gemma 4 26B-A4B base
model with a trained readout) answered all 231 published JevBench v1.2
decisions on 2026-09-21: easy 48/48, standard 71/72, hard 72/111 (64.86%). Jev
1.13.0 scores 48/48, 71/72, 81/111 (72.97%) on the identical items, so DE-1 is
8.1 points below Jev on the hard tier and 16th of the 45 published rows ranked
by hard-tier accuracy.

## Results

Metrics are jevbench's. Accuracy counts a failed or unattempted decision as
wrong; there were none. Latency is the client-side wall time per decision,
including every request the readout made for it.

| run | tier | items | correct | accuracy | Brier | ECE | p50 ms | p95 ms | input tokens/decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DE-1 | easy | 48 | 48 | 100.00% | 0.000 | 0.004 | 19 | 22 | 149 |
| DE-1 | standard | 72 | 71 | 98.61% | 0.019 | 0.033 | 20 | 23 | 154 |
| DE-1 | hard | 111 | 72 | 64.86% | 0.431 | 0.083 | 52 | 156 | 1255 |
| Jev 1.13.0 (published) | easy | 48 | 48 | 100.00% | — | — | — | — | — |
| Jev 1.13.0 (published) | standard | 72 | 71 | 98.61% | — | — | — | — | — |
| Jev 1.13.0 (published) | hard | 111 | 81 | 72.97% | — | — | — | — | — |

The Jev rows are jevbench's published per-item outcomes; Jev was not run here.
The repository report, copied to `reports/JEVBENCH.md`, carries the full table of
all 45 published systems on the same 231 items, the timing breakdown, and the
per-item analysis. That report also documents a frozen-backbone arm, which this
bundle does not carry.

DE-1's hard-tier calibration is ECE 0.083 and Brier 0.431, with a mean total
variation distance of 0.209 to the 10 authored gold distributions (fidelity
79.1 on jevbench's 0–100 scale).

## Hard tier by family

| family | items | DE-1 | Jev 1.13.0 |
|---|---:|---:|---:|
| adversarial | 6 | 6 | 6 |
| ambiguous | 7 | 5 | 6 |
| judge_hard | 17 | 13 | 13 |
| long_policy | 19 | 10 | 12 |
| multi_hop | 18 | 13 | 15 |
| probability | 10 | 6 | 7 |
| routing_hard | 5 | 5 | 5 |
| temporal_numeric | 15 | 4 | 4 |
| tradeoff | 6 | 3 | 5 |
| trap | 8 | 7 | 8 |

DE-1's deficit against Jev is on the families that need a long state read
carefully and combined: `long_policy`, `multi_hop`, and `tradeoff`. It ties Jev
on `temporal_numeric` and `judge_hard` and leads on none of the ten families.
Families hold 5 to 19 items, so differences of one or two items are not
established.

## What was run

| Item | Value |
| --- | --- |
| Tasks | JevBench v1.2 public files: `easy.jsonl` 48, `original.jsonl` (standard tier) 72, `hard.jsonl` 111 |
| Task records | `fstandhartinger/jevbench` at `ee677f01f177102fa50144fa488dff1b5d34aba9` (v1.2.14); each tier's `dataset_hash` matches `datasets/manifest.json` |
| Runner and scoring | jevbench's own `Runner`, `scoring.score_task`, and `summarize` |
| Model side | `evals/harness/adapters/vllm_scoring.py`: restricted softmax over the option-letter slots of one vLLM completion, `systemone_json` scaffold, one top-logprobs request per question plus one full-prompt request per option letter outside the returned top 20 |
| DE-1 | `shisa-ai/shisa-de-1` on `http://127.0.0.1:8021`, vLLM tensor parallel 2 in bf16 on two H20-3e GPUs, tokenizer resolved from the same Hub repository |
| Requests | one at a time; 231 attempted, 231 answered, 0 failed, 231 strict-valid distributions, 0 renormalized |
| Run window | 2026-09-21T19:32:57Z to 19:33:13Z, 16,471 ms total wall, 14 ms warm-up |

DE-1 needed exactly one request per decision on all three tiers, so its
per-decision wall time is its server request time: p50 17.9 ms on easy, 18.6 ms
on standard, 48.6 ms on hard. Its largest single decision is the first hard item
at 745 ms, the cold prefill of a 3.7k-token policy item.

## Limitations

- **231 of 534 decisions.** The judge tier (146 items) and the held-out halves
  of the other tiers are not published, so the Intelligence axis cannot be
  computed on jevbench's tier weights and no JevBench Score is reported.
- **Different transport from the published rows.** Those ran through each
  author's own server via jevbench's `typesafe` adapter; this run uses this
  repository's vLLM letter-slot readout. The items, scoring code, and label sets
  are identical, but the readout is not the one JevBench used for its own rows.
- **Latency is not jevbench's Speed axis.** jevbench's Speed uses the 242-item
  standard-plus-judge run and multiplies self-hosted latency by 2, then adds
  0.15 s; the numbers here are raw client-side wall times with no adjustment.
- **One run, one option order.** Each item was answered once, in the order the
  records list, at temperature 0. No repeat runs and no option-order-reversed
  arm were taken.

## Files

| Path | Contents |
| --- | --- |
| `README.md` | This summary |
| `results/de-1-public/` | Run records: `{easy,standard,hard}.jsonl` (231 per-item records), the three `.summary.json` files, `manifest.json` (model, readout config, dataset hashes), `timing.json` (per-tier latency and readout request counts) |
| `reports/JEVBENCH.md` | Copy of the repository report: the 45-system leaderboard, timing analysis, hard tier by family, limitations, sources. It also documents the frozen-backbone arm, which this bundle does not carry |
| `repro/scripts/jevbench-de1.sh` | Serves the checkpoint from its Hub id and runs all three tiers |
| `repro/scripts/vendor-clone.sh` | Pins `vendor/jevbench` at the commit the run used, along with the other vendored checkouts |
| `repro/evals/harness/jevbench_public.py` | Driver: builds cases from jevbench's records, calls jevbench's runner and scoring, writes records, summaries, and the manifest; `timing` and `report` subcommands |
| `repro/evals/harness/adapters/vllm_scoring.py` | The letter-slot readout, including the fallback request for letters outside the returned top 20 |
| `repro/evals/harness/adapters/base.py` | Backend base class and registry used by the adapter |
| `repro/evals/harness/adapters/__init__.py`, `repro/evals/harness/format.py`, `repro/evals/harness/__init__.py` | Package files the readout imports |
| `repro/evals/harness/test_letter_slot_accounting.py` | Regression test for the fallback token accounting fixed on 2026-09-21 |
| `MANIFEST.md` | SHA-256 and repository source path for every file in this bundle |
| `MANIFEST.sha256` | The same hashes in `sha256sum` format, for verification |

## Reproduce

Prerequisites: two H20-3e GPUs, the conda environments `/root/miniforge3/envs/jev-gpu`
(Python 3.12) and `/root/miniforge3/envs/vllm-ds41f`, and `HF_HOME=/data/huggingface`
as the script defaults. Run from the repository root; the harness uses
package-relative imports and the vendored jevbench checkout, so the copies under
`repro/` are for inspection and verification, not a standalone tree.

```bash
bash scripts/vendor-clone.sh          # pins vendor/jevbench at ee677f01f177
scripts/jevbench-de1.sh               # serves shisa-ai/shisa-de-1 on GPUs 0-1, runs all three tiers
python -m evals.harness.jevbench_public timing \
    --run-dir evals/results/jevbench-de-1-public \
    --raw-dir /data/jevbench-runs/de-1-public/scratch
python -m evals.harness.jevbench_public report \
    --run-dir evals/results/jevbench-de-1-public
```

`scripts/jevbench-de1.sh` takes `SLUG`, `GPUS`, `PORT`, `TP`, `OUT`, and `TIERS`
overrides for a different checkpoint. Its docstring and the copied driver's
usage examples list the repository's two-arm study commands; this bundle carries
the DE-1 arm only.

The runner writes raw responses beside the per-item records under
`/data/jevbench-runs/`, outside the repository. Those raw bodies are not in this
bundle; without them the `timing` step leaves the server-request column empty
and everything else is still derived from the committed records.

## Provenance

- Source repository: `/root/research-jev-universal-classifiers` at commit
  `4b55ddf1903f0ac963baa1a10ea4d83348b7f751` (2026-09-23), branch `main`.
  Every copied file is byte-identical to that commit; `README.md`, `MANIFEST.md`,
  and `MANIFEST.sha256` were written for this bundle.
- JevBench checkout: `fstandhartinger/jevbench` at
  `ee677f01f177102fa50144fa488dff1b5d34aba9` (v1.2.14), protocol `jevbench::v1.2`.
- The run manifest records `cost_basis: local_gpu_no_provider_tariff` and
  `ledger_charged_usd: 0.0`.
- Verify the bundle with `sha256sum -c MANIFEST.sha256` from this directory, or
  compare the records against the repository with
  `diff -r results/de-1-public evals/results/jevbench-de-1-public`.
