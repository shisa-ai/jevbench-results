# JevBench public items: shisa-ai/shisa-de-1

DE-1 answered all 231 published JevBench v1.2 decisions: easy 48/48, standard
71/72, hard 72/111 (64.86%). On the identical items Jev 1.13.0 scores 48/48,
71/72, 81/111 (72.97%). The two systems agree on the easy and standard tiers;
DE-1 is 8.1 points below Jev on the hard tier, and 16th of the 45 published
rows ranked by hard-tier accuracy.

The frozen Gemma 4 backbone answered the same items through the same readout
and scored 48/48, 72/72, 71/111 (63.96%). DE-1's trained readout therefore
changes accuracy by at most one item on any tier, and what it does change is
the calibration: on the hard tier ECE falls from 0.315 to 0.083, the Brier
score from 0.664 to 0.431, and mean distance to the 10 authored gold
distributions from 0.421 to 0.209.

The checkpoint was served from its Hub id, `vllm serve shisa-ai/shisa-de-1`
with tensor parallel 2 in bf16 on two H20-3e GPUs, which is the command the
model card documents. This is the first inference run against the published
copy; the 2026-09-21 release verified file hashes but ran no inference.

## What was run

| Item | Value |
| --- | --- |
| Tasks | JevBench v1.2 public files: `easy.jsonl` 48, `original.jsonl` (standard tier) 72, `hard.jsonl` 111 |
| Task records | `fstandhartinger/jevbench` at `ee677f01f177` (v1.2.14); each tier's `dataset_hash` matches `datasets/manifest.json` |
| Runner and scoring | jevbench's own `Runner`, `scoring.score_task`, and `summarize` |
| Model side | [`evals/harness/adapters/vllm_scoring.py`](../harness/adapters/vllm_scoring.py): restricted softmax over the option-letter slots, `systemone_json` scaffold, one top-logprobs request per question plus one full-prompt request per option letter outside the returned top 20 |
| DE-1 endpoint | `http://127.0.0.1:8021`, `--served-model-name shisa-ai/shisa-de-1`, tokenizer resolved from the same Hub repository |
| Base endpoint | `http://127.0.0.1:8022`, `google/gemma-4-26B-A4B-it`, the same vLLM build and flags |
| Requests | one at a time; 462 attempted across both runs, 462 answered, 0 failed, 462 strict-valid distributions, 0 renormalized |
| Run window | 2026-09-21T19:32:57Z to 19:33:13Z (DE-1), 16,471 ms total wall; 19:37:33Z to 19:38:01Z (base), 28,758 ms |

## Tier results

Metrics are jevbench's. Accuracy counts a failed or unattempted decision as
wrong; there were none. Latency is the client-side wall time per decision: the
render, every request the readout made for that decision, and the parse. The
Timing section separates the first request from it.

| run | tier | items | correct | accuracy | Brier | ECE | p50 ms | p95 ms | input tokens/decision | input tokens/decision, all requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DE-1 | easy | 48 | 48 | 100.00% | 0.000 | 0.004 | 19 | 22 | 149 | 149 |
| DE-1 | standard | 72 | 71 | 98.61% | 0.019 | 0.033 | 20 | 23 | 154 | 154 |
| DE-1 | hard | 111 | 72 | 64.86% | 0.431 | 0.083 | 52 | 156 | 1255 | 1255 |
| Gemma 4 (base) | easy | 48 | 48 | 100.00% | 0.000 | 0.000 | 91 | 118 | 149 | 545 |
| Gemma 4 (base) | standard | 72 | 72 | 100.00% | 0.000 | 0.000 | 44 | 105 | 154 | 373 |
| Gemma 4 (base) | hard | 111 | 71 | 63.96% | 0.664 | 0.315 | 62 | 161 | 1255 | 1350 |

The two token columns differ only for the base run: the records' `usage.input_tokens`
sums the first request of each decision, while the second column adds the prompt
tokens of the fallback requests the readout made (Timing).

Hard-tier fidelity to the 10 published gold distributions is a mean total
variation distance of 0.209 for DE-1 (fidelity 79.1 on jevbench's 0–100 scale)
and 0.421 for the base (57.9).

The standard-tier miss for DE-1 is `original-adequacy-03-1`, the one item the
base answered that DE-1 did not.

## Timing

Per-decision milliseconds from the committed records, plus the server-side
request time and the readout's request count recovered from each run's raw
responses. `wall` is the first to last record timestamp; `mean`, `p50`, `p95`,
`max`, and `first` are over the per-decision latencies the runner recorded;
`server request p50` is the `latency_ms.request` the readout measured around its
first HTTP call; `req/decision` counts every request a decision took, and
`fallbacks` counts the decisions that needed one or more. The committed
machine-readable form is `timing.json` in each run directory, including the
per-missing-letter breakdown.

| run | tier | n | wall | mean | p50 | p95 | max | first | server request p50 | req/decision | fallbacks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DE-1 | easy | 48 | 952 | 20.5 | 19.4 | 21.9 | 74 | 74 | 17.9 | 1.00 | 0/48 |
| DE-1 | standard | 72 | 1488 | 19.8 | 20.0 | 22.9 | 23 | 19 | 18.6 | 1.00 | 0/72 |
| DE-1 | hard | 111 | 7782 | 75.5 | 51.9 | 156.5 | 745 | 745 | 48.6 | 1.00 | 0/111 |
| Gemma 4 (base) | easy | 48 | 3680 | 224.0 | 90.9 | 118.2 | 7118 | 7118 | 18.2 | 3.48 | 41/48 |
| Gemma 4 (base) | standard | 72 | 3688 | 50.7 | 44.2 | 105.4 | 151 | 42 | 18.5 | 2.29 | 49/72 |
| Gemma 4 (base) | hard | 111 | 8668 | 78.6 | 62.0 | 160.5 | 293 | 205 | 48.1 | 1.22 | 13/111 |

- DE-1: 16,471 ms total wall including warm-up and tokenizer load, 10,787 ms of
decision latency, 14 ms warm-up.
- Gemma 4 (base): 28,758 ms total wall, 23,122 ms of decision latency, 25 ms
warm-up.

Per request the two runs are equally fast; they differ in how many requests a
decision takes. Both servers answer an easy or standard decision in about
18 ms p50 and a hard one in about 48 ms p50, with hard-tier p95 near 140 ms for
the 2,000–4,000-token policy items. The trained readout does not change serving
speed, which is expected for the same architecture at the same precision and
tensor parallelism.

The readout asks for the top 20 next-token logprobs and reads the option letters
out of that list. A candidate letter outside the top 20 costs one more request,
for `prompt + letter` with `prompt_logprobs`; the adapter's boundary check pins
appending a letter to exactly one token, so each fallback is another full prefill
of the same prompt. DE-1's letters were always in the top 20: exactly one request
per decision on all three tiers, with no fallback. The base fell back on 41 of 48
easy decisions (2.48 letters missing on average, 3.48 requests per decision), 49
of 72 standard decisions (1.29 missing, 2.29 requests), and 13 of 111 hard
decisions (0.22 missing, 1.22 requests). Both runs saw the same option counts per
tier (easy has the most five-option questions, 27 of 48; hard is dominated by
two-option questions, 38 of 111), so the fallback difference is the model's
letter distribution, not the item mix.

The `request` timer wraps only the first request, so the fallback prefills appear
in a decision's total latency but not in the server-request column. That is the
whole of the base run's wall-time gap. Grouping the base run's decisions by how
many letters were missing shows a flat cost per fallback: the mean time outside
the first request rises from 0.9 ms with no missing letter to 96.6 ms with four
on the easy tier, and from 1.0 ms to 130.5 ms with five on the standard tier,
roughly 20–30 ms per missing letter. On the hard tier only 13 of 111 decisions
fell back and the tier's prompts are 1,255 tokens on average, so the same
per-letter figure is not separable there; its median time outside the first
request is 5.0 ms against DE-1's 3.4 ms. The easy tier is the extreme case, a p50
of 71.9 ms outside the first request against DE-1's 1.3 ms. An earlier version of
this section attributed that gap to warm-up and unidentified client cost; the
base server's first decision did take 7.0 s, but the rest of the tier's cost is
fallback prefills.

Two consequences:

- **The base's input tokens depend on which count is used.** The records'
`usage.input_tokens` sums only the first request of each decision, so for the
base it under-reports prompt tokens per decision: 148.5 against 545.1 on easy,
153.8 against 373.5 on standard, 1,254.7 against 1,350.3 on hard. DE-1's two
figures are equal. The adapter's fallback token accounting was fixed on
2026-09-21 and the regression test is
[`test_letter_slot_accounting.py`](../harness/test_letter_slot_accounting.py);
these records were not regenerated, so both columns are reported above.
- **The gap is a property of the readout, not of the base model.** The same base
would need one request per decision with a larger `logprobs` value, with
`prompt_logprobs` on the first request, or with prefix caching enabled, since
each fallback re-sends a prompt that shares its entire prefix. None of those
configurations was measured here.

DE-1's own largest single value is its first hard decision at 745 ms, the cold
prefill of a 3.7k-token policy item.

## Hard tier by family

| family | DE-1 | Gemma 4 (base) | Jev 1.13.0 | jeff |
|---|---:|---:|---:|---:|
| adversarial | 6/6 | 6/6 | 6/6 | 6/6 |
| ambiguous | 5/7 | 4/7 | 6/7 | 0/7 |
| judge_hard | 13/17 | 14/17 | 13/17 | 8/17 |
| long_policy | 10/19 | 9/19 | 12/19 | 2/19 |
| multi_hop | 13/18 | 13/18 | 15/18 | 7/18 |
| probability | 6/10 | 5/10 | 7/10 | 4/10 |
| routing_hard | 5/5 | 5/5 | 5/5 | 3/5 |
| temporal_numeric | 4/15 | 4/15 | 4/15 | 6/15 |
| tradeoff | 3/6 | 3/6 | 5/6 | 3/6 |
| trap | 7/8 | 8/8 | 8/8 | 4/8 |

DE-1's deficit against Jev is on the families that need a long state read
carefully and combined: `long_policy` (10/19 against 12/19), `multi_hop`
(13/18 against 15/18), and `tradeoff` (3/6 against 5/6). It ties Jev on
`temporal_numeric` (4/15) and `judge_hard` (13/17) and leads on none of the ten
families. This is the same shape as `kev-transfer-v1`, the held-out family
[DE-1.md](../../DE-1.md#known-gaps) already records as unsolved: both suites
reward combining several steps over a long state, and a prompt-scaffold A/B
moved nothing there.

## Published systems on the same 231 public items

Restricted to the 231 public items these runs used, from jevbench's
`results/v1.2/jevbench-v1.2-per-task.json`. The last two columns are jevbench's
own numbers over all 220 hard decisions and its composite score, which neither
local run can be placed on: the judge tier and the held-out halves are not
published.

| system | easy | standard | hard | official hard (220) | official JevBench Score |
|---|---:|---:|---:|---:|---:|
| **DE-1** (this run) | 100.00% | 98.61% | 64.86% | not run (held out) | not computed (judge tier unpublished) |
| **Gemma 4 (base)** (this run) | 100.00% | 100.00% | 63.96% | not run (held out) | not computed (judge tier unpublished) |
| DeepSeek V4.1 Flash (thinking default) | 100.00% | 98.61% | 96.40% | 95.0% | 57.8 (#38) |
| GPT-5.6 Luna (low reasoning effort) | 100.00% | 97.22% | 96.40% | 94.5% | 66.2 (#21) |
| OpenJev (thinking, BF16) | 100.00% | 100.00% | 76.58% | 78.2% | 60.6 (#36) |
| djev (thinking) | 95.83% | 98.61% | 76.58% | 77.7% | 63.5 (#28) |
| reflex-27b (Qwen3.8-27B) | 100.00% | 95.83% | 75.68% | 75.9% | 64.2 (#24) |
| Gemini 3.1 Flash-Lite | 100.00% | 98.61% | 73.87% | 75.0% | 60.9 (#35) |
| SimpleJev Qwen3.8-27B | 100.00% | 97.22% | 73.87% | 75.0% | 67.3 (#15) |
| Jev 1.13.0 (TypeSafe AI) | 100.00% | 98.61% | 72.97% | 74.1% | 75.4 (#1) |
| Winnow-12B Q8 | 100.00% | 95.83% | 72.97% | 70.9% | 72.5 (#5) |
| openjev-sglang (Qwen3.6-35B-A3B on SGLang) | 100.00% | 94.44% | 72.97% | 71.4% | 66.3 (#19) |
| LitJev (Qwen3.8-27B) | 100.00% | 98.61% | 72.07% | 73.2% | 63.7 (#27) |
| classifier.dev (fast tier) | 100.00% | 98.61% | 70.27% | 70.5% | 84.8 |
| djev (Maisa, diffusion-gemma) | 100.00% | 98.61% | 67.57% | 69.5% | 74.3 (#3) |
| decider-35b-a3b (Mapika) | 100.00% | 97.22% | 66.67% | 65.5% | 68.9 (#11) |
| SimpleJev Qwen3.6-35B-A3B | 100.00% | 93.06% | 65.77% | 66.4% | 63.8 (#26) |
| OpenJev (DiffusionGemma 26B-A4B NVFP4, razorback16) | 100.00% | 97.22% | 63.96% | 65.5% | 67.7 (#14) |
| Bespoke Nimble 9B (Bespoke Labs) | 100.00% | 93.06% | 62.16% | 65.5% | 61.8 (#34) |
| SemIf, formerly OpenJev (Qwen3.5-4B, TheoLeeCJ) | 100.00% | 98.61% | 61.26% | 59.5% | 74.7 (#2) |
| jqv (Qwen3-32B zero-shot) | 100.00% | 95.83% | 61.26% | 64.5% | 70.1 (#9) |
| reflex 4B (kshetrajna12) | 100.00% | 94.44% | 60.36% | 63.2% | 71.7 (#6) |
| jev-local (Qwen3.5-9B) | 100.00% | 83.33% | 58.56% | 59.1% | 63.8 (#25) |
| open-alternative-jev (Qwen3.5-4B, IkerMoel) | 100.00% | 83.33% | 56.76% | 56.8% | 69.8 (#10) |
| decider-2b (Mapika) | 100.00% | 84.72% | 49.55% | 47.3% | 64.6 (#22) |
| decision-machine-1 (milliseconds.ai) | 100.00% | 75.00% | 48.65% | 46.8% | 71.5 (#7) |
| system-one (Qwen3-8B, Sean Goedecke) | 100.00% | 88.89% | 48.65% | 50.0% | 56.6 (#39) |
| system-one-open (Gemma 4 E2B LoRA on an L4) | 100.00% | 93.06% | 48.65% | 49.1% | 68.9 (#12) |
| kev 8B (research preview) | 100.00% | 93.06% | 45.05% | 47.3% | 58.3 (#37) |
| kev 0.6B (research preview) | 100.00% | 80.56% | 43.24% | 40.0% | 66.7 (#18) |
| Qwen3.8 27B (Chutes TEE) | 100.00% | 98.61% | 42.34% | 21.4% | 25.5 partial |
| smalljev semantic-v9 | 97.92% | 68.06% | 39.64% | 38.2% | 62.4 (#31) |
| jeff (Logan Markewich, GLiFormer 400M) | 100.00% | 75.00% | 38.74% | 37.7% | 66.9 (#17) |
| open-jev-deberta-v3-large (local CPU) | 100.00% | 43.06% | 37.84% | 36.4% | 64.6 (#23) |
| openJev Verdict (heman10x, ModernBERT-base 151M) | 85.42% | 62.50% | 37.84% | 38.2% | 66.2 (#20) |
| GLiNER2 (Fastino, gliner2.5-base) | 97.92% | 63.89% | 36.94% | 36.4% | 53.0 (#40) |
| GLiNER2 large (Fastino) | 100.00% | 58.33% | 36.94% | 36.4% | 50.5 (#41) |
| kev 4B (research preview) | 100.00% | 88.89% | 36.94% | 42.3% | 62.2 (#32) |
| openJev Verdict 1.4 | 87.50% | 69.44% | 36.94% | 37.7% | 72.5 (#4) |
| Laya (Convai Innovations, ModernBERT-large 421M) | 95.83% | 69.44% | 35.14% | 34.1% | 70.1 (#8) |
| OpenDecision (ModernBERT-large zero-shot) | 87.50% | 59.72% | 34.23% | 33.2% | 67.0 (#16) |
| Certo v1 (AltSlate Labs) | 25.00% | 33.33% | 33.33% | 31.8% | 68.2 (#13) |
| GLiNER2.5 multi (Fastino, 287M) | 91.67% | 44.44% | 33.33% | 37.7% | 63.1 (#30) |
| GLiNER2.5 small (Fastino, 74M) | 85.42% | 41.67% | 31.53% | 33.2% | 62.1 (#33) |
| kev 0.5B | 95.83% | 48.61% | 29.73% | 30.9% | 63.1 (#29) |
| Needle 3 (Cactus, 2-bit, local CPU) | 47.92% | 16.67% | 15.32% | 7.7% | 16.8 partial |
| Needle 3, options as tools (post-hoc adapter mode) | 66.67% | 26.39% | 0.00% |  | 19.2 partial |

The rows above DE-1 on the hard tier are three frontier models that generate
their answers (DeepSeek V4.1 Flash, GPT-5.6 Luna, Gemini 3.1 Flash-Lite), two
decision models run in a thinking mode, Jev 1.13.0, and decision models between
12B and 35B parameters. `classifier.dev` carries no rank because it serves
Jev's own model. DE-1 and the base sit level with OpenJev's DiffusionGemma
26B-A4B run at 63.96%.

## Limitations

- **231 of 534 decisions.** The judge tier (146 items) and the held-out halves
  of the other tiers are not published, so the Intelligence axis cannot be
  computed on jevbench's tier weights and no JevBench Score is reported.
- **Different transport from the published rows.** Those ran through each
  author's own server via jevbench's `typesafe` adapter; these runs use this
  repository's vLLM letter-slot readout. The items, the scoring code, and the
  label sets are identical, but the readout is not the one JevBench used for
  its own rows.
- **Latency is not the Speed axis, and the two runs' wall times are not
  comparable per decision.** jevbench's Speed uses the 242-item standard-plus-judge run and
  multiplies self-hosted latency by 2, then adds 0.15 s; the numbers here are
  raw client-side wall times with no such adjustment. Between the two runs,
  each server request takes the same time; the base run's larger wall times are
  its fallback requests, one per option letter outside the top 20. See Timing.
- **One run per system, one option order.** Each item was answered once, in the
  order the records list, at temperature 0. No repeat runs and no
  option-order-reversed arm were taken.
- **Hard-tier families are jevbench's, not ours.** The family table is
  jevbench's `family` field; the per-family counts are 6 to 19 items, so
  differences of one or two items are not established.
- **The one-item differences are noise.** The standard-tier difference between
  DE-1 and the base is a single item out of 72; the accuracy claim in the
  opening paragraph is that they are level, not that either leads.

## Reproduction

```bash
bash scripts/vendor-clone.sh          # pins vendor/jevbench at ee677f01f177
scripts/jevbench-de1.sh               # serves shisa-ai/shisa-de-1 on GPUs 0-1, runs all three tiers
SLUG=google/gemma-4-26B-A4B-it GPUS=2,3 PORT=8022 scripts/jevbench-de1.sh
python -m evals.harness.jevbench_public timing \
    --run-dir evals/results/jevbench-de-1-public \
    --raw-dir /data/jevbench-runs/de-1-public/scratch \
    --run-dir evals/results/jevbench-gemma4-base-public \
    --raw-dir /data/jevbench-runs/gemma4-base-public/scratch
python -m evals.harness.jevbench_public report \
    --run-dir evals/results/jevbench-de-1-public \
    --run-dir evals/results/jevbench-gemma4-base-public
```

The timing step needs each run's scratch directory, which holds the raw
responses; without it the server-request column is left empty and everything
else is derived from the committed records.

The per-item records, per-tier summaries, and run manifests are committed under
[`evals/results/jevbench-de-1-public/`](../results/jevbench-de-1-public/) and
[`evals/results/jevbench-gemma4-base-public/`](../results/jevbench-gemma4-base-public/);
the raw request and response bodies stay outside the repository in each run's
scratch directory.

## Sources

| ID | Source and evidence type | Retrieved |
| --- | --- | --- |
| J1 | [`fstandhartinger/jevbench`](https://github.com/fstandhartinger/jevbench) at `ee677f01f177`: inspected harness, task records, and `datasets/manifest.json` | 2026-09-21 |
| J2 | [`datasets/public/`](https://github.com/fstandhartinger/jevbench/tree/ee677f01f177/datasets/public): the 231 public task records; MIT | 2026-09-21 |
| J3 | `results/v1.2/jevbench-v1.2-per-task.json` and `jevbench-v1.2-results.json` in the same checkout: the site's published per-item outcomes and scores, reported rather than reproduced here | 2026-09-21 |
| J4 | [Run records](../results/jevbench-de-1-public/) and [base run records](../results/jevbench-gemma4-base-public/): 462 per-item records, six tier summaries, two `timing.json` files with the readout request counts, and both run manifests from these measurements | 2026-09-21 |
| J5 | [vllm_scoring adapter](../harness/adapters/vllm_scoring.py), [letter-slot accounting test](../harness/test_letter_slot_accounting.py), and [jevbench_public driver](../harness/jevbench_public.py): the readout, its fallback accounting, and the jevbench bridge, inspected | 2026-09-21 |
| J6 | [`shisa-ai/shisa-de-1`](https://huggingface.co/shisa-ai/shisa-de-1) model card: the launch command used for the DE-1 server | 2026-09-21 |
| J7 | [DE-1 reference](../../DE-1.md), `Known gaps` items 2 and 4: the checkpoint's own suite results, including `kev-transfer-v1` | 2026-09-21 |
