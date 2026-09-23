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
repository does not carry.

DE-1's hard-tier calibration is ECE 0.083 and Brier 0.431, with a mean total
variation distance of 0.209 to the 10 authored gold distributions (fidelity
79.1 on jevbench's 0–100 scale, which is 100 × (1 − mean TVD)).

Each `*.summary.json` records the numbers behind this table: `ece`,
`brier_mean`, and, on the hard tier, `mean_tvd_to_gold` with
`gold_probability_items`. Every per-item record carries the option distribution
under `probs`, so both can be recomputed from the records alone using jevbench's
definitions: Brier is the multi-class sum over the option set,
`sum_k (p_k - y_k)^2`, with the two-option convention for binary items, and ECE
is top-label confidence in 10 equal-width bins. One trap when recomputing:
`score` items carry their expected level as an integer, and jevbench compares
`str(expected)` against the level labels. The gold distributions are jevbench's
and are not redistributed here; they arrive with the task records that
`run-jevbench.sh` fetches at the pin, and the published probability items are
the 10 that both runs scored.

## Run the model

The checkpoint is served by stock vLLM: no patched kernels, no custom server,
no inference code of ours on the server side. One command starts it on one card
in bf16, from the Hub id:

```sh
./serve-de1.sh
```

Two cards at tensor parallel 2 is the faster setup, and the one the model card
and the committed run used:

```sh
GPUS=0,1 TP=2 ./serve-de1.sh
```

| Env | Default | Meaning |
| --- | --- | --- |
| `GPUS` | `0` | cards to use |
| `TP` | unset | `--tensor-parallel-size`; unset leaves vLLM's choice |
| `GPU_MEM` | unset | `--gpu-memory-utilization`; unset leaves vLLM's default of 0.9 of each card |
| `PORT` | `8021` | endpoint |
| `VLLM_PY` | `python3` | interpreter that can `import vllm`; when it cannot, the script exits before launching and lists interpreters on this machine that can |
| `LOG` | `/tmp/jevbench-de-1-vllm.log` | server log |
| `PIDFILE` | `/tmp/jevbench-de-1-vllm.pid` | recorded server pid |
| `EXTRA` | empty | extra vLLM flags |

Memory: the weights are about 50 GB in bf16, and the KV cache needs room on top
of that. The script sets `--max-model-len 8192`, which bounds the cache for
decisions of this size; one card holds the whole checkpoint, and two cards split
the weights. `GPU_MEM` sizes the cache, not the model, so it does not change the
numbers. The committed run used `--gpu-memory-utilization 0.90` on idle cards.

The script waits for `/v1/models`, prints the served id, and records the server
pid. `./serve-de1.sh --stop` stops that pid, and only that pid; if the pid is
gone it falls back to matching the command line of this checkpoint, never
another model's server. It prints each selected card's used and total memory,
and with `GPU_MEM` set it refuses to launch when a card is short of the target,
because other jobs' use counts against `--gpu-memory-utilization` and a short
card fails inside vLLM minutes later. `SKIP_MEM_CHECK=1` launches anyway. When
the server dies during startup, the script exits with the last log lines.

Needs: a vLLM build that knows the Gemma 4 architecture, on an interpreter that
can `import vllm` (`VLLM_PY`), and `transformers` on the client for the
tokenizer.

## Run JevBench

```sh
./run-jevbench.sh
```

The script fetches jevbench at commit `ee677f01f177` into `vendor/` (the
checkout the committed run used), checks that the endpoint answers, runs the
three public tiers through the readout below, and prints the report tables.

| Env | Default | Meaning |
| --- | --- | --- |
| `PY` | `python3` | interpreter with `httpx` and `transformers` |
| `BASE_URL` | `http://127.0.0.1:8021` | endpoint from `serve-de1.sh` |
| `MODEL`, `TOKENIZER` | `shisa-ai/shisa-de-1` | served id, tokenizer source |
| `OUT` | `runs/de-1-public` | records, summaries, manifest |
| `SCRATCH` | `$OUT/scratch` | raw responses and the ledger |
| `TIERS` | `easy,standard,hard` | which tiers to run |
| `LABEL` | `$MODEL` | `run` field written into every record |
| `LIMIT` | unset | cap items per tier, for a smoke test |
| `FORCE` | `0` | overwrite a run directory that already holds results |
| `JEVBENCH_DIR` | `vendor/jevbench` | existing checkout, skips the clone |

A fresh run lands in `runs/de-1-public` (git-ignored). The committed run of
2026-09-21 is `results/de-1-public`, and reproducing it in place takes
`OUT=results/de-1-public FORCE=1 ./run-jevbench.sh`. A smoke test is
`LIMIT=4 TIERS=easy ./run-jevbench.sh`.

A fresh run on 2026-09-23 through these scripts, against a new server started
with `GPU_MEM=0.5`, reproduced the committed run: 231 of 231 predicted choices
and correctness flags identical, tier accuracies unchanged, per-item
probabilities equal within 8.3e-04 (229 of 231 within 1e-09), and p50 latency
within 1 ms on every tier.

The driver on its own, when one flag needs changing:

```sh
python jevbench_public.py run --base-url http://127.0.0.1:8021 \
    --model shisa-ai/shisa-de-1 --tokenizer shisa-ai/shisa-de-1 \
    --out-dir runs/de-1-public --tiers easy,standard,hard
python jevbench_public.py report --run-dir results/de-1-public
python jevbench_public.py timing --run-dir runs/de-1-public \
    --raw-dir runs/de-1-public/scratch
```

`report` prints the markdown tables for one or more run directories. `timing`
writes `timing.json` beside the records; the server-request split needs the
run's scratch directory, which holds the raw responses.

The client side needs `httpx` and `transformers`:

```sh
python3 -m pip install httpx transformers
```

## How the readout works

DE-1 generates no free-form text. One question becomes one request, and the
answer is read from the logprobs at the first position the model would generate.

1. **Render.** The state, the question, and the lettered options go into a JSON
   payload (`evidence`, `criterion`, `options`) and through the checkpoint's own
   chat template, behind a system line that asks for one uppercase letter. The
   prompt is built client-side by the tokenizer.
2. **Request.** `POST /v1/completions` with `max_tokens: 1`, `temperature: 0`,
   and `logprobs: 20`. `top_logprobs[0]` is the distribution over the first
   generated token, which is where the answer letter goes. Every option letter
   is a single token for this tokenizer (A → 236776, B → 236799, C → 236780),
   and the readout verifies that `prompt + letter` tokenizes as the prompt plus
   exactly that slot token, so the answer boundary cannot shift.
3. **Fallback.** A letter outside the returned top 20 costs one more request,
   `prompt + letter` with `prompt_logprobs: 0`. That re-sends the whole prompt,
   so it is a real prefill and its prompt tokens are counted in the run's usage.
   DE-1's letters were always inside the top 20: exactly one request per
   decision on all three tiers, no fallback.
4. **Normalize.** A softmax over the letters' logprobs is the distribution. The
   argmax is the answer and its probability is the confidence.

`readout/vllm_scoring.py` is that code. `python test-query.py` runs the same
steps against a live server and prints the prompt, the token slots, the returned
top 20, the distribution, and the equivalent `curl` command:

```
1. The prompt the model sees
2. Option letters as single tokens (the answer slots)
3. One completions request: max_tokens 1, temperature 0, logprobs 20
4. The distribution over the option letters
5. The same path as readout/vllm_scoring.py (evaluate)
6. The same request without Python
```

`python -m unittest readout.test_letter_slot_accounting` pins the request count
and the fallback token accounting.

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

## Limitations

- **231 of 534 decisions.** The judge tier (146 items) and the held-out halves
  of the other tiers are not published, so the Intelligence axis cannot be
  computed on jevbench's tier weights and no JevBench Score is reported.
- **Different transport from the published rows.** Those ran through each
  author's own server via jevbench's `typesafe` adapter; this run uses this
  repository's vLLM letter-slot readout. The items, scoring code, and label sets
  are identical, but the readout is not the one JevBench used for its own rows.
  jevbench ships no adapter that reads token logprobs, so DE-1 cannot run
  through its stock adapters at all.
- **Latency is not jevbench's Speed axis.** jevbench's Speed uses the 242-item
  standard-plus-judge run and multiplies self-hosted latency by 2, then adds
  0.15 s; the numbers here are raw client-side wall times with no adjustment.
- **One run, one option order.** Each item was answered once, in the order the
  records list, at temperature 0. No repeat runs and no option-order-reversed
  arm were taken.

## Files

| Path | Contents |
| --- | --- |
| `README.md` | This document |
| `results/de-1-public/` | The committed run: `{easy,standard,hard}.jsonl` (231 per-item records with the option distribution and the readout's token accounting), their `.summary.json` (accuracy, Brier, ECE, TVD to gold), `manifest.json` (model, readout config, dataset hashes), `timing.json` |
| `reports/JEVBENCH.md` | Full report: the 45-system leaderboard, timing analysis, hard tier by family, limitations, sources. It also documents the frozen-backbone arm, which this repository does not carry |
| `serve-de1.sh` | Serve the checkpoint on stock vLLM |
| `run-jevbench.sh` | Fetch jevbench at the pin, run the three public tiers, print the report |
| `test-query.py` | One question against a live server, with every readout step printed |
| `jevbench_public.py` | Driver: jevbench task → case → readout → jevbench scoring; `run`, `report`, `timing` |
| `readout/vllm_scoring.py` | The letter-slot readout |
| `readout/format.py`, `readout/backend.py` | Case and question dataclasses, backend base class |
| `readout/test_letter_slot_accounting.py` | Regression test for the request and token accounting |

## Licensing

This repository is MIT ([LICENSE](LICENSE)). The JevBench v1.2 task records and
scoring code it reports on are MIT, from
[`fstandhartinger/jevbench`](https://github.com/fstandhartinger/jevbench); those
task records are not redistributed here, and the per-item records contain task
ids and this run's outputs. The `shisa-ai/shisa-de-1` checkpoint is Apache-2.0,
the license its model card declares, and is distributed separately at
[`huggingface.co/shisa-ai/shisa-de-1`](https://huggingface.co/shisa-ai/shisa-de-1).

## Provenance

- Source repository: `/root/research-jev-universal-classifiers` at commit
  `4b55ddf1903f0ac963baa1a10ea4d83348b7f751` (2026-09-23), branch `main`.
  `results/de-1-public/` and `reports/JEVBENCH.md` are byte-identical copies of
  it. `readout/`, `jevbench_public.py`, and the test are adapted from it: import
  paths, a root-level entry point, and a docstring written for this repository.
  `README.md`, `LICENSE`, `serve-de1.sh`, `run-jevbench.sh`, and `test-query.py`
  were written for this repository.
- JevBench checkout: `fstandhartinger/jevbench` at
  `ee677f01f177102fa50144fa488dff1b5d34aba9` (v1.2.14), protocol `jevbench::v1.2`.
- The run manifest records `cost_basis: local_gpu_no_provider_tariff` and
  `ledger_charged_usd: 0.0`.
- Compare the records against the source repository with
  `diff -r results/de-1-public evals/results/jevbench-de-1-public`.
