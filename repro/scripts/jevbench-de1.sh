#!/usr/bin/env bash
# Run JevBench's public tiers against shisa-ai/shisa-de-1.
#
# The checkpoint is served exactly as its model card documents it, straight
# from the Hub id (vLLM resolves the tokenizer from the same repository), and
# answered through this repository's vLLM scoring readout. The 231 public
# JevBench items are easy 48, standard 72, hard 111; the judge tier and the
# held-out halves are not published.
#
#   scripts/jevbench-de1.sh                      # serve on GPUs 0-1, run all three tiers
#   scripts/jevbench-de1.sh --serve-only         # launch the server and stop
#   scripts/jevbench-de1.sh --no-serve --port 8021
#   scripts/jevbench-de1.sh --stop
#
# Output goes outside the repository (the runner writes raw responses beside
# the per-item records): /data/jevbench-runs/de-1-public by default. Copy the
# compact records and summaries into evals/results/ to commit them.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/root/miniforge3/envs/jev-gpu/bin/python
VLLM_PY=/root/miniforge3/envs/vllm-ds41f/bin/python
export HF_HOME=${HF_HOME:-/data/huggingface}

SLUG=${SLUG:-shisa-ai/shisa-de-1}
GPUS=${GPUS:-0,1}
TP=${TP:-2}
PORT=${PORT:-8021}
OUT=${OUT:-/data/jevbench-runs/de-1-public}
LOG=${LOG:-/tmp/jevbench-de-1-vllm.log}
TIERS=${TIERS:-easy,standard,hard}
EXTRA=${EXTRA:-}
SERVE=1 RUN=1 STOP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --no-serve) SERVE=0; shift;;
    --serve-only) RUN=0; shift;;
    --stop) STOP=1; shift;;
    --slug) SLUG="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --tp) TP="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --tiers) TIERS="$2"; shift 2;;
    --limit) EXTRA="$EXTRA --limit $2"; shift 2;;
    -h|--help) sed -n '2,18p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

if [ "$STOP" = 1 ]; then
  pkill -f "vllm.entrypoints.cli.main serve $SLUG" && echo "stopped $SLUG" || echo "no $SLUG server found"
  exit 0
fi

if [ "$SERVE" = 1 ]; then
  if curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "server already answering on :$PORT"
  else
    echo "launching vLLM: $SLUG tp=$TP gpus=$GPUS port=$PORT -> $LOG"
    CUDA_VISIBLE_DEVICES="$GPUS" LD_LIBRARY_PATH=/root/miniforge3/envs/vllm-ds41f/lib \
      "$VLLM_PY" -m vllm.entrypoints.cli.main serve "$SLUG" \
      --served-model-name "$SLUG" --host 127.0.0.1 --port "$PORT" \
      --tensor-parallel-size "$TP" --max-model-len 8192 \
      --gpu-memory-utilization 0.90 > "$LOG" 2>&1 &
    echo "waiting for http://127.0.0.1:$PORT/v1/models ..."
    for i in $(seq 1 120); do
      if curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "server ready after ~$((i * 5))s"
        break
      fi
      sleep 5
      if [ "$i" = 120 ]; then echo "server did not become ready; see $LOG" >&2; exit 1; fi
    done
  fi
  echo "--- /v1/models ---"
  curl -s "http://127.0.0.1:$PORT/v1/models" | "$PY" -c \
    'import json,sys; d=json.load(sys.stdin)["data"][0]; print(d["id"]); print(d["root"])'
fi

[ "$RUN" = 1 ] || exit 0

"$PY" -m evals.harness.jevbench_public run \
  --base-url "http://127.0.0.1:$PORT" \
  --model "$SLUG" --tokenizer "$SLUG" \
  --out-dir "$OUT" --tiers "$TIERS" \
  --run-label "${LABEL:-$SLUG}" --force $EXTRA || exit 1

echo
"$PY" -m evals.harness.jevbench_public report --run-dir "$OUT"
