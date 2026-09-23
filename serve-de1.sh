#!/usr/bin/env bash
# Serve shisa-ai/shisa-de-1 on stock vLLM.
#
# vLLM picks tensor parallel size and GPU memory utilization itself. Two cards
# at tensor parallel 2 is faster and is what the model card documents; it is an
# option, not a requirement. --max-model-len 8192 is set here because JevBench
# decisions fit well inside it. The readout runs in the client (see README.md),
# so vLLM only has to answer ordinary completions requests.
#
#   ./serve-de1.sh                    # one card -> http://127.0.0.1:8021
#   GPUS=0,1 TP=2 ./serve-de1.sh      # two cards, faster
#   ./serve-de1.sh --stop
#
# Env: SLUG, GPUS, TP, PORT, GPU_MEM, LOG, PIDFILE, VLLM_PY, EXTRA,
#      SKIP_MEM_CHECK
#
# VLLM_PY must be an interpreter that can import vllm. When it cannot, the
# script exits before launching and lists interpreters on this machine that can.
# That interpreter's own lib directory goes first in LD_LIBRARY_PATH, because
# conda environments carry a newer libstdc++ than the system one.
set -uo pipefail
cd "$(dirname "$0")"

SLUG=${SLUG:-shisa-ai/shisa-de-1}
GPUS=${GPUS:-0}
TP=${TP:-}
PORT=${PORT:-8021}
GPU_MEM=${GPU_MEM:-}
LOG=${LOG:-/tmp/jevbench-de-1-vllm.log}
PIDFILE=${PIDFILE:-/tmp/jevbench-de-1-vllm.pid}
VLLM_PY=${VLLM_PY:-python3}
EXTRA=${EXTRA:-}

STOP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --stop) STOP=1; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

if [ "$STOP" = 1 ]; then
  # Stop only this checkpoint's server: by the recorded pid when it is still the
  # process that was launched, otherwise by a command line that names the slug.
  if [ -f "$PIDFILE" ] && PID=$(cat "$PIDFILE") && kill -0 "$PID" 2>/dev/null \
     && tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null | grep -q "$SLUG"; then
    kill "$PID" && echo "stopped $SLUG (pid $PID)"
  elif pkill -f "vllm.entrypoints.cli.main serve $SLUG"; then
    echo "stopped $SLUG"
  else
    echo "no $SLUG server found"
  fi
  rm -f "$PIDFILE"
  exit 0
fi

HAS_VLLM='import importlib.util as u, sys; sys.exit(0 if u.find_spec("vllm") else 1)'
if ! "$VLLM_PY" -c "$HAS_VLLM" >/dev/null 2>&1; then
  echo "VLLM_PY=$VLLM_PY cannot import vllm; point it at an interpreter that can:" >&2
  for c in "${HOME:-/root}"/miniforge3/envs/*/bin/python /opt/conda/envs/*/bin/python; do
    [ -x "$c" ] || continue
    timeout 20 "$c" -c "$HAS_VLLM" >/dev/null 2>&1 && echo "  VLLM_PY=$c ./serve-de1.sh" >&2
  done
  exit 1
fi
ENV_LIB=$("$VLLM_PY" -c 'import os, sys; print(os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib"))')
[ -d "$ENV_LIB" ] && export LD_LIBRARY_PATH="$ENV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "server already answering on :$PORT"
else
  FLAGS=""
  [ -n "$TP" ] && FLAGS="$FLAGS --tensor-parallel-size $TP"
  [ -n "$GPU_MEM" ] && FLAGS="$FLAGS --gpu-memory-utilization $GPU_MEM"

  # Report free memory per selected card. With GPU_MEM set the script refuses to
  # launch when a card has less free memory than that asks for, because other
  # jobs' use counts against it and a short card fails inside vLLM minutes later
  # with "No available memory for the cache blocks". Without GPU_MEM these lines
  # are what to read if that error appears.
  TIGHT=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    for g in ${GPUS//,/ }; do
      MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$g" 2>/dev/null)
      [ -n "$MEM" ] || continue
      USED=${MEM%%,*}; TOTAL=${MEM##*,}
      FREE=$((TOTAL - USED))
      if [ -n "$GPU_MEM" ]; then
        TARGET=$(awk -v t="$TOTAL" -v m="$GPU_MEM" 'BEGIN{printf "%d", t*m}')
        echo "gpu $g: ${USED} MiB used of ${TOTAL} MiB; gpu_mem $GPU_MEM asks for ~${TARGET} MiB, ${FREE} MiB free"
        [ "$FREE" -lt "$TARGET" ] && TIGHT=1
      else
        echo "gpu $g: ${USED} MiB used of ${TOTAL} MiB, ${FREE} MiB free"
      fi
    done
    if [ "$TIGHT" = 1 ] && [ "${SKIP_MEM_CHECK:-0}" != 1 ]; then
      echo "not enough free memory on the selected cards; lower GPU_MEM, use GPUS=other,index, or set SKIP_MEM_CHECK=1 to launch anyway" >&2
      exit 1
    fi
  fi

  echo "launching vLLM: $SLUG gpus=$GPUS port=$PORT${TP:+ tp=$TP}${GPU_MEM:+ gpu_mem=$GPU_MEM} -> $LOG"
  CUDA_VISIBLE_DEVICES="$GPUS" "$VLLM_PY" -m vllm.entrypoints.cli.main serve "$SLUG" \
    --served-model-name "$SLUG" --host 127.0.0.1 --port "$PORT" \
    --max-model-len 8192 $FLAGS $EXTRA > "$LOG" 2>&1 &
  PID=$!
  echo "$PID" > "$PIDFILE"
  echo "pid $PID (stop with ./serve-de1.sh --stop)"
  echo "waiting for http://127.0.0.1:$PORT/v1/models (log: $LOG)"
  for i in $(seq 1 120); do
    if curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "server ready after ~$((i * 5))s"
      break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "server exited during startup; last lines of $LOG:" >&2
      tail -n 12 "$LOG" >&2
      rm -f "$PIDFILE"
      exit 1
    fi
    if [ "$i" = 120 ]; then
      echo "server did not become ready after 600s; see $LOG" >&2
      exit 1
    fi
    sleep 5
  done
fi

echo "--- /v1/models ---"
curl -s "http://127.0.0.1:$PORT/v1/models" | "$VLLM_PY" -c \
  'import json,sys; d=json.load(sys.stdin)["data"][0]; print(d["id"]); print(d.get("root"))'
