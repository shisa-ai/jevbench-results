#!/usr/bin/env bash
# Run JevBench's public tiers against a served endpoint.
#
# Fetches jevbench at the pinned commit, checks the endpoint, runs the three
# public tiers through the readout in readout/ (see README.md), then writes the
# report and timing.json. Output lands in runs/de-1-public; the committed run
# of 2026-09-21 is in results/de-1-public.
#
#   ./serve-de1.sh                  # in another shell
#   ./run-jevbench.sh
#   LIMIT=4 TIERS=easy ./run-jevbench.sh                  # smoke test
#   OUT=results/de-1-public FORCE=1 ./run-jevbench.sh      # overwrite the committed run
#
# Env: PY, BASE_URL, MODEL, TOKENIZER, OUT, SCRATCH, TIERS, LIMIT, LABEL,
#      FORCE, JEVBENCH_DIR, JEVBENCH_PIN
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
BASE_URL=${BASE_URL:-http://127.0.0.1:8021}
MODEL=${MODEL:-shisa-ai/shisa-de-1}
TOKENIZER=${TOKENIZER:-$MODEL}
OUT=${OUT:-runs/de-1-public}
SCRATCH=${SCRATCH:-$OUT/scratch}
TIERS=${TIERS:-easy,standard,hard}
PIN=${JEVBENCH_PIN:-2fa63fa3226cb369795525ed011800f57dcbd894}
DEST=${JEVBENCH_DIR:-vendor/jevbench}

FETCH=1 RUN=1
while [ $# -gt 0 ]; do
  case "$1" in
    --fetch-only) RUN=0; shift;;
    --no-fetch) FETCH=0; shift;;
    -h|--help) sed -n '2,15p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

if [ ! -d "$DEST/jevbench" ]; then
  if [ "$FETCH" = 1 ]; then
    echo "cloning jevbench @ ${PIN:0:9} into $DEST"
    git clone --quiet https://github.com/fstandhartinger/jevbench.git "$DEST" || exit 1
    git -C "$DEST" checkout --quiet --detach "$PIN" || exit 1
  else
    echo "no jevbench checkout at $DEST (--no-fetch); set JEVBENCH_DIR" >&2
    exit 1
  fi
fi
HEAD=$(git -C "$DEST" rev-parse HEAD)
echo "jevbench: ${HEAD:0:9} in $DEST"
[ "$HEAD" = "$PIN" ] || echo "warning: pinned commit is ${PIN:0:9}; records carry the commit they ran under"
[ "$RUN" = 1 ] || exit 0

if ! curl -sf -m 3 "$BASE_URL/v1/models" >/dev/null; then
  echo "no server at $BASE_URL; start one with ./serve-de1.sh" >&2
  exit 1
fi

EXTRA=""
[ -n "${LIMIT:-}" ] && EXTRA="--limit $LIMIT"
[ "${FORCE:-0}" = 1 ] && EXTRA="$EXTRA --force"

"$PY" jevbench_public.py run \
  --base-url "$BASE_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --out-dir "$OUT" --scratch-dir "$SCRATCH" --tiers "$TIERS" \
  --run-label "${LABEL:-$MODEL}" $EXTRA || exit 1

echo
"$PY" jevbench_public.py report --run-dir "$OUT" || exit 1
"$PY" jevbench_public.py timing --run-dir "$OUT" --raw-dir "$SCRATCH" || exit 1
echo "wrote $OUT: records, summaries, manifest.json, timing.json"
