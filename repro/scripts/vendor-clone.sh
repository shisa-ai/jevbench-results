#!/usr/bin/env bash
# Clone the vendored external implementations reviewed in implementation/README.md
# into vendor/. Each checkout is recorded with its pinned commit in vendor/README.md.
# Safe to re-run: existing checkouts are fetched and reset to the pinned commit only
# when the working tree is clean; dirty trees are left untouched and reported.
set -euo pipefail

cd "$(dirname "$0")/../vendor"

clone_pin() {
  local url="$1" dest="$2" pin="$3"
  if [ ! -d "$dest/.git" ]; then
    git clone "$url" "$dest"
  fi
  if [ -n "$(git -C "$dest" status --porcelain)" ]; then
    echo "SKIP (dirty tree): $dest"
    return 0
  fi
  git -C "$dest" fetch origin --quiet
  git -C "$dest" checkout --quiet --detach "$pin"
  echo "OK: $dest @ $pin"
}

# name, url, pinned commit
clone_pin https://github.com/logan-markewich/jeff.git            jeff          34b32f99a727
clone_pin https://github.com/jaredpalmer/kev.git                kev           20fa6268c8ce
clone_pin https://github.com/TheoLeeCJ/openjev.git              openjev-theo  ca3ba65f1429
clone_pin https://github.com/razorback16/openjev.git            openjev-diff  91d5005effcf
clone_pin https://github.com/featherless-ai/simple-jev.git       simple-jev    0dd5396ffce6
clone_pin https://github.com/kshetrajna12/reflex.git            reflex        21c95dfd0cd4
clone_pin https://github.com/bespokelabsai/nimble.git           nimble        "$(git ls-remote https://github.com/bespokelabsai/nimble.git HEAD | cut -f1)"
clone_pin https://github.com/OmarMujahid/jev-decision-bench.git jev-decision-bench 0883fa781729
clone_pin https://github.com/Knowledgator/GLiFormer.git         gliformer     b5c0a0fd2aac
clone_pin https://github.com/wfzyx/von.git                      von           a94aa368ff4d

# Benchmark harness, not a model implementation: JevBench's frozen task
# records, runner, scoring and composite are what DE-1 is measured on.
clone_pin https://github.com/fstandhartinger/jevbench.git        jevbench      ee677f01f177

# Vision-language readout harness, not a model implementation: its image
# classification suites are what DE-1's vision path is measured on. The Gemma 4
# logit-softcap patch is applied by scripts/glance-vision.sh, not here, so this
# checkout stays clean and re-pins normally.
clone_pin https://github.com/yoheinakajima/glance.git            glance        9aaa93976561966351bab10e0504aa32945766e9

echo "vendor clones updated under $(pwd)"
