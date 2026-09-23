#!/usr/bin/env python3
"""Ask a served model one question and print every step of the readout.

The readout has three steps, and this script shows each one against a live
server:

  1. Render the System One payload (evidence, criterion, lettered options) with
     the model's own chat template.
  2. Send one completions request with `max_tokens: 1`, `temperature: 0` and
     `logprobs: 20`, and read the first-token logprobs. Each option letter is a
     single token; a letter outside the returned top 20 costs one extra request
     that re-sends the prompt with `prompt_logprobs`.
  3. Normalize the letters' logprobs into a distribution with a softmax.

`readout/vllm_scoring.py` is the code that does this for the benchmark. This
script calls the same methods and prints the intermediate values, then runs the
full `evaluate()` path and checks that the two agree.

Usage:
  ./serve-de1.sh                                   # in another shell
  python test-query.py
  python test-query.py --curl                      # print the HTTP request only
  python test-query.py --state '{"ticket": "..."}' --question "..." \
      --options refund=Refund now,trace=Open a carrier trace

Needs `httpx` and `transformers` for the tokenizer; the server needs only vLLM.
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys

from readout.format import Case, Question
from readout.vllm_scoring import LETTERS, VLLMScoring

DEFAULT_STATE = {
    "ticket": "Order 4812 was marked delivered on Monday. The customer says the "
              "parcel never arrived and tracking has not updated since Friday.",
    "account": "customer since 2021, no prior claims",
}
DEFAULT_QUESTION = "What should the support agent do next?"
DEFAULT_OPTIONS = [
    ("refund", "Refund the order now"),
    ("trace", "Open a carrier trace and reply with the case number"),
    ("replace", "Ship a replacement order"),
    ("wait", "Ask the customer to wait three more days"),
]


def parse_options(spec: str | None) -> list[tuple[str, str]]:
    if not spec:
        return DEFAULT_OPTIONS
    options = []
    for part in spec.split(","):
        key, _, description = part.partition("=")
        options.append((key.strip(), description.strip() or key.strip()))
    return options


def curl_command(base_url: str, body: dict) -> str:
    return (
        f"curl -s {shlex.quote(base_url.rstrip('/') + '/v1/completions')} \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -d {shlex.quote(json.dumps(body))}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8021")
    parser.add_argument("--model", default="shisa-ai/shisa-de-1")
    parser.add_argument("--tokenizer", default=None, help="defaults to --model")
    parser.add_argument("--state", default=None, help="JSON state; a built-in example by default")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--options", default=None, help="key=description pairs, comma separated")
    parser.add_argument("--curl", action="store_true", help="print the HTTP request and exit")
    args = parser.parse_args()

    options = parse_options(args.options)
    if len(options) > len(LETTERS):
        print(f"at most {len(LETTERS)} options", file=sys.stderr)
        return 2
    case = Case(
        id="test-query",
        suite="test-query",
        split="test",
        language="en",
        group="test-query",
        source="test-query.py",
        state=json.loads(args.state) if args.state else DEFAULT_STATE,
        questions={"decision": Question(
            qid="decision", type="choice", instructions=args.question, options=dict(options),
        )},
        gold={},
    )
    backend = VLLMScoring({
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer_source": args.tokenizer or args.model,
    })

    # Step 1: the prompt, rendered locally with the model's chat template.
    backend._ensure_tokenizer()
    prompt = backend._prompt(case, args.question, options)

    # Step 2: the request. max_tokens 1 keeps the answer at the first position.
    body = {
        "model": args.model, "prompt": prompt,
        "max_tokens": 1, "temperature": 0.0, "logprobs": 20,
    }
    if args.curl:
        print(curl_command(args.base_url, body))
        return 0

    letters = [LETTERS[i] for i in range(len(options))]
    slots = {letter: backend._slot_id(letter) for letter in letters}
    decoded = {letter: backend._tok.decode([slots[letter]]) for letter in letters}

    print("1. The prompt the model sees")
    print("-" * 72)
    print(prompt)
    print()
    print("2. Option letters as single tokens (the answer slots)")
    print("-" * 72)
    for letter in letters:
        print(f"   {letter}  token {slots[letter]:>6}  {decoded[letter]!r}")
    print()

    top, extra = backend._top_logprobs(prompt)
    usage = extra.get("usage") or {}
    print("3. One completions request: max_tokens 1, temperature 0, logprobs 20")
    print("-" * 72)
    print(f"   POST {args.base_url.rstrip('/')}/v1/completions")
    print(f"   prompt_tokens {usage.get('prompt_tokens')}, sampled token {extra.get('sampled')!r}")
    print("   first-token top 20 (logprobs):")
    for token, logprob in sorted(top.items(), key=lambda kv: -kv[1]):
        marker = " <- answer slot" if token in decoded.values() else ""
        print(f"     {token!r:>12}  {logprob:9.4f}{marker}")
    print()

    # Step 3: the letters' logprobs, then a softmax over them.
    logprobs: dict[str, float] = {}
    missing = [letter for letter in letters if decoded[letter] not in top]
    for letter in letters:
        if letter not in missing:
            logprobs[letter] = top[decoded[letter]]
    for letter in missing:
        logprob, _ = backend._missing_logprob(prompt, letter)
        if logprob is not None:
            logprobs[letter] = logprob
    print("4. The distribution over the option letters")
    print("-" * 72)
    if missing:
        print(f"   {len(missing)} letter(s) outside the top 20 ({', '.join(missing)}): "
              f"one extra request each, prompt + letter with prompt_logprobs")
    peak = max(logprobs.values())
    weights = {letter: math.exp(logprobs[letter] - peak) for letter in logprobs}
    total = sum(weights.values())
    probs = {letter: weight / total for letter, weight in weights.items()}
    for (key, description), letter in zip(options, letters):
        print(f"   {probs[letter]:7.4f}  {letter}  {key}  ({description})")
    best = max(probs, key=probs.get)
    print(f"   answer: {best} ({probs[best]:.4f})")
    print()

    result = backend.evaluate(case)
    if result["status"] != "ok":
        print(f"evaluate() failed: {result.get('error')}", file=sys.stderr)
        return 1
    answer = result["answers"]["decision"]
    meta = result["raw"]["per_question"]["decision"]
    print("5. The same path as readout/vllm_scoring.py (evaluate)")
    print("-" * 72)
    print(f"   choice {answer['choice']}, probabilities "
          f"{json.dumps(answer['probabilities'], sort_keys=True)}")
    print(f"   requests {meta['n_requests']}, missing from top {meta['missing_from_top']}, "
          f"input tokens {result['tokens']['input']}, {result['latency_ms']['total']} ms")
    if answer["choice"] != dict(zip(letters, options))[best][0]:
        print("   mismatch between this script and evaluate()", file=sys.stderr)
        return 1
    print()
    print("6. The same request without Python")
    print("-" * 72)
    print(curl_command(args.base_url, body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
