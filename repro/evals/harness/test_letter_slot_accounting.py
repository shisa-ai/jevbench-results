"""Regression tests for letter-slot readout request and token accounting.

Pins the 2026-09-21 finding behind the JevBench p50 gap. Both HTTP letter-slot
readouts answer a question with one top-logprobs request, then issue one extra
full-prompt request per option letter that is missing from the returned top-20
(``prompt + letter`` with ``prompt_logprobs``). Those fallback requests are real
prefills of the whole prompt, and both adapters dropped their prompt tokens:
the helper returns ``{"usage": {...}}`` while the caller read ``prompt_tokens``
off the wrapper, so ``tokens.input`` counted only the first request. On the
frozen Gemma 4 base run that understated input tokens by 3.7x on the easy tier
(148.5 recorded against 545.1 re-derived).

The invariant asserted here is implementation-independent: ``tokens.input``
must equal the sum of the prompt tokens the server reported for every
completion request the readout made, and ``n_requests`` must count them.

Run: python -m unittest evals.harness.test_letter_slot_accounting
"""
from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from pathlib import Path

import evals.harness.format  # noqa: F401  (makes the relative imports below resolvable)

# evals/harness/adapters/__init__.py imports the torch-based backends, which are
# not installed in the HTTP-only environment, so load the two readouts as
# submodules of a stub package instead of importing the real one.
if "evals.harness.adapters" not in sys.modules:
    _package = types.ModuleType("evals.harness.adapters")
    _package.__path__ = [str(Path(__file__).parent / "adapters")]
    sys.modules["evals.harness.adapters"] = _package

vllm_scoring = importlib.import_module("evals.harness.adapters.vllm_scoring")
ds41_scoring = importlib.import_module("evals.harness.adapters.ds41_scoring")

from .format import Case, Question  # noqa: E402


def _ids(text: str) -> list[int]:
    """One token id per character, so prompt + letter always appends one id."""
    return [ord(char) for char in text]


class _FakeTokenizer:
    """Character tokenizer for the adapters that render prompts locally."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return _ids(text)

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs) -> str:
        return "".join(message["content"] for message in messages) + "\n"


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Answers top-logprobs, prompt_logprobs, and /tokenize requests.

    ``top`` holds the token strings the server reports in the top-20 for the
    first request; option letters outside it trigger the fallback path. Every
    completion request reports its own prompt length as ``prompt_tokens``, so
    the expected total is a property of the requests the readout made.
    """

    def __init__(self, top: dict[str, float]) -> None:
        self.top = top
        self.calls: list[tuple[str, dict]] = []

    def post(self, path: str, content: str | None = None) -> _FakeResponse:
        body = json.loads(content or "{}")
        self.calls.append((path, body))
        if path == "/tokenize":
            return _FakeResponse({"tokens": _ids(body["prompt"])})
        prompt_tokens = len(_ids(body["prompt"]))
        if "prompt_logprobs" in body:
            return _FakeResponse({
                "choices": [{"prompt_logprobs": [None, {"X": {"logprob": -2.0}}]}],
                "usage": {"prompt_tokens": prompt_tokens},
            })
        return _FakeResponse({
            "choices": [{"text": "A", "logprobs": {"top_logprobs": [self.top]}}],
            "usage": {"prompt_tokens": prompt_tokens},
        })


def _case() -> Case:
    return Case(
        id="t-1",
        suite="t",
        split="test",
        language="en",
        group="g",
        source="unit-test",
        state={"note": "state text"},
        questions={
            "q": Question(
                qid="q",
                type="choice",
                instructions="Pick one.",
                options={"a": "first", "b": "second", "c": "third", "d": "fourth", "e": "fifth"},
            )
        },
        gold={"q": "a"},
    )


class TestFallbackRequestAccounting(unittest.TestCase):
    def _run(self, name: str) -> tuple[dict, _FakeClient]:
        # "A", "B", and "C" are in the top-20; "D" and "E" are not, so the
        # readout must issue two fallback requests.
        client = _FakeClient({"A": -0.1, "B": -1.0, "C": -2.0})
        if name == "vllm-scoring":
            backend = vllm_scoring.VLLMScoring({"base_url": "http://stub", "model": "stub"})
            backend._tok = _FakeTokenizer()
        else:
            backend = ds41_scoring.DS41Scoring({"base_url": "http://stub", "model": "stub"})
        backend.client = client
        result = backend.evaluate(_case())
        self.assertEqual(result["status"], "ok", result.get("error"))
        return result, client

    def test_input_tokens_count_every_request(self) -> None:
        for name in ("vllm-scoring", "ds41-scoring"):
            with self.subTest(backend=name):
                result, client = self._run(name)
                completions = [body for path, body in client.calls if path == "/v1/completions"]
                fallbacks = [body for body in completions if "prompt_logprobs" in body]
                expected = sum(len(_ids(body["prompt"])) for body in completions)
                self.assertEqual(len(completions), 3, "one top request plus one per missing letter")
                self.assertEqual(len(fallbacks), 2, "two option letters were outside the top-20")
                self.assertEqual(result["tokens"]["input"], expected)

    def test_request_count_matches_requests_made(self) -> None:
        for name in ("vllm-scoring", "ds41-scoring"):
            with self.subTest(backend=name):
                result, client = self._run(name)
                meta = result["raw"]["per_question"]["q"]
                self.assertEqual(meta["n_requests"], 3)
                self.assertEqual(meta["missing_from_top"], ["D", "E"])
                self.assertEqual(len([p for p, _ in client.calls if p == "/v1/completions"]), 3)


if __name__ == "__main__":
    unittest.main()
