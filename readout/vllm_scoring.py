"""Frozen scorer over a vLLM OpenAI-compatible completions endpoint.

The readout DE-1 was measured with: the prompt is rendered client-side with the
model's tokenizer, one request per question carrying the System One payload
(evidence, criterion, lettered options), and the answer read from the
first-token logprobs over the option-letter slots, with a `prompt_logprobs`
fallback for letters outside the returned top 20.

The served model answers; this module never loads model weights, so the server
can run vLLM in its own environment on any GPU. It needs `httpx` and, for the
tokenizer, `transformers`.
"""
from __future__ import annotations

import json
import math
import time
from typing import Any, ClassVar

import httpx

from .format import Case
from .backend import Backend, register

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


@register
class VLLMScoring(Backend):
    name = "vllm-scoring"
    needs_torch: ClassVar[bool] = True  # renders the chat template via transformers

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.base_url: str = self.config["base_url"].rstrip("/")
        self.model: str = self.config["model"]
        self.tokenizer_source: str = self.config.get("tokenizer_source", self.model)
        self.close_think: str = self.config.get("close_think", "")
        self.enable_thinking: bool = bool(self.config.get("enable_thinking", False))
        self.timeout_s: float = float(self.config.get("timeout_s", 120))
        # Persist resolved defaults for run manifests.
        self.config.update({
            "base_url": self.base_url,
            "model": self.model,
            "tokenizer_source": self.tokenizer_source,
            "close_think": self.close_think,
            "enable_thinking": self.enable_thinking,
            "timeout_s": self.timeout_s,
        })
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_s,
            headers={"Content-Type": "application/json"},
        )
        self._tok = None
        self._slot_cache: dict[str, int] = {}

    def _ensure_tokenizer(self) -> None:
        if self._tok is not None:
            return
        from transformers import AutoTokenizer

        common: dict[str, Any] = {}
        if self.config.get("revision"):
            common["revision"] = self.config["revision"]
        if self.config.get("tokenizer_local_only"):
            common["local_files_only"] = True
        self._tok = AutoTokenizer.from_pretrained(self.tokenizer_source, **common)
        self.config["_resolved_config"] = {
            "base_url": self.base_url,
            "model": self.model,
            "tokenizer": self.tokenizer_source,
            "close_think": self.close_think or None,
            "scaffold": self.config.get("scaffold", "systemone_json"),
            "readout": "first-token top logprobs over letter slots",
        }

    def _slot_id(self, letter: str) -> int:
        if letter in self._slot_cache:
            return self._slot_cache[letter]
        encoded = self._tok.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or self._tok.decode(encoded) != letter:
            raise ValueError(f"answer slot {letter!r} is not one exact token for this tokenizer")
        self._slot_cache[letter] = encoded[0]
        return encoded[0]

    def _prompt(self, case: Case, question_text: str, options: list[tuple[str, str]]) -> str:
        scaffold = self.config.get("scaffold", "systemone_json")
        if scaffold == "plain_qa":
            # Alternate rendering for short-QA items (task #13): the same
            # content (state, question, lettered options) laid out as plain
            # readable text instead of the System One JSON payload. The answer
            # boundary and letter slots are identical to the default scaffold,
            # so the two arms differ only in rendering.
            lines = [case.state_text() if not isinstance(case.state, str) else case.state, ""]
            lines.append(f"Question: {question_text}")
            lines.append("Options:")
            for i, (_, desc) in enumerate(options):
                lines.append(f"({LETTERS[i]}) {desc}")
            lines.append("")
            lines.append("Answer with a single uppercase letter only.")
            user_text = "\n".join(lines)
            prompt = self._tok.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            return prompt + self.close_think
        payload = {
            "evidence": case.state,
            "criterion": question_text,
            "options": [
                {"letter": LETTERS[i], "description": desc}
                for i, (_, desc) in enumerate(options)
            ],
        }
        prompt = self._tok.apply_chat_template(
            [
                {"role": "system", "content": DIRECT_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        return prompt + self.close_think

    def _top_logprobs(self, prompt: str) -> tuple[dict[str, float], dict[str, Any]]:
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 20,
        }
        response = self.client.post("/v1/completions", content=json.dumps(body))
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        top_list = (choice.get("logprobs") or {}).get("top_logprobs") or []
        top: dict[str, float] = dict(top_list[0]) if top_list else {}
        return top, {"usage": data.get("usage"), "sampled": choice.get("text")}

    def _missing_logprob(self, prompt: str, letter: str) -> tuple[float | None, dict[str, Any]]:
        body = {
            "model": self.model,
            "prompt": prompt + letter,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 0,
        }
        response = self.client.post("/v1/completions", content=json.dumps(body))
        response.raise_for_status()
        data = response.json()
        entry = (data["choices"][0].get("prompt_logprobs") or [None])[-1]
        lp = None
        if entry:
            lp = next(iter(entry.values())).get("logprob")
        return lp, {"usage": data.get("usage")}

    def evaluate(self, case: Case) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        confidence: dict[str, Any] = {}
        meta: dict[str, Any] = {}
        started = time.perf_counter()
        request_ms = 0.0
        try:
            self._ensure_tokenizer()
            input_tokens = 0
            for qid, q in case.questions.items():
                if q.type == "noul":
                    options = [("yes", "Yes"), ("no", "No")]
                elif q.type == "choice":
                    options = list((q.options or {}).items())
                    if len(options) > len(LETTERS):
                        raise ValueError(
                            f"choice question {qid!r}: {len(options)} options exceed single-letter slots"
                        )
                else:
                    options = [(str(i), desc) for i, desc in enumerate(q.levels or [])]
                    if len(options) > len(LETTERS):
                        raise ValueError(f"score question {qid!r}: too many levels for letter slots")
                prompt = self._prompt(case, q.instructions, options)
                letters = [LETTERS[i] for i in range(len(options))]
                slot_ids = [self._slot_id(letter) for letter in letters]
                # Boundary verification with the local tokenizer: prompt + letter
                # must tokenize as prompt tokens plus exactly the slot token.
                prompt_ids = self._tok.encode(prompt, add_special_tokens=False)
                for letter, sid in zip(letters, slot_ids):
                    if self._tok.encode(prompt + letter, add_special_tokens=False) != prompt_ids + [sid]:
                        raise ValueError(
                            f"answer boundary changes tokenization for slot {letter!r}; distribution cannot be scored"
                        )
                mark = time.perf_counter()
                top, extra = self._top_logprobs(prompt)
                request_ms += (time.perf_counter() - mark) * 1000
                usage = extra.get("usage") or {}
                input_tokens += usage.get("prompt_tokens") or 0
                token_strs = [self._tok.decode([sid]) for sid in slot_ids]
                logprobs: dict[str, float] = {}
                missing: list[str] = []
                requests_made = 1
                for letter, tok_str in zip(letters, token_strs):
                    if tok_str in top:
                        logprobs[letter] = top[tok_str]
                    else:
                        missing.append(letter)
                for letter in missing:
                    lp, fb_usage = self._missing_logprob(prompt, letter)
                    # The fallback re-sends the full prompt, so count its prompt
                    # tokens: the helper wraps them under "usage" like the top
                    # request does, and reading the wrapper directly silently
                    # dropped them (2026-09-21 accounting fix).
                    input_tokens += ((fb_usage or {}).get("usage") or {}).get("prompt_tokens") or 0
                    requests_made += 1
                    if lp is not None:
                        logprobs[letter] = lp
                if len(logprobs) != len(letters):
                    raise ValueError(
                        f"incomplete distribution for question {qid!r}: "
                        f"missing candidates after fallback: {sorted(set(letters) - set(logprobs))}"
                    )
                vals = [logprobs[letter] for letter in letters]
                mx = max(vals)
                weights = [math.exp(v - mx) for v in vals]
                total = sum(weights)
                probs = [w / total for w in weights]
                if q.type == "noul":
                    answers[qid] = {"noul": probs[0]}
                elif q.type == "choice":
                    keys = [k for k, _ in options]
                    best = max(range(len(keys)), key=lambda i: probs[i])
                    answers[qid] = {
                        "choice": keys[best],
                        "probabilities": {k: round(p, 6) for k, p in zip(keys, probs)},
                    }
                    confidence[qid] = round(probs[best], 6)
                else:
                    best = max(range(len(probs)), key=lambda i: probs[i])
                    answers[qid] = {
                        "score": best,
                        "probabilities": {str(i): round(p, 6) for i, p in enumerate(probs)},
                    }
                    confidence[qid] = round(probs[best], 6)
                meta[qid] = {
                    "missing_from_top": missing,
                    "n_requests": requests_made,
                    "sampled": extra.get("sampled"),
                    "input_tokens": usage.get("prompt_tokens"),
                    # Full-precision candidate logprobs retained for faithful
                    # post-hoc calibration (2026-09-20 review).
                    "logprobs": {k: round(v, 10) for k, v in logprobs.items()},
                }
            return {
                "status": "ok",
                "answers": answers,
                "confidence": confidence,
                "confidence_kind": "top_prob",
                "raw": {"per_question": meta},
                "tokens": {"input": input_tokens},
                "latency_ms": {
                    "total": round((time.perf_counter() - started) * 1000, 2),
                    "request": round(request_ms, 2),
                },
            }
        except Exception as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            kind = "timeout" if "timed out" in str(exc).lower() else "error"
            if "exceed" in str(exc) or "too many levels" in str(exc):
                kind = "unsupported"
            return {
                "status": kind,
                "error": {"kind": kind, "message": str(exc)[:500]},
                "latency_ms": {"total": elapsed},
            }
