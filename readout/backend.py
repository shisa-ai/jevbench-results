"""Backend base class and registry for the readout."""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .format import Case


def instruction_text(value: Any) -> str:
    """Render a question's instructions as text for a text-based readout.

    Suites normally store a string. kev-decision-v1 stores a structured
    {"question", "focus"} object for 199 of its 1336 cases; label- and
    hypothesis-based readouts need prose, so join the object's string fields
    instead of handing a dict to a tokenizer or a label pipeline.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [str(v).strip() for v in value.values() if isinstance(v, str) and v.strip()]
        if parts:
            return " ".join(parts)
    return json.dumps(value, ensure_ascii=False)


class Backend(ABC):
    """A decision backend evaluated against cases in the common format.

    Subclasses set `name` and implement `evaluate`. `evaluate` receives a case
    and returns the answer fields for a result row: `answers`, `confidence`,
    `confidence_kind`, `raw`, `tokens`, and optionally `status`. The runner
    handles timing, retries are not performed, and every attempted case
    produces a row, including failures.
    """

    name: ClassVar[str] = "backend"
    # True when the adapter itself imports torch/transformers and must run in
    # the jev-gpu environment (the model may still be remote, e.g. DS4.1).
    needs_torch: ClassVar[bool] = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def evaluate(self, case: Case) -> dict[str, Any]:
        """Return answer fields for one case; never raises for per-case errors
        (return status="error" with a message instead)."""

    # -- shared helpers -------------------------------------------------

    def serialize_state(self, case: Case) -> Any:
        fmt = self.config.get("state_format", "as-is")
        if fmt == "as-is":
            return case.state
        if fmt == "text":
            return case.state_text()
        raise ValueError(f"unknown state_format {fmt!r}")

    @staticmethod
    def timed(fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        return out, round((time.perf_counter() - t0) * 1000, 2)


_REGISTRY: dict[str, type[Backend]] = {}


def register(cls: type[Backend]) -> type[Backend]:
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(name: str, config: dict[str, Any] | None = None) -> Backend:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown backend {name!r}; known: {known}")
    return _REGISTRY[name](config)


def backend_names() -> list[str]:
    return sorted(_REGISTRY)
