"""Minimal interfaces required to connect a policy and a trainer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from ..schema import ActionTrace, Rollout, Window


class PolicyAdapter(Protocol):
    def generate(self, windows: Sequence[Window], samples_per_window: int) -> Sequence[Rollout]: ...


class TraceDecoder(Protocol):
    def decode(self, rollout: Rollout, window: Window) -> ActionTrace: ...


class TrainerAdapter(Protocol):
    def update(self, records: Sequence[dict[str, Any]]) -> dict[str, float]: ...
