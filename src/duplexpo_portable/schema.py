"""Versioned in-memory contracts shared by all framework adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib.resources import files
import json
import math
from typing import Any, Literal

SCHEMA_VERSION = "duplexpo.v1"
SCHEMA_FILES = {
    "conversation": "conversation.schema.json",
    "window": "window.schema.json",
    "rollout": "rollout.schema.json",
    "grpo_record": "grpo_record.schema.json",
    "recipe": "recipe.schema.json",
}
Speaker = Literal["user", "agent"]
EventKind = Literal["turn", "backchannel"]


def load_json_schema(name: str) -> dict[str, Any]:
    """Load a public JSON Schema from either an installed wheel or source tree."""

    try:
        filename = SCHEMA_FILES[name]
    except KeyError as error:
        raise ValueError(f"unknown schema {name!r}; choose from {sorted(SCHEMA_FILES)}") from error
    path = files("duplexpo_portable").joinpath("schemas", filename)
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")


def _known_schema(value: str) -> None:
    if value != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {value!r}; expected {SCHEMA_VERSION!r}")


@dataclass(frozen=True)
class TranscriptUnit:
    """One time-aligned word or utterance in a canonical two-party transcript."""

    speaker: Speaker
    start_s: float
    end_s: float
    text: str

    def __post_init__(self) -> None:
        if self.speaker not in {"user", "agent"}:
            raise ValueError("speaker must be 'user' or 'agent'")
        _finite_nonnegative("start_s", self.start_s)
        _finite_nonnegative("end_s", self.end_s)
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")
        if not self.text.strip():
            raise ValueError("text must not be empty")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TranscriptUnit":
        return cls(
            speaker=value["speaker"],
            start_s=float(value["start_s"]),
            end_s=float(value["end_s"]),
            text=str(value["text"]),
        )


@dataclass(frozen=True)
class Conversation:
    """Canonical input to construction; source-specific parsing happens upstream."""

    conversation_id: str
    duration_s: float
    units: tuple[TranscriptUnit, ...]
    source: str = "unknown"
    split: str | None = None
    granularity: Literal["word", "utterance"] = "word"
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _known_schema(self.schema_version)
        if not self.conversation_id:
            raise ValueError("conversation_id must not be empty")
        _finite_nonnegative("duration_s", self.duration_s)
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.granularity not in {"word", "utterance"}:
            raise ValueError("granularity must be 'word' or 'utterance'")
        for unit in self.units:
            if unit.end_s > self.duration_s + 1e-6:
                raise ValueError(f"unit ends after conversation duration: {unit}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Conversation":
        return cls(
            conversation_id=str(value["conversation_id"]),
            duration_s=float(value["duration_s"]),
            units=tuple(TranscriptUnit.from_dict(item) for item in value.get("units", [])),
            source=str(value.get("source", "unknown")),
            split=value.get("split"),
            granularity=value.get("granularity", "word"),
            metadata=dict(value.get("metadata", {})),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True)
class Segment:
    speaker: Speaker
    start_s: float
    end_s: float
    text: str
    kind: EventKind
    unit_count: int = 1

    def __post_init__(self) -> None:
        if self.speaker not in {"user", "agent"}:
            raise ValueError("speaker must be 'user' or 'agent'")
        if self.kind not in {"turn", "backchannel"}:
            raise ValueError("kind must be 'turn' or 'backchannel'")
        _finite_nonnegative("start_s", self.start_s)
        _finite_nonnegative("end_s", self.end_s)
        if self.end_s <= self.start_s:
            raise ValueError("segment end_s must exceed start_s")
        if self.unit_count < 1:
            raise ValueError("unit_count must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Segment":
        return cls(
            speaker=value["speaker"],
            start_s=float(value["start_s"]),
            end_s=float(value["end_s"]),
            text=str(value.get("text", "")),
            kind=value["kind"],
            unit_count=int(value.get("unit_count", 1)),
        )


@dataclass(frozen=True)
class Window:
    """A dynamics-critical event with teacher-forced prefix and sampled span."""

    window_id: str
    conversation_id: str
    split: str
    frame_duration_s: float
    sample_start_frame: int
    sample_end_frame: int
    reference_start_frame: int
    reference_end_frame: int
    event_kind: EventKind
    user_segments: tuple[Segment, ...] = ()
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _known_schema(self.schema_version)
        if not self.window_id or not self.conversation_id:
            raise ValueError("window_id and conversation_id must not be empty")
        if not math.isfinite(self.frame_duration_s) or self.frame_duration_s <= 0:
            raise ValueError("frame_duration_s must be finite and positive")
        if self.sample_start_frame < 0 or self.sample_end_frame <= self.sample_start_frame:
            raise ValueError("sample frame range is invalid")
        if self.reference_start_frame < 0 or self.reference_end_frame < self.reference_start_frame:
            raise ValueError("reference frame range is invalid")
        if self.event_kind not in {"turn", "backchannel"}:
            raise ValueError("event_kind must be 'turn' or 'backchannel'")

    @property
    def group_id(self) -> str:
        return self.window_id

    @property
    def sample_start_s(self) -> float:
        return self.sample_start_frame * self.frame_duration_s

    @property
    def sample_end_s(self) -> float:
        return self.sample_end_frame * self.frame_duration_s

    @property
    def reference_start_s(self) -> float:
        return self.reference_start_frame * self.frame_duration_s

    @property
    def reference_end_s(self) -> float:
        return self.reference_end_frame * self.frame_duration_s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Window":
        return cls(
            window_id=str(value["window_id"]),
            conversation_id=str(value["conversation_id"]),
            split=str(value["split"]),
            frame_duration_s=float(value["frame_duration_s"]),
            sample_start_frame=int(value["sample_start_frame"]),
            sample_end_frame=int(value["sample_end_frame"]),
            reference_start_frame=int(value["reference_start_frame"]),
            reference_end_frame=int(value["reference_end_frame"]),
            event_kind=value["event_kind"],
            user_segments=tuple(Segment.from_dict(item) for item in value.get("user_segments", [])),
            source=str(value.get("source", "unknown")),
            metadata=dict(value.get("metadata", {})),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True)
class SpeechInterval:
    start_s: float
    end_s: float | None = None
    termination: Literal["eos", "pad", "cutoff", "unknown"] = "unknown"

    def __post_init__(self) -> None:
        _finite_nonnegative("start_s", self.start_s)
        if self.end_s is not None:
            _finite_nonnegative("end_s", self.end_s)
            if self.end_s < self.start_s:
                raise ValueError("speech interval end_s must not precede start_s")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SpeechInterval":
        end = value.get("end_s")
        return cls(
            start_s=float(value["start_s"]),
            end_s=None if end is None else float(end),
            termination=value.get("termination", "unknown"),
        )


@dataclass(frozen=True)
class ActionTrace:
    """Model-neutral interpretation of a rollout on the conversation timeline."""

    speech: tuple[SpeechInterval, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActionTrace":
        return cls(
            speech=tuple(SpeechInterval.from_dict(item) for item in value.get("speech", [])),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class Rollout:
    """One sampled completion. Supply either a generic trace or decodable actions."""

    rollout_id: str
    window_id: str
    action_ids: tuple[int, ...] = ()
    trace: ActionTrace | None = None
    behavior_logprobs: tuple[float, ...] = ()
    reference_logprobs: tuple[float, ...] = ()
    loss_mask: tuple[bool, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _known_schema(self.schema_version)
        if not self.rollout_id or not self.window_id:
            raise ValueError("rollout_id and window_id must not be empty")
        if self.trace is None and not self.action_ids:
            raise ValueError("rollout requires trace or action_ids")
        for name, values in (
            ("behavior_logprobs", self.behavior_logprobs),
            ("reference_logprobs", self.reference_logprobs),
            ("loss_mask", self.loss_mask),
        ):
            if values and self.action_ids and len(values) != len(self.action_ids):
                raise ValueError(f"{name} must have the same length as action_ids")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.trace is None:
            result.pop("trace")
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Rollout":
        trace_value = value.get("trace")
        return cls(
            rollout_id=str(value["rollout_id"]),
            window_id=str(value["window_id"]),
            action_ids=tuple(int(item) for item in value.get("action_ids", [])),
            trace=None if trace_value is None else ActionTrace.from_dict(trace_value),
            behavior_logprobs=tuple(float(item) for item in value.get("behavior_logprobs", [])),
            reference_logprobs=tuple(float(item) for item in value.get("reference_logprobs", [])),
            loss_mask=tuple(bool(item) for item in value.get("loss_mask", [])),
            metadata=dict(value.get("metadata", {})),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True)
class RewardResult:
    total: float
    components: dict[str, float]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GRPORecord:
    """Stable flat boundary consumed by a trainer adapter."""

    group_id: str
    rollout_id: str
    window: dict[str, Any]
    action_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    reward: float
    advantage: float
    behavior_logprobs: tuple[float, ...] = ()
    reference_logprobs: tuple[float, ...] = ()
    reward_components: dict[str, float] = field(default_factory=dict)
    reward_diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "duplexpo.grpo.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "duplexpo.grpo.v1":
            raise ValueError("unsupported GRPO record schema_version")
        if not self.group_id or not self.rollout_id:
            raise ValueError("group_id and rollout_id must not be empty")
        if str(self.window.get("window_id", "")) != self.group_id:
            raise ValueError("GRPO record window_id must equal group_id")
        if len(self.action_ids) != len(self.loss_mask):
            raise ValueError("action_ids and loss_mask must have equal length")
        for name, values in (
            ("behavior_logprobs", self.behavior_logprobs),
            ("reference_logprobs", self.reference_logprobs),
        ):
            if values and len(values) != len(self.action_ids):
                raise ValueError(f"{name} must have the same length as action_ids")
        if not math.isfinite(self.reward) or not math.isfinite(self.advantage):
            raise ValueError("reward and advantage must be finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GRPORecord":
        return cls(
            group_id=str(value["group_id"]),
            rollout_id=str(value["rollout_id"]),
            window=dict(value["window"]),
            action_ids=tuple(int(item) for item in value["action_ids"]),
            loss_mask=tuple(bool(item) for item in value["loss_mask"]),
            reward=float(value["reward"]),
            advantage=float(value["advantage"]),
            behavior_logprobs=tuple(float(item) for item in value.get("behavior_logprobs", [])),
            reference_logprobs=tuple(float(item) for item in value.get("reference_logprobs", [])),
            reward_components={name: float(item) for name, item in value.get("reward_components", {}).items()},
            reward_diagnostics=dict(value.get("reward_diagnostics", {})),
            metadata=dict(value.get("metadata", {})),
            schema_version=str(value["schema_version"]),
        )
