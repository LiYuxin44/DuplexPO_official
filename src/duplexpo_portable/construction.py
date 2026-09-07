"""Framework-neutral transcript-to-event construction.

The two built-in profiles encode the paper-time Fisher and Seamless rules.
Source adapters only need to emit :class:`Conversation` with canonical
``user``/``agent`` roles and aligned units.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from typing import Any

from .backchannel_lexicon import (
    BACKCHANNEL_BIGRAMS,
    BACKCHANNEL_LEXICON,
    BACKCHANNEL_PENTAGRAMS,
    BACKCHANNEL_QUADGRAMS,
    BACKCHANNEL_TRIGRAMS,
)
from .schema import Conversation, Segment, TranscriptUnit

WORD_RE = re.compile(r"[a-zA-Z]+(?:['-][a-zA-Z]+)?")

SEAMLESS_BACKCHANNEL_PHRASES = frozenset(
    set(BACKCHANNEL_LEXICON)
    | {" ".join(value) for value in BACKCHANNEL_BIGRAMS}
    | {" ".join(value) for value in BACKCHANNEL_TRIGRAMS}
)
DEFAULT_BACKCHANNEL_PHRASES = frozenset(
    set(SEAMLESS_BACKCHANNEL_PHRASES)
    | {" ".join(value) for value in BACKCHANNEL_QUADGRAMS}
    | {" ".join(value) for value in BACKCHANNEL_PENTAGRAMS}
)


@dataclass(frozen=True)
class ConstructionProfile:
    name: str
    max_duration_s: float = 210.0
    word_gap_s: float = 0.30
    turn_merge_gap_s: float = 1.20
    lexical_bc_max_duration_s: float = 0.80
    lexical_bc_max_words: int = 4
    high_order_bc_min_words: int = 4
    high_order_bc_max_duration_s: float = 1.20
    high_order_bc_max_words: int = 6
    fallback_bc_max_duration_s: float = 0.30
    preserve_backchannels: bool = False
    merge_strategy: str = "fisher"
    fragment_max_duration_s: float = 0.30
    fragment_max_words: int = 2
    utterance_bc_max_duration_s: float = 2.0
    utterance_bc_max_words: int = 6
    min_cross_overlap_s: float = 0.10
    excluded_interaction_types: tuple[str, ...] = ()
    backchannel_phrases: frozenset[str] = DEFAULT_BACKCHANNEL_PHRASES

    def __post_init__(self) -> None:
        if self.merge_strategy not in {"fisher", "seamless"}:
            raise ValueError("merge_strategy must be 'fisher' or 'seamless'")
        if self.max_duration_s <= 0 or self.word_gap_s < 0 or self.turn_merge_gap_s < 0:
            raise ValueError("construction durations must be non-negative and max_duration_s positive")

    def fingerprint(self) -> str:
        value = asdict(self)
        value["backchannel_phrases"] = sorted(self.backchannel_phrases)
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def profile_named(name: str) -> ConstructionProfile:
    """Return a frozen built-in recipe, including all behavior-changing knobs."""

    normalized = name.lower().replace("-", "_")
    if normalized in {"fisher", "fisher_v1", "paper_word_v1"}:
        return ConstructionProfile(name="paper_word_v1")
    if normalized in {"seamless", "seamless_v1", "seamless_hq_v1"}:
        return ConstructionProfile(
            name="seamless_hq_v1",
            turn_merge_gap_s=1.0,
            lexical_bc_max_duration_s=2.0,
            lexical_bc_max_words=6,
            fallback_bc_max_duration_s=0.0,
            preserve_backchannels=True,
            merge_strategy="seamless",
            excluded_interaction_types=("charades",),
            backchannel_phrases=SEAMLESS_BACKCHANNEL_PHRASES,
        )
    raise ValueError(f"unknown construction profile: {name}")


def _tokens(text: str) -> list[str]:
    return WORD_RE.findall(" ".join(text.lower().strip().split()))


def _longest_phrase_match(tokens: Sequence[str], phrases: frozenset[str]) -> int:
    longest = 0
    for start in range(len(tokens)):
        for end in range(start + 1, len(tokens) + 1):
            if " ".join(tokens[start:end]) in phrases:
                longest = max(longest, end - start)
    return longest


def _overlap_s(start: float, end: float, other: Segment | TranscriptUnit) -> float:
    return max(0.0, min(end, other.end_s) - max(start, other.start_s))


def _is_fragment(unit: TranscriptUnit, profile: ConstructionProfile) -> bool:
    tokens = _tokens(unit.text)
    duration = unit.end_s - unit.start_s
    if not tokens or duration < 0.12:
        return True
    if unit.text.rstrip().endswith(("...", "-", "--")):
        return True
    if duration <= profile.fragment_max_duration_s and len(tokens) <= profile.fragment_max_words:
        return len(unit.text.strip()) <= 8 or tokens[0] not in profile.backchannel_phrases
    return False


def _is_backchannel(
    segment: Segment,
    other_units: Sequence[TranscriptUnit],
    profile: ConstructionProfile,
    *,
    utterance_mode: bool,
) -> bool:
    duration = segment.end_s - segment.start_s
    tokens = _tokens(segment.text)
    if not tokens or "?" in segment.text:
        return False
    phrase_len = _longest_phrase_match(tokens, profile.backchannel_phrases)

    if utterance_mode:
        lexical = phrase_len > 0
        overlapped = any(
            _overlap_s(segment.start_s, segment.end_s, other) >= profile.min_cross_overlap_s
            for other in other_units
        )
        return (
            lexical
            and duration <= profile.utterance_bc_max_duration_s
            and len(tokens) <= profile.utterance_bc_max_words
        ) or (duration < 1.0 and len(tokens) <= 3 and overlapped)

    if phrase_len >= profile.high_order_bc_min_words:
        return (
            duration <= profile.high_order_bc_max_duration_s
            and len(tokens) <= profile.high_order_bc_max_words
        )
    if phrase_len > 0:
        return duration <= profile.lexical_bc_max_duration_s and len(tokens) <= profile.lexical_bc_max_words
    return profile.fallback_bc_max_duration_s > 0 and duration <= profile.fallback_bc_max_duration_s


def _group_words(units: Sequence[TranscriptUnit], profile: ConstructionProfile) -> list[Segment]:
    if not units:
        return []
    groups: list[list[TranscriptUnit]] = [[units[0]]]
    for unit in units[1:]:
        if unit.start_s - groups[-1][-1].end_s > profile.word_gap_s:
            groups.append([unit])
        else:
            groups[-1].append(unit)
    return [
        Segment(
            speaker=group[0].speaker,
            start_s=group[0].start_s,
            end_s=group[-1].end_s,
            text=" ".join(unit.text for unit in group),
            kind="turn",
            unit_count=len(group),
        )
        for group in groups
    ]


def _initial_segments(
    units: Sequence[TranscriptUnit],
    other_units: Sequence[TranscriptUnit],
    profile: ConstructionProfile,
    granularity: str,
) -> tuple[list[Segment], int]:
    dropped_fragments = 0
    if granularity == "utterance":
        filtered: list[TranscriptUnit] = []
        for unit in units:
            if _is_fragment(unit, profile):
                dropped_fragments += 1
            else:
                filtered.append(unit)
        candidates = [
            Segment(unit.speaker, unit.start_s, unit.end_s, unit.text, "turn", 1)
            for unit in filtered
        ]
    else:
        candidates = _group_words(units, profile)

    result = []
    for segment in candidates:
        kind = "backchannel" if _is_backchannel(
            segment,
            other_units,
            profile,
            utterance_mode=granularity == "utterance",
        ) else "turn"
        result.append(
            Segment(
                segment.speaker,
                segment.start_s,
                segment.end_s,
                segment.text,
                kind,
                segment.unit_count,
            )
        )
    return result, dropped_fragments


def _other_has_substantive_activity(
    other_segments: Sequence[Segment], start_s: float, end_s: float, min_overlap_s: float
) -> bool:
    return any(
        segment.kind == "turn" and _overlap_s(start_s, end_s, segment) > min_overlap_s
        for segment in other_segments
    )


def _merge_turns(
    segments: Sequence[Segment], other_segments: Sequence[Segment], profile: ConstructionProfile
) -> list[Segment]:
    result: list[Segment] = []
    pending: list[Segment] = []

    def flush() -> None:
        if not pending:
            return
        combined_duration = pending[-1].end_s - pending[0].start_s
        combined_kind = (
            "turn"
            if profile.preserve_backchannels
            else ("backchannel" if combined_duration < 1.0 else "turn")
        )
        result.append(
            Segment(
                speaker=pending[0].speaker,
                start_s=pending[0].start_s,
                end_s=pending[-1].end_s,
                text=" ".join(item.text for item in pending),
                kind=combined_kind,
                unit_count=sum(item.unit_count for item in pending),
            )
        )
        pending.clear()

    for segment in segments:
        if segment.kind == "backchannel" and profile.preserve_backchannels:
            flush()
            result.append(segment)
            continue
        if not pending:
            pending.append(segment)
            continue
        gap_start = pending[-1].end_s
        gap_end = segment.start_s
        gap = gap_end - gap_start
        other_active = _other_has_substantive_activity(
            other_segments, gap_start, gap_end, profile.min_cross_overlap_s
        )
        if profile.merge_strategy == "fisher":
            should_merge = gap < profile.turn_merge_gap_s or not other_active
        elif profile.merge_strategy == "seamless":
            should_merge = gap < profile.turn_merge_gap_s and not other_active
        else:
            raise ValueError(f"unknown merge strategy: {profile.merge_strategy}")
        if should_merge:
            pending.append(segment)
        else:
            flush()
            pending.append(segment)
    flush()
    return result


def construct_conversation(
    conversation: Conversation,
    profile: ConstructionProfile,
) -> tuple[list[Segment], list[Segment], dict[str, Any]]:
    """Build canonical segments and a reproducibility/quality audit record."""

    interaction_type = str(conversation.metadata.get("interaction_type", "")).lower()
    if interaction_type in profile.excluded_interaction_types:
        return [], [], {
            "status": "excluded",
            "reason": f"excluded_interaction_type:{interaction_type}",
            "profile": profile.name,
            "profile_sha256": profile.fingerprint(),
        }

    effective_duration = min(conversation.duration_s, profile.max_duration_s)
    units = [
        TranscriptUnit(unit.speaker, unit.start_s, min(unit.end_s, effective_duration), unit.text)
        for unit in sorted(conversation.units, key=lambda item: (item.start_s, item.end_s, item.speaker))
        if unit.start_s < effective_duration and min(unit.end_s, effective_duration) > unit.start_s
    ]
    by_speaker = {
        speaker: [unit for unit in units if unit.speaker == speaker]
        for speaker in ("user", "agent")
    }
    user_initial, user_dropped = _initial_segments(
        by_speaker["user"], by_speaker["agent"], profile, conversation.granularity
    )
    agent_initial, agent_dropped = _initial_segments(
        by_speaker["agent"], by_speaker["user"], profile, conversation.granularity
    )
    user_segments = _merge_turns(user_initial, agent_initial, profile)
    agent_segments = _merge_turns(agent_initial, user_initial, profile)
    status = "kept" if user_segments and agent_segments else "empty_side_after_filtering"
    audit = {
        "status": status,
        "conversation_id": conversation.conversation_id,
        "source": conversation.source,
        "input_duration_s": conversation.duration_s,
        "effective_duration_s": effective_duration,
        "truncated": effective_duration < conversation.duration_s,
        "profile": profile.name,
        "profile_sha256": profile.fingerprint(),
        "user_segment_count": len(user_segments),
        "agent_segment_count": len(agent_segments),
        "user_backchannel_count": sum(item.kind == "backchannel" for item in user_segments),
        "agent_backchannel_count": sum(item.kind == "backchannel" for item in agent_segments),
        "user_fragments_dropped": user_dropped,
        "agent_fragments_dropped": agent_dropped,
    }
    return user_segments, agent_segments, audit


def assign_split(
    conversation_id: str,
    *,
    seed: int = 42,
    train: float = 0.90,
    validation: float = 0.05,
) -> str:
    """Stable conversation-level split; derived samples can never cross splits."""

    if train < 0 or validation < 0 or train + validation > 1:
        raise ValueError("invalid split fractions")
    digest = hashlib.sha256(f"{seed}:{conversation_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "test"


def balanced_round_robin(
    records: Iterable[Any], *, key, seed: int = 42
) -> list[Any]:
    """Deterministically shuffle within sources, then interleave sources 1:1."""

    groups: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        groups[str(key(record))].append(record)
    queues: dict[str, deque[Any]] = {}
    for name in sorted(groups):
        values = groups[name]
        random.Random(f"{seed}:{name}").shuffle(values)
        queues[name] = deque(values)
    result: list[Any] = []
    while any(queues.values()):
        for name in sorted(queues):
            if queues[name]:
                result.append(queues[name].popleft())
    return result
