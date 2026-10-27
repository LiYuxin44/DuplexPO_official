"""Dependency-free examples for mapping Fisher/Seamless metadata to v1 input."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

from ..schema import Conversation, TranscriptUnit

PROMPT_INITIATOR_PATTERNS = (
    (re.compile(r"\bask your partner\b"), 4),
    (re.compile(r"\btell (?:the story|your partner)\b"), 4),
    (re.compile(r"\bgo first\b"), 4),
    (re.compile(r"\bstart by\b"), 3),
    (re.compile(r"\bbegin by\b"), 3),
    (re.compile(r"\bshare\b"), 2),
    (re.compile(r"\bdescribe\b"), 2),
    (re.compile(r"\bexplain\b"), 2),
    (re.compile(r"\bconvince\b"), 2),
    (re.compile(r"\bquestion your partner\b"), 3),
)
PROMPT_RESPONDER_PATTERNS = (
    (re.compile(r"\byour partner will\b"), 4),
    (re.compile(r"\blisten\b"), 3),
    (re.compile(r"\brespond\b"), 3),
    (re.compile(r"\banswer\b"), 3),
    (re.compile(r"\bdiscuss your answer\b"), 3),
    (re.compile(r"\btake your time\b"), 2),
    (re.compile(r"\bfollow along\b"), 2),
    (re.compile(r"\bguess\b"), 2),
)


MAX_UNTIMED_FRACTION = 0.2
"""Default upper bound on the share of a speaker's units without usable timing."""


def _timing(value: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return ``(start, end)`` or ``None`` when either timestamp is missing/non-finite.

    Word- and utterance-level alignments from both Fisher-style and Seamless-style
    sources occasionally carry ``null`` timestamps (numerals, non-lexical marks).
    They cannot be placed on the frame grid and must be dropped rather than
    interpolated, otherwise onset labels become guesses.
    """

    start, end = value.get("start"), value.get("end")
    if start is None or end is None:
        return None
    try:
        start_f, end_f = float(start), float(end)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(start_f) and math.isfinite(end_f)):
        return None
    return start_f, end_f


def _shifted_unit(speaker: str, value: Mapping[str, Any], shift_s: float, text_key: str):
    timing = _timing(value)
    if timing is None:
        return None
    start = timing[0] - shift_s
    end = timing[1] - shift_s
    if end <= 0:
        return None
    start = max(0.0, start)
    if end <= start:
        return None
    text = str(value.get(text_key, "")).strip()
    if not text:
        return None
    return TranscriptUnit(speaker, start, end, text)


def _untimed_audit(
    channels: Mapping[str, Sequence[Mapping[str, Any]]], max_untimed_fraction: float
) -> dict[str, Any]:
    """Count units without timing per canonical speaker and flag excessive loss."""

    dropped = {speaker: sum(1 for value in values if _timing(value) is None) for speaker, values in channels.items()}
    fraction = {
        speaker: (dropped[speaker] / len(values) if values else 0.0) for speaker, values in channels.items()
    }
    return {
        "untimed_units_dropped": dropped,
        "untimed_fraction": fraction,
        "untimed_fraction_exceeded": any(value > max_untimed_fraction for value in fraction.values()),
        "max_untimed_fraction": max_untimed_fraction,
    }


def from_fisher_alignments(
    *,
    conversation_id: str,
    duration_s: float,
    user_alignment: Sequence[Mapping[str, Any]],
    agent_alignment: Sequence[Mapping[str, Any]],
    shift_s: float = 0.0,
    swap_by_first_speaker: bool = True,
    split: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    max_untimed_fraction: float = MAX_UNTIMED_FRACTION,
) -> Conversation:
    """Map historical ``{word,start,end}`` alignments to canonical words.

    The leading agent-side system pseudo-token is discarded. By default the
    earliest speaker becomes canonical ``user``, matching the submitted Fisher
    shard job; disable ``swap_by_first_speaker`` for authoritative roles.
    Words without usable timing are dropped and counted in
    ``metadata["untimed_units_dropped"]``; when a speaker loses more than
    ``max_untimed_fraction`` of its words the conversation is flagged
    (``metadata["untimed_fraction_exceeded"]``) and ``construct_conversation``
    excludes it.
    """

    untimed = _untimed_audit({"user": user_alignment, "agent": agent_alignment}, max_untimed_fraction)
    units = []
    for speaker, alignment in (("user", user_alignment), ("agent", agent_alignment)):
        for index, value in enumerate(alignment):
            if speaker == "agent" and index == 0 and "<system>" in str(value.get("word", "")):
                continue
            unit = _shifted_unit(speaker, value, shift_s, "word")
            if unit is not None and unit.start_s < duration_s:
                units.append(
                    TranscriptUnit(unit.speaker, unit.start_s, min(unit.end_s, duration_s), unit.text)
                )
    roles_swapped = False
    first = {
        speaker: min((unit.start_s for unit in units if unit.speaker == speaker), default=float("inf"))
        for speaker in ("user", "agent")
    }
    if swap_by_first_speaker and first["agent"] < first["user"]:
        roles_swapped = True
        units = [
            TranscriptUnit("user" if unit.speaker == "agent" else "agent", unit.start_s, unit.end_s, unit.text)
            for unit in units
        ]
    final_metadata = dict(metadata or {})
    final_metadata.update(untimed)
    final_metadata.update(
        {
            "role_source": "first_speaker" if swap_by_first_speaker else "explicit",
            "roles_swapped": roles_swapped,
            "canonical_audio_mapping": (
                {"user": "original_agent", "agent": "original_user"}
                if roles_swapped
                else {"user": "original_user", "agent": "original_agent"}
            ),
        }
    )
    return Conversation(
        conversation_id=conversation_id,
        duration_s=duration_s,
        units=tuple(sorted(units, key=lambda item: (item.start_s, item.speaker))),
        source="fisher",
        split=split,
        granularity="word",
        metadata=final_metadata,
    )


def from_seamless_transcripts(
    *,
    conversation_id: str,
    duration_s: float,
    user_transcript: Sequence[Mapping[str, Any]],
    agent_transcript: Sequence[Mapping[str, Any]],
    split: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    max_untimed_fraction: float = MAX_UNTIMED_FRACTION,
) -> Conversation:
    """Map canonical-role ``{transcript,start,end}`` utterances to v1 input.

    Utterances without usable timing are dropped and counted; see
    :func:`from_fisher_alignments` for the ``max_untimed_fraction`` rule.
    """

    untimed = _untimed_audit({"user": user_transcript, "agent": agent_transcript}, max_untimed_fraction)
    units = []
    for speaker, transcript in (("user", user_transcript), ("agent", agent_transcript)):
        for value in transcript:
            unit = _shifted_unit(speaker, value, 0.0, "transcript")
            if unit is not None and unit.start_s < duration_s:
                units.append(
                    TranscriptUnit(unit.speaker, unit.start_s, min(unit.end_s, duration_s), unit.text)
                )
    final_metadata = dict(metadata or {})
    final_metadata.update(untimed)
    return Conversation(
        conversation_id=conversation_id,
        duration_s=duration_s,
        units=tuple(sorted(units, key=lambda item: (item.start_s, item.speaker))),
        source="seamless",
        split=split,
        granularity="utterance",
        metadata=final_metadata,
    )


def _prompt_role_score(prompt: str) -> dict[str, int]:
    text = " ".join((prompt or "").strip().lower().split())
    return {
        "initiator": sum(weight for pattern, weight in PROMPT_INITIATOR_PATTERNS if pattern.search(text)),
        "responder": sum(weight for pattern, weight in PROMPT_RESPONDER_PATTERNS if pattern.search(text)),
    }


def resolve_seamless_roles(
    speaker_0_transcript: Sequence[Mapping[str, Any]],
    speaker_1_transcript: Sequence[Mapping[str, Any]],
    *,
    actor_speaker: str | None = None,
    both_actors: bool = False,
    speaker_0_prompt: str = "",
    speaker_1_prompt: str = "",
) -> tuple[int, int, str, float]:
    """Return ``(user_index, agent_index, source, confidence)`` from HQ-v1 rules."""

    if actor_speaker is not None:
        if actor_speaker not in {"speaker_0", "speaker_1"}:
            raise ValueError("actor_speaker must be speaker_0 or speaker_1")
        return (1, 0, "actor", 1.0) if actor_speaker == "speaker_0" else (0, 1, "actor", 1.0)
    if both_actors:
        return 0, 1, "both_actors", 0.9
    score_0 = _prompt_role_score(speaker_0_prompt)
    score_1 = _prompt_role_score(speaker_1_prompt)
    pref_0 = score_0["initiator"] - score_0["responder"]
    pref_1 = score_1["initiator"] - score_1["responder"]
    if pref_0 >= 2 and pref_1 <= -1:
        return 0, 1, "prompt", min(0.95, 0.65 + 0.05 * (pref_0 - pref_1))
    if pref_1 >= 2 and pref_0 <= -1:
        return 1, 0, "prompt", min(0.95, 0.65 + 0.05 * (pref_1 - pref_0))

    first_0 = min((t[0] for t in map(_timing, speaker_0_transcript) if t is not None), default=float("inf"))
    first_1 = min((t[0] for t in map(_timing, speaker_1_transcript) if t is not None), default=float("inf"))
    return (0, 1, "first_speaker", 0.25) if first_0 <= first_1 else (1, 0, "first_speaker", 0.25)


def from_seamless_pair(
    *,
    conversation_id: str,
    duration_s: float,
    speaker_0_transcript: Sequence[Mapping[str, Any]],
    speaker_1_transcript: Sequence[Mapping[str, Any]],
    actor_speaker: str | None = None,
    both_actors: bool = False,
    speaker_0_prompt: str = "",
    speaker_1_prompt: str = "",
    split: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Conversation:
    """Resolve historical Seamless roles, then emit canonical utterances."""

    user_index, agent_index, role_source, confidence = resolve_seamless_roles(
        speaker_0_transcript,
        speaker_1_transcript,
        actor_speaker=actor_speaker,
        both_actors=both_actors,
        speaker_0_prompt=speaker_0_prompt,
        speaker_1_prompt=speaker_1_prompt,
    )
    transcripts = [speaker_0_transcript, speaker_1_transcript]
    final_metadata = dict(metadata or {})
    final_metadata.update(
        {
            "role_source": role_source,
            "role_confidence": confidence,
            "user_original_speaker_index": user_index,
            "agent_original_speaker_index": agent_index,
            "speaker_0_prompt": speaker_0_prompt,
            "speaker_1_prompt": speaker_1_prompt,
        }
    )
    return from_seamless_transcripts(
        conversation_id=conversation_id,
        duration_s=duration_s,
        user_transcript=transcripts[user_index],
        agent_transcript=transcripts[agent_index],
        split=split,
        metadata=final_metadata,
    )
