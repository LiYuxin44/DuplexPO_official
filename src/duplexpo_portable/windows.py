"""Dynamics-critical window construction and deterministic event sampling."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import random
from collections.abc import Mapping, Sequence
from typing import Any

from .schema import Segment, Window


@dataclass(frozen=True)
class WindowConfig:
    frame_duration_s: float = 0.08
    lead_time_s: float = 1.0
    buffer_s: float = 2.0
    min_window_start_s: float = 2.0
    min_turn_duration_s: float = 2.0
    min_backchannel_duration_s: float = 0.0
    max_turn_windows: int = 3
    max_backchannel_windows: int = 1
    max_rollout_frames: int = 200
    selection_seed: int = 42
    # Minimum number of frames the sampled span must extend past the reference
    # event end so that a timely stop is observable inside the window
    # (3 frames = 0.24 s, the reward's stop tolerance). When the next event's
    # lead would cut the span shorter than this, the span is extended (and may
    # overlap the next window's lead); windows whose reference onset would still
    # fall outside the span are dropped and counted.
    reference_margin_frames: int = 3

    def __post_init__(self) -> None:
        if self.frame_duration_s <= 0:
            raise ValueError("frame_duration_s must be positive")
        if self.reference_margin_frames < 0:
            raise ValueError("reference_margin_frames must be non-negative")
        if self.lead_time_s < 0 or self.buffer_s < 0:
            raise ValueError("lead and buffer must be non-negative")
        if self.max_turn_windows < 0 or self.max_backchannel_windows < 0:
            raise ValueError("window limits must be non-negative")
        if self.max_rollout_frames <= 0:
            raise ValueError("max_rollout_frames must be positive")


def build_windows_with_audit(
    *,
    conversation_id: str,
    split: str,
    source: str,
    duration_s: float,
    agent_segments: Sequence[Segment],
    user_segments: Sequence[Segment] = (),
    conversation_metadata: Mapping[str, Any] | None = None,
    config: WindowConfig = WindowConfig(),
) -> tuple[list[Window], dict[str, int]]:
    """Convert ordered reference events to candidate windows plus drop/extend counts.

    Frames before ``sample_start_frame`` are the teacher-forced prefix. Only
    ``[sample_start_frame, sample_end_frame)`` is sampled and optimized.

    The nominal span follows the paper: ``s_i = max(e_{i-1}, (a_i - L)/Δ)`` and
    ``e_i = min((b_i + B)/Δ, (a_{i+1} - L)/Δ, T)``. When the next event starts
    within the lead time of this event's end, the nominal ``e_i`` can fall
    before the reference end (or even before the reference onset), making the
    event unscorable even for the human reference. The span is therefore
    extended to ``reference_end + reference_margin_frames`` for *this* window
    only; the chaining value used for the next window's ``s_{i+1}`` is left at
    the nominal ``e_i`` so all other windows are unchanged. Windows whose
    reference onset is still outside the span are dropped.
    """

    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    frame_dur = config.frame_duration_s
    total_frames = int(duration_s / frame_dur)
    ordered = sorted(agent_segments, key=lambda item: (item.start_s, item.end_s))
    windows: list[Window] = []
    stats = {"windows_dropped_unsatisfiable": 0, "windows_extended_past_next_lead": 0}
    previous_end = 0
    for index, segment in enumerate(ordered):
        sample_start = max(previous_end, int((segment.start_s - config.lead_time_s) / frame_dur))
        raw_end = int((segment.end_s + config.buffer_s) / frame_dur)
        if index + 1 < len(ordered):
            next_start = max(
                previous_end,
                int((ordered[index + 1].start_s - config.lead_time_s) / frame_dur),
            )
            nominal_end = min(raw_end, next_start)
        else:
            nominal_end = min(raw_end, total_frames)
        reference_start = int(segment.start_s / frame_dur)
        reference_end = int(segment.end_s / frame_dur)
        previous_end = nominal_end
        sample_end = min(raw_end, total_frames, max(nominal_end, reference_end + config.reference_margin_frames))
        extended = sample_end - nominal_end
        if sample_end <= sample_start or sample_end <= reference_start:
            stats["windows_dropped_unsatisfiable"] += 1
            continue
        if extended > 0:
            stats["windows_extended_past_next_lead"] += 1
        metadata = {
            "agent_segment": segment.to_dict(),
            "event_index": index,
            "conversation": dict(conversation_metadata or {}),
        }
        if extended > 0:
            metadata["sample_end_extended_frames"] = extended
        windows.append(
            Window(
                window_id=f"{conversation_id}:event-{index:04d}",
                conversation_id=conversation_id,
                split=split,
                frame_duration_s=frame_dur,
                sample_start_frame=sample_start,
                sample_end_frame=sample_end,
                reference_start_frame=reference_start,
                reference_end_frame=reference_end,
                event_kind=segment.kind,
                user_segments=tuple(user_segments),
                source=source,
                metadata=metadata,
            )
        )
    return windows, stats


def build_windows(**kwargs) -> list[Window]:
    """See :func:`build_windows_with_audit`; returns only the windows."""

    return build_windows_with_audit(**kwargs)[0]


def _stable_seed(conversation_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{conversation_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def historical_token_selection_seed(target_ids: Sequence[int], total_frames: int | None = None) -> int:
    """Reproduce the paper-time event-selection seed from target frame tokens."""

    if not target_ids:
        raise ValueError("target_ids must not be empty")
    length = len(target_ids) if total_frames is None else min(int(total_frames), len(target_ids))
    if length <= 0:
        raise ValueError("total_frames must be positive")
    middle = min(length // 2, length - 1)
    return int(target_ids[0]) + int(target_ids[middle])


def select_windows(
    windows: Sequence[Window], config: WindowConfig = WindowConfig(), *, rng_seed: int | None = None
) -> list[Window]:
    """Select the paper event mix deterministically and clip rollout length."""

    conversation_ids = {window.conversation_id for window in windows}
    if len(conversation_ids) > 1:
        raise ValueError("select_windows accepts one conversation at a time")
    if any(abs(window.frame_duration_s - config.frame_duration_s) > 1e-12 for window in windows):
        raise ValueError("window frame duration differs from selection config")
    min_start_frames = int(config.min_window_start_s / config.frame_duration_s)
    min_turn_frames = int(config.min_turn_duration_s / config.frame_duration_s)
    min_bc_frames = int(config.min_backchannel_duration_s / config.frame_duration_s)
    turns = [
        window
        for window in windows
        if window.event_kind == "turn"
        and window.reference_end_frame - window.reference_start_frame >= min_turn_frames
        and window.sample_start_frame >= min_start_frames
    ]
    backchannels = [
        window
        for window in windows
        if window.event_kind == "backchannel"
        and window.reference_end_frame - window.reference_start_frame >= min_bc_frames
        and window.sample_start_frame >= min_start_frames
    ]
    conversation_id = windows[0].conversation_id if windows else "empty"
    effective_seed = _stable_seed(conversation_id, config.selection_seed) if rng_seed is None else int(rng_seed)
    rng = random.Random(effective_seed)
    if len(turns) > config.max_turn_windows:
        turns = rng.sample(turns, config.max_turn_windows)
    if len(backchannels) > config.max_backchannel_windows:
        backchannels = rng.sample(backchannels, config.max_backchannel_windows)

    selected = []
    for window in sorted(turns + backchannels, key=lambda item: item.sample_start_frame):
        clipped_end = min(
            window.sample_end_frame,
            window.sample_start_frame + config.max_rollout_frames,
        )
        if clipped_end > window.sample_start_frame:
            metadata = dict(window.metadata)
            metadata.update(
                {
                    "selection_seed": config.selection_seed,
                    "effective_selection_seed": effective_seed,
                    "selection_seed_strategy": (
                        "sha256(seed:conversation_id)" if rng_seed is None else "explicit_override"
                    ),
                    "rollout_was_clipped": clipped_end < window.sample_end_frame,
                }
            )
            selected.append(replace(window, sample_end_frame=clipped_end, metadata=metadata))
    return selected


def build_selected_windows(
    *,
    conversation_id: str,
    split: str,
    source: str,
    duration_s: float,
    agent_segments: Sequence[Segment],
    user_segments: Sequence[Segment] = (),
    conversation_metadata: Mapping[str, Any] | None = None,
    config: WindowConfig = WindowConfig(),
    rng_seed: int | None = None,
    audit: dict[str, Any] | None = None,
) -> list[Window]:
    """Build then select windows; if ``audit`` is given, drop/extend counts are added to it."""

    candidates, stats = build_windows_with_audit(
        conversation_id=conversation_id,
        split=split,
        source=source,
        duration_s=duration_s,
        agent_segments=agent_segments,
        user_segments=user_segments,
        conversation_metadata=conversation_metadata,
        config=config,
    )
    if audit is not None:
        audit.update(stats)
    return select_windows(candidates, config, rng_seed=rng_seed)
