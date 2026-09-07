"""Factorized conversational-dynamics reward on model-neutral action traces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable

from .schema import ActionTrace, RewardResult, Rollout, SpeechInterval, Window


@dataclass(frozen=True)
class FCDRConfig:
    """Paper reward constants. ``legacy`` reproduces the reported checkpoint."""

    mode: str = "legacy"
    backchannel_plateau_s: float = 0.5
    backchannel_max_talk_s: float = 1.5
    turn_sigma_early_s: float = 0.2
    turn_sigma_late_s: float = 0.5
    hard_cutoff: float | None = None
    miss_penalty: float = -0.5
    false_alarm_penalty: float = -0.5
    backchannel_overlong_penalty: float = -0.5
    turn_no_eos_scale: float = 0.75
    observable_margin_s: float = 0.16
    interruption_grace_s: float = 0.16
    stop_tolerance_s: float = 0.24
    explicit_eos_bonus: float = 0.05
    near_target_pad_penalty: float = -0.02
    early_stop_penalty: float = -0.15
    late_stop_penalty: float = -0.10

    def __post_init__(self) -> None:
        if self.mode not in {"legacy", "eos_aware_v1"}:
            raise ValueError("mode must be 'legacy' or 'eos_aware_v1'")


class TokenTraceDecoder:
    """Convert BOS/EOS/PAD frame actions to the generic timeline contract."""

    def __init__(self, *, bos_ids: Iterable[int], eos_id: int, pad_id: int):
        self.bos_ids = frozenset(int(item) for item in bos_ids)
        self.eos_id = int(eos_id)
        self.pad_id = int(pad_id)
        if not self.bos_ids:
            raise ValueError("bos_ids must not be empty")

    def decode(self, rollout: Rollout, window: Window) -> ActionTrace:
        bos_position = None
        stop_position = None
        stop_kind = "unknown"
        for index, token in enumerate(rollout.action_ids):
            if bos_position is None:
                if token in self.bos_ids:
                    bos_position = index
                continue
            if token == self.eos_id:
                stop_position, stop_kind = index, "eos"
                break
            if token == self.pad_id:
                stop_position, stop_kind = index, "pad"
                break
        if bos_position is None:
            return ActionTrace(metadata={"decoder": "bos_eos_pad_v1", "bos_found": False})
        start_s = (window.sample_start_frame + bos_position) * window.frame_duration_s
        end_s = None
        if stop_position is not None:
            end_s = (window.sample_start_frame + stop_position) * window.frame_duration_s
        return ActionTrace(
            speech=(SpeechInterval(start_s=start_s, end_s=end_s, termination=stop_kind),),
            metadata={
                "decoder": "bos_eos_pad_v1",
                "bos_found": True,
                "bos_position": bos_position,
                "stop_position": stop_position,
            },
        )


def _next_user_start(window: Window, anchor_s: float) -> float | None:
    for segment in sorted(window.user_segments, key=lambda item: item.start_s):
        if anchor_s <= segment.start_s <= window.sample_end_s:
            return segment.start_s
    return None


def _turn_end_target(window: Window, config: FCDRConfig) -> tuple[float, str, float | None]:
    next_user = _next_user_start(window, window.reference_start_s)
    desired = window.reference_end_s
    target = "natural"
    if next_user is not None and next_user < window.reference_end_s:
        desired = min(window.reference_end_s, next_user + config.interruption_grace_s)
        target = "interrupted"
    return desired, target, next_user


def score_fcdr(
    window: Window,
    trace: ActionTrace,
    config: FCDRConfig = FCDRConfig(),
) -> RewardResult:
    """Score onset, backchannel pattern, and yielding without model dependencies."""

    prediction = trace.speech[0] if trace.speech else None
    has_reference = window.reference_end_frame > window.reference_start_frame
    onset_component = 0.0
    yielding_component = 0.0
    pattern_component = 0.0
    category = "correct_silence"
    desired_end = None
    end_target = None
    next_user = None
    end_observable = None
    onset_error = None
    stop_error = None

    if not has_reference:
        if prediction is None:
            reward, category = 0.0, "correct_silence"
        else:
            reward, category = config.false_alarm_penalty, "false_alarm"
            onset_component = reward
    elif prediction is None:
        reward, category = config.miss_penalty, "miss"
        onset_component = reward
    else:
        onset_error = prediction.start_s - window.reference_start_s
        is_backchannel = window.event_kind == "backchannel" or (
            window.reference_end_s - window.reference_start_s < 1.0
        )
        if is_backchannel:
            category = "backchannel"
            absolute_error = abs(onset_error)
            onset_component = (
                0.5
                if absolute_error <= config.backchannel_plateau_s
                else max(-0.5, 0.5 - (absolute_error - config.backchannel_plateau_s))
            )
        else:
            category = "turn"
            sigma = config.turn_sigma_early_s if onset_error < 0 else config.turn_sigma_late_s
            onset_component = math.exp(-(onset_error**2) / (2 * sigma**2))
        if config.hard_cutoff is not None:
            onset_component = max(config.hard_cutoff, onset_component)
        reward = onset_component

        if reward > 0 and is_backchannel:
            duration = math.inf if prediction.end_s is None else prediction.end_s - prediction.start_s
            if prediction.end_s is None or duration > config.backchannel_max_talk_s:
                updated = min(reward, config.backchannel_overlong_penalty)
                pattern_component = updated - reward
                reward = updated
        elif reward > 0:
            desired_end, end_target, next_user = _turn_end_target(window, config)
            end_observable = window.sample_end_s >= desired_end + config.observable_margin_s
            if config.mode == "legacy":
                if prediction.end_s is None:
                    updated = reward * config.turn_no_eos_scale
                    pattern_component = updated - reward
                    reward = updated
            elif end_observable:
                if prediction.end_s is None:
                    yielding_component = config.late_stop_penalty
                else:
                    stop_error = prediction.end_s - desired_end
                    if stop_error < -config.stop_tolerance_s:
                        yielding_component = config.early_stop_penalty
                    elif stop_error > config.stop_tolerance_s:
                        yielding_component = config.late_stop_penalty
                    elif prediction.termination == "eos":
                        yielding_component = config.explicit_eos_bonus
                    else:
                        yielding_component = config.near_target_pad_penalty
                reward += yielding_component

    if prediction is not None and desired_end is not None and stop_error is None and prediction.end_s is not None:
        stop_error = prediction.end_s - desired_end
    diagnostics = {
        "category": category,
        "event_kind": window.event_kind,
        "has_reference": has_reference,
        "predicted_onset_s": None if prediction is None else prediction.start_s,
        "predicted_stop_s": None if prediction is None else prediction.end_s,
        "predicted_termination": None if prediction is None else prediction.termination,
        "reference_start_s": window.reference_start_s,
        "reference_end_s": window.reference_end_s,
        "sample_end_s": window.sample_end_s,
        "onset_error_s": onset_error,
        "stop_error_s": stop_error,
        "desired_end_s": desired_end,
        "end_target": end_target,
        "next_user_start_s": next_user,
        "end_observable": end_observable,
        "reward_mode": config.mode,
    }
    return RewardResult(
        total=reward,
        components={
            "onset": onset_component,
            "yielding": yielding_component,
            "pattern": pattern_component,
        },
        diagnostics=diagnostics,
    )
