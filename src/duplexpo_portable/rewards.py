"""Factorized conversational-dynamics reward on model-neutral action traces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable

from .schema import ActionTrace, RewardResult, Rollout, SpeechInterval, Window


@dataclass(frozen=True)
class FCDRConfig:
    """Paper reward constants (the reward used for the reported checkpoint)."""

    backchannel_plateau_s: float = 0.5
    backchannel_max_talk_s: float = 1.5
    turn_sigma_early_s: float = 0.2
    turn_sigma_late_s: float = 0.5
    hard_cutoff: float | None = None
    miss_penalty: float = -0.5
    false_alarm_penalty: float = -0.5
    backchannel_overlong_penalty: float = -0.5
    turn_no_eos_scale: float = 0.75
    # Only used to derive the diagnostic ``desired_end_s`` / ``end_observable``
    # fields (never enters the scalar reward).
    observable_margin_s: float = 0.16
    interruption_grace_s: float = 0.16


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


class SilenceEosTraceDecoder:
    """Decode vocabularies without an explicit BOS token.

    Speech starts at the first frame whose action is not in ``silence_ids``
    (e.g. PAD/silence) and not ``eos_id``; it ends at the first ``eos_id`` after
    that (``termination="eos"``) or at the first return to a silence token when
    ``silence_ends_speech`` is set (``termination="pad"``). This covers text-stream
    models that emit PAD while silent, text tokens while speaking and EOS to yield.
    """

    def __init__(self, *, silence_ids: Iterable[int], eos_id: int, silence_ends_speech: bool = False):
        self.silence_ids = frozenset(int(item) for item in silence_ids)
        self.eos_id = int(eos_id)
        self.silence_ends_speech = bool(silence_ends_speech)
        if not self.silence_ids:
            raise ValueError("silence_ids must not be empty")

    def decode(self, rollout: Rollout, window: Window) -> ActionTrace:
        speech: list[SpeechInterval] = []
        start_position = None
        for index, token in enumerate(rollout.action_ids):
            is_silence = token in self.silence_ids
            if start_position is None:
                if not is_silence and token != self.eos_id:
                    start_position = index
                continue
            if token == self.eos_id or (is_silence and self.silence_ends_speech):
                kind = "eos" if token == self.eos_id else "pad"
                speech.append(self._interval(window, start_position, index, kind))
                start_position = None
        if start_position is not None:
            speech.append(self._interval(window, start_position, None, "unknown"))
        return ActionTrace(
            speech=tuple(speech),
            metadata={"decoder": "silence_eos_v1", "bos_found": bool(speech)},
        )

    @staticmethod
    def _interval(window: Window, start: int, stop: int | None, kind: str) -> SpeechInterval:
        frame = window.frame_duration_s
        start_s = (window.sample_start_frame + start) * frame
        end_s = None if stop is None else (window.sample_start_frame + stop) * frame
        return SpeechInterval(start_s=start_s, end_s=end_s, termination=kind)


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
    """Score onset and backchannel pattern of ``trace.speech[0]`` without model dependencies."""

    prediction = trace.speech[0] if trace.speech else None
    has_reference = window.reference_end_frame > window.reference_start_frame
    onset_component = 0.0
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
            # Yielding is only constrained indirectly: a positive full turn that
            # never stops inside the window is scaled by ``turn_no_eos_scale``.
            if prediction.end_s is None:
                updated = reward * config.turn_no_eos_scale
                pattern_component = updated - reward
                reward = updated

    if prediction is not None and desired_end is not None and stop_error is None and prediction.end_s is not None:
        stop_error = prediction.end_s - desired_end
    # Only ``speech[0]`` is scored. A policy can therefore emit a very short first
    # segment (collecting the onset reward and avoiding the no-EOS scale) and then
    # start speaking again for free. This is reported, not penalized.
    restart_after_eos = (
        prediction is not None
        and prediction.end_s is not None
        and any(interval.start_s >= prediction.end_s for interval in trace.speech[1:])
    )
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
        "restart_after_eos": restart_after_eos,
        "extra_speech_segments": max(len(trace.speech) - 1, 0),
    }
    return RewardResult(
        total=reward,
        components={
            "onset": onset_component,
            "pattern": pattern_component,
        },
        diagnostics=diagnostics,
    )
