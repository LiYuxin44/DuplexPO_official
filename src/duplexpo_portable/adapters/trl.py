"""Helpers for Hugging Face TRL's prompt-only GRPO contract."""

from __future__ import annotations

from collections.abc import Mapping

from ..rewards import FCDRConfig, score_fcdr
from ..schema import Rollout, Window


def to_trl_row(window: Window, prompt="") -> dict:
    """Keep DuplexPO fields as extra columns that TRL forwards to rewards."""

    return {
        "prompt": prompt,
        "window_id": window.window_id,
        "duplexpo_window": window.to_dict(),
    }


def make_trl_reward_function(
    windows: Mapping[str, Window],
    decoder,
    config: FCDRConfig = FCDRConfig(),
):
    """Create a TRL-compatible callable using ``completion_ids`` actions."""

    def duplexpo_reward(prompts, completions, completion_ids, window_id, **kwargs):
        del prompts, completions, kwargs
        rewards = []
        for index, (actions, event_id) in enumerate(zip(completion_ids, window_id)):
            rollout = Rollout(
                rollout_id=f"trl:{event_id}:{index}",
                window_id=str(event_id),
                action_ids=tuple(int(item) for item in actions),
            )
            window = windows[str(event_id)]
            rewards.append(score_fcdr(window, decoder.decode(rollout, window), config).total)
        return rewards

    duplexpo_reward.__name__ = "duplexpo_fcdr"
    return duplexpo_reward
