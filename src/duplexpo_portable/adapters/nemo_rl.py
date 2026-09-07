"""Dictionary bridge for NeMo RL policies and per-component rewards."""

from __future__ import annotations

from ..schema import Window


def to_nemo_rl_row(window: Window, prompt="") -> dict:
    return {
        "prompt": prompt,
        "window_id": window.window_id,
        "conversation_id": window.conversation_id,
        "duplexpo_window": window.to_dict(),
    }


def reward_columns(record: dict) -> dict[str, float]:
    """Expose the total plus ``reward/<component>`` metrics used by NeMo RL."""

    result = {"reward": float(record["reward"])}
    for name, value in record.get("reward_components", {}).items():
        result[f"reward/{name}"] = float(value)
    return result
