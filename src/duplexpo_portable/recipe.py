"""Load the machine-readable paper recipe bundled in wheels and source trees."""

from __future__ import annotations

from importlib.resources import files
import hashlib
import json
from typing import Any

from .grpo import GRPOConfig
from .rewards import FCDRConfig
from .windows import WindowConfig

RECIPE_SCHEMA_VERSION = "duplexpo.recipe.v1"


def load_paper_recipe() -> dict[str, Any]:
    path = files("duplexpo_portable").joinpath("recipes/paper_v1.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != RECIPE_SCHEMA_VERSION:
        raise ValueError("unsupported bundled recipe schema")
    return value


def recipe_fingerprint(recipe: dict[str, Any] | None = None) -> str:
    """SHA-256 of canonical JSON, stable across whitespace and key ordering."""

    value = load_paper_recipe() if recipe is None else recipe
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def paper_window_config(*, validation: bool = False) -> WindowConfig:
    value = load_paper_recipe()["windows"]
    return WindowConfig(
        frame_duration_s=float(value["frame_duration_s"]),
        lead_time_s=float(value["validation_lead_time_s"] if validation else value["training_lead_time_s"]),
        buffer_s=float(value["buffer_s"]),
        min_window_start_s=float(value["min_window_start_s"]),
        min_turn_duration_s=float(value["min_turn_duration_s"]),
        min_backchannel_duration_s=float(value["min_backchannel_duration_s"]),
        max_turn_windows=int(value["max_turn_windows"]),
        max_backchannel_windows=int(value["max_backchannel_windows"]),
        max_rollout_frames=int(value["max_rollout_frames"]),
        selection_seed=int(value["selection_seed"]),
        reference_margin_frames=int(value.get("reference_margin_frames", 3)),
    )


def paper_reward_config() -> FCDRConfig:
    value = load_paper_recipe()["reward"]
    return FCDRConfig(**value)


def paper_grpo_config() -> GRPOConfig:
    value = load_paper_recipe()["grpo"]
    return GRPOConfig(
        advantage_epsilon=float(value["advantage_epsilon"]),
        advantage_clip=float(value["advantage_clip"]),
        kl_beta=float(value["kl_beta"]),
        kl_log_ratio_clip=float(value["kl_log_ratio_clip"]),
        kl_value_clip=float(value["kl_value_clip"]),
    )
