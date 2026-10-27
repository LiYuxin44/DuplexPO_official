"""Portable, framework-neutral DuplexPO reference implementation."""

from .construction import ConstructionProfile, construct_conversation, profile_named
from .conformance import ConformanceReport, check_grpo_records, check_policy_adapter, check_rollouts
from .grpo import GRPOConfig, build_grpo_records, normalize_grouped_advantages
from .recipe import load_paper_recipe, paper_grpo_config, paper_reward_config, paper_window_config, recipe_fingerprint
from .rewards import FCDRConfig, SilenceEosTraceDecoder, TokenTraceDecoder, score_fcdr
from .schema import ActionTrace, Conversation, GRPORecord, Rollout, Segment, SpeechInterval, TranscriptUnit, Window, load_json_schema
from .windows import WindowConfig, build_windows, historical_token_selection_seed, select_windows

__all__ = [
    "ActionTrace",
    "ConstructionProfile",
    "ConformanceReport",
    "Conversation",
    "FCDRConfig",
    "GRPOConfig",
    "GRPORecord",
    "Rollout",
    "Segment",
    "SpeechInterval",
    "SilenceEosTraceDecoder",
    "TokenTraceDecoder",
    "TranscriptUnit",
    "Window",
    "WindowConfig",
    "build_grpo_records",
    "build_windows",
    "historical_token_selection_seed",
    "construct_conversation",
    "check_grpo_records",
    "check_policy_adapter",
    "check_rollouts",
    "normalize_grouped_advantages",
    "load_paper_recipe",
    "load_json_schema",
    "paper_grpo_config",
    "paper_reward_config",
    "paper_window_config",
    "recipe_fingerprint",
    "profile_named",
    "score_fcdr",
    "select_windows",
]

__version__ = "0.1.0"
