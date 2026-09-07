"""Portable orchestration: windows + arbitrary rollouts -> trainer records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .grpo import GRPOConfig, build_grpo_records
from .rewards import FCDRConfig, score_fcdr
from .schema import ActionTrace, RewardResult, Rollout, Window


def score_rollouts(
    windows: Mapping[str, Window],
    rollouts: Sequence[Rollout],
    *,
    decoder=None,
    reward_config: FCDRConfig = FCDRConfig(),
) -> list[tuple[Rollout, RewardResult]]:
    scored = []
    seen_ids: set[str] = set()
    for rollout in rollouts:
        if rollout.rollout_id in seen_ids:
            raise ValueError(f"duplicate rollout_id: {rollout.rollout_id}")
        seen_ids.add(rollout.rollout_id)
        try:
            window = windows[rollout.window_id]
        except KeyError as error:
            raise ValueError(f"unknown window_id: {rollout.window_id}") from error
        trace: ActionTrace
        if rollout.trace is not None:
            trace = rollout.trace
        elif decoder is not None:
            trace = decoder.decode(rollout, window)
        else:
            raise ValueError("action-only rollout requires a trace decoder")
        scored.append((rollout, score_fcdr(window, trace, reward_config)))
    return scored


def prepare_grpo_update(
    windows: Mapping[str, Window],
    rollouts: Sequence[Rollout],
    *,
    decoder=None,
    reward_config: FCDRConfig = FCDRConfig(),
    grpo_config: GRPOConfig = GRPOConfig(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored = score_rollouts(
        windows,
        rollouts,
        decoder=decoder,
        reward_config=reward_config,
    )
    records = build_grpo_records(scored, windows, grpo_config)
    group_sizes = Counter(record["group_id"] for record in records)
    audit = {
        "schema_version": "duplexpo.audit.v1",
        "rollout_count": len(records),
        "group_count": len(group_sizes),
        "group_sizes": dict(group_sizes),
        "singleton_groups": sorted(group for group, size in group_sizes.items() if size < 2),
        "all_groups_have_learning_signal": all(size >= 2 for size in group_sizes.values()),
    }
    return records, audit
