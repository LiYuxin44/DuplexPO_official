"""Executable checks for third-party policy and trainer adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .grpo import GRPOConfig, normalize_grouped_advantages
from .schema import Rollout, Window


@dataclass
class ConformanceReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, **asdict(self)}


def check_rollouts(
    windows: Mapping[str, Window],
    rollouts: Sequence[Rollout],
    *,
    expected_samples_per_window: int = 4,
    require_trainable: bool = False,
) -> ConformanceReport:
    """Check grouping, timing domain, action alignment, and KL replay fields."""

    report = ConformanceReport()
    counts = Counter(rollout.window_id for rollout in rollouts)
    rollout_ids = [rollout.rollout_id for rollout in rollouts]
    if len(set(rollout_ids)) != len(rollout_ids):
        report.errors.append("rollout_id values must be globally unique")

    for window_id in sorted(windows):
        count = counts.get(window_id, 0)
        if count != expected_samples_per_window:
            report.errors.append(
                f"{window_id}: expected {expected_samples_per_window} rollouts, found {count}"
            )
    for window_id in sorted(set(counts) - set(windows)):
        report.errors.append(f"rollout references unknown window_id {window_id}")

    actionless = 0
    missing_kl = 0
    for rollout in rollouts:
        window = windows.get(rollout.window_id)
        if window is None:
            continue
        if rollout.trace is not None:
            for interval in rollout.trace.speech:
                tolerance = window.frame_duration_s + 1e-9
                if interval.start_s < window.sample_start_s - tolerance:
                    report.errors.append(f"{rollout.rollout_id}: trace starts before sampled window")
                if interval.start_s > window.sample_end_s + tolerance:
                    report.errors.append(f"{rollout.rollout_id}: trace starts after sampled window")
                if interval.end_s is not None and interval.end_s > window.sample_end_s + tolerance:
                    report.errors.append(f"{rollout.rollout_id}: trace ends after sampled window")
        if not rollout.action_ids:
            actionless += 1
            if require_trainable:
                report.errors.append(f"{rollout.rollout_id}: trainable mode requires action_ids")
        if not rollout.reference_logprobs and not rollout.behavior_logprobs:
            missing_kl += 1
            if require_trainable:
                report.errors.append(
                    f"{rollout.rollout_id}: trainable KL mode requires reference or behavior logprobs"
                )
    if actionless and not require_trainable:
        report.warnings.append(
            f"{actionless} rollout(s) are reward-only traces; add action_ids before policy optimization"
        )
    if missing_kl and not require_trainable:
        report.warnings.append(
            f"{missing_kl} rollout(s) lack KL reference logprobs; valid for reward conformance only"
        )
    report.stats = {
        "window_count": len(windows),
        "rollout_count": len(rollouts),
        "group_sizes": dict(sorted(counts.items())),
        "actionless_rollouts": actionless,
        "rollouts_without_kl_reference": missing_kl,
        "mode": "trainable" if require_trainable else "reward",
    }
    return report


def check_grpo_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_samples_per_window: int = 4,
    require_trainable: bool = False,
    config: GRPOConfig = GRPOConfig(),
) -> ConformanceReport:
    """Validate the stable trainer boundary and independently recompute advantages."""

    report = ConformanceReport()
    required = {
        "schema_version", "group_id", "rollout_id", "window", "action_ids", "loss_mask", "reward", "advantage"
    }
    groups: dict[str, list[int]] = defaultdict(list)
    seen_rollouts: set[str] = set()
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            report.errors.append(f"record {index}: missing fields {sorted(missing)}")
            continue
        if record["schema_version"] != "duplexpo.grpo.v1":
            report.errors.append(f"record {index}: unsupported schema_version")
        rollout_id = str(record["rollout_id"])
        if rollout_id in seen_rollouts:
            report.errors.append(f"duplicate rollout_id {rollout_id}")
        seen_rollouts.add(rollout_id)
        group_id = str(record["group_id"])
        embedded_window = record["window"]
        if not isinstance(embedded_window, Mapping):
            report.errors.append(f"{rollout_id}: embedded window must be an object")
        elif str(embedded_window.get("window_id", "")) != group_id:
            report.errors.append(f"{rollout_id}: embedded window_id differs from group_id")
        groups[group_id].append(index)
        arrays = {}
        for name in ("action_ids", "loss_mask", "reference_logprobs", "behavior_logprobs"):
            raw = record.get(name, [])
            if not isinstance(raw, (list, tuple)):
                report.errors.append(f"{rollout_id}: {name} must be an array")
                arrays[name] = []
            else:
                arrays[name] = list(raw)
        actions = arrays["action_ids"]
        mask = arrays["loss_mask"]
        reference = arrays["reference_logprobs"]
        behavior = arrays["behavior_logprobs"]
        if len(actions) != len(mask):
            report.errors.append(f"{rollout_id}: action_ids/loss_mask length mismatch")
        if reference and len(reference) != len(actions):
            report.errors.append(f"{rollout_id}: reference_logprobs length mismatch")
        if behavior and len(behavior) != len(actions):
            report.errors.append(f"{rollout_id}: behavior_logprobs length mismatch")
        if require_trainable and not actions:
            report.errors.append(f"{rollout_id}: trainable record has no actions")
        if require_trainable and not reference:
            report.errors.append(f"{rollout_id}: KL reference_logprobs are missing")
        for name in ("reward", "advantage"):
            try:
                value = float(record[name])
            except (TypeError, ValueError):
                report.errors.append(f"{rollout_id}: {name} is not numeric")
                continue
            if not math.isfinite(value):
                report.errors.append(f"{rollout_id}: {name} is not finite")
        try:
            if abs(float(record["advantage"])) > config.advantage_clip + 1e-7:
                report.errors.append(f"{rollout_id}: advantage exceeds configured clip")
        except (TypeError, ValueError):
            pass

    for group_id, indices in sorted(groups.items()):
        if len(indices) != expected_samples_per_window:
            report.errors.append(
                f"{group_id}: expected {expected_samples_per_window} records, found {len(indices)}"
            )
        try:
            rewards = [float(records[index]["reward"]) for index in indices]
            expected = normalize_grouped_advantages(
                rewards,
                [group_id] * len(rewards),
                epsilon=config.advantage_epsilon,
                clip=config.advantage_clip,
            )
            actual = [float(records[index]["advantage"]) for index in indices]
            if any(abs(left - right) > 1e-6 for left, right in zip(actual, expected)):
                report.errors.append(f"{group_id}: advantages do not match per-event normalization")
        except (KeyError, TypeError, ValueError):
            pass
    report.stats = {
        "record_count": len(records),
        "group_count": len(groups),
        "group_sizes": {group: len(indices) for group, indices in sorted(groups.items())},
        "mode": "trainable" if require_trainable else "reward",
    }
    return report


def check_policy_adapter(
    adapter,
    windows: Sequence[Window],
    *,
    samples_per_window: int = 4,
    require_trainable: bool = True,
) -> tuple[list[Rollout], ConformanceReport]:
    """Run an arbitrary `PolicyAdapter` and validate its public outputs."""

    generated = list(adapter.generate(windows, samples_per_window))
    report = check_rollouts(
        {window.window_id: window for window in windows},
        generated,
        expected_samples_per_window=samples_per_window,
        require_trainable=require_trainable,
    )
    return generated, report
