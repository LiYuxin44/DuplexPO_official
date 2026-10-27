"""Event-level grouped credit assignment and backend-neutral GRPO objective."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Hashable

from .schema import GRPORecord, RewardResult, Rollout, Window


@dataclass(frozen=True)
class GRPOConfig:
    advantage_epsilon: float = 1e-8
    advantage_clip: float = 5.0
    kl_beta: float = 0.2
    kl_log_ratio_clip: float = 10.0
    kl_value_clip: float = 20.0
    # "frozen": the KL reference must be the frozen SFT policy's logprobs
    # (``Rollout.reference_logprobs``); rollouts that carry behavior logprobs but
    # no reference logprobs are rejected so the KL term cannot silently collapse
    # to an on-policy trust region (log-ratio == 0 at every update).
    # "behavior": explicitly use the rollout behavior logprobs instead.
    kl_reference: str = "frozen"

    def __post_init__(self) -> None:
        if self.kl_reference not in {"frozen", "behavior"}:
            raise ValueError("kl_reference must be 'frozen' or 'behavior'")


def normalize_grouped_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[Hashable],
    *,
    epsilon: float = 1e-8,
    clip: float = 5.0,
) -> list[float]:
    """Population-standardize rewards independently inside every event group."""

    if len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must have equal length")
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[group_id].append(index)
    advantages = [0.0] * len(rewards)
    for indices in groups.values():
        values = [float(rewards[index]) for index in indices]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        for index, value in zip(indices, values):
            advantage = (value - mean) / (std + epsilon) if len(values) > 1 else value - mean
            advantages[index] = max(-clip, min(clip, advantage))
    return advantages


def build_grpo_records(
    scored_rollouts: Sequence[tuple[Rollout, RewardResult]],
    windows: Mapping[str, Window],
    config: GRPOConfig = GRPOConfig(),
) -> list[dict[str, Any]]:
    """Turn scored rollouts into the only records a trainer adapter needs."""

    rewards = [reward.total for _, reward in scored_rollouts]
    group_ids = [rollout.window_id for rollout, _ in scored_rollouts]
    advantages = normalize_grouped_advantages(
        rewards,
        group_ids,
        epsilon=config.advantage_epsilon,
        clip=config.advantage_clip,
    )
    records: list[dict[str, Any]] = []
    for (rollout, reward), advantage in zip(scored_rollouts, advantages):
        try:
            window = windows[rollout.window_id]
        except KeyError as error:
            raise ValueError(f"missing conditioning window {rollout.window_id}") from error
        loss_mask = rollout.loss_mask or tuple(True for _ in rollout.action_ids)
        if config.kl_reference == "behavior":
            kl_reference = rollout.behavior_logprobs
        else:
            kl_reference = rollout.reference_logprobs
            if not kl_reference and rollout.behavior_logprobs and config.kl_beta > 0:
                raise ValueError(
                    f"rollout {rollout.rollout_id} has behavior logprobs but no frozen-reference "
                    "logprobs; supply Rollout.reference_logprobs from the frozen SFT policy, or opt in "
                    "to GRPOConfig(kl_reference='behavior') / --kl-reference behavior"
                )
        records.append(
            GRPORecord(
                group_id=rollout.window_id,
                rollout_id=rollout.rollout_id,
                window=window.to_dict(),
                action_ids=rollout.action_ids,
                loss_mask=loss_mask,
                behavior_logprobs=rollout.behavior_logprobs,
                reference_logprobs=kl_reference,
                reward=reward.total,
                reward_components=dict(reward.components),
                reward_diagnostics=dict(reward.diagnostics),
                advantage=advantage,
                metadata=dict(rollout.metadata),
            ).to_dict()
        )
    return records


def sequence_objective(
    current_logprobs: Sequence[float],
    *,
    advantage: float,
    reference_logprobs: Sequence[float] = (),
    loss_mask: Sequence[bool] = (),
    config: GRPOConfig = GRPOConfig(),
) -> dict[str, float | bool]:
    """Numerical reference for one sampled event; no autograd dependency."""

    if loss_mask and len(loss_mask) != len(current_logprobs):
        raise ValueError("loss_mask and current_logprobs must have equal length")
    if reference_logprobs and len(reference_logprobs) != len(current_logprobs):
        raise ValueError("reference and current logprobs must have equal length")
    mask = list(loss_mask) if loss_mask else [True] * len(current_logprobs)
    policy = -float(advantage) * sum(
        float(logprob) for logprob, keep in zip(current_logprobs, mask) if keep
    )
    kl = 0.0
    if reference_logprobs:
        for current, reference, keep in zip(current_logprobs, reference_logprobs, mask):
            if not keep:
                continue
            log_ratio = float(current) - float(reference)
            exp_input = max(-config.kl_log_ratio_clip, min(config.kl_log_ratio_clip, log_ratio))
            estimate = math.exp(exp_input) - log_ratio - 1.0
            kl += min(config.kl_value_clip, estimate)
    return {
        "policy_loss": policy,
        "kl_loss": kl,
        "total_loss": policy + config.kl_beta * kl,
        "has_kl_reference": bool(reference_logprobs),
        "valid_tokens": float(sum(mask)),
    }


def batch_objective(
    current_logprobs: Sequence[Sequence[float]],
    records: Sequence[dict[str, Any]],
    config: GRPOConfig = GRPOConfig(),
) -> dict[str, float]:
    """Exact paper reduction: sum token terms, divide by real rollout count."""

    if len(current_logprobs) != len(records):
        raise ValueError("current_logprobs and records must have equal length")
    denominator = max(len(records), 1)
    policy = kl = 0.0
    valid_tokens = 0.0
    for logprobs, record in zip(current_logprobs, records):
        terms = sequence_objective(
            logprobs,
            advantage=float(record["advantage"]),
            reference_logprobs=record.get("reference_logprobs", ()),
            loss_mask=record.get("loss_mask", ()),
            config=config,
        )
        policy += float(terms["policy_loss"])
        kl += float(terms["kl_loss"])
        valid_tokens += float(terms["valid_tokens"])
    policy /= denominator
    kl /= denominator
    return {
        "policy_loss": policy,
        "kl_loss": kl,
        "total_loss": policy + config.kl_beta * kl,
        "rollout_count": float(len(records)),
        "valid_tokens": valid_tokens,
    }


def torch_batch_loss(current_logprobs, records, config: GRPOConfig = GRPOConfig()):
    """Optional PyTorch bridge retaining gradients through current logprobs.

    ``current_logprobs`` is a sequence of 1-D tensors, one per GRPO record.
    This import is deliberately local so the core package stays dependency-free.
    """

    import torch

    if len(current_logprobs) != len(records):
        raise ValueError("current_logprobs and records must have equal length")
    if not current_logprobs:
        raise ValueError("at least one rollout is required")
    policy = current_logprobs[0].sum() * 0.0
    kl = current_logprobs[0].sum() * 0.0
    for token_logprobs, record in zip(current_logprobs, records):
        mask = torch.as_tensor(
            record.get("loss_mask") or [True] * token_logprobs.numel(),
            dtype=torch.bool,
            device=token_logprobs.device,
        )
        if mask.numel() != token_logprobs.numel():
            raise ValueError("loss mask length mismatch")
        policy = policy - float(record["advantage"]) * token_logprobs[mask].sum()
        reference = record.get("reference_logprobs") or []
        if reference:
            reference_tensor = torch.as_tensor(
                reference, dtype=token_logprobs.dtype, device=token_logprobs.device
            )
            if reference_tensor.numel() != token_logprobs.numel():
                raise ValueError("reference logprob length mismatch")
            log_ratio = token_logprobs[mask] - reference_tensor[mask]
            estimate = torch.exp(log_ratio.clamp(-config.kl_log_ratio_clip, config.kl_log_ratio_clip))
            estimate = (estimate - log_ratio - 1.0).clamp(max=config.kl_value_clip)
            kl = kl + estimate.sum()
    denominator = max(len(records), 1)
    policy = policy / denominator
    kl = kl / denominator
    return policy + config.kl_beta * kl, {"policy_loss": policy, "kl_loss": kl}
