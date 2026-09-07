"""Command-line reference pipeline over versioned JSONL contracts."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .conformance import ConformanceReport, check_grpo_records, check_rollouts
from .construction import assign_split, construct_conversation, profile_named
from .grpo import GRPOConfig, build_grpo_records
from .io import read_jsonl, write_jsonl
from .pipeline import prepare_grpo_update, score_rollouts
from .recipe import load_paper_recipe, recipe_fingerprint
from .rewards import FCDRConfig, TokenTraceDecoder
from .schema import Conversation, GRPORecord, RewardResult, Rollout, Window
from .windows import WindowConfig, build_selected_windows


def _decoder(args):
    if args.bos_ids is None:
        return None
    bos_ids = [int(item) for item in args.bos_ids.split(",") if item.strip()]
    if args.eos_id is None or args.pad_id is None:
        raise ValueError("--eos-id and --pad-id are required with --bos-ids")
    return TokenTraceDecoder(bos_ids=bos_ids, eos_id=args.eos_id, pad_id=args.pad_id)


def _load_windows(path: str) -> dict[str, Window]:
    windows = [Window.from_dict(value) for value in read_jsonl(path)]
    result = {window.window_id: window for window in windows}
    if len(result) != len(windows):
        raise ValueError("window file contains duplicate window_id values")
    return result


def cmd_construct(args) -> None:
    profile = profile_named(args.profile)
    config = WindowConfig(selection_seed=args.seed)
    recipe = load_paper_recipe()
    recipe_sha256 = recipe_fingerprint(recipe)
    windows: list[dict] = []
    audits: list[dict] = []
    for value in read_jsonl(args.input):
        conversation = Conversation.from_dict(value)
        split_source = "input" if conversation.split else "sha256(seed:conversation_id)"
        split = conversation.split or assign_split(conversation.conversation_id, seed=args.seed)
        user_segments, agent_segments, audit = construct_conversation(conversation, profile)
        audit["split"] = split
        audit["split_source"] = split_source
        audit["recipe_id"] = recipe["recipe_id"]
        audit["recipe_sha256"] = recipe_sha256
        audits.append(audit)
        if audit["status"] != "kept":
            continue
        selected = build_selected_windows(
            conversation_id=conversation.conversation_id,
            split=split,
            source=conversation.source,
            duration_s=min(conversation.duration_s, profile.max_duration_s),
            agent_segments=agent_segments,
            user_segments=user_segments,
            conversation_metadata=conversation.metadata,
            config=config,
            rng_seed=(
                None
                if conversation.metadata.get("window_selection_seed") is None
                else int(conversation.metadata["window_selection_seed"])
            ),
        )
        for window in selected:
            metadata = dict(window.metadata)
            metadata.update(
                {
                    "construction_profile": profile.name,
                    "construction_profile_sha256": profile.fingerprint(),
                    "recipe_id": recipe["recipe_id"],
                    "recipe_sha256": recipe_sha256,
                }
            )
            windows.append(replace(window, metadata=metadata).to_dict())
    count = write_jsonl(args.output, windows)
    if args.audit_output:
        write_jsonl(args.audit_output, audits)
    print(json.dumps({"windows": count, "conversations": len(audits), "output": args.output}))


def cmd_score(args) -> None:
    windows = _load_windows(args.windows)
    rollouts = [Rollout.from_dict(value) for value in read_jsonl(args.rollouts)]
    scored = score_rollouts(
        windows,
        rollouts,
        decoder=_decoder(args),
        reward_config=FCDRConfig(mode=args.reward_mode),
    )
    rows = [
        {
            "schema_version": "duplexpo.scored.v1",
            "window": windows[rollout.window_id].to_dict(),
            "rollout": rollout.to_dict(),
            "reward": reward.to_dict(),
        }
        for rollout, reward in scored
    ]
    count = write_jsonl(args.output, rows)
    print(json.dumps({"scored_rollouts": count, "output": args.output}))


def _load_scored(path: str):
    result = []
    for value in read_jsonl(path):
        rollout = Rollout.from_dict(value["rollout"])
        reward_value = value["reward"]
        reward = RewardResult(
            total=float(reward_value["total"]),
            components={name: float(item) for name, item in reward_value.get("components", {}).items()},
            diagnostics=dict(reward_value.get("diagnostics", {})),
        )
        result.append((rollout, reward))
    return result


def cmd_grpo(args) -> None:
    windows = _load_windows(args.windows)
    records = build_grpo_records(_load_scored(args.input), windows, GRPOConfig(kl_beta=args.kl_beta))
    count = write_jsonl(args.output, records)
    sizes: dict[str, int] = {}
    for record in records:
        sizes[record["group_id"]] = sizes.get(record["group_id"], 0) + 1
    print(
        json.dumps(
            {
                "records": count,
                "groups": len(sizes),
                "singleton_groups": sorted(key for key, size in sizes.items() if size < 2),
                "output": args.output,
            }
        )
    )


def cmd_prepare(args) -> None:
    windows = _load_windows(args.windows)
    rollouts = [Rollout.from_dict(value) for value in read_jsonl(args.rollouts)]
    records, audit = prepare_grpo_update(
        windows,
        rollouts,
        decoder=_decoder(args),
        reward_config=FCDRConfig(mode=args.reward_mode),
        grpo_config=GRPOConfig(kl_beta=args.kl_beta),
    )
    write_jsonl(args.output, records)
    print(json.dumps({**audit, "output": args.output}, sort_keys=True))


def cmd_validate(args) -> None:
    constructors = {
        "conversation": Conversation.from_dict,
        "window": Window.from_dict,
        "rollout": Rollout.from_dict,
        "grpo": GRPORecord.from_dict,
    }
    constructor = constructors[args.kind]
    count = 0
    for value in read_jsonl(args.input):
        constructor(value)
        count += 1
    print(json.dumps({"valid": True, "kind": args.kind, "records": count}))


def cmd_conformance(args) -> None:
    windows = _load_windows(args.windows)
    combined = ConformanceReport(stats={"windows": len(windows)})
    if not args.rollouts and not args.records:
        raise ValueError("conformance requires --rollouts and/or --records")
    if args.rollouts:
        rollouts = [Rollout.from_dict(value) for value in read_jsonl(args.rollouts)]
        report = check_rollouts(
            windows,
            rollouts,
            expected_samples_per_window=args.samples_per_window,
            require_trainable=args.require_trainable,
        )
        combined.errors.extend(report.errors)
        combined.warnings.extend(report.warnings)
        combined.stats["rollouts"] = report.stats
    if args.records:
        records = list(read_jsonl(args.records))
        report = check_grpo_records(
            records,
            expected_samples_per_window=args.samples_per_window,
            require_trainable=args.require_trainable,
        )
        combined.errors.extend(report.errors)
        combined.warnings.extend(report.warnings)
        expected_groups = set(windows)
        record_groups = {str(record.get("group_id")) for record in records}
        if expected_groups != record_groups:
            combined.errors.append(
                "record group IDs differ from window IDs: "
                f"missing={sorted(expected_groups - record_groups)}, "
                f"unknown={sorted(record_groups - expected_groups)}"
            )
        combined.stats["records"] = report.stats
    payload = combined.to_dict()
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if not combined.ok:
        raise SystemExit(1)


def _add_decoder_arguments(parser) -> None:
    parser.add_argument("--bos-ids", help="comma-separated speech BOS token IDs; omit for trace rollouts")
    parser.add_argument("--eos-id", type=int)
    parser.add_argument("--pad-id", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="duplexpo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    construct = subparsers.add_parser("construct", help="canonical transcript JSONL -> selected event windows")
    construct.add_argument("--input", required=True)
    construct.add_argument("--output", required=True)
    construct.add_argument("--audit-output")
    construct.add_argument("--profile", choices=("fisher", "seamless"), required=True)
    construct.add_argument("--seed", type=int, default=42)
    construct.set_defaults(function=cmd_construct)

    score = subparsers.add_parser("score", help="windows + rollouts -> factorized rewards")
    score.add_argument("--windows", required=True)
    score.add_argument("--rollouts", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--reward-mode", choices=("legacy", "eos_aware_v1"), default="legacy")
    _add_decoder_arguments(score)
    score.set_defaults(function=cmd_score)

    grpo = subparsers.add_parser("grpo", help="scored rollouts -> grouped GRPO records")
    grpo.add_argument("--input", required=True)
    grpo.add_argument("--windows", required=True)
    grpo.add_argument("--output", required=True)
    grpo.add_argument("--kl-beta", type=float, default=0.2)
    grpo.set_defaults(function=cmd_grpo)

    prepare = subparsers.add_parser("prepare", help="windows + rollouts -> ready-to-train GRPO records")
    prepare.add_argument("--windows", required=True)
    prepare.add_argument("--rollouts", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--reward-mode", choices=("legacy", "eos_aware_v1"), default="legacy")
    prepare.add_argument("--kl-beta", type=float, default=0.2)
    _add_decoder_arguments(prepare)
    prepare.set_defaults(function=cmd_prepare)

    validate = subparsers.add_parser("validate", help="validate a core JSONL contract")
    validate.add_argument("--kind", choices=("conversation", "window", "rollout", "grpo"), required=True)
    validate.add_argument("--input", required=True)
    validate.set_defaults(function=cmd_validate)

    conformance = subparsers.add_parser(
        "conformance", help="check third-party rollout and trainer-record contracts"
    )
    conformance.add_argument("--windows", required=True)
    conformance.add_argument("--rollouts")
    conformance.add_argument("--records")
    conformance.add_argument("--samples-per-window", type=int, default=4)
    conformance.add_argument("--require-trainable", action="store_true")
    conformance.add_argument("--json-output")
    conformance.set_defaults(function=cmd_conformance)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
