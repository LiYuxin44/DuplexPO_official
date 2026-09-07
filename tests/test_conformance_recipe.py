from dataclasses import asdict
import unittest

from duplexpo_portable.backchannel_lexicon import (
    BACKCHANNEL_BIGRAMS,
    BACKCHANNEL_LEXICON,
    BACKCHANNEL_PENTAGRAMS,
    BACKCHANNEL_QUADGRAMS,
    BACKCHANNEL_TRIGRAMS,
)
from duplexpo_portable.conformance import check_grpo_records, check_policy_adapter, check_rollouts
from duplexpo_portable.construction import profile_named
from duplexpo_portable.grpo import GRPOConfig
from duplexpo_portable.grpo import build_grpo_records
from duplexpo_portable.recipe import (
    load_paper_recipe,
    paper_grpo_config,
    paper_reward_config,
    paper_window_config,
    recipe_fingerprint,
)
from duplexpo_portable.schema import ActionTrace, RewardResult, Rollout, SpeechInterval
from duplexpo_portable.schema import load_json_schema
from duplexpo_portable.rewards import FCDRConfig
from duplexpo_portable.windows import WindowConfig
from tests.test_windows_rewards import demo_windows


def reward_rollouts():
    result = []
    for window in demo_windows():
        for index, offset in enumerate((-0.2, 0.0, 0.2, 0.4)):
            start = window.reference_start_s + offset
            result.append(
                Rollout(
                    rollout_id=f"{window.window_id}:reward:{index}",
                    window_id=window.window_id,
                    trace=ActionTrace((SpeechInterval(start, max(start + 0.2, window.reference_end_s), "eos"),)),
                )
            )
    return result


class ConformanceRecipeTest(unittest.TestCase):
    def test_recipe_matches_executable_defaults(self):
        recipe = load_paper_recipe()
        self.assertEqual(recipe["rollout"]["samples_per_window"], 4)
        self.assertEqual(len(recipe_fingerprint(recipe)), 64)
        self.assertEqual(recipe["construction"]["augmentation"]["waveform_noise"]["snr_db_uniform"], [0.0, 60.0])
        self.assertEqual(paper_window_config().lead_time_s, 1.0)
        self.assertEqual(paper_window_config(validation=True).lead_time_s, 2.0)
        self.assertEqual(paper_reward_config().mode, "legacy")
        self.assertEqual(paper_grpo_config().kl_beta, 0.2)
        self.assertEqual(load_json_schema("recipe")["title"], "DuplexPO paper recipe v1")
        self.assertEqual(asdict(paper_window_config()), asdict(WindowConfig()))
        self.assertEqual(asdict(paper_reward_config()), asdict(FCDRConfig()))
        self.assertEqual(asdict(paper_grpo_config()), asdict(GRPOConfig()))
        fisher = profile_named("fisher")
        self.assertEqual(recipe["construction"]["profile_sha256"]["fisher"], fisher.fingerprint())
        for key, value in recipe["construction"]["fisher"].items():
            if key not in {"granularity", "role_policy"}:
                self.assertEqual(getattr(fisher, key), value)
        seamless = profile_named("seamless")
        self.assertEqual(recipe["construction"]["profile_sha256"]["seamless_hq_v1"], seamless.fingerprint())
        for key, value in recipe["construction"]["seamless_hq_v1"].items():
            if key not in {"granularity", "role_policy"}:
                expected = tuple(value) if key == "excluded_interaction_types" else value
                self.assertEqual(getattr(seamless, key), expected)
        counts = recipe["construction"]["backchannel_lexicon"]
        self.assertEqual(
            [
                len(BACKCHANNEL_LEXICON),
                len(BACKCHANNEL_BIGRAMS),
                len(BACKCHANNEL_TRIGRAMS),
                len(BACKCHANNEL_QUADGRAMS),
                len(BACKCHANNEL_PENTAGRAMS),
            ],
            [counts["unigrams"], counts["bigrams"], counts["trigrams"], counts["quadgrams"], counts["pentagrams"]],
        )

    def test_reward_contract_and_bad_group_count(self):
        windows = {window.window_id: window for window in demo_windows()}
        rollouts = reward_rollouts()
        report = check_rollouts(windows, rollouts)
        self.assertTrue(report.ok)
        self.assertEqual(report.stats["group_sizes"], {key: 4 for key in windows})
        self.assertTrue(report.warnings)
        bad = check_rollouts(windows, rollouts[:-1])
        self.assertFalse(bad.ok)

    def test_record_advantage_recomputation_catches_drift(self):
        rollouts = reward_rollouts()
        scored = []
        for index, rollout in enumerate(rollouts):
            scored.append((rollout, RewardResult(float(index % 4), {"onset": float(index % 4)}, {})))
        windows = {window.window_id: window for window in demo_windows()}
        records = build_grpo_records(scored, windows)
        self.assertTrue(check_grpo_records(records).ok)
        records[0]["advantage"] += 0.1
        self.assertFalse(check_grpo_records(records).ok)
        records[0]["window"] = None
        report = check_grpo_records(records)
        self.assertFalse(report.ok)
        self.assertTrue(any("embedded window" in error for error in report.errors))

    def test_policy_adapter_contract(self):
        class Adapter:
            def generate(self, windows, samples_per_window):
                for window in windows:
                    for index in range(samples_per_window):
                        yield Rollout(
                            rollout_id=f"{window.window_id}:{index}",
                            window_id=window.window_id,
                            action_ids=(1, 2),
                            behavior_logprobs=(-0.1, -0.2),
                            loss_mask=(True, True),
                        )

        rollouts, report = check_policy_adapter(Adapter(), demo_windows())
        self.assertEqual(len(rollouts), 8)
        self.assertTrue(report.ok, report.errors)


if __name__ == "__main__":
    unittest.main()
