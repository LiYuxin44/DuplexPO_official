import math
import unittest

from duplexpo_portable.adapters.nemo_rl import reward_columns
from duplexpo_portable.adapters.trl import to_trl_row
from duplexpo_portable.adapters.verl import attach_verl_group_index, to_verl_row
from dataclasses import replace

from duplexpo_portable.grpo import (
    GRPOConfig,
    batch_objective,
    build_grpo_records,
    normalize_grouped_advantages,
    sequence_objective,
)
from duplexpo_portable.pipeline import prepare_grpo_update
from duplexpo_portable.rewards import score_fcdr
from duplexpo_portable.schema import ActionTrace, Rollout, SpeechInterval
from tests.test_windows_rewards import demo_windows


class GrpoPipelineTest(unittest.TestCase):
    def test_group_normalization(self):
        values = normalize_grouped_advantages([-0.5, 0.0, 0.5, 1.0], ["a"] * 4)
        scale = math.sqrt(0.3125) + 1e-8
        self.assertAlmostEqual(values[0], -0.75 / scale)
        self.assertAlmostEqual(values[-1], 0.75 / scale)
        self.assertEqual(normalize_grouped_advantages([3.0], ["one"]), [0.0])

    def test_objective(self):
        result = sequence_objective(
            [-1.0, -2.0],
            advantage=2.0,
            reference_logprobs=[-1.0, -2.0],
            loss_mask=[True, True],
        )
        self.assertAlmostEqual(result["policy_loss"], 6.0)
        self.assertAlmostEqual(result["kl_loss"], 0.0)
        records = [{"advantage": 2.0, "reference_logprobs": [-1.0, -2.0], "loss_mask": [True, True]}]
        self.assertAlmostEqual(batch_objective([[-1.0, -2.0]], records)["total_loss"], 6.0)

    def test_behavior_only_rollouts_need_explicit_opt_in(self):
        windows = {window.window_id: window for window in demo_windows()}
        window = next(iter(windows.values()))
        start = window.reference_start_s
        scored = []
        for index in range(2):
            rollout = Rollout(
                f"{window.window_id}:b{index}",
                window.window_id,
                action_ids=(1, 2),
                behavior_logprobs=(-0.1, -0.2),
                loss_mask=(True, True),
                trace=ActionTrace((SpeechInterval(start + 0.2 * index, start + 1.0, "eos"),)),
            )
            scored.append((rollout, score_fcdr(window, rollout.trace)))
        with self.assertRaises(ValueError):
            build_grpo_records(scored, windows)
        # no KL at all -> nothing to anchor, allowed
        self.assertEqual(len(build_grpo_records(scored, windows, GRPOConfig(kl_beta=0.0))), 2)
        # explicit opt-in -> behavior logprobs become the reference
        records = build_grpo_records(scored, windows, GRPOConfig(kl_reference="behavior"))
        self.assertEqual(list(records[0]["reference_logprobs"]), [-0.1, -0.2])
        # frozen reference supplied -> used verbatim, behavior kept separately
        frozen = [(replace(r, reference_logprobs=(-0.3, -0.4)), s) for r, s in scored]
        records = build_grpo_records(frozen, windows)
        self.assertEqual(list(records[0]["reference_logprobs"]), [-0.3, -0.4])
        self.assertEqual(list(records[0]["behavior_logprobs"]), [-0.1, -0.2])

    def test_end_to_end_and_adapter_shapes(self):
        windows = {window.window_id: window for window in demo_windows()}
        rollouts = []
        for window in windows.values():
            for index, offset in enumerate((-0.4, 0.0, 0.4, 0.8)):
                rollouts.append(
                    Rollout(
                        f"{window.window_id}:r{index}",
                        window.window_id,
                        trace=ActionTrace(
                            (
                                SpeechInterval(
                                    max(0.0, window.reference_start_s + offset),
                                    max(window.reference_end_s, window.reference_start_s + offset + 0.2),
                                    "eos",
                                ),
                            )
                        ),
                    )
                )
        records, audit = prepare_grpo_update(windows, rollouts)
        self.assertEqual(len(records), 8)
        self.assertTrue(audit["all_groups_have_learning_signal"])
        self.assertEqual(set(audit["group_sizes"].values()), {4})

        first_window = next(iter(windows.values()))
        self.assertEqual(to_trl_row(first_window)["window_id"], first_window.window_id)
        self.assertEqual(to_verl_row(first_window)["extra_info"]["index"], first_window.window_id)
        attached = attach_verl_group_index({}, records)
        self.assertEqual(len(attached["non_tensor_batch"]["uid"]), 8)
        columns = reward_columns(records[0])
        self.assertIn("reward/onset", columns)


if __name__ == "__main__":
    unittest.main()
