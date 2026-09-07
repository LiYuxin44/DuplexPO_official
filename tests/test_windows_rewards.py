import unittest

from duplexpo_portable.construction import construct_conversation, profile_named
from duplexpo_portable.rewards import FCDRConfig, TokenTraceDecoder, score_fcdr
from duplexpo_portable.schema import ActionTrace, Rollout, SpeechInterval
from duplexpo_portable.windows import WindowConfig, build_selected_windows, historical_token_selection_seed
from tests.test_construction import demo_conversation


def demo_windows():
    conversation = demo_conversation()
    user, agent, _ = construct_conversation(conversation, profile_named("fisher"))
    return build_selected_windows(
        conversation_id=conversation.conversation_id,
        split="train",
        source="demo",
        duration_s=conversation.duration_s,
        agent_segments=agent,
        user_segments=user,
        conversation_metadata=conversation.metadata,
        config=WindowConfig(),
    )


class WindowsAndRewardsTest(unittest.TestCase):
    def test_exact_window_frames(self):
        windows = demo_windows()
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].event_kind, "backchannel")
        self.assertEqual(
            (windows[0].sample_start_frame, windows[0].sample_end_frame),
            (25, 62),
        )
        self.assertEqual(
            (windows[1].sample_start_frame, windows[1].sample_end_frame),
            (62, 130),
        )
        self.assertEqual(windows[0].metadata["selection_seed_strategy"], "sha256(seed:conversation_id)")
        self.assertEqual(windows[0].metadata["conversation"]["note"], "synthetic protocol example")

    def test_historical_token_selection_seed(self):
        self.assertEqual(historical_token_selection_seed([10, 11, 12, 13, 14]), 22)
        self.assertEqual(historical_token_selection_seed([10, 11, 12, 13, 14], total_frames=4), 22)

    def test_selection_rejects_mixed_conversations(self):
        from dataclasses import replace
        from duplexpo_portable.windows import select_windows

        windows = demo_windows()
        with self.assertRaises(ValueError):
            select_windows([windows[0], replace(windows[1], conversation_id="other")])

    def test_trace_reward(self):
        backchannel, turn = demo_windows()
        bc_result = score_fcdr(
            backchannel,
            ActionTrace((SpeechInterval(3.0, 3.4, "eos"),)),
        )
        self.assertAlmostEqual(bc_result.total, 0.5)
        miss = score_fcdr(backchannel, ActionTrace())
        self.assertEqual(miss.total, -0.5)
        exact_turn = score_fcdr(turn, ActionTrace((SpeechInterval(6.0, 8.4, "eos"),)))
        self.assertGreater(exact_turn.total, 0.99)
        no_stop = score_fcdr(turn, ActionTrace((SpeechInterval(6.0, None, "unknown"),)))
        self.assertAlmostEqual(no_stop.total, exact_turn.components["onset"] * 0.75)

    def test_token_trace_decoder(self):
        window = demo_windows()[0]
        rollout = Rollout("r", window.window_id, action_ids=(0, 1, 9, 9, 2))
        trace = TokenTraceDecoder(bos_ids={1}, eos_id=2, pad_id=0).decode(rollout, window)
        self.assertAlmostEqual(trace.speech[0].start_s, 2.08)
        self.assertAlmostEqual(trace.speech[0].end_s, 2.32)
        self.assertEqual(trace.speech[0].termination, "eos")

    def test_eos_aware_bonus(self):
        turn = demo_windows()[1]
        result = score_fcdr(
            turn,
            ActionTrace((SpeechInterval(turn.reference_start_s, turn.reference_end_s, "eos"),)),
            FCDRConfig(mode="eos_aware_v1"),
        )
        self.assertAlmostEqual(result.components["yielding"], 0.05)


if __name__ == "__main__":
    unittest.main()
