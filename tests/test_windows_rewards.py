import unittest

from duplexpo_portable.construction import construct_conversation, profile_named
from duplexpo_portable.rewards import FCDRConfig, TokenTraceDecoder, score_fcdr
from duplexpo_portable.schema import ActionTrace, Rollout, Segment, SpeechInterval
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

    def test_restart_after_eos_is_reported_not_scored(self):
        turn = demo_windows()[1]
        start = turn.reference_start_s
        single = score_fcdr(turn, ActionTrace((SpeechInterval(start, start + 0.16, "eos"),)))
        restarted = score_fcdr(
            turn,
            ActionTrace(
                (
                    SpeechInterval(start, start + 0.16, "eos"),
                    SpeechInterval(start + 0.4, turn.sample_end_s, "unknown"),
                )
            ),
        )
        # Only speech[0] enters the scalar reward (legacy behaviour, unchanged) ...
        self.assertAlmostEqual(restarted.total, single.total)
        # ... but the exploit is visible in diagnostics.
        self.assertFalse(single.diagnostics["restart_after_eos"])
        self.assertTrue(restarted.diagnostics["restart_after_eos"])
        self.assertEqual(restarted.diagnostics["extra_speech_segments"], 1)
        self.assertEqual(set(restarted.components), {"onset", "pattern"})

    def test_backchannel_label_ceiling_matches_reward(self):
        from duplexpo_portable.construction import profile_named

        limit = FCDRConfig().backchannel_max_talk_s
        for name in ("fisher", "seamless"):
            profile = profile_named(name)
            for field_name in (
                "lexical_bc_max_duration_s",
                "high_order_bc_max_duration_s",
                "fallback_bc_max_duration_s",
                "utterance_bc_max_duration_s",
            ):
                self.assertLessEqual(getattr(profile, field_name), limit, f"{name}.{field_name}")

    def test_window_extended_when_next_lead_cuts_reference(self):
        from duplexpo_portable.windows import WindowConfig, build_windows_with_audit

        # backchannel 3.0-3.9 s immediately followed by the agent's own turn at 4.2 s:
        # the nominal end (4.2 - 1.0 = 3.2 s) would fall before the backchannel ends.
        segments = (
            Segment("agent", 3.0, 3.9, "yeah", "backchannel", 1),
            Segment("agent", 4.2, 9.0, "I think so too and more words", "turn", 6),
        )
        windows, stats = build_windows_with_audit(
            conversation_id="x", split="train", source="demo", duration_s=20.0,
            agent_segments=segments, config=WindowConfig(),
        )
        self.assertEqual(len(windows), 2)
        bc, turn = windows
        self.assertEqual(stats["windows_extended_past_next_lead"], 1)
        self.assertEqual(stats["windows_dropped_unsatisfiable"], 0)
        self.assertGreaterEqual(bc.sample_end_frame, bc.reference_end_frame + 3)
        self.assertEqual(bc.metadata["sample_end_extended_frames"], bc.sample_end_frame - 40)
        # the following window is unchanged: it still starts at the nominal chaining point
        self.assertEqual(turn.sample_start_frame, 40)
        # human reference now scores as a proper backchannel instead of "never stopped"
        human = score_fcdr(bc, ActionTrace((SpeechInterval(3.0, 3.9, "eos"),)))
        self.assertAlmostEqual(human.total, 0.5)

    def test_window_dropped_when_onset_unreachable(self):
        from duplexpo_portable.windows import WindowConfig, build_windows_with_audit

        # reference onset after the declared duration: no frame inside the conversation to act on
        segments = (Segment("agent", 20.3, 20.8, "ok", "backchannel", 1),)
        windows, stats = build_windows_with_audit(
            conversation_id="x", split="train", source="demo", duration_s=20.0,
            agent_segments=segments, config=WindowConfig(),
        )
        self.assertEqual(windows, [])
        self.assertEqual(stats["windows_dropped_unsatisfiable"], 1)

    def test_untimed_units_are_dropped_and_counted(self):
        from duplexpo_portable.adapters.datasets import from_fisher_alignments, from_seamless_transcripts
        from duplexpo_portable.construction import construct_conversation, profile_named

        user = [{"word": "hi", "start": 0.5, "end": 0.8}, {"word": "3", "start": None, "end": None}]
        agent = [{"word": "yes", "start": 2.0, "end": 2.3}, {"word": "sure", "start": 2.4, "end": 2.8}]
        conversation = from_fisher_alignments(
            conversation_id="c", duration_s=10.0, user_alignment=user, agent_alignment=agent,
            swap_by_first_speaker=False,
        )
        self.assertEqual(len(conversation.units), 3)
        self.assertEqual(conversation.metadata["untimed_units_dropped"], {"user": 1, "agent": 0})
        self.assertTrue(conversation.metadata["untimed_fraction_exceeded"])  # 1/2 > 0.2
        _, _, audit = construct_conversation(conversation, profile_named("fisher"))
        self.assertEqual(audit["status"], "excluded")
        self.assertEqual(audit["reason"], "untimed_fraction_exceeded")

        kept = from_seamless_transcripts(
            conversation_id="s", duration_s=10.0,
            user_transcript=[{"transcript": "hello there", "start": 0.5, "end": 1.5}] * 5
            + [{"transcript": "x", "start": None, "end": 3.0}],
            agent_transcript=[{"transcript": "yes indeed", "start": 2.0, "end": 3.0}],
        )
        self.assertEqual(kept.metadata["untimed_units_dropped"], {"user": 1, "agent": 0})
        self.assertFalse(kept.metadata["untimed_fraction_exceeded"])  # 1/6 <= 0.2
        _, _, audit = construct_conversation(kept, profile_named("seamless"))
        self.assertEqual(audit["status"], "kept")
        self.assertEqual(audit["untimed_units_dropped"], {"user": 1, "agent": 0})


if __name__ == "__main__":
    unittest.main()
