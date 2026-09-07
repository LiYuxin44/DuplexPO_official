"""Golden edge cases extracted from the paper-time Fisher/Seamless builders."""

import unittest

from duplexpo_portable.construction import construct_conversation, profile_named
from duplexpo_portable.schema import Conversation, TranscriptUnit


def conversation(conversation_id, units, *, granularity="word"):
    return Conversation(conversation_id, 6.0, tuple(units), granularity=granularity)


class HistoricalConstructionTest(unittest.TestCase):
    def test_fisher_short_gap_always_merges(self):
        value = conversation(
            "short-gap",
            [
                TranscriptUnit("agent", 1.0, 1.2, "yeah"),
                TranscriptUnit("agent", 1.5, 1.8, "provide"),
                TranscriptUnit("agent", 1.85, 2.3, "details"),
                TranscriptUnit("user", 4.0, 4.8, "continue"),
            ],
        )
        _, agent, _ = construct_conversation(value, profile_named("fisher"))
        self.assertEqual([(item.start_s, item.end_s, item.kind) for item in agent], [(1.0, 2.3, "turn")])

    def test_fisher_long_silent_gap_merges(self):
        value = conversation(
            "long-silent",
            [
                TranscriptUnit("agent", 1.0, 1.5, "elephant"),
                TranscriptUnit("agent", 3.0, 3.5, "telescope"),
                TranscriptUnit("user", 4.0, 4.8, "continue"),
            ],
        )
        _, agent, _ = construct_conversation(value, profile_named("fisher"))
        self.assertEqual([(item.start_s, item.end_s) for item in agent], [(1.0, 3.5)])

    def test_fisher_substantive_other_speaker_splits_long_gap(self):
        value = conversation(
            "occupied-gap",
            [
                TranscriptUnit("agent", 1.0, 1.5, "elephant"),
                TranscriptUnit("user", 1.7, 2.8, "substantive content"),
                TranscriptUnit("agent", 3.0, 3.5, "telescope"),
            ],
        )
        _, agent, _ = construct_conversation(value, profile_named("fisher"))
        self.assertEqual([(item.start_s, item.end_s) for item in agent], [(1.0, 1.5), (3.0, 3.5)])

    def test_seamless_drops_fragments_and_blocks_merge_across_other_turn(self):
        value = conversation(
            "seamless-edge",
            [
                TranscriptUnit("agent", 0.1, 0.2, "x"),
                TranscriptUnit("user", 0.5, 1.7, "a substantive explanation"),
                TranscriptUnit("agent", 1.0, 1.4, "yeah"),
                TranscriptUnit("agent", 1.6, 2.6, "first response"),
                TranscriptUnit("user", 2.4, 3.5, "another substantive point"),
                TranscriptUnit("agent", 3.0, 4.1, "second response"),
            ],
            granularity="utterance",
        )
        _, agent, audit = construct_conversation(value, profile_named("seamless"))
        self.assertEqual(audit["agent_fragments_dropped"], 1)
        self.assertEqual([item.kind for item in agent], ["backchannel", "turn", "turn"])
        self.assertEqual([(item.start_s, item.end_s) for item in agent], [(1.0, 1.4), (1.6, 2.6), (3.0, 4.1)])


if __name__ == "__main__":
    unittest.main()
