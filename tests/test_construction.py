import unittest

from duplexpo_portable.adapters.datasets import from_fisher_alignments, from_seamless_pair
from duplexpo_portable.construction import (
    assign_split,
    balanced_round_robin,
    construct_conversation,
    profile_named,
)
from duplexpo_portable.schema import Conversation, TranscriptUnit


def demo_conversation():
    units = (
        TranscriptUnit("user", 0.5, 0.9, "tell"),
        TranscriptUnit("user", 0.95, 1.3, "me"),
        TranscriptUnit("user", 1.35, 1.9, "more"),
        TranscriptUnit("agent", 3.0, 3.4, "yeah"),
        TranscriptUnit("user", 3.1, 4.2, "about that"),
        TranscriptUnit("agent", 6.0, 6.6, "I"),
        TranscriptUnit("agent", 6.65, 7.3, "think"),
        TranscriptUnit("agent", 7.35, 8.4, "so too"),
        TranscriptUnit("user", 9.0, 10.0, "great"),
    )
    return Conversation(
        "demo-001",
        12.0,
        units,
        source="demo",
        split="train",
        metadata={"note": "synthetic protocol example"},
    )


class ConstructionTest(unittest.TestCase):
    def test_word_profile_preserves_backchannel(self):
        profile = profile_named("fisher")
        user, agent, audit = construct_conversation(demo_conversation(), profile)
        self.assertEqual(audit["status"], "kept")
        self.assertEqual([segment.kind for segment in agent], ["backchannel", "turn"])
        self.assertEqual(agent[0].text, "yeah")
        self.assertAlmostEqual(agent[1].end_s - agent[1].start_s, 2.4)
        self.assertEqual(audit["profile_sha256"], profile.fingerprint())
        self.assertTrue(user)

    def test_split_is_stable_and_conversation_level(self):
        first = assign_split("same-session", seed=42)
        self.assertEqual(first, assign_split("same-session", seed=42))
        self.assertIn(first, {"train", "validation", "test"})

    def test_unknown_schema_is_rejected(self):
        value = demo_conversation().to_dict()
        value["schema_version"] = "duplexpo.v2"
        with self.assertRaises(ValueError):
            Conversation.from_dict(value)
        value.pop("schema_version")
        with self.assertRaises(KeyError):
            Conversation.from_dict(value)

    def test_balanced_round_robin(self):
        records = [
            {"source": "a", "id": 1},
            {"source": "a", "id": 2},
            {"source": "b", "id": 3},
            {"source": "b", "id": 4},
        ]
        mixed = balanced_round_robin(records, key=lambda row: row["source"], seed=7)
        self.assertEqual([row["source"] for row in mixed], ["a", "b", "a", "b"])

    def test_seamless_exclusion_is_audited(self):
        conversation = Conversation(
            "excluded",
            2.0,
            (
                TranscriptUnit("user", 0.1, 0.5, "hello"),
                TranscriptUnit("agent", 0.7, 1.1, "hi"),
            ),
            granularity="utterance",
            metadata={"interaction_type": "charades"},
        )
        user, agent, audit = construct_conversation(conversation, profile_named("seamless"))
        self.assertEqual((user, agent), ([], []))
        self.assertEqual(audit["status"], "excluded")

    def test_fisher_source_adapter_shifts_and_removes_system_token(self):
        conversation = from_fisher_alignments(
            conversation_id="f",
            duration_s=3.0,
            user_alignment=[{"word": "hello", "start": 1.0, "end": 1.4}],
            agent_alignment=[
                {"word": "<system> prompt <system>", "start": 0.0, "end": 0.1},
                {"word": "yeah", "start": 1.5, "end": 1.8},
            ],
            shift_s=0.5,
        )
        self.assertEqual([unit.text for unit in conversation.units], ["hello", "yeah"])
        self.assertAlmostEqual(conversation.units[0].start_s, 0.5)

    def test_fisher_source_adapter_swaps_earliest_agent(self):
        conversation = from_fisher_alignments(
            conversation_id="swap",
            duration_s=3.0,
            user_alignment=[{"word": "later-user", "start": 1.0, "end": 1.4}],
            agent_alignment=[{"word": "first-agent", "start": 0.5, "end": 0.8}],
        )
        self.assertTrue(conversation.metadata["roles_swapped"])
        self.assertEqual(conversation.units[0].speaker, "user")
        self.assertEqual(conversation.units[0].text, "first-agent")

    def test_seamless_role_hierarchy(self):
        speaker_0 = [{"transcript": "first", "start": 0.1, "end": 0.5}]
        speaker_1 = [{"transcript": "second", "start": 0.6, "end": 1.0}]
        by_actor = from_seamless_pair(
            conversation_id="actor",
            duration_s=2.0,
            speaker_0_transcript=speaker_0,
            speaker_1_transcript=speaker_1,
            actor_speaker="speaker_0",
        )
        self.assertEqual(by_actor.metadata["role_source"], "actor")
        self.assertEqual(by_actor.metadata["agent_original_speaker_index"], 0)
        by_prompt = from_seamless_pair(
            conversation_id="prompt",
            duration_s=2.0,
            speaker_0_transcript=speaker_0,
            speaker_1_transcript=speaker_1,
            speaker_0_prompt="Ask your partner to share",
            speaker_1_prompt="Listen and respond",
        )
        self.assertEqual(by_prompt.metadata["role_source"], "prompt")
        self.assertEqual(by_prompt.metadata["user_original_speaker_index"], 0)


if __name__ == "__main__":
    unittest.main()
