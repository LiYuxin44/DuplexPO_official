# Data construction

## Canonicalization before this package

Dataset-specific code resolves audio/transcript files, participant roles, and
licenses, then emits canonical `Conversation` JSONL. The portable core never
assumes Fisher filenames, Seamless directory layout, SHAR, or Lhotse.
`adapters/datasets.py` provides dependency-free mappings for the historical
alignment/utterance dictionaries once the caller has explicitly resolved roles.

For Fisher, the paper-time source contained word alignments for two channels,
produced by running the [NeMo Forced Aligner](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html)
(NFA) on each channel against its transcript; each word becomes
`{"word", "start", "end"}` in seconds. Any word-level aligner with the same
output shape can be used instead. Words whose `start`/`end` are missing are
dropped and counted (`untimed_units_dropped`); a conversation is excluded when
more than 20% of either side's words are untimed. The submitted constructor
used `swap_by_first_speaker`, so the earliest speaker became canonical `user`
and the audio-channel mapping was swapped with it. The built-in `fisher`
profile performs:

1. clip to 210 seconds;
2. group same-speaker words separated by no more than 0.30 seconds;
3. detect short lexical backchannels (with a conservative duration-only fallback);
4. merge gaps under 1.2 seconds as natural pauses; for longer gaps, split only
   when the other speaker has substantive activity;
5. derive the final event type from the merged duration, as in the historical
   window builder. In practice an overlapped acknowledgment remains standalone
   because the other speaker occupies the following gap.

For Seamless Interaction, the input is utterance-level. The `seamless` profile
also drops truncated/very short fragments, rejects `charades`, recognizes short
overlapped acknowledgments, always keeps them standalone, and only merges turns
across a sub-1.0-second gap with no substantive cross-speaker activity. Role
resolution follows explicit actor metadata, then asymmetric prompt cues, then a
low-confidence first-speaker fallback. `from_seamless_pair` implements and logs
that hierarchy; callers with authoritative roles can instead use
`from_seamless_transcripts` directly.

## Backchannel vocabulary

`backchannel_lexicon.py` retains the paper-time English taxonomy and complete
1--5 gram inventory. Fisher uses all orders; the historical Seamless HQ path
used unigrams through trigrams. This inventory is not a language-universal
truth: pass a replacement `backchannel_phrases` set in a `ConstructionProfile`
for another corpus or language. The complete phrase set is part of the profile
SHA-256, so changing it cannot silently masquerade as the same constructed
dataset.

## Leak prevention and reproducibility

- Preserve an official conversation/session split when it exists.
- Otherwise `assign_split` hashes `seed:conversation_id`; split before event
  expansion. The audit records `split_source`, so this fallback cannot be
  mistaken for the official paper split.
- Every output window records the source, split, profile name/fingerprint, and
  deterministic selection seed.
- `balanced_round_robin` independently shuffles source queues and interleaves
  them, matching the paper's equal Fisher/Seamless sampling intent.
- Keep the audit JSONL. It records truncation, dropped fragments, counts, and
  exclusion reasons.

## Augmentation boundary

The paper used SpecAugment plus waveform noise with probability 0.5 and SNR
uniformly sampled from 0--60 dB. Augmentation belongs in the model input adapter,
because a metadata-only package cannot safely mutate arbitrary audio/tensor
representations. Record its RNG seed and realized parameters in rollout metadata
if exact replay is required.
