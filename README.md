# DuplexPO Portable

`DuplexPO_portable` is a framework-neutral reference package for the two parts
of DuplexPO that are useful outside the original training stack:

1. constructing dynamics-critical events from aligned full-duplex dialogue;
2. turning arbitrary policy rollouts into event-grouped GRPO training records.

The original `DuplexPO_release` directory is intentionally untouched. This new
package has no runtime dependency on NeMo, Lhotse, TRL, verl, a checkpoint, or a
particular speech-token vocabulary.

## Stable boundary

```text
canonical Conversation
        -> transcript construction
selected Window (one GRPO group per conversational event)
        -> any policy/runtime
Rollout -> pluggable TraceDecoder -> ActionTrace
        -> factorized conversational-dynamics reward
GRPORecord {window, group_id, reward, advantage, action_ids, loss_mask, ref_logprobs}
        -> any trainer backend
```

The schemas under `schemas/` are the public compatibility boundary. Framework
adapters are deliberately thin and disposable.

## Quick start

From this directory, run the complete synthetic example without installing any
dependencies:

```bash
PYTHONPATH=src python -m duplexpo_portable.cli construct \
  --profile fisher \
  --input examples/conversations.jsonl \
  --output /tmp/duplexpo-windows.jsonl \
  --audit-output /tmp/duplexpo-audit.jsonl

PYTHONPATH=src python -m duplexpo_portable.cli prepare \
  --windows /tmp/duplexpo-windows.jsonl \
  --rollouts examples/rollouts.jsonl \
  --output /tmp/duplexpo-grpo.jsonl

PYTHONPATH=src python -m duplexpo_portable.cli conformance \
  --windows /tmp/duplexpo-windows.jsonl \
  --rollouts examples/rollouts.jsonl \
  --records /tmp/duplexpo-grpo.jsonl
```

For token-based models, emit `action_ids` instead of `trace` and add, for
example, `--bos-ids 1 --eos-id 2 --pad-id 0`. For a different action language,
implement one method:

```python
class MyTraceDecoder:
    def decode(self, rollout, window):
        return ActionTrace(speech=(SpeechInterval(start_s=..., end_s=...),))
```

Then call `prepare_grpo_update(...)`. The policy and trainer never need to
share internal classes with this package.

## Paper recipe represented here

- 80 ms frame grid;
- 1.0 s training lead and 2.0 s post-event buffer;
- at most three eligible full turns and one backchannel per conversation;
- four rollouts per selected event (the caller controls generation count);
- per-event population mean/std reward normalization, epsilon `1e-8`, clipped
  to `[-5, 5]`;
- sampled-token-only policy gradient;
- Schulman KL estimator against frozen-reference logprobs when supplied,
  otherwise rollout-behavior logprobs, with beta `0.2`;
- the reported `legacy` FCDR reward plus the post-paper `eos_aware_v1` option.

See `docs/ALGORITHM.md` for equations and fidelity notes.

## Repository map

- `src/duplexpo_portable/construction.py`: Fisher/Seamless construction profiles,
  leak-safe splitting, quality audit, and balanced mixing.
- `src/duplexpo_portable/backchannel_lexicon.py`: paper-time English 1--5 gram taxonomy.
- `src/duplexpo_portable/windows.py`: critical-window construction and selection.
- `src/duplexpo_portable/rewards.py`: token-neutral FCDR and optional BOS/EOS/PAD decoder.
- `src/duplexpo_portable/grpo.py`: grouped advantages and reference loss.
- `src/duplexpo_portable/recipes/paper_v1.json`: wheel-bundled, machine-readable recipe.
- `src/duplexpo_portable/conformance.py`: executable checks for outside adapters.
- `src/duplexpo_portable/adapters/`: generic, TRL, verl, and NeMo RL bridges.
- `schemas/`: JSON Schema compatibility contracts.
- `examples/`: synthetic inputs plus hashed, byte-for-byte conformance outputs.
- `tests/`: dependency-free unit and end-to-end tests.
- `docs/RESEARCH.md`: design survey of related open-source systems.
- `docs/COMPLETION_AUDIT.md`: requirement-by-requirement evidence and boundaries.

## What is not bundled

Fisher and Seamless Interaction audio/transcripts, model checkpoints, and
third-party model code are not redistributed. Dataset adapters must canonicalize
licensed source data into `Conversation` records. Audio paths or tensors can be
carried through `metadata`; the construction and GRPO protocol itself does not
read audio.

## License and release status

The portable DuplexPO code in this repository is released under the
[MIT License](LICENSE). Dataset audio/transcripts, model checkpoints, and
third-party model code are not included or relicensed. Complete the remaining
items in `docs/RELEASE_CHECKLIST.md` when preparing a tagged public release.
