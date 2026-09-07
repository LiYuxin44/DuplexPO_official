# Connecting another framework

Only three operations are required.

## Policy

Implement `PolicyAdapter.generate(windows, samples_per_window)`. Each window
contains frame bounds and source metadata, so the adapter can slice its native
audio/tensor representation and prefill the prefix. Return `Rollout` objects;
use a shared `window_id` for all samples from the same event.

## Action interpretation

If the model directly emits speech intervals, attach `ActionTrace`. If it emits
tokens, implement `TraceDecoder.decode(rollout, window)`. The included
`TokenTraceDecoder` covers one BOS, one EOS, and PAD-as-stop frame vocabularies.

## Trainer

Recompute current token logprobs and consume the flat GRPO records. For PyTorch,
`torch_batch_loss` is an executable reference; production trainers may express
the same formula in JAX or their own distributed tensor layer.

The supplied adapters illustrate field mappings:

- `adapters/trl.py`: prompt-only row plus extra window columns and a reward callable;
- `adapters/verl.py`: event ID mapped to the `uid`/group-index convention;
- `adapters/nemo_rl.py`: named `reward/<component>` columns;
- `adapters/base.py`: framework-independent structural protocols.

These modules do not import the target frameworks. That keeps installation
lightweight and makes version drift a local adapter concern.

Run `duplexpo conformance` on exported windows/rollouts/records before connecting
the backend. Add `--require-trainable` to require action IDs and replayable KL
logprobs rather than accepting reward-only traces. Programmatic adapters can use
`check_policy_adapter` directly.

## Conformance test for a new adapter

Given the synthetic examples, verify that it:

1. returns exactly four rollouts for each selected event;
2. preserves `window_id` through generation and collation;
3. produces an action-aligned `loss_mask`;
4. preserves reference/behavior logprobs at every valid action;
5. matches `sequence_objective` numerically on a small fixed batch;
6. produces zero advantage when all rewards in an event are identical;
7. never mixes events with the same batch position but different IDs.
