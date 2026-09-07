# Related open-source designs and resulting choices

This survey focuses on public interfaces rather than reproducing an entire
trainer.

| Project | Useful public seam | Limitation for DuplexPO | Adopted here |
|---|---|---|---|
| Hugging Face TRL GRPOTrainer | Prompt-only rows; extra dataset columns are forwarded to reward callables; custom rollout functions return prompt IDs, completion IDs, and logprobs | The primary abstraction is a text completion, not a frame-aligned duplex event | Callable reward adapter and extra-column pattern, while retaining a richer neutral `Window` |
| Open R1 | A recipe layer separately resolves dataset, model, tokenizer, and registered reward functions before constructing GRPOTrainer | It is an application recipe, not a reusable event protocol | Keep recipes/adapters outside the mathematical core |
| verl | Advantage estimators and policy losses are registry entries; GRPO groups samples through an explicit index and applies response masks | Native structures are tensor/DataProto-specific | Make `group_id` and `loss_mask` mandatory at the trainer boundary |
| NeMo RL | A policy interface separates generation from the GRPO loop and stores named reward components | The policy abstraction is native to its distributed runtime | Expose `PolicyAdapter` as a small protocol and preserve named FCDR components |

Primary references:

- [TRL GRPOTrainer documentation](https://huggingface.co/docs/trl/grpo_trainer)
- [TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats)
- [Open R1 GRPO entrypoint](https://github.com/huggingface/open-r1/blob/main/src/open_r1/grpo.py)
- [verl core algorithms](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py)
- [NeMo RL GRPO guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/grpo.md)
- [DuplexPO paper](https://arxiv.org/pdf/2607.07148)

## Design decisions

1. **Protocol first.** JSON-compatible, versioned records are the stable API;
   Python classes merely validate them.
2. **Construction is a first-class artifact.** A selected event window is
   serializable and auditable before any policy is loaded.
3. **Rewards consume behavior, not model tokens.** `ActionTrace` represents
   predicted speech intervals. A pluggable decoder translates each model's
   action vocabulary.
4. **Grouping is explicit.** Every rollout carries the event `window_id`; no
   trainer is expected to infer group boundaries from batch position.
5. **Masking is explicit.** Only sampled actions receive policy gradient. The
   teacher-forced prefix is described by the window, not mixed into `loss_mask`.
6. **Framework adapters contain no algorithm.** They only rename/repackage the
   same records. This avoids semantic drift across backends.
7. **Provenance travels with data.** The construction profile fingerprint,
   source, split, selection seed, and quality audit remain serializable.

## Why not expose the original NeMo model class as the main API?

The original class couples perception, token embedding, cache behavior,
distributed dummy forwards, rollouts, reward, and loss. That snapshot remains
valuable for exact historical provenance in `../DuplexPO_release`, but it is a
poor integration boundary. The portable layer extracts only the invariants an
outside implementation must agree on.
