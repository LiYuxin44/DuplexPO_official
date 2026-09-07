# Completion audit

Audit date: 2026-09-06

| Objective requirement | Authoritative evidence | Result |
|---|---|---|
| Research comparable open-source GRPO designs | `RESEARCH.md` cites official TRL, Open R1, verl, NeMo RL, and paper sources and records the adopted interface decisions | Complete |
| Preserve the previous release | All implementation files are confined to sibling `DuplexPO_portable`; `../DuplexPO_release` remains present and its files were not rewritten | Complete |
| Open the training-data construction method | Fisher and Seamless adapters, role mapping, full historical 1--5 gram taxonomy, segmentation/merge/filter rules, truncation, split audit, balanced mixing, and augmentation specification are present | Complete |
| Open the paper GRPO algorithm | Critical windows, FCDR, per-event population normalization, sampled-action mask, historical loss reduction, and KL estimator are executable and documented | Complete |
| Connect to arbitrary trainer frameworks by I/O | Five versioned JSON Schemas, self-contained GRPO records, generic Protocols, TRL/verl/NeMo mappings, and `conformance` API/CLI are present | Complete |
| Reproducible reference behavior | Machine-readable recipe, profile/recipe SHA-256 values, frozen conformance vectors, and historical source hashes are present | Complete |

## Verification evidence

- Dependency-free unit/end-to-end suite: 26 tests passed.
- Historical parity: three Fisher construction edge cases, one Seamless HQ edge
  family, critical-window tuples, six grouped-advantage values, and three legacy
  reward cases matched the retained implementations.
- JSON Schema validation: conversation (1), rollout (8), window (2), GRPO record
  (8), and recipe (1) passed Draft 2020-12 validation.
- Packaging: sdist and `py3-none-any` wheel built; the wheel installed in a clean
  Python 3.9 venv; bundled recipe and all five schemas loaded successfully.
- Frozen vector checksums: all entries in `examples/SHA256SUMS` passed.
- Static portability check: production modules have no mandatory third-party
  imports and contain no machine paths, credentials, W&B IDs, or checkpoint URIs.

## Deliberate, documented boundaries

- Licensed Fisher/Seamless media and checkpoints are not redistributed.
- SpecAugment/noise application is delegated to the model input adapter; its
  paper settings remain machine-readable in the recipe.
- The portable window-selection RNG uses conversation-ID hashing by default.
  Historical token-derived seeding is available through an explicit override.
- A repository owner must choose the public license and author/citation metadata
  before publication; these are governance decisions, not missing algorithm code.
