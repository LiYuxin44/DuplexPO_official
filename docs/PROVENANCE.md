# Local extraction provenance

The portable implementation was checked against the retained paper-time source
snapshot without changing that snapshot.

| Component | Historical artifact SHA-256 | Portable status |
|---|---|---|
| English backchannel taxonomy | `5e46a4f9e5a40b3413de81ad77e2b3e06bed3d3b1c9c6482e84b6a0474d1c681` | Content copied to `backchannel_lexicon.py`; portable file differs only by a final newline |
| Fisher constructor | `0dbb86e6fa18f3496a9f639214d0e6d7677566caedf321886d6afd75c2837850` | Short-gap, long-silent-gap, and occupied-gap parity verified |
| Seamless HQ constructor | `7cc4da2fcb3fcbb553028e12f50c8ea7256012bf3038705b70612c577281af57` | Fragment, overlap, backchannel, and merge parity verified |
| Critical-window builder | `f26cbabf67a628182904807464ca7b263b45374ee8b17808e19c665fae3f79f7` | Fixed fixture parity verified |
| Event advantage normalization | `42ac7f3672d198f3077b9cb933b3f595241fa10c5d6a573fbb3d5f828a3cfc1d` | Fixed fixture parity verified |
| Legacy FCDR | `0255a2aa96df911ea6081df048e94455774960ab7b12be529f98ae79e8be0518` | BOS/EOS, no-EOS, and miss fixture parity verified |
| Historical integrated model/loss | `133e047cd01519c049bcf2d3b37fa77059497c50454cc7b35a65731a309951a4` | Objective reduction and token-derived selection seed extracted |

The portable event sampler deliberately replaces the historical token-derived
RNG seed with a conversation-ID hash by default. Exact historical replay remains
available through `historical_token_selection_seed`/the explicit override, and
each emitted window records which strategy was used.

Public releases should replace this working-tree note with immutable repository
commit IDs and a tagged conformance report.
