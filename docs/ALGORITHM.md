# DuplexPO event-level GRPO

## Critical windows

For agent event `(a_i, b_i)`, frame duration `Delta`, lead `L`, and buffer `B`:

```text
g_i = floor(a_i / Delta)
h_i = floor(b_i / Delta)
s_i = max(previous_window_end, floor((a_i - L) / Delta))
e_i = min(floor((b_i + B) / Delta),
          floor((a_{i+1} - L) / Delta), T)
```

The prefix before `s_i` is teacher-forced. Generation and gradient are confined
to `[s_i, e_i)`. `WindowConfig` encodes the reported 80 ms grid, 1 s training
lead, 2 s buffer, 200-frame rollout cap, and deterministic turn/backchannel
selection.

When the next agent event starts within `L` of `b_i`, the nominal `e_i` can fall
before `h_i` (or even before `g_i`), so not even the human reference can score
well inside the window. `build_windows` therefore extends `e_i` to at least
`h_i + reference_margin_frames` (3 frames = 0.24 s) for that window only; the
chaining value `previous_window_end` used for `s_{i+1}` stays at the nominal
`e_i`, so all other windows are unchanged. Windows whose onset is still outside
the span are dropped. Both cases are counted in the construction audit
(`windows_extended_past_next_lead`, `windows_dropped_unsatisfiable`).

Backchannel label ceilings in every construction profile are at most
`backchannel_max_talk_s` (1.5 s) so that a reference backchannel is never
scored as "overlong" by the reward.

The historical implementation seeded event subsampling with the sum of the
first and middle target frame-token IDs. That tokenization-specific rule is
available as `historical_token_selection_seed` and can be supplied through the
`rng_seed` override (or `metadata.window_selection_seed` in the CLI). The
portable default instead hashes `seed:conversation_id`, so it is stable across
tokenizers and frameworks. Output metadata records the effective strategy and
seed; these two modes must not be presented as bit-identical selection.

## Factorized reward

FCDR operates on the first predicted speech interval, independent of how the
model encodes it.

- Full turns use an asymmetric Gaussian onset reward: sigma 0.2 s when early,
  sigma 0.5 s when late.
- Backchannels receive 0.5 inside a +/-0.5 s plateau, then decrease linearly to
  -0.5.
- Missing speech and false alarms receive -0.5.
- An absent/overlong backchannel stop is capped at -0.5.
- An otherwise positive full turn without a stop inside the window is
  multiplied by 0.75.

Only the first predicted speech interval is scored. The diagnostics therefore
report `restart_after_eos` and `extra_speech_segments` so that a policy that
emits a short first segment and then starts again can be detected in logs.

## Event-grouped credit assignment

For rollout reward `r_(i,k)` inside event group `i`:

```text
A_(i,k) = clip((r_(i,k) - mean_i) / (population_std_i + 1e-8), -5, 5)
```

With a singleton group, the advantage is zero. Four rollouts per event were used
in the paper. Grouping by the event—not by the source conversation or global
batch—is essential.

## Policy and KL loss

For sampled action positions `M_(i,k)` and current logprob `log pi_theta`:

```text
L_policy = -1/N * sum_(i,k) A_(i,k) * sum_(t in M_(i,k)) log pi_theta(a_t)
```

The sampled-token mask excludes the teacher-forced prefix. This implementation
matches the paper checkpoint's reduction: token terms are summed, then divided
by the number of real rollouts `N` (not by the token count).

With reference logprob `log pi_ref`, `x = log pi_theta - log pi_ref`:

```text
KL_hat = exp(x) - x - 1
L = L_policy + beta/N * sum KL_hat,  beta = 0.2
```

The reference is a frozen policy when available; otherwise stored rollout
behavior logprobs provide the semi-offline reference. The reference code clips
the exponential input to `[-10, 10]` and each estimate above at 20, matching the
historical implementation.

## Backend responsibilities

The portable package constructs rewards and advantages. The backend must:

1. generate multiple rollouts for each `window_id`;
2. preserve behavior/reference logprobs if KL is enabled;
3. recompute differentiable current logprobs for sampled actions only;
4. apply `loss_mask` and the loss above;
5. retain any backend-specific distributed padding/dummy forwards required by
   its sharding implementation.

## Reported training scale

Values are the ones used for the paper checkpoint (`recipes/paper_v1.json`,
`training` block):

- full-parameter fine-tuning of the LLM backbone (speech encoder and codec
  frozen), 64 x A800 80 GB, bfloat16, gradient clip 1.0;
- 4 rollouts per window (GRPO group size), temperature 1.0, top-p 0.9;
- the batch unit is the conversation; every selected window of each
  conversation in the batch is used, so the number of windows (and rollouts)
  per step is not fixed;
- step count and conversations per step are not part of the recipe: they are
  chosen per model and setup. The reported run converged within a few tens of
  steps, i.e. inside the 100-step warm-up;
- AdamW, lr `1e-5` with 100 warm-up steps and inverse-square-root decay to
  `1e-6`. This is the **full-parameter** learning rate. With LoRA adapters the
  policy does not move within tens of steps at `1e-5`; use roughly 10x
  (about `1e-4`) and tune.

To reproduce: build windows with `duplexpo construct --profile {fisher,seamless}`,
generate 4 rollouts per window with your policy, run `duplexpo prepare` to
obtain GRPO records, and optimise with the loss above; monitor the mean reward
per event kind and the `restart_after_eos` diagnostic rather than a fixed step
budget.
