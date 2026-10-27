# I/O protocol v1

The JSON Schemas in `src/duplexpo_portable/schemas/` are normative and are bundled in the wheel;
`load_json_schema("conversation")` (or `window`, `rollout`, `grpo_record`,
`recipe`) loads them after installation. Python dataclasses provide stricter
cross-field validation.

## 1. Conversation

A source adapter emits one record per complete conversation. `speaker` must
already be canonicalized to `user` or `agent`. `units` may be words or
utterances, selected by `granularity`. Split assignment must happen on
`conversation_id` before segments or windows are expanded.

Required information:

```json
{
  "conversation_id": "session-42",
  "duration_s": 37.2,
  "granularity": "word",
  "units": [
    {"speaker": "user", "start_s": 0.1, "end_s": 0.4, "text": "hello"}
  ]
}
```

Use `metadata` for audio URIs, language, role-resolution evidence, dataset
licenses, and other source-specific fields.

## 2. Window

Each record is one conversational event and therefore one GRPO group.

- frames `[0, sample_start_frame)` are the teacher-forced context;
- frames `[sample_start_frame, sample_end_frame)` are sampled;
- `[reference_start_frame, reference_end_frame)` describes expected agent
  speech timing but is never fed as an answer during rollout;
- `event_kind` is `turn` or `backchannel`;
- `user_segments` make interruption/yielding targets observable to reward code.

Conversation-level metadata is copied to `window.metadata.conversation`, so a
policy adapter can retain audio/object-store URIs, language, channel mappings,
or native sample IDs without changing the stable top-level schema.

## 3. Rollout and ActionTrace

The rollout runtime returns a unique `rollout_id`, its `window_id`, and either:

- model-native `action_ids`, decoded by a `TraceDecoder`; or
- a model-neutral `trace.speech` list with absolute timeline seconds.

If policy optimization is performed later or in another process, also attach
behavior or frozen-reference token logprobs. `loss_mask` must identify sampled
actions only. Prefix tokens are never policy-gradient targets.

## 4. GRPORecord

The package emits one flat record per rollout:

```json
{
  "group_id": "session-42:event-0003",
  "rollout_id": "sample-2",
  "window": {"window_id": "session-42:event-0003", "...": "..."},
  "action_ids": [12, 44, 2],
  "loss_mask": [true, true, true],
  "reference_logprobs": [-0.2, -1.1, -0.3],
  "reward": 0.81,
  "advantage": 1.24
}
```

A record is self-contained: the embedded `window` carries conditioning bounds
and conversation metadata required to reconstruct the policy input. A backend
recomputes differentiable current-policy logprobs for `action_ids` and applies
the formula in `ALGORITHM.md`. It must not re-normalize advantages
across conversations or a global batch.

## Compatibility rule

Readers must reject an unknown major schema version. New optional metadata may
be added in v1; renaming fields, changing units, or changing interval semantics
requires v2.
