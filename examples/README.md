# Conformance vectors

`conversations.jsonl` and `rollouts.jsonl` are synthetic and redistributable.
The files in `expected/` are frozen outputs from recipe `duplexpo-paper-v1`.

Regenerate into a temporary directory and compare byte-for-byte:

```bash
duplexpo construct --profile fisher \
  --input examples/conversations.jsonl \
  --output "$TMPDIR/windows.jsonl" \
  --audit-output "$TMPDIR/audit.jsonl"

duplexpo prepare \
  --windows "$TMPDIR/windows.jsonl" \
  --rollouts examples/rollouts.jsonl \
  --output "$TMPDIR/grpo_records.jsonl"

cmp "$TMPDIR/windows.jsonl" examples/expected/windows.jsonl
cmp "$TMPDIR/audit.jsonl" examples/expected/audit.jsonl
cmp "$TMPDIR/grpo_records.jsonl" examples/expected/grpo_records.jsonl
```

The bundled rollout vector is reward-only so it remains independent of any
model vocabulary. A policy adapter must additionally pass `duplexpo conformance
--require-trainable`, which requires action IDs and behavior/frozen-reference
logprobs.
