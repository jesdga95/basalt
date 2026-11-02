# Jev decisions

Finite decisions use the resident model's target logits rather than generated text. Add one setting to the
existing config, keeping its model and engine blocks:

```json
{ "server": { "jev": 1 } }
```

Rebuild the native binary and run `bin/basalt serve --config basalt_config.json` as usual. The command-line
equivalent is `--jev 1`; `--jev 0` overrides the file and disables it. Omission and `0` keep Jev off. Put its settings under `server`,
where the API lives; the server enables the native `--jev` capability itself.

The engine requires a version 2 `.basalt` pack. An existing version 1 pack can be migrated with upstream's
[pack migration](../../tools/migrations/README.md), preserving its weights and quantization.

## Runtime and settings

| Setting or component | Interaction |
| --- | --- |
| One chat session | Supported with defaults or `parallel: 1`; Jev adds no chat slots and never divides the context. |
| Multiple chat sessions | Existing chat parallelism remains. A decision wave waits for active chats, then leases the verifier exclusively. Native chat and decision work exclude each other at lease boundaries. |
| Weights | Uses the loaded `.basalt` pack and its tokenizer/template. Experts, dense tensors, PLE/n-gram table, MTP and vision weights keep their stored formats. |
| KV | Keeps the selected `kv` format, including `fp16`, and the existing context and memory policies. |
| Experts and prefill | Keeps expert placement, resident experts, auto-sizing, prefill and adaptive-tier settings. Decision work can warm the same expert cache as chat. |
| MTP | Keeps chat `spec` and the drafter. Decisions prepare target state only. A draft-valid prefix bounds which checkpoints a subsequent chat may reuse; chat-only cache entries and parked conversations keep their ordinary draft state. |
| Prompt cache | Shared prefixes use existing checkpoints, including recurrent state, indexer tails and PLE history. Their count remains bounded by `prompt-cache`; `0` works by recomputing prefixes. More checkpoints improve reuse at the existing RAM cost. |
| Conversation cache | Follows the existing RAM/disk policy; chat conversations can be parked or evicted as with other requests. Enabling Jev reserves no extra conversation pool. |
| Sampling and grammar | Chat settings remain unchanged. Decision rows use allowed-token probabilities, independently of chat temperature, top-k, penalties, relaxed drafting and output grammar. |
| Vision | Chat vision stays configured as before. Decision inputs are text/state JSON; these routes do not encode images. |

`jev` is a numeric decision lane limit, `0..8`, separate from `parallel`. `1` uses the solo target scorer with
zero dedicated GPU KV storage. Higher values allow independent suffixes to advance together through native batch
windows. Native `INFO` reports the supported lanes and row capacity; responses report the actual largest wave.
The verifier row capacity is the maximum of `spec`, chat batch width, `jev`, and two. This sizes target scratch
without changing the chat speculation length, chat slots or context division. No second model is loaded.

Compatible rows fork from their longest shared prefix. Each suffix consumes its own token rows, advancing across
as many bounded windows as needed. Resident unrotated FP16 and INT8 KV support these waves. Other KV layouts and
layer-split stages use exact solo scoring. A wave that exceeds workspace is split into smaller waves,
down to solo scoring when necessary. Peer expert tiers remain available.

Short waves share read-only GDN recurrence and copy only writable QSA KV/indexer state and PLE history. Continuing
waves also copy FP32 recurrence per lane, so each sequence can commit intermediate windows independently. For
the model's default geometry this recurrence is about 112 MiB per lane, in addition to prompt-sized KV/PLE.
The arena is allocated on demand against actual available memory and the server's native free-memory floor,
rather than a fixed arena ceiling. Allocations include a staging allowance and recheck free memory afterward.
Buffers and graphs are reused between decisions and released before chat work. Expert placement, chat KV
precision and context capacity remain unchanged; limited free memory can reduce effective decision width.

After 60 seconds without decision work, the native event loop releases Jev's branch buffers, graphs and prefix
checkpoints. Its excess elastic KV backing returns to the ordinary engine baseline; fixed session allocations
remain part of the loaded model. Chat work and model sleep release this temporary state immediately. Parked
conversations and other chat slots keep their caches. Cleanup waits on a deadline, with no Jev polling thread.

The host submits independent tree nodes together; greedy paths proceed in dependent waves. Requests already
queued at an engine boundary may share a sequence batch, with no collection timer or identical-request fan-out.
Each caller completes when its own rows finish, rather than waiting for unrelated groups. Cache-opt-out rows
establish their fresh prefix in a separate transaction.
Forced single-token edges need no target score. Prefix checkpoints remain
bounded by the existing host cache budget; the branch KV prefix is copied, rather than copy-on-write mapped.

Advanced settings are optional:

| Server setting | Default | Range | Meaning |
| --- | --- | --- | --- |
| `jev-max-pending` | `16` | `0..256` | Concurrent decisions beside one admitted caller; excess receives HTTP 429. |
| `jev-timeout` | `300` | `1..3600` seconds | Queue/scoring deadline. Cold load or wake uses the existing bounded engine lifecycle; an expired request stops when that operation returns. |
| `jev-max-nodes` | `8192` | `1..65536` | Total compiled divergent nodes across contexts and fields. |
| `jev-tree-max` | `128` | `1..255` | Largest domain evaluated exhaustively in automatic mode. |

Disconnects and deadlines remove queued callers; shared work stops when no unfinished caller still needs it.
Native continuing waves check `STOP` between windows and drain the reply. An engine that fails to stop within ten seconds
is ended so the next request starts a synchronized process. The same API key, Host and CORS rules protect these
routes; replies carry `Cache-Control: no-store`. Each request body is limited to 4 MiB, with separate text and
domain limits below. No model requests or polling run while Jev is idle.

## Finite fields

`POST /v1/decision` (also `/decision`) accepts:

```json
{
  "instructions": "Classify each message.",
  "contexts": ["I was charged twice."],
  "schema": {
    "urgent": { "type": "boolean", "description": "Needs immediate action?" },
    "topic": { "type": "enum", "description": "Main topic", "choices": ["billing", "other"] }
  },
  "mode": "auto",
  "cache_prompt": true
}
```

The optional `model` must name the served model or an alias. Each result contains `decision` (field values),
`fields` (selected probability, scored node count and tree/greedy mode), and context usage. Top-level usage counts
prompt, cached and computed tokens and scored rows; timings include prefill, scoring and total elapsed time.
Repeated prefix visits count as input work, rather than generated output tokens. `execution` reports
`requested_lanes`, `effective_lanes`, the active `kv` format and `dedicated_kv_bytes` (the branch arena plus device
staging, including indexer/PLE storage; zero on the solo path). `largest_batch` and `scored_windows` report physical
windows; `deduplicated_rows` is zero because callers are scored independently. `native_scoring_ms` is native scoring wall
time, including branch setup and copies; `total_ms` also includes admission, formatting and tokenization.

- `boolean`: `true` or `false`.
- `enum`/`choice`/`selection`: `choices`, 1..255 distinct strings. JSON Schema `enum` works too.
- `integer`: inclusive `minimum` and `maximum`, at most 255 values, signed 64-bit bounds.
- `number`: inclusive bounds and positive `step`, at most 255 grid points and nine decimal places. Both endpoints
  must lie on the grid and survive JSON number conversion exactly.

A JSON Schema object wrapper with `properties` is accepted. In that form number grids use `multipleOf`, aligned
to zero, instead of `step`. Only finite scalar fields are supported; nested objects, arbitrary strings and arrays
are rejected. Compact fields require `description`; wrapper fields may omit it. Limits: 32 fields, 256 contexts,
1 MiB combined context text, and 1 MiB combined instructions/schema. Choice strings and field names are bounded.

`tree` evaluates every divergent node and returns the normalized constrained distribution's winning value.
`greedy` walks the locally strongest allowed edge; its probability is the chosen path's probability, not an
exhaustive distribution. `auto` uses tree mode up to `tree_max` (request override), greedy above it. Numeric fields
can set `aggregate: "mean"` or `"median"` (`x-aggregate` in a wrapper); these require tree mode and select the nearest
grid value for a mean. Tree numeric fields include `interval_p10_p90`.

Probabilities are conditional on the allowed branches, not calibrated real-world confidence. Complete candidate
continuations are tokenized together to preserve BPE boundaries. Forced edges cost no scoring row. With
`cache_prompt: false`, the first score rebuilds its prefix; subsequent rows share that request's fresh prefix.
Oversized prompts receive HTTP 400 rather than truncation.

## SystemOne questions

`POST /v1/systemone` accepts TypeSafe-shaped questions:

```json
{
  "state": "I was charged twice.",
  "questions": {
    "needs_help": { "type": "noul", "instructions": "Does the customer need help?" },
    "topic": {
      "type": "choice",
      "instructions": "Pick the main topic.",
      "criteria": { "billing": "A charge or payment", "other": "Another topic" }
    },
    "severity": { "type": "score", "criteria": ["Low", "Medium", "High"] }
  }
}
```

`state`, instructions and criterion descriptions may be text, JSON objects, arrays or null. There are at most
32 questions and 1 MiB of combined state/rubrics. Question IDs are response keys only: renaming an ID changes no
model input. A single question's rubric precedes the changing state for prefix reuse; multiple questions share
the state and carry isolated rubrics.

The response has `model`, `answers`, input/output usage, `execution` and `timings`; output tokens are zero.
Execution reports actual width and workspace, while timings separate target prefill and native scoring.
Shared physical-group costs are accounted once, on the group's first row; caller elapsed time includes waiting.

Answer shapes:

- `noul`: `noul` is always `P(true)`, including when false is more likely. Optional criteria keys are `true`/`false`.
- `choice`: `choice`, a probability map and confidence `(N * P(winner) - 1) / (N - 1)`; a singleton has confidence 1.
- `score`: 2..10 ordered criteria, zero-based expected `score`, probability map and index-to-criterion `legend`.
  Confidence is one minus expected distance from the modal index divided by uniform distance from the center,
  floored at zero. It is a distribution-shape measure, not a correctness guarantee.

All SystemOne questions use exhaustive tree scoring. `/v1/decision` uses the same native scoring seam.

## Design references

- [Codacus parallel-decision](https://github.com/thecodacus/llama.cpp/tree/parallel-decision): finite domains,
  complete-continuation tokenization, compressed tries and constrained target probabilities.
- [NInfer Jev](https://github.com/Neroued/ninfer/pull/387): SystemOne contracts and bounded shared-model admission.
- [Precision](../precision/decisions.md): the target head and stored weight formats.
