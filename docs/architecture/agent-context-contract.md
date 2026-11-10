# Agent context sharing — implementation contract

Binding interface between parallel implementation streams. Design: [agent-context-sharing.md](agent-context-sharing.md).

## Stream claims (as of 2026-11-10) — avoid duplicate work
Another local AI already holds worktrees `ctx-a-registry` (A), `ctx-d-scheduler` (D) and `ctx-engine-bc` (B+C). Those streams stay theirs.
This base branch (`ctx/00-base`) = design doc + Stateful Threads v1 merge + this contract; **rebase those branches onto it** (`git rebase ctx/00-base`) so the Rust workspace and thread API they extend exist.
Claude holds E (adapters/Mantle) and the later serial integration (engine `CONTEXT_*` commands, `serve/service.py`, docs). Do not touch those paths.

## File ownership (no stream edits another's files)
| Stream | Owns |
| --- | --- |
| A registry/fork API | `crates/basalt-context-registry/**`, `crates/migrations/0006_*.sql`.., `crates/basalt-thread-api/src/main.rs` routes `/v1/contexts*`, `serve/api/context_proxy.py`, `serve/tests/test_context_proxy.py` |
| B snapshot manifest | `include/basalt/context/{fingerprint,manifest,snapshot_io}.hpp`, `src/context/{fingerprint,manifest,snapshot_io}.cpp`, `tests/context/{fingerprint,manifest,snapshot_io}_test.cpp` |
| C pages/COW/tiers | `include/basalt/context/{page_store,tier_manager}.hpp`, matching `src/context`, `tests/context` files |
| D scheduler | `serve/scheduler/**`, `serve/tests/test_scheduler*.py` |
| E adapters/Mantle | `serve/mantle/**`, `serve/api/context_adapter.py`, `serve/tests/test_mantle*.py`, `test_context_adapter*.py` |
| Integration (later, serial) | `src/program/serve/**`, `serve/service.py`, `serve/handler.py`, `serve/config.py`, `CMakeLists.txt`, docs |

New C++ files need no CMake edit (`cmake/context.cmake` globs). Common types: `include/basalt/context/ids.hpp`.

## HTTP contract (Rust API, authoritative; owner scope from the existing auth header, tenant = thread owner)
- `POST /v1/contexts` `{thread_id, branch_id?, upto_message_id?}` → `201 {context_id, state, version, model_fingerprint(hex), prefix_hash(hex), prefix_tokens, tier}`. One context per transcript head: a repeated create at the branch's live head returns that context (the branch's own before forked children) instead of provisioning a second row.
- `POST /v1/contexts/{id}/fork` headers `Idempotency-Key`; body `{expected_version}` → `201|200 {context_id(child), parent_id, state, version}`; same key+payload → same child (200); different payload → `409`.
- `POST /v1/contexts/{id}/resume` body `{lease_ttl_ms?}` → `200 {context_id, lease:{token,expires_at}, source:"hot|warm|cold|replay"}`; one fenced writer lease per context. `lease.token` is a 64-bit integer.
- `POST /v1/contexts/{id}/lease/release` `{token}`; `POST /v1/contexts/{id}/transition` `{event, expected_version, lease_token?, tier?, locator?, validated?, model_fingerprint?, prefix_hash?, prefix_tokens?, engine_epoch?}` (engine/scheduler only). `tier` must equal the FSM's tier for the transition; `locator`/`validated` are accepted only with `ready|warm|chill`; the three manifest fields only with `ready` (a `ready` out of RESTORING must reproduce the stored identity, so omitting them is the norm); `engine_epoch` (UUID) with `materialize|restore|ready`.
- Explicit engine thread tapes (Python server, same auth; solo engine only, 503 while a request holds the engine for >2 s): `POST /v1/engine/threads/attach` `{thread, head, ids[]}` → `200`; `POST /v1/engine/threads/fork` `{parent, parent_head, child, child_head, keep_tokens?}` → `{shared_prefix_tokens}`; `409` stale head/unknown parent/conflict, `422` bad body. Engine line: `THREAD_FORK <parent> <parent_head> <child> <child_head> [keep]` → `THREAD_OK THREAD_FORK <parent> <child> <shared>`. Names are `[A-Za-z0-9_-]{1,128}`. The child shares the token prefix; KV reuse stays under the exact-prefix check. Not tied to the registry: contexts persist through `CONTEXT_*` snapshots.
- `GET /v1/contexts/{id}` → state, tier, position, `locator`, `lineage[]`, `validated`, `children`, `refs`; cross-tenant → `404`.
- `DELETE /v1/contexts/{id}` → tombstone `202`; GC deferred while children/leases hold refs. Never deletes the thread.
- `POST /v1/contexts/{id}/payload/released` (engine only, after it deleted the snapshot payload of a tombstoned context) → `200 {context_id, state, tier:"none", locator:null, version}`; clears locator/tier/validated/epoch so the sweeper can purge; non-tombstoned → `422`; idempotent.
- Errors: `{error:{code,message}}`; 404 unknown/foreign, 409 conflict/stale version/lease, 422 invalid.

## Context FSM (events)
`record`→RECORDED; `materialize`: RECORDED|COLD|FAILED→MATERIALIZING; `ready`: MATERIALIZING|RESTORING→HOT; `cool`: HOT→COOLING; `warm`: COOLING→WARM; `chill`: WARM|COOLING→COLD; `restore`: WARM|COLD→RESTORING; `fail`: MATERIALIZING|RESTORING|COOLING→FAILED; `reset`: FAILED→RECORDED. Every transition is CAS on `version` and, for HOT-writing events, on lease token. Eviction (`cool`) requires zero active leases.

## Engine manifest (C++ ↔ registry)
Manifest = `{format_version, model_fingerprint, prefix_hash, prefix_tokens, parent_prefix_hash?, generation, inventory[{kind: kv|gdn|ple|mtp_draft|tail, bytes, digest}], payload_digest}`. Published atomically (write tmp, fsync, digest verify, rename, fsync dir). Fingerprint covers weights id, tokenizer/template digest, state schema version, positional config, KV/state precision, vision/cvec inputs.

## Engine context commands (serve protocol, solo session, gated on `--context-store-dir`)
The registry stores the engine's restore metadata verbatim as the context `locator`, grammar `g<generation>,<meta>`:

```
CONTEXT_FINGERPRINT <chat_template_sha256_hex|->  -> CONTEXT_OK FINGERPRINT <fingerprint_hex> <epoch_hex32>
CONTEXT_SNAP <context_id> <generation>            -> CONTEXT_OK SNAP <id> <gen> <bytes> <tokens> <prefix_hash_hex> <meta>
CONTEXT_RESTORE <context_id> <gen> <ids_file> <meta> -> CONTEXT_OK RESTORE <id> <gen> <tokens>
CONTEXT_DROP <context_id>                         -> CONTEXT_OK DROP <id> <removed>
```

Every refusal is `ERR CONTEXT: <reason>`: the caller rebuilds the conversation from its transcript instead.
The snapshot's prefix runs to the request's last emitted token (its stop token included): the engine commits
nothing past the stop or the answer's cap, so the transcript the client holds re-renders to exactly the stored
prefix and `CONTEXT_RESTORE`'s `prefix_hash` check can pass from the transcript alone.
`<meta>` is the restore metadata text (running-state sizes, then per-KV-group geometry and buffer sizes; runs of
identical groups collapse to `<count>x<group>`); `<ids_file>` line 1 = comma-separated prefix token IDs, line 2 =
`<start>:<hash>` image keys. `epoch_hex32` is a fresh engine epoch in UUID simple form, patched into the registry
with `materialize|restore|ready`. The serve-side conductor is `serve/api/context_lifecycle.py`: attach leases and
restores (demoting a dead engine's HOT through `cool`+`chill` first), settle publishes `materialize`+`ready` after
a rebuilt conversation, an idle reaper cools HOT contexts to disk, and a DELETE drains the payload through
`CONTEXT_DROP` + `payload/released`.

## Scheduler (Python) hooks
`serve.scheduler.AgentScheduler`: `submit(Request(tenant, agent, cost_tokens, kind=prefill|decode, deadline))` → awaitable admission `Grant`; per-tenant/per-agent bounds, aging, cancellation deadlines, metrics dict for `/metrics`. `solo` bypass when only one request is in flight (single-stream fast path untouched).

## Mantle handoff (Python)
`serve.mantle.Handoff` JSON: `{plan, evidence[], task_dag, transcript_ref, source_fingerprint, target_model}`; raw KV never crosses models; destination re-prefills from transcript.
