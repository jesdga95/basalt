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
- `POST /v1/contexts` `{thread_id, branch_id?, upto_message_id?}` → `201 {context_id, state, version, model_fingerprint(hex), prefix_hash(hex), prefix_tokens, tier}`
- `POST /v1/contexts/{id}/fork` headers `Idempotency-Key`; body `{expected_version}` → `201|200 {context_id(child), parent_id, state, version}`; same key+payload → same child (200); different payload → `409`.
- `POST /v1/contexts/{id}/resume` body `{lease_ttl_ms?}` → `200 {context_id, lease:{token,expires_at}, source:"hot|warm|cold|replay"}`; one fenced writer lease per context.
- `POST /v1/contexts/{id}/lease/release` `{token}`; `POST /v1/contexts/{id}/transition` `{event, expected_version, lease_token?, tier?, locator?, validated?}` (engine/scheduler only); `ready` also accepts `model_fingerprint?`, `prefix_hash?`, `prefix_tokens?`, `engine_epoch?`.
- Engine fork (Python server, same auth as the API): `POST /v1/engine/threads/fork` `{parent, parent_head, child, child_head, keep_tokens?}` → `{shared_prefix_tokens}`; `409` stale head/conflict, `422` bad body, `503` engine unavailable. Engine line: `THREAD_FORK <parent> <parent_head> <child> <child_head> [keep]` → `THREAD_OK THREAD_FORK <parent> <child> <shared>`. The child shares the token prefix; KV reuse stays under the exact-prefix check.
- Scheduler: `basalt serve --scheduler` (off by default); identity from `X-Basalt-Tenant` / `X-Basalt-Agent` (<=64 chars of `[A-Za-z0-9_.-]`); a queue-full or expired admission is a 503-class `GpuBusy`.
- `GET /v1/contexts/{id}` → state, tier, position, `lineage[]`, `validated`, `children`, `refs`; cross-tenant → `404`.
- `DELETE /v1/contexts/{id}` → tombstone `202`; GC deferred while children/leases hold refs. Never deletes the thread.
- Errors: `{error:{code,message}}`; 404 unknown/foreign, 409 conflict/stale version/lease, 422 invalid.

## Context FSM (events)
`record`→RECORDED; `materialize`: RECORDED|COLD|FAILED→MATERIALIZING; `ready`: MATERIALIZING|RESTORING→HOT; `cool`: HOT→COOLING; `warm`: COOLING→WARM; `chill`: WARM|COOLING→COLD; `restore`: WARM|COLD→RESTORING; `fail`: MATERIALIZING|RESTORING|COOLING→FAILED; `reset`: FAILED→RECORDED. Every transition is CAS on `version` and, for HOT-writing events, on lease token. Eviction (`cool`) requires zero active leases.

## Engine manifest (C++ ↔ registry)
Manifest = `{format_version, model_fingerprint, prefix_hash, prefix_tokens, parent_prefix_hash?, generation, inventory[{kind: kv|gdn|ple|mtp_draft|tail, bytes, digest}], payload_digest}`. Published atomically (write tmp, fsync, digest verify, rename, fsync dir). Fingerprint covers weights id, tokenizer/template digest, state schema version, positional config, KV/state precision, vision/cvec inputs.

## Scheduler (Python) hooks
`serve.scheduler.AgentScheduler`: `submit(Request(tenant, agent, cost_tokens, kind=prefill|decode, deadline))` → awaitable admission `Grant`; per-tenant/per-agent bounds, aging, cancellation deadlines, metrics dict for `/metrics`. `solo` bypass when only one request is in flight (single-stream fast path untouched).

## Mantle handoff (Python)
`serve.mantle.Handoff` JSON: `{plan, evidence[], task_dag, transcript_ref, source_fingerprint, target_model}`; raw KV never crosses models; destination re-prefills from transcript.
