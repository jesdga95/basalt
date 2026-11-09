# Agent context sharing and lifecycle — design proposal

Status: **proposed, not implemented**. Scope: Basalt CUDA/Blackwell fork. Inspiration: [inferd](https://github.com/Incept5-AI/inferd); no code is imported. This design extends existing Basalt behavior, not its current CLI contract.

## Problem and baseline

Basalt already has `--parallel` (1..8), per-slot root/turn checkpoints, elastic KV, RAM conversation snapshots, and `--conversation-cache-disk-mib`. See [running](../running.md) and [configuration](../configuration/index.md). These features reuse some shared prefixes, but do not expose a stable, forkable agent context independent of a slot or a client replaying the transcript. Model loading, slot admission and cache eviction are coupled through the shared VRAM budget.

**Goal:** many long-lived agents sharing immutable model-compatible prefix state, with independent branches and deterministic recovery after eviction or engine restart. Preserve the single-request fast path and existing API semantics.

**Non-goals:** a Metal backend; copying inferd kernels; assuming arbitrary KV compatibility across models; treating conversation storage as an exact replay of sampled output; immediately increasing `--parallel` above 8 without a scheduler/quality benchmark.

## Layers and ownership

| Layer | Source of truth | Responsibility |
| --- | --- | --- |
| API / server | PostgreSQL-backed thread/event log (new integration contract) | auth, tenant ownership, request/idempotency key, message order, immutable input/output events, branch ancestry |
| Context registry | transactional metadata + durable manifest | context IDs, version, model fingerprint, prefix identity, state, parent/child relations, reference counts / leases |
| Inference scheduler | live bounded queues and leases | admission, fairness, prefill/decode interleaving, cancel/retry, VRAM pressure, BIG/LITTLE hand-off |
| Context storage | GPU pages → pinned/host RAM → disk snapshot | share immutable prefix pages, private writable delta, checksums, tier transitions, recovery |
| Engine kernels | Basalt existing CUDA implementation | exact computation of KV and non-KV recurrent/draft state |

Do not conflate **transcript** (durable truth), **tokenized prefix** (model-specific reproducible input), and **running state** (model-specific disposable acceleration). Engine caches never overwrite raw events.

### Data and identity

`thread_id` and `response_id` identify user-facing lineage; `context_id` identifies an internal inference snapshot. Use opaque IDs, not a GPU slot number. Model fingerprint includes weights/quantization, tokenizer and chat template, engine state schema, positional configuration, KV/state precision, vision/control-vector inputs, and all settings that affect state. Prefix identity uses canonical **token IDs plus relevant non-text inputs**, not only visible prompt text; hash with domain separation. Snapshot manifest includes parent context, token position, checkpoints, page inventory, recurrent state inventory, CRC/cryptographic digest, and generation fencing token.

A context is shareable only when its model fingerprint and prefix identity match, its state is complete/validated, and its access scope permits reuse. Cross-tenant sharing is disabled by default; never key security on prefix hash alone. BIG and LITTLE models cannot share raw KV/recurrent state: transfer the canonical transcript and structured handoff, then prefill under the destination fingerprint.

### State machine

```text
RECORDED -> MATERIALIZING -> HOT
   |             |            |
   |             v            v
   +----------> FAILED       COOLING -> WARM -> COLD
                  ^             |        |       |
                  |             +------> RESTORING
                  |                       |
                  +-----------------------+
                             |
                             v
                            HOT
```

Transitions use compare-and-swap version and lease/fencing token. `RECORDED` is always reconstructable from immutable events even if no valid binary snapshot exists. `HOT` means allocated device pages; `WARM` means RAM; `COLD` means verified disk snapshot or replay-only marker. `FAILED` never means the transcript is lost. A fork creates a new child context referencing an immutable parent prefix and allocating private continuation pages (copy-on-write on attempted mutation). Parent deletion cannot reclaim blocks with active child references or leases.

### API contract (proposed, versioned)

- `POST /v1/contexts`: create from `thread_id` or immutable input events; return `context_id`, fingerprint, version.
- `POST /v1/contexts/{id}/fork`: idempotent child creation with expected parent version and `Idempotency-Key`.
- `POST /v1/contexts/{id}/resume`: resume from validated snapshot or canonical replay; token generation requires a fenced exclusive writer lease per child.
- `GET /v1/contexts/{id}`: state, tier, position, lineage, validation status, never expose cross-tenant cache contents.
- `DELETE /v1/contexts/{id}`: tombstone and deferred GC; never delete the underlying thread.
- Existing Chat/Responses/Anthropic-compatible routes continue to work; `previous_response_id`/thread binding is added at the server adapter, with explicit replay fallback.

Idempotency keys are scoped by tenant + operation + target; identical payload replays return the same result, conflicting payloads return a conflict. Store request outcome and transition with one durable transaction; external engine effects use an outbox plus fencing/compensation, not a fictional distributed atomic transaction.

### Memory and scheduling

1. **Phase A — metadata fork:** keep existing per-slot checkpoint implementation. Add registry, fingerprint, immutable transcript, and correct fork/resume under conservative resource limits.
2. **Phase B — shared paged state:** refcount immutable prefix pages; private delta pages; lazy allocation; copy-on-write; admission based on real GPU/RAM/disk budgets. Include *all* Qwen3.8 model-specific recurrent/non-KV state and MTP drafter state or explicitly recompute it. Do not assume KV pages alone are sufficient.
3. **Phase C — tiered snapshots:** device → RAM → disk, atomic manifest publish after payload fsync and digest verification. Eviction proceeds only after no active writer/reader lease; use generation fencing across reload. Restore validates fingerprint, schema, length, digests and ownership; mismatch deterministically rebuilds from transcript.
4. **Phase D — fair agent scheduler:** bounded per-tenant and per-agent admission, prefill/decode budgets, aging to avoid starvation, cancellation deadlines, and metrics for queueing/p95 TTFT. Preserve batch/solo compatibility and detect shared expert/KV VRAM pressure rather than blindly increasing parallel slots.
5. **Phase E — Mantle handoff:** serialize BIG plan / evidence / task DAG and model-specific checkpoints separately. On BIG→LITTLE transition persist output, release BIG GPU residency, load LITTLE and prefill; on promotion run reverse handoff. Measure total job throughput and quality, not only aggregate tokens/sec.

No production API flags/config values are added until end-to-end implementation exists; this document is a design contract, not a claim that endpoints or a PostgreSQL schema are present today.

## Safety, failure modes and verification

| Failure or risk | Required behavior / independently observable evidence |
| --- | --- |
| Fork mutates shared prefix | randomized parent + many children concurrent divergence tests; parent output and page hash unchanged |
| Two retries allocate different children | same idempotency key returns same child; conflicting payload rejected |
| Cancel/worker crash during tier move | previous complete manifest remains readable; no half-published snapshot |
| Engine restart / GPU reset | context restored or rebuilt from raw events; checksum and token position checked |
| Model update or KV precision change | fingerprint mismatch rejects binary reuse and triggers model-specific replay |
| Recurrent/draft state omitted | parity check against no-cache prefill/decode; disable reuse if parity fails |
| Cross-tenant context collision | authorization and isolation tests; no cross-scope cache hit or inspection |
| RAM/VRAM exhaustion | bounded admission, fair queueing, no OOM; preserve existing single-stream path |
| BIG→LITTLE handoff | raw state never reused; structured handoff produces traceable task result |

Measure on the same hardware and power limit: single-stream tok/s; 1/2/4/8 slot throughput; p50/p95 TTFT; effective tokens re-prefilled; shared-prefix hit rate; per-context GPU/RAM/disk bytes; restores per tier; errors; cancellation/timeout latency; completed multi-agent jobs and correctness. A/B with a cold-cache baseline and a long shared-prefix workload; report variance, not invented speedups. Run Linux and Windows build/CPU checks, CUDA parity and quality gates as applicable.

## Roadmap and shipped criteria

- **PR A — registry / durable contract:** types, migrations, event+idempotency ledger, FSM transition tests, branch ancestry, authorization. **Complete when** repeated requests and restart recover one canonical child.
- **PR B — snapshot correctness:** full model-state inventory, fingerprinted manifest, restore/rebuild, corruption injection, disk atomicity. **Complete when** uncached and restored continuations pass parity/quality criteria.
- **PR C — prefix COW and tiering:** shared pages, safe eviction, refcount/lease GC, contention tests. **Complete when** concurrent forks preserve parent and shrink measured memory/re-prefill.
- **PR D — scheduler:** admission fairness, prefill/decode priorities, metrics, mixed short/long context load; benchmark solo regression. **Complete when** latency/throughput and no-starvation gates pass.
- **PR E — user API + Mantle:** response/thread adapters, BIG/LITTLE handoff, retries and cancellations, GPU-restart E2E. **Complete when** a real multi-agent workflow survives an eviction and restart with audited results.
- **QA / red team:** independent design–code gap audit, security and fault injection, CPU/GPU E2E; do not manufacture findings when no new defects are evidenced.

Stack these PRs in dependency order, preserving a runnable parent branch. A merged design doc **does not** mark any implementation stage complete.

## Attribution and references

- [inferd project](https://github.com/Incept5-AI/inferd): architecture inspiration and comparison; independently implement capabilities in Basalt, after verifying licensing and actual source semantics before any code borrowing.
- [Basalt concurrent/elastic KV behavior](../running.md#concurrent-requests---parallel-n) and [disk conversations](../running.md#conversations-on-disk---conversation-cache-disk-mib).
- [Basalt configuration](../configuration/index.md#conversation-cache).
