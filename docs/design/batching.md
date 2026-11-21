# Batched decoding

How several conversations share one pass through the weights, for engine contributors. A model that follows this page gets
`--parallel N` batching from the shared engine; the rest of it is the model's state and its stateful kernels.

## The idea

Every model already runs windows of 2 to 8 rows for speculative verify: the anchor and its drafts go through the weights in
one pass. The products, the experts, the head, the norms and the mixes are row-agnostic: a row's result does not depend on
the other rows of its call. A batch is the same window with rows from several conversations. Only what a row reads of its
own conversation differs:

- attention reads its conversation's K/V at its own positions;
- per-conversation state (recurrent state, short convolutions' rings, window rings, compressor state) is read and written per
  conversation;
- each conversation accepts or rejects its own drafts and keeps its own bookkeeping.

## The interface

The interface exists today (`include/basalt/model/adapter.hpp`); the dense stack (Qwen3.8-27B) implements it and the shared
serve engine (`src/program/serve/session_engine.cpp`) drives it. Every model batches through it.

- **Sessions are slots.** A model makes one `ModelSession` per slot (`LoadRequest::sessions`). A session owns its
  conversation's state: caches, rings, recurrent state, the drafter's context, its sampler and its last logits.
- **`int batch_rows()`** is the rows one batched pass holds in all, 0 for a model that does not batch.
- **`decode_batch(sessions, reqs, n)`** runs the windows `reqs[i]` of the distinct sessions `sessions[i]` as one pass. Each
  session ends exactly as after its own `decode_window(reqs[i])`: its picks, its last logits, its drafter's context.
- **`draft_batch(peers, anchors, pos, n, count)`** starts the drafts of several sessions in one pass where the drafter can
  (the default runs them in turn).

### The row-tagged window

A batch window is a list of segments. Segment i is one session's window: rows `row0[i] .. row0[i + 1] - 1`, tagged with that
session and positions `pos0[i]`, `pos0[i] + 1`, ... A drafted window is a segment of several rows; a batch of plain decodes
is segments of one row each. Rows of a segment always sit at consecutive positions of one conversation, which is what the
short convolutions and recurrences need: a row's convolution reads the rows before it in its own segment, then its
conversation's ring. Arbitrary per-row tags would buy nothing (a conversation's rows are always a run) and cost every
convolution a search.

Positions and tokens are staged on the card per row, so a captured graph replays at any positions. The segment table
(which rows belong to which session, and each session's cache addresses) is fixed per composition: the sessions and their
row counts. A model captures one graph per composition and keeps the most recently used ones.

### Scheduling slots x draft rows

The shared engine schedules; a model only states its cap. Each round, every busy slot drafts its next window. With no more
slots than rows one pass serves every slot (`batch_shares`): each takes one row, then each further row goes to the slot whose
next draft is likeliest kept (the joint of its drafts' calibrated chances), and rows no draft wants go round the slots. With more
slots than rows each may take the cap and the windows are packed first fit into passes of at most `cap` rows
(`batch_groups`). A lone window runs as `decode_window`, the solo path. The engine keeps one draft policy for solo windows and
one for batch windows, whose rounds cost differently.

## Shared code and what a model provides

Shared code never names a model (`tools/check_model_layering.sh`).

| Piece | Where |
| --- | --- |
| The session interface: `batch_rows`, `decode_batch`, `draft_batch` | `model/adapter.hpp` |
| Slot admission, rounds, row shares by the drafts' chances, groups, per-slot accept and output | `src/program/serve/session_engine.cpp`, `batch_plan.hpp` |
| The driver: `decode_batch` for any model (checks, segments, head rows, graph per composition, stream order, one sync, picks) | `model/batch_driver.hpp` (`BatchSession`) |
| A batch window's segments (rows, per-row positions and tokens) and the graph cache keyed by composition | `model/batch_window.hpp` |
| The device form of the segment table and a row's segment lookup, for any model's kernels | `kernels/window_segs.hpp` |
| Segment-aware shared kernels: attention (`gqa_decode_attn_segs`), GDN recurrence (`dec_gdn_rec_seg`) | `kernels/gqa_attn.hpp`, `kernels/gdn_window.hpp` |
| The conformance check: solo windows against the same windows batched, bitwise | `model/batch_check.hpp`, `--diag-batch-check` |
| Serve identity and throughput across slot counts against one slot | `tools/batch_identity.py` |

### Per-slot state and shared scratch

A session holds only its conversation: caches, rings, recurrent state, the drafter's context, its sampler and its last
logits. Everything one forward call needs and drops when it ends (row buffers, activations, expert staging, attention and
product workspaces, the prompt path's chunk buffers, a drafter's tap) is the model's shared scratch, allocated once for every
session. That is safe because the engine runs one call at a time: each session call syncs its stream before it returns, and
the driver syncs every session's stream before a batch runs on the leader's. A model sizes its memory as per-session state
times the slots plus the shared scratch once, so more slots cost only their caches.

### Adding batching to a model

1. Split the session's allocations: per-slot state in the session, the forward scratch in one object the sessions share.
2. Make each stateful kernel take a `kernels::WindowSegs` beside each segment's state: attention over that segment's cache,
   convolutions that stop at the segment's first row, recurrences from the segment's own state. Prove each with a kernel
   parity test: several conversations in one call bitwise equal to each alone.
3. Derive the session from `model::BatchSession` and write its hooks: `batch_rows` (the widest row-invariant window),
   `batch_name`, `batch_context`, `batch_shared`, `batch_stream`, `batch_reserve` (room for positions), `batch_stage` (rows'
   tokens and positions to the card, the launch bounds into the graph key), `batch_forward` (launches only: the layers over
   the segments and the head), `batch_take` (a segment's logits rows into its session, its history) and `batch_draw` (its
   draws, queued).
4. Run the checks: `basalt --model FILE --batch 4 --tokens-file PROMPT --diag-batch-check` (every row of every composition
   bitwise its solo window), then `tools/batch_identity.py ENGINE --model FILE --slots 1,2,4,8` (every slot's tokens, alone
   and together, equal to a one-slot engine's).

## Flash-Next and 27B

- **Qwen3.8-27B** (the dense stack, `src/dense`) has its own `decode_batch`: it packs segments (`BatchSeg`) into one window
  of up to 16 rows, attention through `gqa_decode_attn_segs`, GDN through `dec_gdn_rec_seg_split`, one graph per composition
  (`Model::batch_graphs`). On the driver its `BatchSeg` packing becomes `batch_stage`, the layer walk `batch_forward` (its
  kernels already read segments), the logits and MTP or DFlash taps per segment `batch_take`, its sampler `batch_draw`;
  `Model::batch_graphs` goes for `BatchShared`. Its sessions already share the stack's buffers.
- **Qwen3.8-Flash-Next** (`src/models/qwen4exp`) has its own runtime (`serve::Runtime`) and verifier: `Verifier::stage_batch`
  takes per-row slot ids and lengths over its elastic KV arena. It is the most tuned path in the engine and already batches
  slots x drafts in one pass. It does not migrate: a rewrite onto the shared runtime gains nothing measurable and risks its
  decode numbers. New models copy the shared interface, not its runtime.

## Retrofits

Every model below runs `--parallel N` today as N sessions whose windows run one after another (`batch_rows() == 0`).

### Inkling 2 Mini (first)

- **State:** per visit (48 layer applications, the loop's core three times) a `VisitCache`: K/V (a window ring on sliding
  layers, a cell per position on full ones) and four convolution rings (key and value inputs, attention output, experts'
  sum), all indexed by position. The block-attention mixes, loop controllers and messages live inside one forward: no state.
  The DSpark drafter's context is per session.
- **Kernels that read state:** the attention's preparation (key and value convolutions, cache and ring writes), the attention
  itself, `post_attn` (the attention output's convolution), `moe_finish` (the experts' sum's convolution), and `mix_norm`'s
  L2 read-ahead of older cells. Each takes a segment table (`Segments` in its args: per segment its first row and its caches);
  without one (n = 0) the call is one conversation's, as today.
- **The fused small-window attention** (1 to 3 rows, one cluster kernel) stays solo-only; a batch takes the prepare, split
  and combine kernels, which give the same bits.
- **Cap:** 8 rows, the products' column limit (`kDecodeRows`).
- **On the driver:** `batch_stage` stages every row's token and position, `batch_forward` runs the visits with each
  segment's caches, then the head over every row into the shared batch logits, and commits each session's drafter context
  from its rows of the shared tap; `batch_take` copies a segment's head rows into its session's logits; `batch_draw` samples
  with the session's own sampler into its pinned picks.
- **Shared:** the forward's row buffers and expert scratch, the prompt path (its chunk buffers) and the drafter's tap.

### Hy4-flash

- **State:** MLA latent rows per sparse layer and the indexer's pooled keys (one per 4 positions), sliding-window rings, the
  Engram rows (looked up from the conversation's own preceding tokens), the DFly2 drafter's context. Hyper-connections are
  stateless.
- **Kernels:** the latent cache write and the sparse attention take per-segment caches; the indexer's pooled key of a partial
  group of 4 is per-segment state; Engram hashing reads each row's own history (per-segment token ring).
- **Experts:** one tier shared by every session. Most decode time goes to experts on the bus and in RAM, so the MoE caveat
  below decides its cap.

### DeepSeek-V4.1-Flash

- **State:** the window ring, the compressed KV rows and index keys of the kv-source layers, and the compressor's partial-group
  state (`st_kv`, `st_sc`): order-dependent, so a segment's rows update its own state in order, like a recurrence. Engram as
  Hy4.
- **Kernels:** the compressor and the indexer take segments the way GDN does (`dec_gdn_rec_seg`: each segment from its own
  state); the window and compressed attention take per-segment caches.
- **Experts:** mostly in RAM and in the file on most machines; judged on two cards only.

## The MoE caveat

Rows from different conversations route independently. With k of E experts a row, R independent rows touch about
`E (1 - (1 - k / E)^R)` distinct experts a layer: Inkling (8 of 128) 8 at one row, 15.5 at 2, 29 at 4, 51 at 8. Draft rows of
one conversation overlap more than rows of different conversations, so 4 slots x 1 row reads more experts than 1 slot x 4
rows.

- **Experts on the card:** a window's cost is mostly fixed (launches, dense weights, attention); the union grows the expert
  reads only. Inkling's decode is far from the card's bandwidth at one row, so a batch is close to free.
- **Experts in RAM (Hy4, DeepSeek):** a RAM expert costs a bus copy or a CPU product per distinct expert touched, and the CPU
  cost grows with its rows. The union grows those directly, so a batch costs close to linear in its slots where most routed
  pairs land in RAM. Such a model's cap comes from a measured cost per window by rows and slots (the batch draft policy
  already prices rounds separately), not from its kernels' limit, and lands only when two-slot and four-slot aggregate
  throughput beat in-turn windows on every placement.

## Lossless identity

- **Row invariance** is the contract every kernel in a window keeps: a row's bits do not depend on how many rows share its
  call, where it sits among them, or which conversations the others belong to. Products take a fixed reduction order per row
  whatever the column count; experts compute per (row, expert) entry; attention splits a row's keys by its own key count,
  never by the launch's bound; convolutions read only their own segment.
- **A request alone in its slot** runs the solo path: one busy slot is a group of one, which runs `decode_window`, the same
  code as `--parallel 1`. Its tokens are bitwise those of `--parallel 1`.
- **Concurrent requests** each give the tokens they give alone, greedy: their rows' logits are bitwise the same, and verify
  keeps exactly the greedy tokens, whatever drafts a smaller row share left in the window.
- **Sampling:** each session draws with its own sampler, counter and history, so a seeded request draws the same alone or
  beside others.
- **No new defaults:** KV formats (fp8, 4-bit) stay opt-in; batching changes no flag's default.

## Tests and bench

- **Kernel parity:** each segment-aware kernel against the same segments run one call each, bitwise
  (`<model>_kernels_parity`).
- **Batch check** (`--diag-batch-check`, `model/batch_check.hpp`): through the session interface only, windows of 1 to 3
  rows of several sessions alone, then batched in order, in reverse order and in pairs; every row's pick and each window's
  last logits bitwise the solo window's.
- **CLI identity:** greedy tokens of the branch against main, plain and drafted (`--spec 1` and default), short and 16K
  prompts: `--parallel 1` is unchanged.
- **Serve identity** (`tools/batch_identity.py`): each prompt first in a one-slot engine (`--batch 1`), then in an engine
  with `--batch N` alone in slot 0 and all together, one per slot; every slot's tokens must equal the one-slot run's, greedy,
  at 2, 4 and 8 slots.
- **Throughput:** the same tool's aggregate tokens per second (every slot's tokens over the wall time from the first slot's
  start to the last one's end) at 1, 2, 4 and 8 slots, drafting on and with `--spec 1`, main against the branch, the bench
  fixtures (conv, struct, prose). Labelled with the cards and power (400 W RTX 5090, 180 W RTX 5060 Ti). Main's numbers are
  the in-turn windows.
- **CTest:** `model_layering` and the CPU tests without a GPU.
