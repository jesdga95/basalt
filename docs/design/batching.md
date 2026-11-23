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

The interface is `include/basalt/model/adapter.hpp`; the shared serve rounds (`src/program/serve/batch_round.cpp`) drive it,
from the shared engine (`session_engine.cpp`) and from a model's own serving runtime alike. Every model batches through it:
one batching path, no model-specific batching mode.

- **Sessions are slots.** A model makes one `ModelSession` per slot (`LoadRequest::sessions`). A session owns its
  conversation's state: caches, rings, recurrent state, the drafter's context, its sampler and its last logits.
- **`int batch_rows()`** is the rows one batched pass holds in all, 0 for a model that does not batch.
- **`decode_batch(sessions, reqs, n)`** runs the windows `reqs[i]` of the distinct sessions `sessions[i]` as one pass. Each
  session ends exactly as after its own `decode_window(reqs[i])`: its picks, its last logits, its drafter's context.
- **`decode_kept(sessions, kept, n)`**, after the accept and before any drafts: the rows each window keeps. A model that
  settles a window's state at once (a verifier that rolls the rejected rows back) commits here; by default the next
  window's position says it.
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
next draft is likeliest kept (the joint of its drafts' calibrated chances), and rows no draft wants go round the slots. A
drafter that sets its own windows' length (`own_window`: a stop rule in its chain) states no chances to weigh: each window
takes the rows it would take alone, shared max-min fairly when they do not fit (`batch_fair_shares`: the windows wanting
least served in full, the others splitting what is left), and the policy only weighs a lookup proposal against its drafts.
With more slots than rows each may take the cap and the windows are packed first fit into passes of at most `cap` rows
(`batch_groups`). A lone window runs as `decode_window`, the solo path. The engine keeps one draft policy for solo windows and
one for batch windows, whose rounds cost differently. A runtime may hold a slot out of a round (`slot_ready`, Flash-Next's
elastic KV beside a prompt read); it keeps its place.

## Shared code and what a model provides

Shared code never names a model (`tools/check_model_layering.sh`).

| Piece | Where |
| --- | --- |
| The session interface: `batch_rows`, `decode_batch`, `decode_kept`, `draft_batch`, `own_window` | `model/adapter.hpp` |
| Rounds, row shares by the drafts' chances, groups, per-slot accept and output, for every runtime | `src/program/serve/batch_round.cpp`, `batch_plan.hpp` |
| Slot admission and the slots' conversations | `src/program/serve/request_finish.cpp`, `request_restore.cpp` |
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
   tokens and positions to the card, the launch bounds into the graph key), `batch_forward` (the layers over the segments and
   the head; false with an error when the window fails), `batch_take` (a segment's logits rows into its session, its
   history) and `batch_draw` (its draws, queued). A model whose forward only launches is captured by the driver, a graph per
   composition. A model whose host steers its forward between launches (Hy4 places its experts between per-layer graphs)
   returns false from `batch_captured`: the driver runs its forward directly every window and the model replays its own
   graphs of the stretches between. A session whose state the batch path does not cover yet returns false from
   `batch_ready`, and the batch's windows run in turn.
4. Run the checks: `basalt --model FILE --batch 4 --tokens-file PROMPT --diag-batch-check` (every row of every composition
   bitwise its solo window), then `tools/batch_identity.py ENGINE --model FILE --slots 1,2,4,8` (every slot's tokens, alone
   and together, equal to a one-slot engine's).

## Models on the driver

Every model batches through the shared driver and the shared rounds: no separate batching path.

- **Qwen3.8-27B** (the dense stack, `src/dense`) is on the driver. A window holds up to 16 rows (`kBatch`); its kernels read
  their own segment tables (the attention's `GqaSeg` per segment, GDN's `BatchTable` through `dec_gdn_rec_seg_split`), which
  carry each segment's cache and recurrent-state addresses beside its rows. `batch_reserve` settles a session (the last
  window's GDN commit on its side stream; the driver readies every session before it waits on any, so the commits run side by
  side) and prepares its draw; `batch_stage` packs the rows' tokens, rope positions (each session's offset) and the segment
  table (`stage_batch`); `batch_forward` is the layer walk and the head over every row, each segment's logits and block-drafter
  or MTP rows into its session (`batch_body`); `batch_take` its bookkeeping, `batch_draw` its sampler. The window scratch is the
  model's, shared by every session. A layer split runs its two cards' graphs itself (`batch_captured` false: `forward_batch`
  keeps each card's in its model's `BatchShared`). The block drafter's batched pass (`draft_batch`) stays the model's.
- **Qwen3.8-Flash-Next** (`src/models/qwen4exp`) keeps its own serving runtime (prompt split, elastic KV arena, peer
  expert tier, conversation cache); its slots batch through the shared rounds and driver (the retrofit below).

## Retrofits

How each model below went onto the driver: its state, the kernels that read it, its cap and what its sessions share.

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

### Hy4-flash (second)

- **State:** MLA latent rows per sparse layer and the indexer's pooled keys (one per 4 positions), sliding-window rings, the
  indexer's rows of the last positions, the Engram history (the conversation's own tokens), the DFly2 drafter's tap, the
  sampler and the logits. Hyper-connections are stateless.
- **Kernels:** they already take a position per row (`mla_query`, the row spans of `mla_attend`); only a cache's base is per
  conversation. The batch runs each segment's cache write, attention and indexer (ring rows, pooled keys, selection) as one
  launch of its rows over its own state; every other launch takes every row. The Engram's lookups read each row's own
  history on the host.
- **On the driver:** `batch_captured` is false: the host waits for the router's ids and places each layer's experts between
  three graphs a layer. The graphs hold no session state (the per-segment launches run between them), so they are keyed by
  the row count only. `batch_ready` is false for a session holding a picture's rotary table.
- **Cap:** 8 rows, the experts' group. With frozen tiers every row's pick and logits equal its solo window's.
- **Checks:** both with the tier frozen and its size pinned (`--diag-tier-frozen --expert-cache N`): each session costs
  VRAM, so an unpinned tier holds fewer experts at more slots. The serve check runs on one card: with a second card a
  prompt's experts split between the cards by measured rates, so two engine starts can differ (in turn as well); together
  equals alone on two cards too.
- **Experts:** one tier shared by every session. Most decode time goes to experts on the bus and in RAM, so the MoE caveat
  below decides whether more rows pay.

### DeepSeek-V4.1-Flash

- **State:** the window ring, the compressed KV rows and index keys of the kv-source layers, and the compressor's partial-group
  state (`st_kv`, `st_sc`): order-dependent, so a segment's rows update its own state in order, like a recurrence. Engram as
  Hy4.
- **Kernels:** the compressor and the indexer take segments the way GDN does (`dec_gdn_rec_seg`: each segment from its own
  state); the window and compressed attention take per-segment caches.
- **On the driver:** the session derives `BatchSession`, `batch_rows` 8 (a decode window's `kMaxT`), `batch_captured` false
  (the host places experts layer by layer, so the forward runs directly every window). Per segment: the attention through
  that session's own ops (caches, ring, compressor state: the same call as its solo window of that many rows), the Engram
  reads and projection, the DSpark feature rows. Every row: the mixes, norms, router, shared and routed experts and the
  head. `draft_batch` runs each session's DSpark block in turn (one shared drafter) and holds its drafts for `draft_end`.
- **Experts:** mostly in RAM and in the file on most machines. On two cards batching is slower than the windows in turn
  were (the MoE caveat below): a batch's rows reach a wider union of RAM experts, and the 8-row cap trims each slot's
  drafts. That is a limit of the hardware, accepted for one batching path. Two cards (400 W RTX 5090 + 180 W RTX 5060 Ti),
  drafted, tiers unfrozen, 256 tokens a slot, aggregate tok/s, ABBA (main: windows in turn):

| Slots | Main, in turn (A1 / A2) | Batched (B1 / B2) | Change |
| --- | --- | --- | --- |
| 1 | 89.3 / 87.0 | 87.0 / 92.8 | none: one slot runs its solo window, the same code |
| 2 | 85.0 / 82.7 | 85.4 / 80.4 | -1.1% |
| 4 | 71.5 / 70.0 | 63.8 / 62.1 | -11.0% |
| 8 | 76.9 / 77.2 | 54.6 / 51.1 | -31.4% |

### Qwen3.8-Flash-Next

- **State:** per slot a session state carved like the solo session's (on one card slot 0 is the solo session): the GDN
  recurrences and convolution histories, the QSA layers' INT8 K/V (elastic: cells backed out of the expert cache's tail on
  demand) with the indexer's tails and pooled rows, and the PLE history. The drafter's K/V is the drafter's store of the
  slot.
- **Kernels:** the verifier's batch window (`Verifier::run_slot_segs`) already ran segments of several slots over their own
  states, one graph per row count; a segment's rows are its slot's window.
- **On the driver:** a slot is a `QfnSlotSession` (`serve/slot_session.cpp`), `batch_captured` false (the host serves each
  layer's CPU experts). `batch_forward` runs the window on the verifier, which draws every row with its slot's sampling, and
  `batch_draw` hands back the picks. A window's kept rows are committed at `decode_kept`, beside the drafts when drafts
  follow. `draft_batch` runs the draft chain of each slot that goes on, stopped by its own length rule (`own_window`, so the
  windows take max-min fair shares); when the window holds as many rows as slots, the slots likeliest to use drafts make
  them and the others only advance their draft K/V. Between rounds the runtime backs each slot's next cells (a slot whose
  cells wait for a prompt read beside it sits the round out, `slot_ready`) and steps the adaptive tier. The runtime's own
  serving (prompt split, peer tier, conversation cache, the solo loop) is unchanged.
- **Cap:** 8 rows, the verifier's window (`kVerifyMaxT`). `--batch 1` with `--kv int8` is one slot (the batch identity
  tool's reference).
- **Checks:** `--diag-batch-check` runs in the serving engine, where the slots exist; a slot reads the check's prompt
  through its own windows.
- **Row invariance:** a window's CPU experts take a kernel by the window's row count, so a row's bits can depend on the
  other rows when its experts are in RAM. Solo windows against the same windows batched agree bitwise (the batch check);
  a slot decoding beside others can part from its tokens alone after a few dozen tokens on one card, token for token as it
  did before the move (plain decode, tiers pinned, both builds' tokens equal at every slot count).
- **Gate:** no slot count slower than Flash-Next's own rounds were. Two cards (400 W RTX 5090 + 180 W RTX 5060 Ti), int8
  KV, expert cache pinned at 8600, tiers adapting, 256 tokens a slot (the bench fixtures in turn), aggregate tok/s, ABBA:

| Slots | Drafted: own rounds (A1 / A2) | On the driver (B1 / B2) | Plain: own rounds (A1 / A2) | On the driver (B1 / B2) |
| --- | --- | --- | --- | --- |
| 1 | 195.6 (cold) / 235.9 | 233.2 / 225.1 | 127.7 / 129.2 | 129.2 / 130.2 |
| 2 | 206.4 / 200.0 | 207.4 / 205.4 | 161.1 / 161.9 | 162.6 / 163.2 |
| 4 | 266.8 / 261.2 | 264.9 / 263.8 | 227.0 / 236.6 | 229.5 / 236.6 |
| 8 | 307.9 / 309.3 | 307.7 / 309.7 | 313.9 / 311.3 | 311.8 / 308.7 |

## The MoE caveat

Rows from different conversations route independently. With k of E experts a row, R independent rows touch about
`E (1 - (1 - k / E)^R)` distinct experts a layer: Inkling (8 of 128) 8 at one row, 15.5 at 2, 29 at 4, 51 at 8. Draft rows of
one conversation overlap more than rows of different conversations, so 4 slots x 1 row reads more experts than 1 slot x 4
rows.

- **Experts on the card:** a window's cost is mostly fixed (launches, dense weights, attention); the union grows the expert
  reads only. Inkling's decode is far from the card's bandwidth at one row, so a batch is close to free.
- **Experts in RAM (Hy4, DeepSeek):** a RAM expert costs a bus copy or a CPU product per distinct expert touched, and the CPU
  cost grows with its rows. The union grows those directly, so a batch costs close to linear in its slots where most routed
  pairs land in RAM. Such a model still batches every round, through the same path: on DeepSeek with two cards four
  slots batched took CPU lookups from 13.3% to 17.6% of routed pairs and tier drops from 5.0K to 38.8K a run, and runs
  slower than its windows in turn did (the table under DeepSeek-V4.1-Flash).

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
