# Several GPUs

How a run uses more than one card from 0.5.7 on, for engine contributors. The user picks the cards and the mode; shared code
places the work and reports it; each model fills a small contract per mode. The engine never prices or picks a mode.

## The idea

Today each model decides for itself what a second card does: `plan_multi_gpu` prices mechanisms, alike cards get one rule and
unlike cards another, and a card may idle without being asked to. Users then ask why their second card is idle (#527).
Batching had the same problem and fixed it the same way (see [batching](batching.md)).

`gpu` says which cards, `gpu-mode` what they do together, `gpu-share` in what proportions. Shared code owns card selection,
shares, placement, transfers, refusals and the start lines. A model states what it can divide and runs its own forward over
what it was given; a mode it cannot run is refused with the reason and the modes it runs. How a mode works inside stays as it
is: which experts a peer holds, the overlapped split windows, the GDN and head splits.

## The settings

The same three names everywhere: flags (`--gpu`, `--gpu-mode`, `--gpu-share`) of `basalt`, `basalt-bench` and `basalt-probe`,
and entry options of the config (`serve/config.py` `MODEL_OPTIONS`), so each model in a config sets its own.

| Setting | Values | Default | Meaning |
| --- | --- | --- | --- |
| `gpu` | `0`, `0,1`, `1,0,2`, ..., `all` | GPU 0 alone, with one warning that names the other visible cards | The cards, numbered as `nvidia-smi` numbers them. The first card listed is the main card. `all` means every visible card in that order, so GPU 0 is the main card. |
| `gpu-mode` | `peer`, `split`, `tensor` | none | Required when `gpu` names two or more cards. Not allowed with one card. |
| `gpu-share` | positive numbers, one per card, such as `3,1` | see [shares](#shares) | Relative shares, in the same order as `gpu`. Not allowed with one card. |

- **peer:** the main card runs every layer and the others help: experts and prompt stages in a MoE model, prompt stages in a
  dense one (the 27B's prompt split).
- **split:** the layers are divided between the cards; KV and state stay with their layers; the cuts follow the shares.
- **tensor:** every layer sharded. Reserved: the name and hooks exist, no model runs it at 0.5.7 ([tensor](#tensor-reserved)).

### Parsing and errors

Parsing stays in `program/gpu_select.cpp`, before any CUDA call. The server checks the same rules through `--gpu-plan` with the
model, so a config error stops the start before any load. An old value gets the error any bad value gets: no transition
release, no message aimed at removed values. Exact texts:

```
gpu "auto": expected a card number like 0, a list like 0,1 (the first is the main card), or all
gpu 0,1 names 2 cards: set gpu-mode to peer, split or tensor
gpu-mode "auto": expected peer, split or tensor
gpu-mode peer: gpu names one card (gpu "0"); a mode needs two or more
gpu-share "3": expected one positive number per card of gpu (2 cards), like 3,1
gpu-share: gpu names one card (gpu "0"); shares need two or more
gpu-mode split: Hy4-flash does not run it; it runs peer
gpu-mode tensor: Qwen3.8-27B does not run it (not built); it runs peer or split
gpu-mode peer: Hy4-flash takes at most 4 cards (the main card and 3 peers); gpu names 5
gpu-mode peer: GPU 1 (RTX 5060 Ti) would hold nothing: the main card holds every expert; use gpu 0
gpu-mode split: GPU 1 (RTX 5060 Ti) takes layers 30-47 at share 0.38 and needs 15.2 GiB, 13.8 GiB free; lower its gpu-share, max-context or parallel
gpu-mode split: GPU 1 (RTX 4060 Ti) is sm_89; split and tensor run on sm_120 cards (peer takes sm_89 in a -DPEER_ADA=ON build)
gpu 1,0: GPU 1 (RTX 4060 Ti) is sm_89; the main card must be sm_120
```

`gpu` in a text is the setting as written (`gpu "all"`, `gpu "0"` when unset), the model its adapter's `model_name()`; "(not built)"
marks a mode no model runs (tensor at 0.5.7). The checks that need the shares (nothing to hold, does not fit) are the shared
deal's. The number checks stay (past the visible cards, listed twice). In the config `gpu` and `gpu-share` may be a number, a
list or a string (`check_gpu`, `gpu_text`); the server forwards `--gpu-mode` and `--gpu-share` beside the narrowed `--gpu`.
A visible card `gpu` does not name is no error: every model warns once at start, in the CLI, the server and the bench (`gpu_select.cpp`):
`GPU 1 (RTX 5060 Ti) unused; to use more cards set gpu (0,1 or all) and gpu-mode`. It stays through the rework; the refusals are for named cards.

### The start line

One line before the load and one after, the same for every model; the first replaces `gpus: ... expert tier` (`cli/gpu.cpp:72`):

```
gpus: peer; RTX 5090 (nvidia-smi 0) main, share 0.80; RTX 5060 Ti (nvidia-smi 1) peer, share 0.20 (bandwidth)
gpus: RTX 5090 dense, KV, 9,812 experts (21.3 GiB); RTX 5060 Ti 5,120 experts (13.9 GiB), prompt layers 38-47
```

The tag says where the shares came from (`gpu-share`, `free VRAM`, `bandwidth`); peer shows the prompt stages' shares, split
the layers'. Each card's part of the second line is the model's `gpu_report(card)`; the shared layer prints it. One card: `gpus: RTX 5090 (nvidia-smi 0)`.

## Shares

The shares are normalized to sum to 1 and stay fixed for the run. What a share divides depends on the mode:

| Mode | What the share divides | Default without `gpu-share` |
| --- | --- | --- |
| split | each card's bytes: its layers' weights plus their KV and state at the context and slots; the cut is the layer nearest the shares, a part that does not fit is refused | free VRAM at start |
| peer, experts | the experts outside the main card, dealt hottest first to the peers by share (replaces `spread_experts`' bandwidth weighting); the main card always fills first | free VRAM of the peers |
| peer, prompt stages | the layers of a long prompt's chunk: the main card reads the first part, each stage card the next part by share | memory bandwidth |
| peer, main card's hot share | only when `gpu-share` is given: the hottest experts are interleaved main/peer by share (today's alike-card 0.45, `load/model.cpp:121-136`) | off: the main card keeps the hottest |
| tensor | heads per card | equal (the PR shards evenly) |

## The shared layer

| Piece | Where |
| --- | --- |
| Parsing `gpu`, `gpu-mode`, `gpu-share`; `GpuMode`, `GpuModes` and their refusals; card table; `CUDA_VISIBLE_DEVICES` narrowing; `--gpu-plan` JSON with mode and shares | `program/gpu_select.hpp` (kept, trimmed) |
| The card set a model receives, mode support, refusals, shares, start lines | `model/multi_gpu.hpp` (new) |
| Peer's driver: the deal by block, expert or stage, the would-hold-nothing refusal, each card's part (step 4) | `model/peer_hooks.hpp` |
| Split's driver: the cut by the shares or `--diag-gpu-cut`, `--diag-gpu-same`, the refusals, each card's part (step 6) | `model/split_hooks.hpp` |
| Expert deal by share, layer cuts by share and bytes, prompt-stage cuts | `model/multi_gpu.cpp` (from `gpu_select.cpp:266-300`, `dense/split_plan.cpp`) |
| Transfers: peer access for every pair, device-to-device or host-staged copies, link probes | `core/card_link.hpp` (from `cli/gpu.cpp:76-97` and Flash-Next's `probe_pcie_h2d_gbps`) |
| Prompt stages and peer expert storage (shared already) | `core/prompt_stage.hpp`, `core/peer_experts.hpp`, `core/expert_tier.cpp` |
| The conformance check | `model/gpu_check.hpp`, `--diag-gpu-check`, `tools/gpu_identity.py` |

```cpp
// GpuCards (model/multi_gpu.hpp): the cards main first (device, info, index, label, free), the mode, gpu-share, diag_cut
// include/basalt/model/peer_hooks.hpp (built): like BatchSession, a shared driver over the hooks a model writes
enum class PeerDeal { None, Blocks, Experts };     // experts by layer block (split_by_bytes) or one by one (deal_experts)
struct PeerSpec { PeerDeal deal; std::vector<uint64_t> unit; int block_most; uint64_t reserve; bool main_holds_all;
                  int stage_layers, stage_lo; };   // unit: per layer one expert's bytes (Blocks), per expert hottest first
struct PeerPlacement { std::vector<int> first, end; int homes; std::vector<int> owner; std::vector<int64_t> held;
                       std::vector<uint64_t> bytes; std::vector<int> stage_first; };   // per card, 0 = the main card
class PeerHooks {
public:   // peer_card on every peer, peer_room, deal_peers (refuses a peer with nothing), peer_place; then the parts line
    bool peer_open(const GpuCards&, std::string& err);  std::string peer_report(size_t card) const;  void peer_drop_stages();
protected:
    virtual bool peer_card(size_t card, int device, std::string& err) = 0;  // its own buffers, whatever it is dealt
    virtual uint64_t peer_room(size_t card) const = 0;        // what is left for the deal, once every peer is open
    virtual PeerSpec peer_spec() const = 0;  virtual bool peer_place(const PeerPlacement&, std::string& err) = 0;
    virtual std::string peer_main() const = 0;                // the main card's part of the parts line
};
class SplitHooks {   // built (step 6): split_open = split_cut over split_spec, then split_place; --diag-gpu-same: all on the main card
public:   bool split_open(const GpuCards&, std::string& err);  std::string split_report(size_t card) const;  // "layers 0-31, " + part
protected: virtual SplitSpec split_spec(const GpuCards&) const = 0;  virtual std::string split_part(size_t card) const = 0;
    virtual bool split_place(const SplitCut&, const std::vector<int>& devices, std::string& err) = 0; };  // per card its device
struct TensorHooks {  // tensor, reserved: tensor_check(cards) says why it can't run (empty: it can); tensor_open(cards)
};
virtual std::string gpu_report(int card) const;     // LoadedModel: the card's part after the load; peer()->peer_report by default
// ModelAdapter, in place of max_expert_cards, second_card_reads_prompts, second_card_runs_layers; every model states it
virtual program::GpuModes gpu_modes() const = 0;   // the most cards per mode, 0 = not run; peer_ada: its peer runs on sm_89
// LoadedModel: each mode's hooks, null for a mode the model does not run (peer() built; split(), tensor() in their steps)
virtual PeerHooks* peer() const { return nullptr; }  virtual SplitHooks* split() { return nullptr; }  virtual TensorHooks* tensor() { return nullptr; }
```
The flow is the same for every model: parse and narrow the cards; check `gpu_modes()` (an unsupported mode or too many cards is
refused before the load, and in `--gpu-plan`); the model's spec in, the shared deal or cut out (a part that does not fit is
refused with its numbers); `*_open`; both start lines. `GpuCards` replaced `LoadRequest::peer_devices`, `card_gbps`, `diag_split_at`.

### Who runs what at 0.5.7

| Model | peer | split | tensor |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next | yes, up to 4 cards (main + `kDmMaxX` + 1 peers): experts and prompt stages | yes, up to 8 cards (`SplitDrive::kMax`) | refused |
| DeepSeek-V4.1-Flash | yes, up to 5 cards (`kMaxSecondCards` 4): experts by layer block | refused | refused |
| Hy4-flash | yes, up to 4 cards (`kMaxSecondCards` 3): experts by layer block | refused | refused |
| Qwen3.8-27B | yes, 2 cards: one prompt stage | yes, 2 cards (#388 widens it) | refused until the tensor PR lands |
| Inkling 2 Mini | yes, 2 cards: spilled experts on the peer (step 9) | yes, 2 cards | refused |

## Today's code, and what happens to it

D = deleted. H = moves behind a hook. K = kept as a mechanic inside its mode.

### Shared

| What | Where | Auto decision | Fate |
| --- | --- | --- | --- |
| `gpu auto` (fastest as main), `choose_roles`, `plan_multi_gpu` (tier vs split vs tensor, alike only), `price_prompt_split` and its cuts | `program/gpu_select.cpp`, `program/multi_gpu_plan.cpp` | main card, roles, mode, prompt split | D (steps 1-2; `all` takes the listed order) |
| `prompt_split_draft_room`, `prompt_one_card_chunks`, `prompt_chunk_list` | `multi_gpu_plan.cpp:367-407` | none (chunk geometry) | K, moved to `prefill` |
| `spread_experts`, `tiers_holding`; the start line's "later cards idle" warning | `gpu_select.cpp:266-300`, `cli/gpu.cpp:47-75` | deal by bandwidth; idle cards | H (deal by share); idle becomes a refusal, new line |
| Peer access for every pair | `cli/gpu.cpp:76-97` | none | K, moved to `core/card_link` |
| `max_expert_cards`, `second_card_reads_prompts`, `second_card_runs_layers` | `model/adapter.hpp:359-367`, `generate/configure.cpp:27-31` | the 27B's layer stage when it does not fit | D, replaced by `gpu_modes()` |
| `--gpu-plan` JSON, `select_gpus` for the launchers | `gpu_select.cpp:360-395` | none | K, gains mode and shares |
| Server `--gpu` help, `check_gpu`, error text | `serve/config.py:578`, `serve/engine/vision.py:490-507`, `serve/server.py:131-133` | none | K, new values and texts |

### Qwen3.8-Flash-Next (`src/models/qwen4exp`)

| What | Where | Auto decision | Fate |
| --- | --- | --- | --- |
| `plan_multi_gpu` call, sets `layer_split = "auto"` | `load/qfn_run.cpp:58-79` | tier vs split | D: mode decides |
| `peer_streamed_share` (alike cards share long chunks) | `qfn_run.cpp:83`, `:98-104` | alike/unlike | D (step 2): every peer takes the routed-only chunks' rows up to `kPeerSlowRows`; the FP4 streamed share deleted in step 5 (its 5120-token cap sat below `stream_all_min` 6144, so no chunk reached it; rows past 2559 lose on a slower peer, ledger QP-124) |
| `split_alike`, `split_cap` (prompt split off on 3+ unlike cards) | `qfn_run.cpp:87-93` | alike/unlike | D |
| `layer_split` string parsing, `split_auto`, `split_same` | `qfn_run.cpp:111-170` | split on one GPU when one visible | D (the string); `split_same` stays as a diagnostic |
| Layer-split placement priced by SMs x clock | `load/layer_plan.cpp:63-170` | the cut | D (step 2): `split_cut` over dense weights + sessions per layer |
| Peer hot share 0.45 if the peer is not slower | `load/model.cpp:121-136`, `engine_local.cpp:76` | alike/slower | H: only with `gpu-share` given |
| A tier holding nothing idles | `load/token_path.cpp:36-60` | idle card | D: refused by the shared driver (step 5) |
| Prompt split pricing at start and per part | `load/prompt_split.cpp:130-330`, `serve/prompt_split.cpp:9-51` | on/off, stages, cuts, "context too short" | D (step 2): fixed cuts from the Stages shares over layers 2..n; a part longer than a chunk splits while no other slot decodes (chunk floor 4096, routed-only rows and the draft layer's room kept) |
| `--peer-exclusive` picked by RAM (`choose_auto`) | `load/resident_plan.cpp:251-273` | RAM fit, not a mode | K |
| Head split by bandwidth, GDN split if the peer is faster | `decode/verify/head_split.cpp:24-50`, `gdn_split.cpp:26` | per-op proportion | K (decision 6) |
| Overlapped split windows | `decode/verify/pipeline.cpp`, `qfn_run.cpp:173` | none | K |
| `--no-peer-prefill` | `flags.cpp:113` | | D (step 5): peer mode helps prompts; use `gpu 0` |
| `--peer-adapt-swaps`, `--peer-reserve-mib`, `--peer-exclusive` | `flags.cpp:111-112`, `cli/arguments.cpp:118` | | K: peer-mode tuning |

### The other models

| Model | What | Where | Auto decision | Fate |
| --- | --- | --- | --- | --- |
| DeepSeek, Hy4 | peer cards capped at 4 and 3, extra cards idle | `model.hpp` `kMaxSecondCards` | idle cards | D: `gpu_modes()`, refused before the load |
| DeepSeek, Hy4 | layer blocks sized to rooms, the refusal, the parts line, prompt share prices | `core/expert_tier.cpp` | none | H (step 4): `ExpertTier` is their `PeerHooks`; `deal_peers` caps the rooms at a given share |
| DeepSeek | its peer card must be sm_120, checked in its plan | `deepseek41/model.cpp` | | D (step 4): `GpuModes::peer_ada` false, refused before the load |
| 27B | `plan_split` (alike/unlike), `second_card_runs_layers`, the stage priced per read (`kPriceMargin` 0.95), the stage off under `--kv-resident` | `dense/split_plan.cpp:46-58`, `dense/stage.cpp:177`, `qwen35/adapter.cpp:319-345` | the mode, the cut, on/off per read | D (steps 2, 7): `split_spec` + `split_cut`, `stage_cuts` within the last 20 layers; `--kv-resident` is a peer refusal |
| 27B, Inkling | `--diag-split-at` | `qwen35/adapter.cpp:653`, `inkling2/adapter.cpp:58` | | D (step 2): one shared `--diag-gpu-cut L` (decision 7) |
| Inkling | `plan_cards`: `plan_multi_gpu`, "CUDA1 idles" | `inkling2/model.cpp:234-237`, `:603-664` | split vs main alone | H (steps 8, 9): `SplitHooks`, cut by share; `PeerHooks` |
| Inkling | `max_expert_cards` 1 | `inkling2/adapter.cpp:48` | | D: `gpu_modes()` |

Docs that describe the auto logic go with it: `docs/configuration/gpus.md` (rewritten), `configuration/index.md:86-87`,
`models/index.md:18-19`, the multi-card sections of `models/qwen38-flash-next.md:30`, `qwen38-27b.md:61-110` and `inkling2.md:67-87,161`,
`platforms.md:6`, `adding-a-model.md:34-39`, and the GDN-split section of `precision/qwen38-flash-next.md`.

## Migration per model

- **Flash-Next, peer (built, step 5):** `TierPeers` (`load/peers.hpp`) is its `PeerHooks`: `peer_card` prepares a card's tier,
  `peer_room` is its free memory less its prompt buffers and splits, `peer_spec` deals `Experts` (the ranked pairs the main card
  lacks, hottest first; stages over layers 2..n), `peer_place` fills each tier, `peer_main` the main card's part; one path for
  one peer or several; `peer_drop_stages` when the stages' buffers do not fit. Gate: no loss (64K 9,823 vs 9,835 tok/s). Greedy
  identical to step 3. Step 6: a stage read unlike the main card (the file's bf16 hyper-connection matrices where the main card
  widens its 8-bit copies; cuBLAS splitting the N=4 and N=1 products by the card's SMs), so cuts 37-39 gave another token 29. The
  image now holds the widened copies; `--diag-main-card-bits` alone makes the other cards run the main card's cuBLAS algorithms
  (cublasLt) and plan by its SM count: bitwise after a 50K prompt at cuts 34-42. Always on it cost 0.6% of the 64K prefill (9,810 vs
  9,869); a normal run lets each card use its own (9,864 vs 9,844 for step 5, ABBA x1), so its bar is the BF16 KLD: 0.06174 against 0.06221 on one card (100 chunks, paired -0.0005 [-0.0013, +0.0004]), top-1 91.84% against 91.80%.
- **Flash-Next, split (built, step 6):** `LayerSplit` (`load/split.hpp`) is its `SplitHooks`: `split_spec` states each layer's
  dense weights and session (`pool_bytes_by_layer`, `plan_bytes_by_layer`, `hc8_plan_bytes`, `session_bytes`) and each card's
  room, `split_place` takes `split_at` and the stages' devices, `split_part` each card's experts. The cut follows the shares
  (free VRAM: K=32; `gpu-share 3,1`: 36; `1,1`: 23); as fast as step 5 (pp16384 4,541 vs 4,502, conv 180 vs 180; v0.5.6 cannot
  split unlike cards). Bitwise against `--diag-gpu-same` at cuts 30 and 40 under `--diag-main-card-bits`, tokens and logits.
- **DeepSeek, Hy4 (built, step 4):** `ExpertTier` derives `PeerHooks` once for both: `peer_card` opens a card's plain slots and
  prompt share, `peer_room` is what they leave, `peer_spec` deals `Blocks` (one 256-rounded slot per layer, the reserve
  `--peer-reserve-mib`), `peer_place` sets the blocks and the prompt-share prices, `peer_main` the main card's part. The models
  pass `GpuCards` in `ExpertTierSpec::gpus` and return the tier from `peer()`; `ExpertTier::peer_take` holds their one prompt
  share pricing. Gate: the 15K check bitwise vs step 3; greedy identical to v0.5.6 with the tier frozen; decode no loss.
- **Qwen3.8-27B (built, step 7):** `Loaded` derives both hooks. Peer is the one prompt stage (`stage_lo` 51; the Stages shares cut
  it, 51 at the default bandwidth shares), decoding stays on the main card; a peer that would read nothing (`--kv-resident`, no prompt
  to read at the context, no room for its layers) is refused. Split cuts the bytes per layer plus cells and state by the shares (41
  at the default free-VRAM shares), each layer's cells and state on its card; `batch_captured` stays false. Both greedy identical.
- **Inkling, split (built, step 8):** `InklingModel` derives `SplitHooks`: `split_spec` is the old `plan_cards` byte accounting (per layer
  its experts, dense rows and visits' caches; per card the head, scratch and drafter), `split_place` takes the cut and devices, the cut
  follows the shares. Bitwise against one card (tokens and logit rows after a 75K prompt; no cuBLAS, so no `--diag-main-card-bits`); speed equal to step 7.
- **Inkling, peer (built, step 9):** the hottest experts that spilled from the main card go to the peer's VRAM (`PeerSpec` `Experts`; room
  = free less one prompt lane and 1 GiB; no RAM copy). Decode: `CpuExperts`' doorbell thread has `core::PeerExperts` compute those pairs
  (`launch_layout`, rows straight into the pool's) beside the pool; the merge is unchanged. Prompt: per layer the rows with a pair on the peer
  (`peer_rows_plan`, gather) go to a second `Lane`, the peer's partial sums come back (`peer_rows_add`); main drops those pairs (id -1). The
  elastic spill copies peer experts to the main card too; a layer whose spilled experts are all copied runs on the main card alone (bitwise).
- **Tensor:** every model refuses; the hooks are declared.

## Conformance

This works like `batch_check` (see [batching](batching.md)), in two runs since a mode and its reference are two loads. `basalt
--tokens-file P --max-new N --diag-gpu-check --diag-gpu-trace T` reads the prompt in one prefill call (a long one takes the model's chunks
and any prompt stage), then N greedy steps, and writes each step's logits and pick to T. The same command on the reference with
`--diag-gpu-ref T` feeds T's picks (no step on two texts) and compares: bitwise steps, greedy identity, KLD (mean, worst), top-1
(`model/gpu_check.hpp`; exit 0 only when bitwise). `tools/gpu_identity.py` compares greedy tokens through the serving engine (`--ref-engine`:
another build), Flash-Next's only check; `--logits` adds its rows' logits bitwise. `--diag-gpu-same` with `gpu-mode split` and `--diag-gpu-cut L`
puts both Flash-Next stages on CUDA0 (the split's reference). Flash-Next's bitwise runs also take `--diag-main-card-bits`. What differed was expert placement, not kernels: a decode's
missed experts ran on the CPU pool in one run and on a card in the other. `--diag-fixed-experts` fixes it: a prompt's experts all
run on the GPU and a window's verifiers see none in VRAM, so the CPU pool computes every window row. Under `--diag-main-card-bits` a split plans by the main
card (`core::plan_sms`, set in `qfn_run.cpp`: MMQ's stream-k block count, `dense_tc`'s one-tile path) and every product off the main
card runs its cuBLAS algorithm (`gemm.cu`); a normal run keeps each card's own, so the KLD bar below holds it.

| Mode, part | Reference | Bar |
| --- | --- | --- |
| peer, prompt stages (27B; Flash-Next under `--diag-main-card-bits`) | the same chunks with every layer on the main card, same placement | bitwise: KLD 0, top-1 100%, greedy identical |
| peer, prompt stages, Flash-Next normal run | one card | KLD and top-1 the same or better (100 chunks) |
| peer, experts (Flash-Next, DeepSeek, Hy4) | per expert: its rows on the main card | bitwise rows (kernel parity test per expert format) |
| peer, experts, whole model | v0.5.6 in the same mode, tier frozen (`--diag-tier-frozen`); a single card is not bitwise: its spill runs on the CPU | greedy identical to v0.5.6; decode-path KLD vs the BF16 base and top-1 the same or better |
| peer, Inkling | single card (CPU experts) | BF16 KLD (decode path) and top-1 the same or better; a peer's rows equal the main card's for the same experts (bitwise); every spilled expert copied back: bitwise vs one card |
| split, every model | single card where the model fits at the check's context; otherwise the same split with both stages on one GPU (`split_same`) | bitwise: KLD 0, greedy identical |
| tensor | single card | KLD and top-1 the same or better within the gate's noise; not bitwise (partial sums meet with one BF16 rounding) |

KLD runs use `/mnt/nvme/kld/*-bf16-base.kld` through the decode path (rule 5); traces go in a scratch directory. Step 3: the 27B stage and
Inkling's split are bitwise vs one card; DeepSeek's peer vs one card is its CPU vs GPU expert rounding (KLD 0.01; BF16 gate 0.2805, 0.2794).

## Tensor (reserved)

Community work (issue #409, azido-azide-azide): dense `--tp 2` in their fork (fork PR #36, merged there, not yet opened
upstream). It reports 1.5x decode and 1.2x prefill over the layer split on 2x RTX 5060 Ti with the 27B. Its constraints, which
`TensorHooks::tensor_check` states:
- exactly 2 cards, dense (`qwen35`) only; Flash-Next MoE tensor parallel is out of scope;
- head counts divisible by the card count, block-aligned quantized slices (`dense::tp::check`);
- rank 0 keeps the head, sampler, drafter and its KV; rank 1 its layers' cells and state; lockstep with equal exchange counts;
- partials meet in an FP32 mailbox with one BF16 rounding (`core::TpMailbox`): not bitwise against one card; it moves
  `gdn_head_split.hpp` to `core/`. When it lands, `--tp` becomes `gpu-mode tensor` and its pair comes from `gpu`.

## Decisions (Jay took each default, 2026-11-23)
1. **Default shares:** split and the peers' expert deal: free VRAM at start. Peer prompt stages: memory bandwidth (the slow card
   must not take a third of a chunk's layers).
2. **Dense peer scope:** prompt stages only. KV on a second card is what split does.
3. **Split/KV balance:** the share divides weights plus KV and state at the configured context and slots; a part that does not fit is refused.
4. **Mixed modes:** one mode per run; no split with a peer on a third card.
5. **The tensor PR:** reserved as above; taken behind `TensorHooks` once opened upstream and shown not to regress one card or the split.
6. **Per-op splits inside peer** (Flash-Next head and GDN splits): mechanics on the cards' bandwidth as before.
7. **Diagnostic cut:** one shared `--diag-gpu-cut L` replaces each model's `--diag-split-at`.
8. **Idle cards:** a peer with nothing to hold, or more cards than a mode takes, is refused with the reason or the maximum.
9. **`all` and older cards:** `all` takes every visible card in order; a card the mode cannot use is refused with its reason (peer
   takes sm_89 only in a `-DPEER_ADA=ON` build, split and tensor never; the main card must be sm_120).
10. **Community bench rows:** `basalt-bench --community` takes the user's `gpu` and `gpu-mode` and records both in the row.
11. **Open PRs:** superseded, closed once the rework lands: #212, #352, #564. Rebased onto the hooks: #388 (dense split on 3+ cards),
    #442, #473, #589 (older peers), #501, #509 (Flash-Next prompt split, peer).

## Build plan

Everything happens on the unpushed box branch `mgpu`, one agent at a time. Every step builds, passes
`CUDA_VISIBLE_DEVICES= ctest -L cpu -LE avx512` and keeps the single-card output bitwise identical to v0.5.6 (greedy, one model per step touched).

| # | Step | Gate |
| --- | --- | --- |
| 1 | Settings: parse `gpu` / `gpu-mode` / `gpu-share`, the error texts, `all`, `--gpu-plan` with mode and shares, server config keys and forwarding, bench and probe | unit tests for every error text; `serve/tests`; `--gpu-plan` JSON for each mode |
| 2 | Shared layer: `model/multi_gpu.hpp`, refusals, shares, start lines, `core/card_link`; every pricing and auto pick deleted (`plan_multi_gpu`, `price_prompt_split`, `plan_split`, the alike gates, the 27B stage's pricing, Flash-Next's `predict`, `choose_roles`); every model takes `GpuCards` and the shared cut, deal and stage cuts | builds; cut, deal and refusal tests; single-card identity on three models; two-card starts per mode |
| 3 | Conformance: `model/gpu_check.hpp`, `--diag-gpu-check`, `tools/gpu_identity.py` | the check passes on the 27B prompt stage (bitwise) before any migration |
| 4 | DeepSeek and Hy4 peer on the hooks (done) | greedy identical to v0.5.6 on two cards; two-card decode ABBA x1 no loss (DeepSeek is judged on two cards) |
| 5 | Flash-Next peer on the hooks; the FP4 streamed share measured or deleted (done) | no speed loss vs v0.5.6 with Jay's config (see below); greedy identical; prompt stages bitwise |
| 6 | Flash-Next split on the hooks (done) | bitwise vs `split_same`; it runs on 5090 + 5060 Ti, numbers reported |
| 7 | Qwen3.8-27B peer and split on the hooks (done) | both modes greedy identical to one card; 64K prefill in peer mode gains over one card, decode no worse; split's cells on the second card |
| 8 | Inkling split on the hooks (from `inksplit`) | bitwise vs single card at a context it fits (done) |
| 9 | Inkling peer: spilled experts on the peer through the `CpuExperts` shape (done) | pp16384 and decode gain vs one card at `--expert-cache` 1500 and 2400 (64K equal at 2400); KLD and top-1 same or better; a peer is refused when nothing spills |
| 10 | Docs (done): `configuration/gpus.md` rewritten, index rows, model pages, `adding-a-model.md` (a "Several GPUs" step like batching's), 0.5.7 changelog's breaking note | docs under 300 lines each; serve tests; no-GPU ctest |

After the docs land, Jay's config (his call) becomes `"gpu": "all", "gpu-mode": "peer"` for Flash-Next, DeepSeek and Hy4, `"gpu": "0"`
for the 27B and Inkling; every entry says `"gpu": "auto"` today, so the old config stops loading on `mgpu`.
