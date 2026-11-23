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
| Expert deal by share, layer cuts by share and bytes, prompt-stage cuts | `model/multi_gpu.cpp` (from `gpu_select.cpp:266-300`, `dense/split_plan.cpp`) |
| Transfers: peer access for every pair, device-to-device or host-staged copies, link probes | `core/card_link.hpp` (from `cli/gpu.cpp:76-97` and Flash-Next's `probe_pcie_h2d_gbps`) |
| Prompt stages and peer expert storage (shared already) | `core/prompt_stage.hpp`, `core/peer_experts.hpp`, `core/expert_tier.cpp` |
| The conformance check | `model/gpu_check.hpp`, `--diag-gpu-check`, `tools/gpu_identity.py` |

```cpp
// include/basalt/model/multi_gpu.hpp (sketch; GpuMode and GpuModes live in program/gpu_select.hpp, the parser needs them)
struct GpuCards {                                   // what the load request carries, main first
    struct Card { int device; program::GpuCard info; int index; std::string label; uint64_t free; };
    GpuMode mode = GpuMode::One;
    std::vector<Card> cards;
    std::vector<double> given;                      // gpu-share normalized; empty: each use's default (shares_for)
    int diag_cut = 0;                               // --diag-gpu-cut
};
struct PeerHooks {                                  // peer: the main card runs every layer
    virtual std::vector<PeerUnit> peer_units() = 0;           // experts outside the main card, hottest first, with bytes
    virtual StageSpec peer_stages() const = 0;                // layers a stage may read, their bytes, a chunk's rows
    virtual Result peer_open(const GpuCards&, const PeerPlacement&) = 0;   // its experts and stages on each card
};
struct SplitHooks {                                 // split: layers divided, KV with its layers
    virtual SplitSpec split_spec(const GpuCards&) const = 0;  // per layer: weights + KV/state; per card: fixed bytes
    virtual Result split_open(const GpuCards&, const std::vector<LayerRange>&) = 0;
};
struct TensorHooks {                                // tensor: reserved
    virtual std::string tensor_check(int cards) const = 0;    // why it can't run; empty = it can
    virtual Result tensor_open(const GpuCards&) = 0;
};
virtual std::string gpu_report(int card) const;     // LoadedModel, whatever the mode: the card's part after the load
// ModelAdapter, in place of max_expert_cards, second_card_reads_prompts, second_card_runs_layers; every model states it
virtual program::GpuModes gpu_modes() const = 0;   // the most cards per mode, 0 = not run
// LoadedModel: each mode's hooks, null for a mode the model does not run
virtual PeerHooks* peer() { return nullptr; }  virtual SplitHooks* split() { return nullptr; }  virtual TensorHooks* tensor() { return nullptr; }
```

The flow is the same for every model: parse and narrow the cards; check `gpu_modes()` (an unsupported mode or too many cards is
refused before the load, and in `--gpu-plan`); the model's spec in, the shared deal or cut out (a part that does not fit is
refused with its numbers); `*_open`; both start lines. `GpuCards` replaced `LoadRequest::peer_devices`, `card_gbps` and `diag_split_at` (step 2).

### Who runs what at 0.5.7

| Model | peer | split | tensor |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next | yes, up to 4 cards (main + `kDmMaxX` + 1 peers): experts and prompt stages | yes, up to 8 cards (`SplitDrive::kMax`) | refused |
| DeepSeek-V4.1-Flash | yes, up to 5 cards (`kMaxSecondCards` 4): experts by layer block | refused | refused |
| Hy4-flash | yes, up to 4 cards (`kMaxSecondCards` 3): experts by layer block | refused | refused |
| Qwen3.8-27B | yes, 2 cards: one prompt stage | yes, 2 cards (#388 widens it) | refused until the tensor PR lands |
| Inkling 2 Mini | yes, 2 cards: spilled experts on the peer (new) | yes, 2 cards | refused |

## Today's code, and what happens to it

D = deleted. H = moves behind a hook. K = kept as a mechanic inside its mode.

### Shared

| What | Where | Auto decision | Fate |
| --- | --- | --- | --- |
| `gpu auto`: every sm_120 card, fastest as main | `program/gpu_select.cpp:115-130`, `:158-215` | main-card pick by bandwidth | D (`all` takes the listed order) |
| `choose_roles`, `RoleInputs`, `GpuRole` | `gpu_select.cpp:250-264`, `gpu_select.hpp:59-73` | role per card from the model's two booleans | D |
| `plan_multi_gpu`, `MgModel`, `MgPlan`, `cards_alike`, the 5% margin | `program/multi_gpu_plan.cpp:14-180`, `.hpp:16-75` | expert tier vs layer split vs tensor, alike only | D |
| `price_prompt_split`, `prompt_split_windows`, `prompt_split_order`, `PsCard` | `multi_gpu_plan.cpp:183-366` | prompt split on or off, its cuts, stage count | D |
| `prompt_split_draft_room`, `prompt_one_card_chunks`, `prompt_chunk_list` | `multi_gpu_plan.cpp:367-407` | none (chunk geometry) | K, moved to `prefill` |
| `spread_experts`, `tiers_holding` | `gpu_select.cpp:266-300` | deal weighted by bandwidth; idle a tier with nothing | H (deal by share); the idle tier becomes a refusal |
| Start line `gpus: ...` and the "later cards idle" warning | `cli/gpu.cpp:47-75` | idle cards beyond the model's cap | D (refused instead), new line |
| Peer access for every pair | `cli/gpu.cpp:76-97` | none | K, moved to `core/card_link` |
| `max_expert_cards`, `second_card_reads_prompts`, `second_card_runs_layers` | `model/adapter.hpp:359-367`, `generate/configure.cpp:27-31` | the 27B's layer stage when it does not fit | D, replaced by `gpu_modes()` |
| `--gpu-plan` JSON, `select_gpus` for the launchers | `gpu_select.cpp:360-395` | none | K, gains mode and shares |
| Server `--gpu` help, `check_gpu`, error text | `serve/config.py:578`, `serve/engine/vision.py:490-507`, `serve/server.py:131-133` | none | K, new values and texts |

### Qwen3.8-Flash-Next (`src/models/qwen4exp`)

| What | Where | Auto decision | Fate |
| --- | --- | --- | --- |
| `plan_multi_gpu` call, sets `layer_split = "auto"` | `load/qfn_run.cpp:58-79` | tier vs split | D: mode decides |
| `peer_streamed_share` (alike cards share long chunks) | `qfn_run.cpp:83`, `:98-104` | alike/unlike | D (step 2): every peer takes the routed-only chunks' rows up to `kPeerSlowRows`, the unlike path Jay's cards run; the FP4 streamed share (`T_fp4`) is left unreached for step 5 to measure or delete |
| `split_alike`, `split_cap` (prompt split off on 3+ unlike cards) | `qfn_run.cpp:87-93` | alike/unlike | D |
| `layer_split` string parsing, `split_auto`, `split_same` | `qfn_run.cpp:111-170` | split on one GPU when one visible | D (the string); `split_same` stays as a diagnostic |
| Layer-split placement priced by SMs x clock | `load/layer_plan.cpp:63-170` | the cut | D (step 2): `split_cut` over dense weights + sessions per layer |
| Peer hot share 0.45 if the peer is not slower | `load/model.cpp:121-136`, `engine_local.cpp:76` | alike/slower | H: only with `gpu-share` given |
| A tier holding nothing idles | `load/token_path.cpp:36-60` | idle card | D (step 2): refused |
| Prompt split pricing at start and per part | `load/prompt_split.cpp:130-330`, `serve/prompt_split.cpp:9-51` | on/off, stages, cuts, "context too short" | D (step 2): fixed cuts from the Stages shares over layers 2..n; a part longer than a chunk splits while no other slot decodes (chunk floor 4096, routed-only rows and the draft layer's room kept) |
| `--peer-exclusive` picked by RAM (`choose_auto`) | `load/resident_plan.cpp:251-273` | RAM fit, not a mode | K |
| Head split by bandwidth, GDN split if the peer is faster | `decode/verify/head_split.cpp:24-50`, `gdn_split.cpp:26` | per-op proportion | K (open point 6) |
| Overlapped split windows | `decode/verify/pipeline.cpp`, `qfn_run.cpp:173` | none | K |
| `--no-peer-prefill` | `flags.cpp:113` | | D: peer mode helps prompts; use `gpu 0` |
| `--peer-adapt-swaps`, `--peer-reserve-mib`, `--peer-exclusive` | `flags.cpp:111-112`, `cli/arguments.cpp:118` | | K: peer-mode tuning |

### The other models

| Model | What | Where | Auto decision | Fate |
| --- | --- | --- | --- | --- |
| DeepSeek | peer cards capped at 4, extra cards idle with a warning | `deepseek41/model.cpp:256-272`, `model.hpp:44` | idle cards | D: refused before the load |
| DeepSeek, Hy4 | layer blocks sized to rooms, prompt share rates | `core/expert_tier.cpp:107-140`, `:288-325`, `core/tier_policy.hpp:130` | none | H: blocks by share, rooms capped at the share (step 2; the default is the rooms); a peer with no home refused |
| Hy4 | peer cards capped at 3 | `hy4flash/model.cpp:223-228`, `model.hpp:28` | idle cards | D: refused |
| 27B | `plan_split`: no split if it fits; otherwise alike = half the bytes, unlike = the faster card fullest | `dense/split_plan.cpp:46-58`, `qwen35/adapter.cpp:319-330` | alike/unlike, main card alone | D (step 2): `dense::split_spec` + `split_cut` |
| 27B | `second_card_runs_layers` | `qwen35/adapter.cpp:630-645` | split vs prompt stage | D |
| 27B | prompt stage priced per read, `kPriceMargin` 0.95 | `dense/stage.cpp:177`, `:316-365`, `:680` | on/off and cut per read | D (step 2): `stage_cuts` within the last 20 layers, the main card keeps the last chunk; the start's timing is gone |
| 27B | stage off under `--kv-resident` | `qwen35/adapter.cpp:339-340` | | K, now a peer-mode refusal |
| 27B, Inkling | `--diag-split-at` | `qwen35/adapter.cpp:653`, `inkling2/adapter.cpp:58` | | D (step 2): one shared `--diag-gpu-cut L` (open point 7) |
| Inkling | `plan_cards`: `plan_multi_gpu`, "CUDA1 idles" | `inkling2/model.cpp:234-237`, `:603-664` | split vs main alone | H: split cut by share (step 2); peer is new |
| Inkling | `max_expert_cards` 1 | `inkling2/adapter.cpp:48` | | D: `gpu_modes()` |

Docs that describe the auto logic go with it: `docs/configuration/gpus.md` (rewritten), `configuration/index.md:86-87`,
`models/index.md:18-19`, the multi-card sections of `models/qwen38-flash-next.md:30`, `qwen38-27b.md:61-110` and `inkling2.md:67-87,161`,
`platforms.md:6`, `adding-a-model.md:34-39`, and the GDN-split section of `precision/qwen38-flash-next.md`.

## Migration per model

- **Flash-Next, peer** (today's expert tier): the main card fills first; the peers' deal follows the shares, and the default (free
  VRAM) gives one peer exactly what it holds today. Prompt stages take a fixed cut from the bandwidth share, no pricing.
  `PeerHooks` wraps `resident_plan.cpp`'s `peer.open` and the stage set-up of `load/prompt_split.cpp`.
- **Flash-Next, split** (today's auto layer split, now chosen): `split_spec` reports the per-layer bytes the placement already
  computes (`pool_bytes_by_layer`, `plan_bytes_by_layer`, `hc8_plan_bytes`, `session_bytes`, `layer_plan.cpp:84-100`); the
  shared cut replaces `predict`.
- **DeepSeek, Hy4:** peer only. `ExpertTier` already owns the peers; `split_by_bytes` takes the shares as weights on the rooms.
  `kMaxSecondCards` goes into `gpu_modes()`.
- **Qwen3.8-27B:** peer is the one prompt stage, its cut from the share in place of `price()`; decoding stays on the main card.
  Split is the layer stage, its cut from `plan_split` without the alike/unlike branches. With `gpu 0` a model that does not
  fit is refused and pointed at `gpu 0,1` with `gpu-mode split`. The split's batch path (`batch_captured` false) stays.
- **Inkling, split:** `plan_cards` keeps its byte accounting (`mm.layer_bytes`, `layer_caches`, `split_handoffs`) as
  `split_spec`; the cut comes from the share. The placement code (branch `inksplit`) stays separable and becomes the hook. A
  split card still holds every expert of its layers.
- **Inkling, peer (new):** spilled experts move from RAM to the peer's VRAM. `CpuExperts` (`inkling2/cpu_experts.hpp`, #536)
  already sends a layer's rows and choices off the card after the router and merges them before the combine. Peer mode keeps
  that shape: `start` copies rows and ids to the peer (device to device through `card_link`), the peer computes its pairs with
  the main card's expert kernels, `finish` adds them into `out`. Experts that fit neither card stay on the CPU path. Gate: it
  helps prefill and does not hurt decode.
- **Tensor:** every model refuses; the hooks are declared.

## Conformance

This works like `batch_check` (see [batching](batching.md)), in two runs since a mode and its reference are two loads. `basalt
--tokens-file P --max-new N --diag-gpu-check --diag-gpu-trace T` reads the prompt in one prefill call (a long one takes the model's chunks
and any prompt stage), then N greedy steps, and writes each step's logits and pick to T. The same command on the reference with
`--diag-gpu-ref T` compares them: bitwise steps, greedy identity, KLD, top-1 (`model/gpu_check.hpp`; exit 0 only when bitwise).
`tools/gpu_identity.py` compares greedy tokens through the serving engine (`--ref-engine`: another build), Flash-Next's only check.
`--diag-gpu-same` with `gpu-mode split` and `--diag-gpu-cut L` puts both Flash-Next stages on CUDA0, its split's reference; it stalls
in the first prompt window (0 layers served) while the real split runs: step 6 fixes it.

| Mode, part | Reference | Bar |
| --- | --- | --- |
| peer, prompt stages (Flash-Next, 27B) | the same chunks with every layer on the main card, same placement | bitwise: KLD 0, top-1 100%, greedy identical |
| peer, experts (Flash-Next, DeepSeek, Hy4) | per expert: its rows on the main card | bitwise rows (kernel parity test per expert format) |
| peer, experts, whole model | v0.5.6 in the same mode, tier frozen (`--diag-tier-frozen`); a single card is not bitwise: its spill runs on the CPU | greedy identical to v0.5.6; decode-path KLD vs the BF16 base and top-1 the same or better |
| peer, Inkling | single card (CPU experts) | KLD and top-1 the same or better; peer rows bitwise equal to main-card rows |
| split, every model | single card where the model fits at the check's context; otherwise the same split with both stages on one GPU (`split_same`) | bitwise: KLD 0, greedy identical |
| tensor | single card | KLD and top-1 the same or better within the gate's noise; not bitwise (partial sums meet with one BF16 rounding) |

KLD runs use `/mnt/nvme/kld/*-bf16-base.kld` through the decode path (rule 5); traces go in a scratch directory. Step 3: the 27B stage
and Inkling's split are bitwise vs one card; DeepSeek's experts on a card vs the CPU differ at KLD 9+ (no peer), so it is held to v0.5.6.

## Tensor (reserved)

Community work (issue #409, azido-azide-azide): dense `--tp 2` in their fork (fork PR #36, merged there, not yet opened
upstream). It reports 1.5x decode and 1.2x prefill over the layer split on 2x RTX 5060 Ti with the 27B. Its constraints, which
`TensorHooks::tensor_check` states:

- exactly 2 cards, dense (`qwen35`) only; Flash-Next MoE tensor parallel is out of scope;
- head counts divisible by the card count, block-aligned quantized slices (`dense::tp::check`);
- rank 0 keeps the head, sampler, drafter and its KV; rank 1 its layers' cells and state; lockstep with equal exchange counts;
- partials meet in an FP32 mailbox with one BF16 rounding (`core::TpMailbox`): not bitwise against one card;
- it moves `gdn_head_split.hpp` to `core/`. When it lands, `--tp` becomes `gpu-mode tensor` and its pair comes from `gpu`.

## Open points

1. **Default shares. DEFAULT, Jay to confirm:** split and the peers' expert deal: free VRAM at start. Peer prompt stages:
   memory bandwidth, since a free-VRAM cut hands a quarter-speed RTX 5060 Ti a third of each chunk's layers, makes it the
   pipeline's slow stage and would fail the Flash-Next gate.
2. **Dense peer scope. DEFAULT, Jay to confirm:** prompt stages only. KV on a second card is what split does.
3. **Split/KV balance. DEFAULT, Jay to confirm:** the share divides weights plus KV and state at the configured context and
   slots (as `plan_split` counts them). A card whose part does not fit is refused.
4. **Mixed modes. DEFAULT, Jay to confirm:** one mode per run; no split with a peer on a third card (Flash-Next already refuses
   a peer with a layer split, `cli/options.cpp:128`).
5. **The tensor PR's constraints. DEFAULT, Jay to confirm:** reserve as above; take the PR behind `TensorHooks` once opened
   upstream and shown not to regress one card or the layer split.
6. **Per-op splits inside peer** (Flash-Next head and GDN splits). **DEFAULT, Jay to confirm:** kept as mechanics on the cards'
   bandwidth as today; moving them to the shares would change Jay's decode.
7. **Diagnostic cut. DEFAULT, Jay to confirm:** one shared `--diag-gpu-cut L` replaces each model's `--diag-split-at`.
8. **Idle cards. DEFAULT, Jay to confirm:** a peer with nothing to hold, or more cards than a mode takes, is refused with the reason or the maximum (today they idle).
9. **`all` and older cards. DEFAULT, Jay to confirm:** `all` takes every visible card in order and skips none; a card the mode
    cannot use is refused with its reason (peer takes sm_89 only in a `-DPEER_ADA=ON` build, split and tensor never; the main
    card must be sm_120), so a machine with an older card names the cards it wants.
10. **Community bench rows. DEFAULT, Jay to confirm:** `basalt-bench --community` takes the user's `gpu` and `gpu-mode` and
    records both in the row (it forces `auto` today).
11. **Open PRs this touches. DEFAULT, Jay to confirm:** superseded, closed once the rework lands: #212, #352, #564. Rebased
    onto the hooks: #388 (dense split on 3+ cards), #442, #473, #589 (older peers), #501, #509 (Flash-Next prompt split, peer).

## Build plan

Everything happens on the unpushed box branch `mgpu`, one agent at a time. Every step builds, passes
`CUDA_VISIBLE_DEVICES= ctest -L cpu -LE avx512` and keeps the single-card output bitwise identical to v0.5.6 (greedy, one model per step touched).

| # | Step | Gate |
| --- | --- | --- |
| 1 | Settings: parse `gpu` / `gpu-mode` / `gpu-share`, the error texts, `all`, `--gpu-plan` with mode and shares, server config keys and forwarding, bench and probe | unit tests for every error text; `serve/tests`; `--gpu-plan` JSON for each mode |
| 2 | Shared layer: `model/multi_gpu.hpp`, refusals, shares, start lines, `core/card_link`; every pricing and auto pick deleted (`plan_multi_gpu`, `price_prompt_split`, `plan_split`, the alike gates, the 27B stage's pricing, Flash-Next's `predict`, `choose_roles`); every model takes `GpuCards` and the shared cut, deal and stage cuts | builds; cut, deal and refusal tests; single-card identity on three models; two-card starts per mode |
| 3 | Conformance: `model/gpu_check.hpp`, `--diag-gpu-check`, `tools/gpu_identity.py` | the check passes on the 27B prompt stage (bitwise) before any migration |
| 4 | DeepSeek and Hy4 peer on the hooks | greedy identical to v0.5.6 on two cards; two-card decode ABBA x1 no loss (DeepSeek is judged on two cards) |
| 5 | Flash-Next peer on the hooks; the FP4 streamed share measured or deleted | no speed loss vs v0.5.6 with Jay's config (see below); greedy identical; prompt stages bitwise |
| 6 | Flash-Next split on the hooks; `split_same` set by `--diag-gpu-check` | bitwise vs `split_same`; it runs on 5090 + 5060 Ti, numbers reported |
| 7 | Qwen3.8-27B peer and split on the hooks; `second_card_runs_layers` gone | both modes bitwise vs single card; 64K prefill in peer mode no loss vs v0.5.6 on two cards |
| 8 | Inkling split on the hooks (from `inksplit`) | bitwise vs single card at a context it fits |
| 9 | Inkling peer: spilled experts on the peer through the `CpuExperts` shape | prefill gains and decode doesn't lose vs single card at 524,288 fp8; KLD and top-1 same or better; peer rows bitwise |
| 10 | Docs: `configuration/gpus.md` rewritten, index rows, model pages, `adding-a-model.md` (a "Several GPUs" step like batching's), 0.5.7 changelog's breaking note | docs under 300 lines each; serve tests; no-GPU ctest |

The Flash-Next gate (step 5): the "Qwen 3.8 FN" entry of `/mnt/nvme/models/basalt_config.json` (read only: `kv int8`,
`parallel 4`, `max-context 262144`, `pool-workers 8`), v0.5.6 with `gpu auto` against `mgpu` with `gpu all` and `gpu-mode peer`,
all else equal. `bench.sh` x2 (the second counts), `~/tools/prefill-any.sh 64k` and a 16K prompt (the fixed stage cut), pinned
ABBA x1, 400 W + 180 W. Any loss outside noise stops the step.

After the docs land, Jay's config becomes `"gpu": "all", "gpu-mode": "peer"` for Flash-Next, DeepSeek and Hy4, and `"gpu": "0"`
for the 27B and Inkling. That edit is Jay's call. Every entry says `"gpu": "auto"` today, so the old config stops loading on `mgpu`.
