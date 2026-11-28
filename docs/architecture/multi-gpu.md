# Several GPUs

How a run uses more than one card, for engine contributors. The user picks the cards and the mode; shared code places the work
and reports it; each model fills a small contract per mode. The engine never prices or picks a mode, and a card is never left
idle without the user having named it. Batching is built the same way (see [batching](batching.md)). The user-facing side is
[configuration/gpus.md](../configuration/gpus.md).

## The settings

`gpu` says which cards, `gpu-mode` what they do together, `gpu-share` in what proportions. The same three names are used by the
flags (`--gpu`, `--gpu-mode`, `--gpu-share`) of `basalt`, `basalt-bench` and `basalt-probe`, and by the entry options of the
server config (`serve/config.py` `MODEL_OPTIONS`), so each model in a config sets its own.

| Setting | Values | Default | Meaning |
| --- | --- | --- | --- |
| `gpu` | `0`, `0,1`, `1,0,2`, ..., `all` | GPU 0 alone, with one warning that names the other visible cards | The cards, numbered as `nvidia-smi` numbers them. The first card listed is the main card. `all` is every visible card in order, so GPU 0 is the main card. |
| `gpu-mode` | `peer`, `split`, `tensor` | none | Required when `gpu` names two or more cards; an error with one card. |
| `gpu-share` | positive numbers, one per card, such as `3,1` | see [shares](#shares) | Relative shares in the order of `gpu`; an error with one card. |

- **peer:** the main card runs every layer and the others help: experts and prompt stages in a mixture-of-experts model, prompt
  stages in a dense one.
- **split:** the layers are divided between the cards in order; each layer's KV and state stay on its card; the cuts follow the
  shares.
- **tensor:** every layer sharded across the cards. Reserved: the name parses and `GpuModes::tensor` exists, but no model runs it
  and there is no hook contract for it yet.

One mode per run: a split with a peer on a third card does not exist. Inside peer mode the models keep their per-operation splits
(Flash-Next's head and GDN splits, which follow the cards' bandwidth) and overlapped split windows; those are mechanics of the
mode, not part of the shared layer.

### Parsing and refusals

Parsing is `program/gpu_select.cpp`, before any CUDA call. The server asks the engine for the same checks with `--gpu-plan` (one
JSON line with the cards, mode and shares; it takes the model, so the model's modes are checked too), so a config error stops the
start before any load. `choose_gpus` checks the cards and `check_gpu_modes` the model's `gpu_modes()`; the shared deal and cut
(`model/multi_gpu.cpp`) refuse what needs the shares. Texts, as printed (`gpu` is the setting as written, the model its adapter's
`model_name()`):

```
gpu "auto": expected a card number like 0, a list like 0,1 (the first is the main card), or all
gpu 0,1: GPU 0 is listed twice
gpu-mode "auto": expected peer, split or tensor
gpu-mode peer: gpu names one card (gpu "0"); a mode needs two or more
gpu-share: gpu names one card (gpu "0"); shares need two or more
gpu-mode split: Hy4-flash does not run it; it runs peer
gpu-mode tensor: Qwen3.8-27B does not run it (not built); it runs peer or split
gpu-mode peer: Hy4-flash takes at most 4 cards (the main card and 3 peers); gpu names 5
gpu-mode peer: GPU 1 (RTX 5060 Ti) would hold nothing: ...; use gpu 0
gpu-mode split: GPU 1 (RTX 5060 Ti) would hold nothing: its share ...
gpu 1,0: GPU 1 (RTX 4060 Ti) is sm_89; the main card must be sm_120
```

- The main card must be sm_120. Split and tensor need every card sm_120. Peer also takes an sm_89 card in a `-DPEER_ADA=ON` build,
  and only for a model whose `GpuModes::peer_ada` is true (its peer kernels are among those that build compiles for sm_89).
- A part that does not fit its card is refused with its numbers (what it needs, what is free) and the settings to lower
  (`gpu-share`, `max-context`, `parallel`). A peer that would hold nothing is refused, with `use gpu 0`.
- A visible card `gpu` does not name is no error: when `gpu` is unset every model warns once at start, in the CLI, the server and
  the bench: `GPU 1 (RTX 5060 Ti) unused; to use more cards set gpu (0,1 or all) and gpu-mode`.
- In the config `gpu` and `gpu-share` may be a number, a list or a string; the server forwards `--gpu-mode` and `--gpu-share`
  beside a `--gpu` narrowed to the chosen cards (`CUDA_VISIBLE_DEVICES` is set to them before any CUDA call, so inside the
  engine the main card is CUDA device 0).

### The start line

One line before the load and one after, the same for every model:

```
gpus: peer; <name> (nvidia-smi 0) main, share <s0>; <name> (nvidia-smi 1) peer, share <s1> (bandwidth)
gpus: <name> dense, KV, <n> experts (<size>); <name> <n> experts (<size>), prompt layers <a>-<b>
```

The tag says where the shares came from (`gpu-share`, `free VRAM`, `bandwidth`, or `equal` when nothing is known); peer shows the
prompt stages' shares, split the layers'. Each card's part of the second line is the model's `gpu_report(card)`; the shared layer
prints it. One card prints `gpus: <name> (nvidia-smi 0)`.

## Shares

Shares are normalized to sum to 1 and stay fixed for the run. `shares_for(cards, ShareUse)` returns them with their origin. A
given `gpu-share` is used for every use; otherwise the default depends on what is divided:

| Mode, part | What the share divides | Default without `gpu-share` |
| --- | --- | --- |
| split | each card's bytes: its layers' weights plus their KV and state at the context and slots; the cut is the layer boundary nearest the shares | free VRAM at start |
| peer, experts | the experts outside the main card, dealt hottest first to the peers by share; the main card always fills first | free VRAM of the peers |
| peer, prompt stages | the layers of a long prompt's chunk: the main card reads the first part, each stage card the next part by share | memory bandwidth (a slow card must not take a large part of a chunk's layers) |
| peer, the main card's hot share | only when `gpu-share` is given (Flash-Next): the hottest experts are interleaved between the main card and a peer by share | off: the main card keeps the hottest |
| tensor | heads per card (reserved) | equal |

## The shared layer

| Piece | Where |
| --- | --- |
| Parsing `gpu`, `gpu-mode`, `gpu-share`; `GpuMode`, `GpuModes` and their refusals; the card table; `CUDA_VISIBLE_DEVICES` narrowing; the `--gpu-plan` JSON | `include/basalt/program/gpu_select.hpp`, `src/program/gpu_select.cpp` |
| `GpuCards` (the cards main first, the mode, the shares, the diagnostic cut), `shares_for`, `split_cut`, `deal_experts`, `stage_cuts`, the start lines | `include/basalt/model/multi_gpu.hpp`, `src/model/multi_gpu.cpp` |
| Peer's driver | `include/basalt/model/peer_hooks.hpp` |
| Split's driver | `include/basalt/model/split_hooks.hpp`, `src/model/split_hooks.cpp` |
| Transfers: peer access for every pair, device-to-device or host-staged copies, link probes | `include/basalt/core/card_link.hpp` |
| Prompt stages and peer expert storage | `core/prompt_stage.hpp`, `core/peer_experts.hpp`, `core/expert_tier.cpp` |
| The conformance check | `include/basalt/model/gpu_check.hpp`, `--diag-gpu-check`, `tools/gpu_identity.py` |

The flow is the same for every model: parse and narrow the cards; check `gpu_modes()` (an unsupported mode, or more cards than the
mode takes, is refused before the load and in `--gpu-plan`); the model states what it can divide, the shared layer deals or cuts
it (a part that does not fit is refused with its numbers); the model's `*_open` places its part; both start lines print.

### What a model declares

```cpp
// ModelAdapter (model/adapter.hpp): the most cards each mode takes, 0 = the model does not run it
virtual program::GpuModes gpu_modes() const = 0;   // {peer, split, tensor, peer_ada}
// LoadedModel: the hooks of each mode a model runs; null for a mode it does not
virtual PeerHooks* peer() const { return nullptr; }
// the card's part of the start line after the load; by default the peer hooks' report
virtual std::string gpu_report(int card) const;
```

A model that runs split derives `SplitHooks` and calls `split_open` itself during its load; `gpu_modes()` is what makes the
shared layer refuse the modes the model lacks, so every model states it, including one that runs a single card only (all zeros).

### Peer

`PeerHooks` (`model/peer_hooks.hpp`) is a driver over the hooks a model writes. `peer_open` runs `peer_card` on every peer, reads
each `peer_room`, takes `peer_spec`, deals it with `deal_peers` (which refuses a peer that would hold nothing), and calls
`peer_place` with the result; `peer_report(card)` is the card's part for the start line.

| Hook | Contract |
| --- | --- |
| `peer_card(card, device, err)` | Open the peer's own buffers, whatever it is dealt (a prompt share's staging, ring buffers). |
| `peer_room(card)` | The bytes left for the deal once every peer is open. |
| `peer_spec()` | What is divided (`PeerSpec`): `deal` is `None`, `Blocks` or `Experts`; `unit` the bytes per layer (`Blocks`) or per expert hottest first (`Experts`); `block_most`; `reserve` (left free past what is dealt); `main_holds_all`; and `stage_layers`/`stage_lo` for prompt stages (0: none). |
| `peer_place(placement, err)` | Take the `PeerPlacement`: the layer range and expert count per card (blocks), each expert's owner and each card's held count and bytes (experts), the stage cuts. |
| `peer_main()` | The main card's part of the start line. |

`Blocks` gives each peer a contiguous block of layers and the same number of experts in every layer of it, sized to its room
(`core::policy::split_by_bytes`), capped at its share when `gpu-share` is given. `Experts` deals one by one, hottest first, each to
the open peer with the fewest for its share, a peer closing at the first expert that no longer fits (`deal_experts`). Stages are
cut by `stage_cuts` over `[stage_lo, stage_layers)` from the bandwidth (or given) shares. `peer_drop_stages()` clears the stages
when their buffers turn out not to fit.

### Split

`SplitHooks` (`model/split_hooks.hpp`): `split_open` cuts the layers with `split_cut` over the model's `split_spec`, refuses a
card that would hold nothing or whose part does not fit, then calls `split_place`.

| Hook | Contract |
| --- | --- |
| `split_spec(cards)` | `SplitSpec`: per layer its weights plus its KV and state at the context and slots; per card what it holds whatever its layers (head, drafter, scratch); per card the bytes it has past its reserves. |
| `split_place(cut, devices, err)` | Card `i` runs layers `[cut.first[i], cut.first[i+1])` on `devices[i]`. |
| `split_part(card)` | The card's own part beside its layers for the start line; empty: nothing more to say. |

`split_report(card)` prefixes `layers a-b` to the model's part. The cut is the layer boundary where the cumulative bytes come
nearest the shares of every card's bytes.

### Tensor (reserved)

No model runs tensor and no hook class exists. The intended shape is the same as the others: the shared layer parses `tensor`,
checks `GpuModes::tensor`, and a model-side hook states why it cannot run (or opens it).

## Which model runs what

The most cards each mode takes, from each adapter's `gpu_modes()`. A model also refuses a mode with its reason.

| Model | peer | split | tensor |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next | 4 cards: experts dealt one by one, and prompt stages | 8 cards | refused |
| DeepSeek-V4.1-Flash | 5 cards: experts by layer block | refused | refused |
| Hy4-flash | 4 cards: experts by layer block | refused | refused |
| ERNIE-5.5-Flash | 4 cards: experts by layer block | refused | refused |
| Qwen3.8-27B | 2 cards: one prompt stage; decode stays on the main card | 2 cards | refused |
| Inkling 2 Mini | 2 cards: the experts that spilled from the main card, held on the peer | 2 cards | refused |
| Gemma 4 (single card only) | refused | refused | refused |

`peer_ada` is set for Flash-Next only. The block-deal models (DeepSeek, Hy4, ERNIE) share `core::ExpertTier`, which is their
`PeerHooks`. Flash-Next's peer is `TierPeers` and its split `LayerSplit` (`src/models/qwen4exp/load`); the 27B's `Loaded`
and Inkling's `InklingModel` derive both hooks.

Model-specific peer behaviour worth knowing when extending:

- **Flash-Next** takes its prompt stage cuts over layers 2..n and splits a part into stages only while no other slot decodes; a
  peer takes a prompt chunk's routed-only rows up to `kPeerSlowRows`. `--diag-main-card-bits` makes a stage on another card compute
  the main card's bits (its cuBLAS algorithms and the main card's SM count in kernel planning) so a peer or split is comparable
  bitwise against one card.
- **27B** peer: the prompt stage reads within the last layers of the model (`stage_lo`); a peer that would read nothing
  (`--kv-resident`, no prompt long enough at the context, no room for its layers) is refused. Split keeps each layer's cells and
  state on its card.
- **Inkling** peer: spilled experts live on the peer's VRAM only (no RAM copy); `core::PeerExperts` computes those pairs beside
  the CPU pool in decode, and a prompt layer's rows with a pair on the peer go to a second lane whose partial sums come back.

## Conformance

`model/gpu_check.hpp` checks a mode against a reference through the session interface only, so it names no model, in two runs
since a mode and its reference are two loads. `basalt --tokens-file P --max-new N --diag-gpu-check --diag-gpu-trace T` reads the
prompt in one prefill call (a long one takes the model's chunks and any prompt stage), then N greedy steps, and writes each step's
logits and pick to T. The same command on the reference with `--diag-gpu-ref T` feeds T's picks (no step on two texts) and
compares: bitwise steps, greedy identity, KLD (mean, worst), top-1; exit 0 only when bitwise.

- `tools/gpu_identity.py` compares greedy tokens through the serving engine (`--ref-engine`: another build) for each fixture
  prompt; `--logits` also compares every accepted row's logits bitwise (Flash-Next). It is the only check Flash-Next's serving
  engine offers.
- `--diag-gpu-same` with `gpu-mode split` and `--diag-gpu-cut L` puts every part of a split on the main card: the split's
  bitwise reference when the model does not fit one card.
- `--diag-fixed-experts` (Flash-Next) fixes expert placement so runs agree: a prompt's experts all run on the GPU and a window's
  verifiers see none in VRAM. Without it a decode's missed experts run on the CPU pool in one run and on a card in the other.
- `--diag-tier-frozen` (the block-deal models) freezes the peers' expert tier for comparisons.

| Mode, part | Reference | Bar |
| --- | --- | --- |
| peer, prompt stages (27B; Flash-Next under `--diag-main-card-bits`) | the same chunks with every layer on the main card, same placement | bitwise: KLD 0, top-1 100%, greedy identical |
| peer, prompt stages, Flash-Next normal run | one card | KLD and top-1 the same or better |
| peer, experts | per expert: its rows on the main card | bitwise rows (a kernel parity test per expert format) |
| peer, experts, whole model | the same mode with the tier frozen; a single card is not bitwise, its spill runs on the CPU | greedy identical; decode-path KLD against the BF16 base and top-1 the same or better |
| peer, Inkling | one card (CPU experts) | KLD and top-1 the same or better; a peer's rows equal the main card's for the same experts; every spilled expert copied back: bitwise |
| split | one card where the model fits at the check's context; otherwise the same split with every part on one GPU (`--diag-gpu-same`) | bitwise: KLD 0, greedy identical |
| tensor (reserved) | one card | KLD and top-1 the same or better; not bitwise if partial sums meet with a different rounding |

KLD is [wikitext KLD](../precision/quality.md#wikitext-kld) through the decode path; the preferred quality check is the
[quality suite](../precision/quality-suite.md). A peer's expert sum can reorder against one card's, which shows as rounding-level
KLD, not a bug.

## Adding a mode to a model

1. State `gpu_modes()` (the most cards per mode, 0 for a mode it lacks; `peer_ada` only if the peer kernels are built for sm_89).
2. Derive the hooks of the mode and write the contract above; keep what a card computes in the model, leave the deal, cut,
   shares, refusals and start line to the shared layer. If another model already deals the same thing, reuse its driver input
   (`Blocks` or `Experts`) rather than a new deal.
3. Override `gpu_report` when the model has a split or extra words for a peer.
4. Check it against its reference in the table above with `--diag-gpu-check`, and with `tools/gpu_identity.py` through the server.
