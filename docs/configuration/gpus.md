# GPUs

Three settings say how a run uses its cards. The engine never picks a mode or a cut for you: `gpu-mode` decides, `gpu-share` sets
the proportions, and a mode the model cannot run, or a card that would hold nothing, is refused with the reason before the load.

They mean the same in the server (`"gpu"`, `"gpu-mode"` and `"gpu-share"` in a model entry, or `--gpu`, `--gpu-mode`, `--gpu-share`),
in `basalt` itself, in `basalt-bench` and in `basalt-probe`.

| Setting | Values | Default |
| --- | --- | --- |
| `gpu` | a number (`0`), a list (`0,1`, `1,0,2`) or `all` | GPU 0 alone; one warning at start names the other visible cards (only when `gpu` is unset) |
| `gpu-mode` | `peer`, `split`, `tensor` | none; required when `gpu` names two or more cards, an error with one |
| `gpu-share` | one positive number per card, in the order of `gpu` (`3,1`) | by the mode ([shares](#shares)); an error with one card |

- Cards are numbered as `nvidia-smi` numbers them. The first card listed is the main card; `all` is every visible card in order, so
  GPU 0 is the main card. When `CUDA_VISIBLE_DEVICES` is set, the numbers count within its list.
- The main card must be Blackwell (sm_120). A later card must be sm_120 too, except a peer card of sm_89 (Ada) in a build with
  `-DPEER_ADA=ON` ([older peer card](#an-older-peer-card)).
- A number past the visible cards or a card listed twice stops the start with an error.
- **Idle cards:** with `gpu` unset and more cards visible, every model warns once at start (`GPU 1 (RTX 5060 Ti) unused; to use
  more cards set gpu (0,1 or all) and gpu-mode`), in the CLI, the server and the bench. It is a warning, not an error. Naming a
  card (`gpu 0`) is a choice and prints nothing.
- **Old values:** `gpu: auto`, a `gpu` list without a `gpu-mode`, and the flags the old multi-card logic had are errors, with the
  valid forms in the message ([upgrading](../changelog/0.5.7.md#upgrading)).

```
gpus: peer; RTX 5090 (nvidia-smi 0) main, share 0.80; RTX 5060 Ti (nvidia-smi 1) peer, share 0.20 (bandwidth)
```

The start prints this line before the load and each card's part after it. `--gpu-plan` prints the choice as JSON, with the mode and
the shares, and checks the model's modes without loading it.

## Modes

- **peer:** the main card runs every layer and decodes; the other cards help. What they do depends on the model: hold and compute
  experts of a mixture of experts, read the last layers of a long prompt, or both.
- **split:** the layers are divided between the cards in order; each card holds its layers' weights and their KV cache and
  recurrent state, and a window or prompt chunk crosses from one card to the next. The cut follows the shares.
- **tensor:** every layer sharded across the cards. The name is reserved; no model runs it yet.

| Model | peer | split | tensor |
| --- | --- | --- | --- |
| [Qwen3.8-Flash-Next](../models/qwen38-flash-next.md#cards) | up to 4 cards: experts and prompt layers | up to 8 cards | refused |
| [DeepSeek-V4.1-Flash](../models/deepseek41.md#cards) | up to 5 cards: experts, a block of layers each | refused | refused |
| [Hy4-flash](../models/hy4-flash.md#cards) | up to 4 cards: experts, a block of layers each | refused | refused |
| [Qwen3.8-27B](../models/qwen38-27b.md#cards) | 2 cards: the last layers of long prompts | 2 cards | refused |
| [Inkling 2 Mini](../models/inkling2.md#two-cards) | 2 cards: the experts that spill from the main card | 2 cards | refused |
| [ERNIE-5.5-Flash](../models/ernie55.md#cards) | up to 4 cards: experts, a block of layers each | refused | refused |

A mode a model refuses is an error that names the modes it runs, for example `gpu-mode split: Hy4-flash does not run it; it runs
peer`. More cards than a mode takes is refused with the most it takes. One mode per run.

## Shares

Shares are normalized to sum to 1 and stay fixed for the run. What a share divides depends on the mode:

| Mode, part | What the share divides | Default without `gpu-share` |
| --- | --- | --- |
| split | each card's bytes: its layers' weights plus their KV and state at the context and slots; the cut is the layer nearest the shares | free VRAM at start |
| peer, experts | the experts the main card does not hold, dealt hottest first to the peers; the main card fills first | free VRAM of the peers |
| peer, prompt layers | the layers of a long prompt's chunk: the main card reads the first part, each card the next by share | memory bandwidth |
| peer, the main card's hot share | only with `gpu-share`: Flash-Next interleaves the hottest experts between the main card and a peer by share | off: the main card keeps the hottest |

A part that does not fit its card is refused with the numbers (what it needs, what is free); lower that card's `gpu-share`,
`max-context` or `parallel`. A peer that would hold nothing (the main card holds every expert, or its room is gone) is refused too,
with `use gpu 0`.

## Peer mode

- **Experts (Flash-Next, DeepSeek, Hy4-flash, Inkling, ERNIE-5.5).** Each peer holds experts the main card does not and computes their rows in
  every decode window, following the conversation like the main card's cache. The rows are the main card's kernels' rows, so where an
  expert sits does not change them. Flash-Next and Inkling deal the experts one by one, hottest first; DeepSeek, Hy4-flash and ERNIE-5.5 give each
  peer a contiguous block of layers, sized to its room.
- **Prompt rows.** A peer also computes its experts' share of each prompt chunk: any chunk on a peer as fast as the main card, chunks
  up to 2,559 rows on a slower one (a slower card costs a longer chunk more than the experts it saves the main card's link).
- **Prompt layers (Flash-Next, Qwen3.8-27B).** A long fresh part of a prompt is read as a pipeline over the cards while no other
  conversation decodes: the main card reads the first layers of a chunk and hands it on, a peer reads the next layers while the cards
  before it read the next chunks, and the main card reads the last chunk whole. Decode never uses the stage.
  - Flash-Next's cuts come from the shares over layers 2 to the last. A part shorter than a chunk, a chunk under 4,096 tokens, a
    chunk short enough to be read routed-only on a slower peer, and a prompt that arrives while other slots decode stay on the main
    card.
  - A peer's weights, prompt state and buffers go into its tier's last slots for the read, so outside a prompt decode does not change.
  - A loaded control vector is copied to every stage card.
- **Per-operation splits (Flash-Next).** A peer whose memory is faster than the main card's takes a share of each GDN mixer and of the
  head ([the GDN split](../precision/qwen38-flash-next.md#gdn-split-across-two-cards)).
- **Tuning:** `peer-reserve-mib` (default 600) is the VRAM a peer keeps free beside its experts and prompt buffers; `peer-exclusive`
  keeps the peer's experts out of RAM when RAM is short; `gpu 0` keeps everything on the main card.

Measured, from the ledger's rows:

- **Flash-Next:** no speed loss against the earlier peer path (64K prefill 9,823 against 9,835 tok/s, conversation decode within
  noise). A normal peer run scores BF16 KLD 0.06174 against 0.06221 on one card and top-1 91.84% against 91.80%.
- **Qwen3.8-27B:** a 64K prefill reads 17% faster than on one card (7,410 against 6,336 tok/s); decode stays on the main card.
- **Inkling 2 Mini:** a peer helps only when experts spill beyond what the main card's elastic copies cover. At default settings one
  card is fastest up to 1M tokens, and a peer is refused when the main card holds every expert
  ([Inkling](../models/inkling2.md#two-cards)).
- **Rounding:** a peer's prompt reads differ from one card's at summation-order rounding, about 0.01 KLD on `--diag-gpu-check`, the
  same as any reordering of the experts' sum. `--diag-main-card-bits` makes Flash-Next's other cards compute the main card's bits,
  and the prompt stages are then bitwise ([diagnostics](diagnostics.md)).

## Split mode

Each card runs a block of layers, one after another, with the window handed to the next card once. The cut is made at start by the
shares, before any weight is placed; every card holds only the dense weights of its own layers, and its leftover VRAM goes to that
model's experts or caches.

- **Flash-Next:** the experts no card holds are read from one pinned RAM copy that all stages share, and each card's adaptive swaps stay
  within its own layers. With two cards, a server and the drafter, one conversation at a time, the split overlaps consecutive windows:
  the first card starts window K+1 on the guess that K is accepted whole while the second card verifies K; a guess that fails puts the
  first card's state back and runs the window again, so the text is the same either way (`pipelined windows: N launched, M held` per
  request).
- **Qwen3.8-27B:** the output is the main card's alone to the bit ([27B](../models/qwen38-27b.md#cards)).
- **Inkling 2 Mini:** a split holds every expert of its layers on its card; it is about 25% slower than one card at the contexts where
  one card holds or copies back its experts ([Inkling](../models/inkling2.md#two-cards)).
- A split costs a fixed part of a window per crossing: use it for a model or context that does not fit one card, and a peer where it
  does.

## Two 16 GB cards

`gpu "0,1"` with `gpu-mode "peer"` on two cards of 16 GB (two RTX 5080s, say) has each card hold experts in what its VRAM has left. The
engine sizes the second card to leave room for its prompt buffers (about 200 MiB, computed) and `peer-reserve-mib` more (default
600 MiB, at least 128). If the start prints `prefill peer: the peer's buffers do not fit`, raise `peer-reserve-mib` (a display or
another program may be using the card) and its experts shrink by the same amount. `peer GPU 1: N experts` says how many it holds.

## An older peer card

One older architecture can be built for the peer's kernels: sm_89 (Ada), beside an sm_120 main card, with the `-DPEER_ADA=ON` build
flag (off by default to keep build times down). Without it, or for any other pre-Blackwell card, the card is refused at the start.
What such a peer does:

- It holds experts and computes their rows in every decode window with the standard-CUDA kernels (Q8_1/Q2_0 and the GGUF i-quant
  rows, dp4a arithmetic). It reads its experts' share of a prompt chunk under the slower-card rule (chunks up to 2,559 rows).
- It does not run the main card's work: the dense weights, the KV cache, the drafter, TMA and the FP4 tensor-core expert kernels stay
  on the Blackwell card, so an NVFP4 pack's FP4 experts are main-card only.
- Only Flash-Next's peer kernels are built for it; the other models refuse an sm_89 card. Split and tensor need sm_120 cards
  (`gpu-mode split: GPU 1 (RTX 4060 Ti) is sm_89; split and tensor run on sm_120 cards`).

Three or more cards are community-tested: the path is checked on two cards. Reports from machines with three or four cards are
welcome ([community](../community.md)).
