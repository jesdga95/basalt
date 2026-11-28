# DeepSeek-V4.1-Flash

Runs with the same command as any model: `basalt serve PACK`. What it supports next to the others: [models](index.md). Rounding and
quality: [precision](../precision/deepseek41.md).

## Pack

Packs on [Hugging Face](https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt), built from Unsloth's GGUFs:
`DeepSeek-V4.1-Flash-1017-UD-IQ1_M` (about 260 GB) and `-UD-IQ2_XXS` (about 285 GB, closer to BF16, slower when experts spill to disk). Its layout and the DSpark section: [model file](../model-file/deepseek.md#deepseek-pack).

Unsloth's `-UD-Q2_K_XL` (about 343 GB, MXFP4 and IQ3_S down experts, a Q5_K Engram table) has no published pack; it converts
from the GGUF and is the closest to BF16 ([precision](../precision/deepseek41.md#against-the-bf16-reference)):

```sh
python3 tools/convert/convert.py --gguf DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00001-of-00007.gguf --dspark DRAFTER.gguf \
    --out DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL.basalt
```

An NVFP4 pack, `DeepSeek-V4.1-Flash-1017-NVFP4` (about 428 GB), is built from NVIDIA's NVFP4 checkpoint: NVFP4 routed
experts, the FP8 projections as Q8_0, the MXFP4 Engram table (a UD set's Engram may be IQ4_NL, Q5_K, Q6_K, MXFP4 or
MXFP8; [how](../model-file/deepseek.md#deepseek-nvfp4-pack)). It is meant for
machines whose RAM holds most of its experts (about 320 GB or more); with less, most experts are read from the SSD and decode
is slow (2.8 tok/s with 172.8 GiB of experts on the SSD, 121 GiB of RAM, 400 W + 180 W).

## Memory

- **Start:** experts are dealt out in routing order and loaded once per start, the same way on one card and two. A peer card
  ([cards](#cards)) holds experts nowhere else has. When RAM cannot take the rest, the main
  card holds the most routed ones too (homes, sized to stay through a prompt at the full context); then RAM, most routed first; the
  rest are read from the pack by the CPU. When RAM takes them all, the main card's slots copy the RAM tier's head instead. Either way
  the main card fills the rest of its memory with copies, down to what a short prompt needs. The start line
  `expert homes of 384 per layer: ...` gives the split.
- **RAM:** the checkpoint blocks (`--prompt-cache` + 1 per slot, about 3 MB each) are allocated page-locked at start, with a
  spare VRAM buffer of the same size, before the RAM tier is sized. The prompt path's host buffers are allocated at
  start, before the RAM tier: the pinned ring of file
  reads (only when experts may stay in the file, at most 240 experts), the staging of a short read, a peer card's prompt
  rows, and a chunk's Engram reads. A picture's decoded pixels and patch rows are ordinary memory that grows while a request's pictures are read and
  is freed after them (the largest picture takes a few hundred MiB); with `--vision dynamic` nothing is held for pictures at rest, so the RAM tier,
  the homes and the copies are planned exactly as with `--vision off`. The tier then leaves `--ram-reserve-mib` free beside the driver's tracking of its
  pinned pages (`RAM floor X GiB of N GiB free`). N places short puts N experts of every layer in the file, with a warning naming
  the RAM to free.
- **Engram table:** read by row from the pack by default (`--ple-io direct`), so it takes no RAM of its own; `--ple-io ram` maps
  the table at start (reserving its RAM before the expert tier is sized) and locks it in RAM once the experts load (or touches it
  into the OS page cache when locking is refused).
- **Slots follow routing:** an expert that keeps missing takes the slot used longest ago, on either card.
- **Experts in the file:** with experts left in the file, the RAM tier is also a cache of it. A decode step that routes to a file
  expert reads it whole into the place of the layer's pinned expert used longest ago, around the OS file cache, while the CPU
  computes the experts already in RAM; the CPU then computes the ones read. A pinned expert the CPU keeps computing takes a VRAM
  slot. `TIER final: N experts read into the RAM tier` counts the reads at exit.
- **Context:** `--max-context` from 1,024 up to the pack's 1,048,576, default 8,192. Caches grow with position (a short chat maps
  a few MiB, a 1M context about 0.97 GiB) into the memory the tier leaves free. A value past the maximum, or whose caches need more
  than half the free memory, is refused; a quarter or more gives a warning.
- **VRAM:** the tier is sized once, at the start, and its size does not change while serving: a prompt borrows slots inside that
  total and gives the same number back. `--expert-cache` is a count per all layers (divided by the layer count): honoured exactly,
  or the start is refused with one line (what fits, the free VRAM, the reserve); `auto` takes what fits. `--vram-reserve-mib` or
  `POST /v1/vram` sets what stays free and plans the size again once, from the free memory then. The caches, prompts and images
  take the tier's copies first (homes stay) and halve their chunks to fit above the reserve.
  `--vision dynamic` has no card memory at rest: a request's pictures borrow the encoder's weights and work buffers from the tier's copies
  beside the homes (all the copies when those do not hold them), and the tier takes them back and refills them when the pictures are read.
  The homes keep 1.5 GiB free beside them when the pack carries a projector, with vision on or off, so the plan is the same either way.
- **Cache precision:** `--kv fp8` (default) stores the caches as the model's reference rounds them, 920 bytes per position;
  `--kv fp16` keeps the unrounded values, 3,200 bytes per position (a 1M context about 3.2 GiB). [precision](../precision/deepseek41.md#where-each-stage-rounds)
- **Refused flags:** `--rope-*`, `--yarn-*` (YaRN is part of the model).

## Cards

`gpu` names the cards and `gpu-mode` what they do ([GPUs](../configuration/gpus.md)); `"gpu": "0,1", "gpu-mode": "peer"`, or `"gpu":
"all"`. The model runs `peer` only, on up to five cards (the main card and four peers); `split` and `tensor` are refused with the
modes it runs.

- **Peer:** each card after the main one holds the second-card experts of one contiguous block of layers, the blocks sized to each
  card's free memory (or by `gpu-share`) so that every layer keeps the same number of peer slots. A peer that would hold nothing is
  refused. The peer computes its experts in every decode window and its share of each prompt chunk. An sm_89 peer is refused: its
  kernels are built for sm_120 only.
- **Quality:** against one card a peer differs at CPU-against-GPU expert rounding (KLD 0.01 on the reference's own picks, BF16 KLD gate
  0.2805 against 0.2794); the same text with the tier frozen.

## Long prompts

- Chunks are 16,384 tokens, halved down to 256 to fit memory.
- A read of more than 2,048 tokens gives back the copies the main card holds beside the short-prompt workspaces; from 4,096 tokens,
  a main card without homes gives back all its slots. Afterwards the same number of slots reopens, empty, and refills from routing
  order; if the card can no longer hold them, that is an error, not a smaller tier.
- With a peer card, each layer of a chunk splits its experts: the card holding that layer computes its own and a share of the
  RAM ones, read over its own PCIe link, while the main card computes the rest; the share is priced from each card's link and SM
  clock. Each peer card's prompt buffers come out of its own memory before its slots are sized.
- `--no-prompt-skip` runs every row through every layer. Without it, rows further than 2,668 positions from the read's end stop
  at layer 20 (the math is the same; rounding differs; [precision](../precision/deepseek41.md#prompt-skip)).
- A next turn that starts with the held tokens (or the checkpoint) reads only the new tail. `pp=N/Mc` in the `done` line shows how many
  tokens were kept.

## Drafting

On when the pack has the DSpark drafter. One block pass drafts up to 5 tokens and states each one's chance of acceptance. Each
window takes the drafts that pay: of the windows of 1 to 6 rows, the one with the most expected tokens per millisecond, from those
chances (calibrated by how often drafts of that chance were accepted) and the round times this machine measured for each window
size (`include/basalt/spec/draft_policy.hpp`). The block drafts at most one token more than the last window used, and at least 2.
`--spec N` caps the window (default 6: the last accepted token plus the whole block; at most 8). `--spec 0` turns it off. Each
drafter stage keeps its 32 most routed experts (`--draft-experts`, default 32; 64, 96, 128 or 0 for all), ranked by
`data/profiles/deepseek41-40x384-dspark.prof`. The start line:

```
basalt deepseek41: drafter on GPU 0: 2.41 GiB, 32 of 128 experts per stage, windows of at most 6 tokens (--spec 0: off)
```

### What it costs

Measured on the UD-IQ1_M pack, 400 W + 180 W, 121 GiB of RAM, `--pool-workers 8`:

- The drafter takes 2.41 GiB of the main card with 32 experts per stage, 7.47 GiB with all 128 (which leaves 28 to 29 experts per
  layer in the file). A block takes 1.4 ms.
- Greedy, 800 tokens, cold, tok/s (tokens per window):

| | Code | Conversation | Prose |
|---|---|---|---|
| Plain | 62.2 | 64.7 | 65.2 |
| `--spec 3`, 32 experts | 82.6 (2.64) | 78.1 (2.26) | 69.6 (1.96) |
| `--spec 3`, all 128 | 56.4 (2.69) | | 51.8 (1.89) |

- Long conversation (125 to 141K tokens, thinking and tools, temperature 1, turns after the first): plain 62.15 tok/s, `--spec 3` with
  32 experts 63.15.
- **Where it pays:** code and structured output. On long prose at temperature 1 it is about even.

## Parallel slots

Slots share one decode stream through the shared batch driver ([batching](../architecture/batching.md)), as on every model: a round
runs the busy slots' windows as one batch of up to 8 rows. Each slot drafts as a single stream does, from the drafter's window
rows of its own conversation. The slots share one drafter, so `--parallel` adds no drafter memory. A request alone in its slot
gives the same greedy tokens as with `--parallel 1` (tiers pinned), and a batch's rows give bitwise their solo windows' picks and
logits (`--diag-batch-check`, frozen tier).

With experts in RAM a batch's rows reach a wider union of experts than one slot's window, and the 8-row cap trims each slot's
drafts (about 2 rows a slot at four slots, against 6-row windows alone), so on two cards batching is slower than running each
slot's window in turn. Aggregate tok/s, two cards (400 W + 180 W), drafted, tiers unfrozen, 256 tokens a slot, ABBA:

| Slots | In turn (A1 / A2) | Batched (B1 / B2) | Change |
| --- | --- | --- | --- |
| 1 | 89.3 / 87.0 | 87.0 / 92.8 | none: one slot runs its solo window, the same code |
| 2 | 85.0 / 82.7 | 85.4 / 80.4 | -1.1% |
| 4 | 71.5 / 70.0 | 63.8 / 62.1 | -11.0% |
| 8 | 76.9 / 77.2 | 54.6 / 51.1 | -31.4% |

## Lossy flags

Off by default. `--route-prior` and `--route-tail-skip` change output, and each is named in one start warning. What they do on
every MoE model: [quality](../precision/quality.md#lossy-routing-flags); what they cost and gain here:
[precision](../precision/deepseek41.md#lossy-routing-flags). Without them, output still depends on which experts are resident, so a
different RAM tier size can change the text ([precision](../precision/deepseek41.md#experts-and-placement)).

## Limits

- **Two-card idle sleep** (peer mode) happens only when the GPU state fits in free RAM with 8 GiB to spare. On 121 GiB of RAM, two cards held
  38 GiB of GPU state against 21 GiB free, so the model stayed awake. One card slept in 1.9 s.
- **Not measured:** restore drift from the disk tier, and second-card sleep on a larger machine.
