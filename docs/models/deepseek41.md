# DeepSeek-V4.1-Flash

Runs with the same command as any model: `basalt serve PACK`. What it supports next to Flash-Next: [models](index.md). Rounding and
quality: [precision](../precision/deepseek41.md).

## Pack

Packs on [Hugging Face](https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt), built from Unsloth's GGUFs:
`DeepSeek-V4.1-Flash-1017-UD-IQ1_M` (about 260 GB) and `-UD-IQ2_XXS` (about 285 GB, closer to BF16, slower when experts spill to disk). Its layout and the DSpark section: [model file](../model-file.md#deepseek-pack).

A test pack, `DeepSeek-V4.1-Flash-1017-NVFP4-test` (about 428 GB), is built from NVIDIA's NVFP4 checkpoint: NVFP4 routed
experts, the FP8 projections as Q8_0, the MXFP4 Engram table ([how](../model-file.md#deepseek-nvfp4-pack)). It is meant for
machines whose RAM holds most of its experts (about 320 GB or more); with less, most experts are read from the SSD and decode
is slow (2.8 tok/s with 172.8 GiB of experts on the SSD, 121 GiB of RAM, 400 W + 180 W).

## Memory

- **Start:** experts are dealt out in routing order and loaded once per start, the same way on one card and two. A second card
  (`--gpu 0,1`) holds experts nowhere else has; one second card is used, later cards idle. When RAM cannot take the rest, the main
  card holds the most routed ones too (homes, sized to stay through a prompt at the full context); then RAM, most routed first; the
  rest are read from the pack by the CPU. When RAM takes them all, the main card's slots copy the RAM tier's head instead. Either way
  the main card fills the rest of its memory with copies, down to what a short prompt needs. The start line
  `expert homes of 384 per layer: ...` gives the split.
- **RAM:** the prompt path's and the images' host buffers are allocated at start, before the RAM tier: the pinned ring of file
  reads (only when experts may stay in the file, at most 240 experts), the staging of a short read, the second card's prompt
  rows, a chunk's Engram reads, and each slot's largest image. The tier then leaves `--ram-reserve-mib` free beside the driver's tracking of its
  pinned pages (`RAM floor X GiB of N GiB free`). N places short puts N experts of every layer in the file, with a warning naming
  the RAM to free.
- **Slots follow routing:** an expert that keeps missing takes the slot used longest ago, on either card.
- **Context:** `--max-context` from 1,024 up to the pack's 1,048,576, default 8,192. Caches grow with position (a short chat maps
  a few MiB, a 1M context about 0.97 GiB) into the memory the tier leaves free. A value past the maximum, or whose caches need more
  than half the free memory, is refused; a quarter or more gives a warning.
- **VRAM:** `--expert-cache` caps the slots; `--vram-reserve-mib` or `POST /v1/vram` sets what stays free. The caches, prompts and
  images take the tier's copies first (homes stay) and halve their chunks to fit above the reserve.
- **Cache precision:** `--kv fp8` (default) stores the caches as the model's reference rounds them, 920 bytes per position;
  `--kv fp16` keeps the unrounded values, 3,200 bytes per position (a 1M context about 3.2 GiB). [precision](../precision/deepseek41.md#where-each-stage-rounds)
- **Refused flags:** `--rope-*`, `--yarn-*` (YaRN is part of the model).

## Long prompts

- Chunks are 16,384 tokens, halved down to 256 to fit memory.
- A read of more than 2,048 tokens gives back the copies the main card grew into its spare memory; from 4,096 tokens, a main card
  without homes gives back all its slots. Afterwards the grown copies reopen from routing order, the other slots empty.
- With a second card, each layer of a chunk splits its experts: the second card computes its own and a share of the RAM ones,
  read over its own PCIe link, while the main card computes the rest; the share is priced from each card's link and SM clock.
  Its prompt buffers come out of the second card's memory before its slots are sized.
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

Slots share one decode stream: a batch window runs each busy slot's window in turn. Each slot drafts as a single stream does, from
the drafter's window rows of its own conversation. The slots share one drafter, so `--parallel` adds no drafter memory. A request
alone in its slot gives the same greedy tokens as with `--parallel 1` (tiers pinned).

## Lossy flags

None. Output depends on which experts are resident, so a different RAM tier size can change the text
([precision](../precision/deepseek41.md#experts-and-placement)).

## Limits

- **Two-card idle sleep** happens only when the GPU state fits in free RAM with 8 GiB to spare. On 121 GiB of RAM, two cards held
  38 GiB of GPU state against 21 GiB free, so the model stayed awake. One card slept in 1.9 s.
- **Not measured:** restore drift from the disk tier, and second-card sleep on a larger machine.
