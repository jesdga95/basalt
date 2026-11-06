# DeepSeek-V4.1-Flash

Runs with the same command as any model: `basalt serve PACK`. What it supports next to Flash-Next: [models](index.md). Rounding and
quality: [precision](../precision/deepseek41.md).

## Pack

One pack, `DeepSeek-V4.1-Flash-1017-UD-IQ1_M`, on [Hugging Face](https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt), about 260 GB, built
from Unsloth's UD-IQ1_M GGUF. Its layout and the DSpark section: [model file](../model-file.md#deepseek-pack).

## Memory

- **Start:** experts are dealt out in routing order and loaded once per start. One card: as many as fit go to a pinned RAM tier,
  most routed first; VRAM slots copy the tier's head; the rest are read from the pack by the CPU. Two cards (`--gpu 0,1`): the second
  card holds experts nowhere else has. One second card is used; later cards idle. The start line
  `expert homes of 384 per layer: ...` gives the split.
- **RAM:** the RAM tier leaves 16 GiB free (`RAM floor 16.0 GiB of N GiB free`) for the OS, serving and the page cache of
  the pack's reads. N places short puts the last N experts of every layer in the file, with a warning naming the RAM to free.
- **Slots follow routing:** an expert that keeps missing takes the slot used longest ago, on either card.
- **Context:** `--max-context` from 1,024 up to the pack's 1,048,576, default 8,192. Caches grow with position (a short chat maps
  a few MiB, a 1M context about 3.2 GiB) into the memory the tier leaves free. A value past the maximum, or whose caches need more
  than half the free memory, is refused; a quarter or more gives a warning.
- **VRAM:** `--expert-cache` caps the slots; `--vram-reserve-mib` or `POST /v1/vram` sets what stays free. Prompts take the tier's
  copies first and halve their chunks to fit above the reserve.
- **Refused flags:** `--kv` (fp16 only), `--rope-*`, `--yarn-*` (YaRN is part of the model).

## Long prompts

- Chunks are 16,384 tokens, halved down to 256 to fit memory.
- A read of 4,096 tokens or more (2,048 on two cards) gives the main card's copies back for its duration. They reopen afterwards:
  from routing order on two cards, empty on one, so the first tokens after a long prompt decode slower on one card.
- `--no-prompt-skip` runs every row through every layer. Without it, rows further than 2,668 positions from the read's end stop
  at layer 20 (the math is the same; rounding differs; [precision](../precision/deepseek41.md#prompt-skip)).
- A next turn that starts with the held tokens (or the checkpoint) reads only the new tail. `pp=N/Mc` in the `done` line shows how many
  tokens were kept.

## Drafting

On when the pack has the DSpark drafter. `--spec N` sets the window (default 3: the last accepted token plus up to 2 drafts; at most
8). `--spec 0` turns it off. Each drafter stage keeps its 32 most routed experts (`--draft-experts`, default 32; 64, 96, 128 or 0 for
all), ranked by `data/profiles/deepseek41-40x384-dspark.prof`. The start line:

```
basalt deepseek41: drafter on GPU 0: 2.41 GiB, 32 of 128 experts per stage, windows of at most 3 tokens (--spec 0: off)
```

### What it costs

Measured on the UD-IQ1_M pack, 400 W + 180 W, 121 GiB of RAM, `--pool-workers 8`:

- The drafter takes 2.41 GiB of the main card with 32 experts per stage, 7.47 GiB with all 128 (which leaves 28 to 29 experts per
  layer in the file). A block takes 2.3 ms.
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

No throughput gain: at 400 W + 180 W, one stream 60.1 tok/s, two slots 58.4 and four 54.9 in total. Slots share one decode stream
and draft nothing.

## Lossy flags

None. Output depends on which experts are resident, so a different RAM tier size can change the text
([precision](../precision/deepseek41.md#experts-and-placement)).

## Limits

- **Two-card idle sleep** happens only when the GPU state fits in free RAM with 8 GiB to spare. On 121 GiB of RAM, two cards held
  38 GiB of GPU state against 21 GiB free, so the model stayed awake. One card slept in 1.9 s.
- **Not measured:** restore drift from the disk tier, and second-card sleep on a larger machine.
