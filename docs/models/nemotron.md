# Nemotron 3.5 Super

NVIDIA's Nemotron 3.5 Super (120B total, 12B active; OpenMDW-1.1), the language model of
[NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4).
Runs with the same command as any model: `basalt serve PACK`. What it supports next to the others: [models](index.md).
Rounding and quality: [precision](../precision/nemotron.md).

## Pack

One file, `Nemotron-3.5-Super-NVFP4.basalt` (68.5 GiB), built from NVIDIA's NVFP4 checkpoint alone:

- Routed experts: the checkpoint's NVFP4 codes and scales, bit for bit (60.54 GiB, the MTP head's included).
- Mamba, attention and output projections: the checkpoint's FP8 codes and per-matrix scales, bit for bit.
- Shared experts: the checkpoint's NVFP4, bit for bit. Latent projections and token embedding BF16 as stored.
- Norms, router, its bias, convolutions, `dt` bias, `D` and `A` F32.

- The multi-token-prediction (MTP) head: an attention layer (BF16 as stored) and a MoE layer, the drafter.

The vision tower is not converted: text only. Build it: [model file](../model-file.md#nemotron-35-super-pack).

## Architecture

The `nemotronh` architecture (code in `src/models/nemotronh`): 88 layers, 4,096 wide, no rotary positions.

- **Mamba-2 (40 layers):** 128 heads of 64, a state of 128 per head in 8 groups, a causal convolution of 4. The state is a
  fixed size whatever the context. The gated norm comes after the gate, in groups of 1,024.
- **Attention (8 layers):** 32 query heads, 2 key/value heads of 128, every position.
- **Latent MoE (40 layers):** the token goes down to a 1,024-wide latent (BF16), through 22 of 512 routed experts chosen
  by sigmoid with a correction bias (normalized, times 5), and back up. The experts are squared-ReLU with no gate:
  `down(relu(up x)^2)`, 1,024 to 2,688 and back. A shared squared-ReLU expert (4,096 to 5,376) runs on the full width.

The Mamba-2 kernels (`src/kernels/cuda/ssd.cu`) and the squared-ReLU expert form (`NativeExpertLayout::relu2`, CPU and GPU)
are shared, for any later model that uses them.

## Memory

- **One card.** The main card holds the dense weights (7.72 GiB), the attention caches, the Mamba state and as many expert
  slots as the rest of the card allows. The experts that do not fit are in page-locked RAM: about 64 GB of system RAM
  for all 512 x 40. With less, the rest are read from the file.
- **Per position:** 8 attention layers, 8 KiB with `--kv fp16` (the only format).
- **No second card.** The expert tier does not place this model's experts on peer cards (`max_expert_cards` 0).

## Decode

Each token runs its 40 MoE layers' 22 experts where they are: the card's from their slots, a few RAM misses the card
fetches over the bus, the rest by the CPU workers from RAM. The host never waits for the CPU's share: the card's weighted
sum waits for a word the host writes once the CPU's rows landed, the host queues the next layers meanwhile, and the card
reads the next weights into L2 while it waits (the decode of Hy4-flash and DeepSeek). On an RTX PRO 4000 (power limited
to 145 W) with 59 GiB of experts in RAM and 38 experts a layer on the card: **30.7 ms a token (32.6 tok/s)**. Windows of
2, 4 and 8 rows take 43, 66 and 125 ms.

**The MTP head drafts** (`--spec N`, default 4: windows of up to N rows; `0` off). It takes the main model's last hidden
row and the token just picked, and runs one attention and one MoE layer through the same code and expert tier as the
main model, its own attention cache kept with the conversation's. Each further draft feeds the head its own output. The
window then verifies every row at once and keeps the drafts the main model agrees with, so only speed changes. A 200-token
chat reply at temperature 0:

| `--spec` | Tokens a window | Draft, verify (ms a window) | ms a token | tok/s |
| --- | --- | --- | --- | --- |
| 0 | 1.00 | 0, 31.8 | 31.8 | 31.4 |
| 2 | 1.93 | 2.2, 46.4 | 25.2 | 39.7 |
| 3 | 2.79 | 4.3, 58.7 | 22.6 | 44.2 |
| 4 | 3.32 | 6.3, 67.1 | 22.1 | **45.2** |
| 6 | 4.47 | 10.3, 94.7 | 23.5 | 42.6 |
| 8 | 4.81 | 13.9, 124.1 | 28.7 | 34.9 |

The first draft is kept 88% of the time, the fourth 67%. Past 4 the window's cost grows faster than what it keeps.

**Prompt lookup** (`--suffix-draft N`, default 3, `0` off) fills a window from an earlier repeat of the last N or more
tokens, longer than the head's. Rewriting a 1,700-token Python file at temperature 0 (700 tokens, the same text each way):

| Drafts | Tokens a window | tok/s |
| --- | --- | --- |
| none | 1.0 | 30.4 |
| MTP head | 4.0 | 49.2 |
| prompt lookup | 14.3 | 62.0 |
| both (the default) | 14.6 | 59.2 |

With both, the head still makes its first draft each window (a lookup's proposal is matched against it) and reads the
kept rows, about 16 ms a window: on text that copies the prompt that costs 5%, on any other the head is what drafts.

Which experts sit on the card changes the rounding of the rows they compute (the card's and the CPU's 8-bit
activations differ), so a near tie can go either way between runs with different drafting; with the placement fixed the
windows are bitwise one-row decode ([precision](../precision/nemotron.md#against-the-reference)).

## Long prompts

A prompt reads in chunks of up to 4,096 rows. Each chunk stages, layer by layer, every expert its rows route to on the
card (from its slot, RAM or the file), 16 at a time on a copy stream while the group before computes. The routed experts run
on llama.cpp's MMQ (int8 tensor cores, q8_1 activations against the NVFP4 blocks), the Mamba projections on the FP8
tensor cores, the shared expert on the FP4 tensor cores. Staging dominates: a chunk copies nearly every expert of every
layer, so longer chunks are faster.

| Prompt | Chunk | Time | tok/s |
| --- | --- | --- | --- |
| 4,000 tokens | 512 | 12.2 s | 329 |
| 4,000 tokens | 2,048 | 4.5 s | 883 |
| 12,000 tokens | 2,048 | 13.6 s | 882 |
| 12,000 tokens | 4,096 | 9.5 s | 1,258 |

RTX PRO 4000 at 145 W, one session, the experts in RAM. The engine uses 4,096.

## Chat, tools, thinking

ChatML turns (`<|im_start|>`, `<|im_end|>`) with the checkpoint's chat template. Thinking is on by default, in
`<think>` ... `</think>`, returned separately as reasoning. Tool calls are Qwen's `<tool_call><function=...>` form, several
at once allowed. Reasoning effort: `none` turns thinking off; `low` and `medium` set the template's efficient-reasoning
switch; `high` and `max` think fully. A number is refused.

## Not yet

- A quality base for `basalt-bench --quality` (needs a BF16 run of the full model). Until then the check is the float64
  reference ([precision](../precision/nemotron.md#against-the-reference)).
- Images, other `--kv` formats, a second card, idle sleep.
