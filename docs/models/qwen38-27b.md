# Qwen3.8-27B

A dense model. Runs with the same command as any model: `basalt serve PACK`. What it supports next to the others: [models](index.md). Rounding and quality: [precision](../precision/qwen38-27b.md).

## Pack

One file from QUASAR's quantization-aware NVFP4 checkpoint
([QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4), Apache-2.0), with the
DFlash 2 block drafter ([z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)):

- GDN, attention and MLP projections: QUASAR's NVFP4 codes and scales, moved bit for bit with their tensor scales.
- GDN a and b projections (48 rows each): QUASAR's NVFP4 values rounded once to BF16.
- Output head: QUASAR's BF16 head as FP8 rows (E4M3 codes, an FP32 scale per row), rounded once by the converter.
- The MTP layer and the block drafter: BF16 projections as Q8_0; at load the block drafter's projections are rounded to
  NVFP4 (drafts only: the target model checks every draft). Vision: the BF16 tower, as stored.
- The BF16 token embedding stays in page-locked RAM (2.4 GiB); everything else sits on the main card (16.0 GiB with
  the block drafter, 14.0 GiB without a drafter).

Download it from [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-27B-Basalt) (`hf download jesdga/Qwen3.8-27B-Basalt Qwen3.8-27B-QUASAR-NVFP4.basalt --local-dir models`),
or build the same file yourself: [model file](../model-file.md#qwen38-27b-pack). NVIDIA's NVFP4 checkpoint converts with the
same command; how each compares against BF16: [precision](../precision/qwen38-27b.md#quality-against-bf16).

## Architecture

The `qwen35` architecture (Qwen3.5, 3.6 and 3.8 dense): 64 layers, 3 gated delta-net layers to 1 gated attention layer, a dense
SwiGLU FFN in each, interleaved M-RoPE for pictures. It runs on Basalt's shared dense stack (`src/dense`), which reads every
width from the pack; the GDN and attention kernels are shared with Flash-Next.

## Memory

- **Context:** `--max-context`, default 8,192, up to the model's 262,144; past it with `--rope-scaling yarn --rope-scale F`
  (up to 262,144 x F; [configuration](../configuration/index.md#context-and-kv-cache)), which also changes short prompts'
  output, so leave it off below 262,144 ([precision](../precision/qwen38-27b.md#context-past-262144)). The KV cache is allocated at start for every slot:
  `--kv fp16` (default) takes 64 KiB per position over the 16 attention layers, `--kv int8` 33 KiB (int8 codes with an fp16
  scale per 64 values; [precision](../precision/qwen38-27b.md)). On a 32 GB card one slot holds about 190K positions in fp16
  and the full 262,144 in int8. The start refuses a context that does not fit and names the free memory.
- **K/V streaming:** `--kv-resident N` (default 0, off) keeps each attention layer's first N positions on the card and the
  rest in page-locked RAM, so a context or `--parallel` whose cache does not fit the card still starts. Attention reads every
  position for every token, so each decoded token reads all positions past N over the bus: decode slows with every position
  in RAM. A prompt chunk copies each layer's positions in RAM to the card once (one layer's worth of VRAM, kept for the run)
  while the layers before it run, so prompt reads stay close. The output is bitwise the same as with every position on the
  card. With `--kv-resident` below the context the second card does not read prompts ([below](#long-prompts-on-two-cards)).
  One RTX 5090 at 400 W on a PCIe 5.0 x8 link (28.8 GB/s measured host to card), `--kv int8`, a 250,038-token prompt
  (repeating text, which drafts well), 64 to 256 tokens decoded:

  | Positions in RAM | Prompt read | Decode, `--spec 0` | Decode, drafted |
  | --- | --- | --- | --- |
  | none (`--kv-resident 0`) | 4,345 tok/s | 45.4 tok/s | 79.0 tok/s |
  | 32K | 4,251 tok/s | 14.6 tok/s | |
  | 64K | 4,212 tok/s | 8.6 tok/s | 15.7 tok/s |
  | 128K | 4,183 tok/s | 4.7 tok/s | 8.4 tok/s |
  | 200K | 4,171 tok/s | 3.1 tok/s | |

  Each position in RAM costs a token 33 KiB of bus reads (`--kv fp16`: 64 KiB; 60K positions at a 120K prompt decode at
  6.3 tok/s against 59.9). Decode reads them at about 24 GB/s of the link's 28.8, so the link sets the speed.
- **Slots:** `--parallel N` gives each slot its own cache and recurrent state (157 MiB). Slots share the weights.
- **Cards:** the main card decodes and holds everything. A second Blackwell card (`gpu "0,1"`, or `"auto"` when two are visible) is the
  prompt line: it reads the last layers of long prompts ([below](#long-prompts-on-two-cards)). Further cards idle.

## Drafting

On by default with the pack's drafter; `--spec N` caps the window (default 16, at most 16), `--spec 0` turns it off. The block
drafter runs its block of 8 rows, and a wider block (up to the window) after windows whose drafts were all taken, and offers
every draft of the block; the draft policy reads each draft's confidence to pick the window. Without it the MTP layer drafts one
token per step. `--suffix-draft N` (default 3, `0` = off)
also fills a window from an earlier repeat of at least N tokens in the context, where the draft policy expects it to pay.
Output is exactly plain decoding's: a row's arithmetic does not depend on the window's rows, drafted or not, alone or batched.
A sampled request without a `seed` draws the block drafter's drafts and keeps them by speculative sampling: its tokens follow the
model's distribution exactly, and more drafts are kept than the drafter's best guesses would give
([precision](../precision/qwen38-27b.md#sampling-with-drafts)).

## Long prompts on two cards

With a second Blackwell card (`gpu "0,1"`, [GPUs](../configuration/gpus.md)) that card holds the last 20 layers (4 GiB) and the
KV cache of their attention layers. A prompt part longer than two chunks (8,192 tokens) is read as a pipeline: the main card
reads the first layers of a chunk and hands its rows over through page-locked RAM, the second card reads the remaining layers of
that chunk while the main card reads the next one, and each chunk's KV cells come back as it finishes. The main card reads the
last one or two chunks whole once the second card's recurrent state is back, so decoding, the conversation cache and later
requests find the session a read on the main card alone leaves: the output is the same to the bit (the second card runs the
main card's kernels on the same rows). Each read is priced from both cards' rates, timed at start on random rows and refitted
from every split read, and is split, at the layer the price picks, only when that is at least 5% faster. Log lines start with
`prompt line:`. Decoding never uses the second card. Without room on it the start warns and prompts stay on the main card.

The start line names it (`gpus: RTX 5090 (nvidia-smi 0) main, RTX 5060 Ti (nvidia-smi 1) prompt line`), then `prompt line: layers 44-63
(4.00 GiB) on CUDA1`, the two cards' timed rates and the page-locked RAM (0.4 GiB). It costs about 5 s at start (the upload and the
timing) and 0.4 GiB of the main card. The second card holds the 4 GiB of layers, their attention layers' cells and a chunk's scratch: 6 to
9 GiB, 8.7 GiB at 262,144 positions in `int8`. A context of 16,384 or less holds no prompt the line would split, and with `--kv-resident` below the context (K/V in RAM, [memory](#memory)) the line stays off; in both cases nothing goes to
the second card.

The line works with everything a read on the main card does: `--parallel` slots (a slot's read runs while the others wait, as on one
card), pictures and video (a picture's rows go through the main card; the text before and after reads on the line), the conversation cache and
its checkpoints, the thinking budget. A cancelled request is met within two chunks on either card (0.9 s into a 16K read and 1.4 s into a 40K one, against 0.5 and 0.8 s on one card)
and the next request finds the state it would find after a cancel on one card.

## Images and video

`--vision dynamic`: the encoder's weights wait in page-locked RAM and go to the card for each request's pictures. Video works as
for Flash-Next ([video](../running.md#video)); frames decode on the CPU.

## Limits

- `--kv-resident` keeps the MTP drafter's own cache on the card (the block drafter's is a short window).
- Slots whose windows fit 16 rows together (every slot in plain decoding, two slots of 8-row drafted windows) share one pass through the weights each round, and their block drafters run two blocks a pass; a window that does not fit takes a pass of its own. Prompt reads run per slot.
- Decoding runs on the main card: a second card reads long prompts only (no layer split, no second-card drafting).

## Speed

`basalt-bench` on an RTX 5090 at 400 W, `--kv fp16` unless stated, the pack with the block drafter, cold first runs (one card except the last two rows):

| | Result |
| --- | --- |
| Prompt read, 4k / 16k / 32k / 64k tokens | 12,735 / 10,850 / 8,751 / 6,201 tok/s |
| Decode, plain (`--spec 0`, conversation) | 90 tok/s |
| Decode, drafted (conversation, struct, prose) | 262, 776, 194 tok/s |
| Decode, `--parallel 2` / `--parallel 4`, every slot on the same chat, all slots together | 408 / 566 tok/s |
| Prompt read, 8k / 16k / 32k / 64k, one card, then with an RTX 5060 Ti (180 W) as the prompt line | 12,072 / 10,716 / 8,654 / 6,137, then 13,606 / 12,699 / 10,518 / 7,570 tok/s (+13% / +19% / +22% / +23%) |
| The same with `--kv int8` | 12,822 / 12,070 / 10,644 / 8,637, then 14,610 / 14,828 / 13,214 / 10,991 tok/s (+14% / +23% / +24% / +27%) |
