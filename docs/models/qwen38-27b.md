# Qwen3.8-27B

Basalt's first dense model. Runs with the same command as any model: `basalt serve PACK`. What it supports next to the others:
[models](index.md). Rounding and quality: [precision](../precision/qwen38-27b.md).

## Pack

One file from NVIDIA's NVFP4 checkpoint ([nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)), with the
DFlash 2 block drafter ([z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)) when built with it:

- MLP and output head: NVIDIA's NVFP4, moved bit for bit with their tensor scales.
- GDN and attention projections: NVIDIA's FP8 values as Q8_0.
- The MTP layer and the block drafter: BF16 projections as Q8_0. Vision: the BF16 tower, as stored.
- The BF16 token embedding stays in page-locked RAM (2.4 GiB); everything else sits on the main card (17.2 GiB with the
  MTP layer, 19.1 GiB with the block drafter).

How to build it: [model file](../model-file.md#qwen38-27b-pack).

## Architecture

The `qwen35` architecture (Qwen3.5, 3.6 and 3.8 dense): 64 layers, 3 gated delta-net layers to 1 gated attention layer, a dense
SwiGLU FFN in each, interleaved M-RoPE for pictures. It runs on Basalt's shared dense stack (`src/dense`), which reads every
width from the pack; the GDN and attention kernels are shared with Flash-Next.

## Memory

- **Context:** `--max-context`, default 8,192, up to the model's 262,144. The KV cache is allocated at start for every slot:
  `--kv fp16` (default) takes 64 KiB per position over the 16 attention layers, `--kv int8` 33 KiB (int8 codes with an fp16
  scale per 64 values; [precision](../precision/qwen38-27b.md)). On a 32 GB card one slot holds about 150K positions in fp16
  and the full 262,144 in int8. The start refuses a context that does not fit and names the free memory.
- **Slots:** `--parallel N` gives each slot its own cache and recurrent state (157 MiB). Slots share the weights.
- **Cards:** one. The second card idles (see [GPUs](../configuration/gpus.md)).

## Drafting

On by default with the pack's drafter; `--spec N` caps the window (default 8, at most 16), `--spec 0` turns it off. The block
drafter runs its block of 8 rows, and a wider block (up to the window) after windows whose drafts were all taken; the draft
policy reads each draft's confidence to pick the window. Without it the MTP layer drafts one token per step. `--suffix-draft N` (default 3, `0` = off)
also fills a window from an earlier repeat of at least N tokens in the context, where the draft policy expects it to pay.
Output is exactly plain decoding's: a row's arithmetic does not depend on the window's rows, drafted or not, alone or batched.

## Images and video

`--vision dynamic`: the encoder's weights wait in page-locked RAM and go to the card for each request's pictures. Video works as
for Flash-Next ([video](../running.md#video)); frames decode on the CPU.

## Limits

- Slots whose windows fit 16 rows together (up to 16 slots plain, two slots of 8-row drafted windows) share one pass through the weights each round, and their block drafters run two blocks a pass; a window that does not fit takes a pass of its own. Prompt reads run per slot.
- No layer split and no second-card drafting.
