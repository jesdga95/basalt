# Qwen3.8-Flash-Next

Runs with the same command as any model: `basalt serve PACK`. What it supports next to DeepSeek: [models](index.md). Rounding and
quality: [precision](../precision/qwen38-flash-next.md).

## Pack

Nine packs on [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt), all named `Qwen3.8-FN-` plus:

- `NVFP4-DQ6_K` (default) and `NVFP4-DQ8`: NVIDIA's NVFP4 experts. `D` is the dense projections' format: Q6_K, or Q8_0 for DQ8.
  DQ8 is slower than DQ6_K and has a lower KLD ([precision](../precision/qwen38-flash-next.md#6-bit-dense-projections)).
- `UD-Q4_K_XL`, `UD-Q5_K_XL`, `UD-Q6_K_XL` and `Q8_0`: Unsloth's GGUF types.
- `GSQ-RCO-IQ3_S`, `GSQ-RCO-IQ3_XXS` and `GSQ-RCO-Q2_0`: GSQ-RCO quants.

Sizes are on the card. The layout, and how to build a pack: [model file](../model-file.md).

## Memory

- **Context:** `--max-context`, default 8,192. The KV cache grows with it. `--kv fp16` (default), `int8` (half the VRAM), `q4_0`
  and `k8v4` trade precision for VRAM ([precision](../precision/qwen38-flash-next.md#measured-costs)).
- **Experts:** the main card's cache holds the most routed ones, system RAM the rest, and an adaptive tier moves them as the
  conversation changes ([adaptive tier](../configuration/adaptive.md)).
- **Cards:** every card after the main one is an expert tier, up to three. Matched cards may split the layers instead
  ([GPUs](../configuration/gpus.md)).
- **Elastic KV:** with `--max-context` above 32,768 (and not with `--kv-resident`, `--vram-elastic` or a layer split), each session
  backs its first 32,768 cells at start. More cells come from the expert cache when a request needs them, and go back when it does
  not. An idle conversation's KV moves to RAM after 15 s. The startup line `elastic KV: ...` shows the figures.

## Long prompts

- `--prefill auto` reads up to 16,384 tokens per chunk (32,768 for NVFP4). The CPU computes 0.6 of the prompt experts that stream
  (`--prefill-cpu-share`).
- A fresh part of 64 tokens or fewer goes through decode windows (`--short-read`).
- With several cards, a long prompt can be read as a pipeline over them ([GPUs](../configuration/gpus.md#long-prompts-across-the-cards)).
- Prompts beside decoding slots: [running](../running.md#prompts-beside-decoding-slots).

## Drafting

On when the pack has the MTP drafter. `--spec N` sets the window (default 8: the last accepted token plus up to 7 drafts).
`--spec 0` turns it off. `--suffix-draft` (default 3) also fills a window from an earlier repeat in the context. Output is exactly
plain decoding's ([precision](../precision/qwen38-flash-next.md#drafting-does-not-change-output)). Structured output drafts best
and prose least.

## Parallel slots

`--parallel N` (1 to 8) serves N requests at once ([running](../running.md#concurrent-requests---parallel-n)).

## Lossy flags

Off by default. `--relaxed-drafting`, `--route-prior` and `--route-tail-skip` each change output, and each is named in one start
warning. What they do and what they cost: [precision](../precision/qwen38-flash-next.md#relaxed-drafting).

## Limits

- **No video.**
- **Two-card idle sleep** can drop the second card's experts until restart (seen at context 8,192 without elastic KV, not at 64K or
  262,144).
