# Gemma 4 precision

Where Gemma 4 rounds and how its output compares. Method: the [quality suite](quality-suite.md). Guide:
[Gemma 4](../models/gemma4.md).

## Summary

- **Against the BF16 model:** on the quality suite's full set the 12B (FP8) scores mean KLD 0.0126 and top-1 97.34%, the
  26B A4B (NVFP4 experts) 0.0293 and 96.25%, the 31B (NVFP4 MLP) 0.0319 and 96.34% ([below](#against-the-reference)).
- **Experts in RAM do not cost quality:** the 26B A4B on a 16 GB card, with a fifth of its experts on the CPU, scores as on
  a 32 GB card that holds them all.
- **Greedy tokens (26B A4B):** on four prompts, greedy tokens follow HF transformers' BF16 model on the CPU until a
  near-tie: two run to the end of the reference, one parts at token 18 of 40, one at token 3, where HF's two best tokens
  are tied. Wikitext KLD is not used for this model: the instruction model after BOS on raw text is ill-conditioned (two
  BF16 implementations differ by about 0.37 KLD there).
- **The dense models, same prompts:** the 12B (FP8) matches 101 of 134 reference tokens (two run to the end; the partings are
  0.25 and 0.38 nats apart in HF). The 31B (MLP NVFP4) matches 95 of 135: three run to the end, the fourth parts at its first
  token, where HF's two best are 1.25 apart; HF itself with the pack's NVFP4 MLP weights brings them to 0.13 apart, so the
  weights' rounding, not the engine's arithmetic, moves that pick.
- **Drafting does not change output.** Greedy drafted decoding gives plain decoding's tokens.
- **Batch windows do not change output.** A row's arithmetic does not depend on the rows beside it.

## Weights

| Tensors | Stored as | From the BF16 checkpoint |
| --- | --- | --- |
| Routed experts | NVFP4 (E2M1, an E4M3 scale per 16 values, an FP32 scale per matrix) | rounded once; the router's per-expert scale folded into the down scale |
| Routed experts of the `--experts-mxfp8` layers | MXFP8 (E4M3, a power-of-two scale per 32 values), the down rows zero-padded to whole 128-value blocks | rounded once, the router's per-expert scale folded into the down weights first |
| Attention and MLP projections | FP8 E4M3, an FP32 scale per matrix | rounded once; the MLP's inner width zero-padded to a multiple of 128 |
| MLP projections with `--mlp-nvfp4` (the 31B) | NVFP4 planes (E2M1, an E4M3 scale per 16 values, an FP32 scale per matrix) | rounded once, as the experts |
| Output head (tied embedding) | FP8 E4M3 rows, an FP32 scale per row | rounded once |
| Token embedding | BF16, in page-locked RAM | as stored |
| Router | F32, its input scale folded in | as stored |
| Norms, layer scalars | F32 | as stored |
| Assistant drafter's projections and head | FP8 E4M3 as above | drafting only: cannot change output |

## Where each stage rounds

| Stage | Format | Arithmetic |
| --- | --- | --- |
| Projections, MLP, head | FP8 E4M3 weights | decode: bf16 activations, fp32 sums; prompt chunks: activations rounded to E4M3 per row (scale amax / 448) on the FP8 tensor cores, fp32 accumulation |
| MLP with `--mlp-nvfp4` | NVFP4 planes | decode: bf16 activations, fp32 sums; prompt chunks: activations rounded to FP4 per row (one term, an E4M3 scale per 16 values, an FP32 scale per row) on the FP4 block-scaled tensor cores, fp32 accumulation |
| Routed experts | NVFP4 | decode: q8_1 activations, fp32 sums, the tail-scaled GELU gate in fp32; prompt chunks: two FP4 terms per activation row on the FP4 block-scaled tensor cores, the GELU gate in fp32, H rounded to two FP4 terms |
| Routed experts | MXFP8 | decode: q8_1 activations, fp32 sums, the GELU gate in fp32; prompt chunks: activations rounded to E4M3 per 32 values (a power-of-two scale) on the MXFP8 block-scaled tensor cores, the GELU gate in fp32, H rounded to fp16, then to E4M3 per 32 values |
| Routed experts in RAM | NVFP4, MXFP8 | on the CPU: Q8_0 activations, fp32 sums, the GELU gate in fp32; against the card's q8_1 path within about 1e-7 relative (`native_expert_parity --gemma4`) |
| Router | F32 | fp32 logits, softmax over the top 8 |
| Attention | fp16 or E4M3 caches | fp16 tensor cores with fp32 scores, softmax and sums; decode: the query as two fp16 terms (fp32-precise scores), probabilities fp16; prompt chunks: queries and probabilities fp16; a full layer's keys rotated from its stored values in fp32, then rounded to fp16 |
| Norms, residual, softcap | F32 | fp32 |

Prompt chunks round differently from decoded rows; a read of up to 16 tokens runs as decode windows and rounds as decoded rows.

## Attention caches

- **`--kv fp16` (default):** keys and values as fp16; a full layer whose keys are its values stores the normed values once.
- **`--kv fp8`:** E4M3 codes with a unit scale (keys and values are normed, so they fit without a scale).

## Against the reference

`basalt-bench --quality` against each model's vLLM BF16 reference ([quality suite](quality-suite.md)): the full set, assistant
rows, conversation-bootstrap error bars, one RTX 5090 at 400 W, drafting off, the prompt before a conversation's first scored
row read as prompt chunks and the rest in verify windows of 16. Conversations longer than 5,120 tokens (7 of 417) are skipped
at the default `--quality-max-context`.

| Pack | Conversations, rows | KLD mean | Top-1 | PPL ratio |
| --- | --- | ---: | ---: | ---: |
| 12B, FP8 | 410, 208,899 | 0.0126 ± 0.0006 | 97.34 ± 0.08% | 1.0138 |
| 26B A4B, NVFP4 experts, FP8 attention and MLP | 410, 216,477 | 0.0293 ± 0.0015 | 96.25 ± 0.12% | 1.0284 |
| 31B, NVFP4 MLP, FP8 attention | 410, 203,794 | 0.0319 ± 0.0014 | 96.34 ± 0.11% | 1.0338 |

For scale, Unsloth's UD-Q4_K_M GGUF of the 26B A4B (llama.cpp) scores 0.0146 ± 0.0008 and 97.40 ± 0.08% over all 417
conversations (220,423 rows): it includes the 7 long ones, so it is not the same row set.

The smoke slice (21 conversations, 11,864 rows) of the 26B A4B: 0.0236 ± 0.0045 and 96.92% on an RTX 5090 that holds every
expert; 0.0235 ± 0.0044 and 96.90% on an RTX 5060 Ti alone with 24 of each layer's 128 experts in RAM on the CPU.
