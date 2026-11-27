# Gemma 4 precision

Where Gemma 4 rounds and how its output compares. Method: [quality](quality.md). Guide: [Gemma 4](../models/gemma4.md).

## Summary

- **Against the reference (26B A4B):** on four prompts, greedy tokens follow HF transformers' BF16 model on the CPU until a
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
| Router | F32 | fp32 logits, softmax over the top 8 |
| Attention | fp16 or E4M3 caches | fp16 tensor cores with fp32 scores, softmax and sums; decode: the query as two fp16 terms (fp32-precise scores), probabilities fp16; prompt chunks: queries and probabilities fp16; a full layer's keys rotated from its stored values in fp32, then rounded to fp16 |
| Norms, residual, softcap | F32 | fp32 |

Prompt chunks round differently from decoded rows; a read of up to 16 tokens runs as decode windows and rounds as decoded rows.

## Attention caches

- **`--kv fp16` (default):** keys and values as fp16; a full layer whose keys are its values stores the normed values once.
- **`--kv fp8`:** E4M3 codes with a unit scale (keys and values are normed, so they fit without a scale).
