# Inkling 2 Mini precision

Where Inkling 2 Mini rounds and how its output compares. Method: [quality](quality.md). Guide: [Inkling 2 Mini](../models/inkling2.md).

## Summary

- **Against the reference:** the default pack (`NVFP4-DQ8`) scores mean KLD 0.00305 and top-1 97.76% over 100 chunks; the
  `NVFP4-MX8` pack, with the release's MXFP8 experts native, 0.00297 and 97.83%, and 0.00303 and 97.72% with its RAM experts
  on the CPU ([below](#against-the-reference)).
- **The reference is not a true BF16 model.** No BF16 checkpoint exists: the release ships its routed experts in NVFP4 and
  MXFP8. The base is the official release with its experts dequantized to BF16, so its KLD is not comparable with the other
  models' true-BF16 KLDs.
- **The experts are the release's bits.** Layers 0 to 19 keep the release's NVFP4 codes and scales as they are; layers 20 to 23 keep the release's MXFP8 codes and UE8M0 scales bit for bit in the MX8 pack, and as Q8_0 of those values in the DQ8 pack.
- **Drafting does not change output.** Greedy drafted decoding gives plain decoding's tokens.

## Weights

The pack, from the official release ([guide](../models/inkling2.md#pack)). Every rounding is deterministic and runs once in the
converter.

| Tensors | Stored as | From the release |
| --- | --- | --- |
| Routed experts, layers 0 to 19 | NVFP4 (E2M1, an E4M3 scale per 16 values, an FP32 scale per matrix) | bit for bit; the FP32 scale is the release's amax over its denominator, rounded once |
| Routed experts, layers 20 to 23 | MX8 pack: MXFP8 (E4M3, a UE8M0 scale per 32 values, type 46). DQ8 pack: Q8_0 | MX8: bit for bit. DQ8: the release's values rounded to Q8_0 by ggml's reference rounding (0.54% RMS error) |
| Attention, gate, short-convolution and output projections | Q8_0 | ggml's reference rounding of the BF16 values |
| Output head | BF16 | as stored; the engine rounds it to Q8_0 as it loads |
| Token embedding | BF16, in page-locked RAM | as stored |
| Norms, attention-residual queries, router, biases, convolution taps, relative tables, clip scales | F32 | as stored |
| Loop controllers | BF16 | as stored |
| Vision encoder and projector | BF16 | as stored |
| DSpark drafter | BF16 in the file; projections rounded to Q8_0 as they load | drafting only: cannot change output |

The release marks NVFP4 activation quantization as required (its own serving runs W4A4); transformers and llama.cpp run
dequantized weights, and so does the reference.

Formats measured for layers 20 to 23 over 20 chunks: native MXFP8 0.00265 KLD, Q8_0 of the MXFP8 values 0.00256 KLD, NVFP4 0.00431. Quantizing `w.proj`/`w.out` to MXFP8 raised 20-chunk KLD to 0.00421 (97.16% top-1, 4.9x weight L2 error vs Q8_0), so `w.proj`/`w.out` remain Q8_0. Keeping attention in BF16 instead of Q8_0 scored 0.00218 for 1.2 GB more reads a token; it was not taken.

## Where each stage rounds

| Stage | Format | Arithmetic |
| --- | --- | --- |
| Routed experts | NVFP4, MXFP8 (Q8_0 in the DQ8 pack) | decode: q8_1 activations, fp32 sums (MXFP8 decoded to fp32 per block, matching CPU AVX2/AVX-512); experts in RAM on the CPU: Q8_0 activations, fp32 sums (MXFP8 codes decoded to fp32 by F16C, a weight block once for all the tokens of a window, every token's dot the same bits as the table-lookup scalar row on AVX2 and AVX-512), the card's softcapped SwiGLU; prompt chunks: NVFP4 layers on FP4 block-scaled tensor cores (two FP4 terms per activation row, fp16 SwiGLU/down intermediates), MXFP8 layers on block-scaled MXFP8 tensor cores (E4M3 activations + UE8M0 scales, fp32 accumulation, fp16 SwiGLU intermediate); older Q8_0 layers via MMQ |
| Q8_0 projections (`w.proj`, `w.out`), head | Q8_0 (loaded into `q8_planes` for `w.proj`/`w.out`) | q8_1 activations via `native_q8_planes_mmvq` (decode, bitwise `native_q8_0_mmvq`); block-32 Q8_0 2-CTA TMA int8 tensor cores (`prefill::q8`, prompt); fp32 logits, softcap 30 |
| Router | F32 | fp32 sigmoid, top 8 |
| Attention-residual mixes, controllers, norms | F32, BF16 weights | fp32 |
| Attention caches | fp16 (`--kv fp8`: E4M3, unit scale) | decode: fp32 scores and softmax; prompt: fp16 tensor cores, fp32 accumulation |
| Short-convolution state | fp32 | fp32 |

A decode row's arithmetic does not depend on how many rows share its window. Prompt chunks round differently from decoded rows.

The prompt path's NVFP4 experts, measured over 20 chunks: one FP4 term per activation (the release's W4A4) 0.00407 KLD, two
terms 0.00264; the same chunks through MMQ (q8_1 activations) 0.00252. Two terms were taken: the prompt reads about 1.5 times
as fast as through MMQ.

## Attention caches

- **`--kv fp16` (default):** keys and values as fp16.
- **`--kv fp8`:** E4M3 codes with a unit scale. Keys are L2-clipped and values RMS-normed by construction, so they fit E4M3's
  range without a scale. It halves a position's bytes.

## Against the reference

The base: the official release with its routed experts dequantized to BF16 (llama.cpp 29b52d398, `convert_hf_to_gguf
--outtype bf16`), through `llama-perplexity`, wikitext-2, 100 chunks of 512, `-b 8192 -ub 8192`; `inkling2-bf16-base.kld`, base
PPL 7.0714. transformers' implementation on the CPU agrees with it (chunks 0 and 1: KLD 0.0017 and 0.0019, top-1 98.4% and
97.7%).

| Run | Mean KLD | Top-1 | PPL |
| --- | ---: | ---: | ---: |
| DQ8 pack, 100 chunks, `--kv fp16` | 0.00305 | 97.76% | 7.0695 (base 7.0633 over the scored rows) |
| DQ8 pack, 20 chunks (the quick gate) | 0.00264 | 97.80% | 7.6642 (base 7.6534) |
| DQ8 pack, 90 chunks, RTX 5060 Ti alone, 1,047 experts in RAM on the CPU | 0.00298 | 97.70% | |
| DQ8 pack, the same 90 chunks, the whole model on the card | 0.00304 | 97.76% | |
| DQ8 pack, 100 chunks, layer split: 0 to 11 on the RTX 5090, 12 to 23 on an RTX 5060 Ti | 0.00304 | 97.70% | 7.0718 (base 7.0633) |
| MX8 pack, 100 chunks, the whole model on the card | 0.00297 | 97.83% | |
| MX8 pack, 20 chunks | 0.00265 | | |
| MX8 pack, 100 chunks, experts in RAM on the CPU | 0.00303 | 97.72% | |

RTX 5090 at 400 W, the whole model on the card, unless the row says otherwise. On the 5060 Ti the CPU computes the experts in
RAM (a Ryzen 9 9950X, AVX-512).

## Drafting

The DSpark drafter only proposes; the target checks every draft, and a decode row rounds the same in any window. Greedy output
with drafting equals plain greedy output (prose, struct and conversation, after short and 16K-token prompts, and a
2,405-token prompt checked).
