# Inkling 2 Mini precision

Where Inkling 2 Mini rounds and how its output compares. Method: [quality](quality.md). Guide: [Inkling 2 Mini](../models/inkling2.md).

## Summary

- **Against the reference:** over 100 WikiText-2 chunks, FP16 scores mean KLD 0.00310 and INT8 0.00344, with 97.78% and 97.79% top-1 agreement respectively ([below](#against-the-reference)).
- **At 256k context:** FP8 saves 4.25 GiB and INT8 saves 3.74 GiB of total VRAM versus FP16 with the same expert placement ([measurements](#vram-at-256k-context)).
- **The reference is not a true BF16 model.** No BF16 checkpoint exists: the release ships its routed experts in NVFP4 and
  MXFP8. The base is the official release with its experts dequantized to BF16, so its KLD is not comparable with the other
  models' true-BF16 KLDs.
- **The experts are the release's bits.** Layers 0 to 19 keep the release's NVFP4 codes and scales as they are.
- **Drafting does not change output.** Greedy drafted decoding gives plain decoding's tokens.

## Weights

The pack, from the official release ([guide](../models/inkling2.md#pack)). Every rounding is deterministic and runs once in the
converter.

| Tensors | Stored as | From the release |
| --- | --- | --- |
| Routed experts, layers 0 to 19 | NVFP4 (E2M1, an E4M3 scale per 16 values, an FP32 scale per matrix) | bit for bit; the FP32 scale is the release's amax over its denominator, rounded once |
| Routed experts, layers 20 to 23 | Q8_0 | the release's MXFP8 values (E4M3, a UE8M0 scale per 32) rounded to Q8_0 by ggml's reference rounding (0.54% RMS error) |
| Attention, gate, short-convolution and output projections | Q8_0 | ggml's reference rounding of the BF16 values |
| Output head | BF16 | as stored; the engine rounds it to Q8_0 as it loads |
| Token embedding | BF16, in page-locked RAM | as stored |
| Norms, attention-residual queries, router, biases, convolution taps, relative tables, clip scales | F32 | as stored |
| Loop controllers | BF16 | as stored |
| Vision encoder and projector | BF16 | as stored |
| DSpark drafter | BF16 in the file; projections rounded to Q8_0 as they load | drafting only: cannot change output |

The release marks NVFP4 activation quantization as required (its own serving runs W4A4); transformers and llama.cpp run
dequantized weights, and so does the reference.

Formats measured for layers 20 to 23 over 20 chunks: Q8_0 of the MXFP8 values 0.00256 KLD, NVFP4 0.00431. Keeping attention in
BF16 instead of Q8_0 scored 0.00218 for 1.2 GB more reads a token; it was not taken.

## Where each stage rounds

| Stage | Format | Arithmetic |
| --- | --- | --- |
| Routed experts | NVFP4, Q8_0 | decode: q8_1 activations, fp32 sums; prompt chunks: NVFP4 layers on the FP4 block-scaled tensor cores, each activation row as two FP4 terms under one per-row scale (its rounding and the rounding of what that leaves, about 8 bits), fp32 accumulation; Q8_0 layers MMQ (int8 mma); each matrix's FP32 scale applied after |
| Q8_0 projections, head | Q8_0 | q8_1 activations (decode); MMQ (prompt); fp32 logits, softcap 30 |
| Router | F32 | fp32 sigmoid, top 8 |
| Attention-residual mixes, controllers, norms | F32, BF16 weights | fp32 |
| Attention caches | fp16, int8 (`--kv int8`: signed codes with an FP16 scale per 64 channels), or fp8 (E4M3, unit scale) | decode: fp32 scores and softmax; prompt: values dequantized to fp16 tiles, fp32 accumulation |
| Short-convolution state | fp32 | fp32 |

A decode row's arithmetic does not depend on how many rows share its window. Prompt chunks round differently from decoded rows.

The prompt path's NVFP4 experts, measured over 20 chunks: one FP4 term per activation (the release's W4A4) 0.00407 KLD, two
terms 0.00264; the same chunks through MMQ (q8_1 activations) 0.00252. Two terms were taken: the prompt reads about 1.5 times
as fast as through MMQ.

## Attention caches

- **`--kv fp16` (default):** keys and values as fp16.
- **`--kv fp8`:** E4M3 codes with a unit scale. Keys are L2-clipped and values RMS-normed by construction, so they fit E4M3's
  range without a scale. A row uses 128 bytes.
- **`--kv int8`:** signed INT8 codes, with an FP16 scale for each 64-channel group. A row uses 144 bytes, about 44% less than
  FP16's 256 bytes and 12.5% more than FP8.

## Against the reference

The base: the official release with its routed experts dequantized to BF16 (llama.cpp 29b52d398, `convert_hf_to_gguf
--outtype bf16`), through `llama-perplexity`, wikitext-2, 100 chunks of 512, `-b 8192 -ub 8192`; `inkling2-bf16-base.kld`, base
PPL 7.0714. transformers' implementation on the CPU agrees with it (chunks 0 and 1: KLD 0.0017 and 0.0019, top-1 98.4% and
97.7%).

| Run | Mean KLD | Top-1 | PPL |
| --- | ---: | ---: | ---: |
| 100 chunks, `--kv fp16` | 0.00310 | 97.78% | 7.0695 (base 7.0644 over the scored rows) |
| 100 chunks, `--kv fp8` | 0.00443 | 97.29% | 7.0702 (base 7.0644) |
| 100 chunks, `--kv int8` | 0.00344 | 97.79% | 7.0762 (base 7.0644) |
| 20 chunks (the earlier quick gate) | 0.00264 | 97.80% | 7.6642 (base 7.6534) |

The FP16 and FP8 quality runs were on an RTX 5090 at 400 W. INT8's 100-chunk quality run and the decode comparison were at 475 W, with the whole model on the card. The fixed-seed ABBA decode comparison (one cold repetition per process) averaged 5.20 / 7.61 / 4.75 ms per window and 192.3 / 131.4 / 210.7 passes/s for INT8 (conv / struct / prose), versus 5.43 / 7.61 / 4.77 ms and 184.3 / 131.4 / 209.6 passes/s for FP16. Completion lengths and draft acceptance varied, so tok/s is not a clean comparison.

A second fixed-context comparison isolates FP8 from FP16: RTX 5090 at 475 W, `--max-context 32768`, `--spec 5`, one cold `conv` fixture per process, fixed seed 1, and `--expert-cache 3072` (3,070 expert slots on the GPU and 2 in RAM for every run). The order was FP16–FP8–FP8–FP16. The prompt and context capacity were identical; cache formats produced different completions, so window time and passes/s are the comparable decode measures.

The fixture is the standard short `conv` prompt, so this compares decode at the same 32k capacity and prompt, not decode after filling 32k tokens. Attempts to run a 16k active-context WikiText prompt or the 16k prefill fixture exited with `f32_rows_dot: an illegal memory access was encountered`; those runs were excluded.

| KV format | Mean ms/window | Mean passes/s | Mean tok/s | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 5.406 | 185.0 | 384.6 | 1,158 |
| FP8 | 5.524 | 181.1 | 415.5 | 1,200 |

FP8 averaged 2.2% more time per decode window (2.1% fewer passes/s). Its higher tok/s reflects the longer completion: FP16 stopped at 1,158 tokens while FP8 reached the 1,200-token limit. The two FP16 runs measured 5.409 and 5.403 ms/window; FP8 measured 5.615 and 5.433, so the within-format spread is visible in the table's two-run means.

## VRAM at 256k context

Measured after `READY 262144`, before any prompt was submitted, on the same RTX 5090 and model. `--expert-cache 3072` pinned placement to 3,070 GPU expert slots plus 2 in RAM; the cache allocation reserves the full context capacity at startup. `nvidia-smi` idle use was 1,895 MiB, and each format was measured in a fresh process.

| KV format | Total VRAM used | Above idle | Saved vs FP16 |
| --- | ---: | ---: | ---: |
| FP16 | 30,192 MiB (29.48 GiB) | 28,297 MiB | — |
| FP8 | 25,840 MiB (25.23 GiB) | 23,945 MiB | 4,352 MiB (4.25 GiB) |
| INT8 | 26,366 MiB (25.75 GiB) | 24,471 MiB | 3,826 MiB (3.74 GiB) |

At the same context capacity and expert placement, FP8 used 4.25 GiB less total VRAM than FP16; INT8 used 3.74 GiB less. FP8 used 526 MiB less than INT8 for this model's compressed attention cache.

The INT8 row writer parity test matched the CPU reference for all 257 rows of width 128. Dequantized max absolute error was 0.00559 (FP16 rounding: 0.000485); max relative error for values with |x| >= 0.01 was 0.332 (FP16: 0.000487). Its microbench took 2.81 µs for 4,096 rows at 957 GB/s.

## Drafting

The DSpark drafter only proposes; the target checks every draft, and a decode row rounds the same in any window. Greedy output
with drafting equals plain greedy output (prose, struct and conversation, after short and 16K-token prompts, and a
2,405-token prompt checked).
