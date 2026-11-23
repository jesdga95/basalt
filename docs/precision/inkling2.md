# Inkling 2 Mini precision

Where Inkling 2 Mini rounds and how its output compares. Method: [quality](quality.md). Guide: [Inkling 2 Mini](../models/inkling2.md).

## Summary

- **Against the reference:** the default pack (`NVFP4-MX8`, the release's MXFP8 experts native) scores mean KLD 0.00297 and top-1 97.83% over 100 chunks, and 0.00303 and 97.72% with its RAM experts on the CPU; the `NVFP4-DQ8` pack 0.00305 and 97.76% ([below](#against-the-reference)).
- **INT8 KV:** on the DQ8 pack, the final rebased 100-chunk run scored mean KLD 0.00343 with 97.47% top-1; its row is 144 bytes versus 256 for FP16.
- **At 256k context:** FP8 saves 4.25 GiB and INT8 saves 3.74 GiB of total VRAM versus FP16 with the same expert placement ([measurements](#earlier-vram-use-at-256k-context)).
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
- **`--kv int8`:** signed INT8 codes, with an FP16 scale for each 64-channel group: the scale is the group's largest
  magnitude over 127 rounded to fp16, a code the value over it rounded to nearest even and clamped to ±127. A row uses 144
  bytes (128 codes, 2 scales, 12 zero bytes), about 44% less than FP16's 256 bytes and 12.5% more than FP8. Decode scores
  keys in fp32 from code × scale (exact), values go to fp16 tiles rounded once from the same product.

## Against the reference

The base: the official release with its routed experts dequantized to BF16 (llama.cpp 29b52d398, `convert_hf_to_gguf
--outtype bf16`), through `llama-perplexity`, wikitext-2, 100 chunks of 512, `-b 8192 -ub 8192`; `inkling2-bf16-base.kld`, base
PPL 7.0714. transformers' implementation on the CPU agrees with it (chunks 0 and 1: KLD 0.0017 and 0.0019, top-1 98.4% and
97.7%).

| Run | Mean KLD | Top-1 | PPL |
| --- | ---: | ---: | ---: |
| DQ8 pack, 100 chunks, `--kv fp16` (final rebased build, 475 W) | 0.00304 | 97.81% | 7.0737 (base 7.0644 on these rows) |
| DQ8 pack, 100 chunks, `--kv fp8` (final rebased build, 475 W) | 0.00452 | 97.04% | 7.0766 (base 7.0644 on these rows) |
| DQ8 pack, 100 chunks, `--kv int8` (final rebased build, 475 W) | 0.00343 | 97.47% | 7.0788 (base 7.0644 on these rows) |
| DQ8 pack, 100 chunks, `--kv fp8` (pre-rebase result) | 0.00443 | 97.29% | 7.0702 (base 7.0644) |
| DQ8 pack, 100 chunks, `--kv int8` (pre-rebase result) | 0.00344 | 97.79% | 7.0762 (base 7.0644) |
| DQ8 pack, 20 chunks (the quick gate) | 0.00264 | 97.80% | 7.6642 (base 7.6534) |
| DQ8 pack, 90 chunks, RTX 5060 Ti alone, 1,047 experts in RAM on the CPU | 0.00298 | 97.70% | |
| DQ8 pack, the same 90 chunks, the whole model on the card | 0.00304 | 97.76% | |
| DQ8 pack, 100 chunks, `gpu-mode split`: layers 0 to 11 on the RTX 5090, 12 to 23 on an RTX 5060 Ti | 0.00304 | 97.70% | 7.0718 (base 7.0633) |
| MX8 pack, 100 chunks, the whole model on the card | 0.00297 | 97.83% | |
| MX8 pack, 20 chunks | 0.00265 | | |
| MX8 pack, 100 chunks, experts in RAM on the CPU | 0.00303 | 97.72% | |
| MX8 pack, 100 chunks, `--expert-cache 1500`, one card with the spilled experts on the CPU | 0.00444 | 97.15% | |
| MX8 pack, 100 chunks, `--expert-cache 1500`, `gpu-mode peer` (the spilled experts on the second card) | 0.00440 | 97.25% | |
| 100 chunks, `--kv int8` (#552's build) | 0.00344 | 97.79% | 7.0762 (base 7.0644) |
| 100 chunks, `--kv fp8` (#552's build) | 0.00443 | 97.29% | 7.0702 (base 7.0644) |
| 100 chunks, `--kv fp16` (#552's build), the same set of runs as the two rows above | 0.00310 | 97.78% | 7.0695 (base 7.0644) |

The final rebased quality rows above use the DQ8 pack `/mnt/storage/Projects/basalt/models/Inkling-2-Mini-NVFP4-q8.basalt` and the BF16-derived reference `/mnt/storage/kld/inkling2-q6cr-eval/inkling2-bf16-base.kld`, with the same 100 wikitext-2 chunks (25,500 scored rows), T=3, and RTX 5090 power limit 475 W. The scored-row base PPL was 7.0644 for each run. Earlier quality rows and 256K VRAM readings were measured at 400 W unless noted. On the RTX 5060 Ti, the CPU computes experts in RAM (Ryzen 9 9950X, AVX-512).

## Rebased decode: INT8 vs FP8

RTX 5090 at 475 W. One cold process per measurement, same standard `conv` prompt, fixed seed 20261122, `--max-context 32768`, `--spec 5`, and `--expert-cache 3072`; each run generated 1,200 tokens. ABBA order was INT8, FP8, FP8, INT8. Throughput is end-to-end generation throughput; acceptance varied between runs, so this is not an isolated KV-kernel comparison.

| Run | KV format | tok/s | Accept % | Draft len |
| --- | --- | ---: | ---: | ---: |
| A1 | INT8 | 381.8 | 67.2% | 1.48 |
| B1 | FP8 | 388.6 | 62.4% | 1.77 |
| B2 | FP8 | 388.9 | 63.1% | 1.75 |
| A2 | INT8 | 383.8 | 67.2% | 1.54 |
| Mean | INT8 | 382.8 | — | — |
| Mean | FP8 | 388.8 | — | — |

FP8 was 1.6% faster in this short-context decode ABBA.

## Pinned-placement decode windows

Three `basalt-bench -t conv -r 3` repetitions per format, same q8 pack, seed 20261122, 32K context, speculative depth 5, and 3,070 expert slots on the RTX 5090 at 475 W. Every run reported 99.9998% VRAM expert hits. These are end-to-end verify-window measurements; draft acceptance differed across formats, so they do not isolate KV kernel cost.

| KV format | Mean tok/s | Mean ms/window | Mean passes/s | Mean acceptance |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 402.9 | 5.53 | 181.0 | 70.2% |
| FP8 | 386.4 | 5.40 | 185.4 | 64.6% |
| INT8 | 387.9 | 5.26 | 190.3 | 67.6% |

A kernel-only timing from `inkling2_kernels_parity`, one row and 2,998 keys on the same RTX 5090 at 475 W: FP16 12.5 µs, FP8 10.1 µs, INT8 11.7 µs. CUDA 13.3 `nvcc --time` phase durations for the changed translation units are listed below.

## 16K prefill memory check

RTX 5090 at 475 W. The standard prefill fixture requested with `-p 16384` contains 15,810 tokens. Each run used `--max-context 32768`, `--max-new 1`, and `--spec 0`. The FP16 illegal access reproduced on the PR before the fix; current main and the fixed PR both pass FP16. These are single-run historical tok/s figures, so the apparent 3.3% FP8 drop remains unresolved without repeated pinned-placement `basalt-bench` window latency and passes/s.

CUDA 13.3 `nvcc --time` phase durations for the changed translation units (cicc / ptxas, ms; sm_120):

| File | cicc | ptxas |
| --- | ---: | ---: |
| `attn.cu` | 2282.8 | 1294.7 |
| `attn_prompt.cu` | 569.8 | 179.1 |
| `glue.cu` | 777.2 | 281.3 |
| `ops.cu` | 604.4 | 294.2 |
| `prompt_ops.cu` | 396.4 | 154.0 |
| `kv_int8_parity.cu` | 316.8 | 20.8 |

| Revision | KV format | Prefill tok/s | Result |
| --- | --- | ---: | --- |
| Main at tested revision (`8667cc8abd`) | FP16 | 17,653 | Pass |
| Main at tested revision (`8667cc8abd`) | FP8 | 18,258 | Pass |
| Rebased PR, before staging fix | FP16 | — | Failed during request |
| Rebased PR, final head (`df17c5807b`) | FP16 | 17,639 | Pass |
| Rebased PR, final head (`df17c5807b`) | FP8 | 17,647 | Pass |
| Rebased PR, final head (`df17c5807b`) | INT8 | 17,118 | Pass |

The regression was specific to the PR branch's FP16 staging allocation: it used the 144-byte INT8 row size instead of FP16's 256-byte row size. Sizing staging for the widest format fixes it.

## Fully resident 512K capacity

These runs allocated the full native 512K cache (`--max-context 524288`) on the RTX 5090 at 475 W, with the same q8 model and `--expert-cache 3072`; `--kv-resident` stayed at its default, so the cache remained fully resident. The 12-token `conv.ids` prompt started generation near position 12, and each run generated 1,200 tokens (ending near position 1,212). They demonstrate resident capacity, not decoding at position 524,288.

| KV format | Decode tok/s | Accept % | Draft len | Result |
| --- | ---: | ---: | ---: | --- |
| FP8 | 353.8 | 60.9% | 1.93 | Pass |
| INT8 | 313.0 | 67.2% | 1.51 | Pass |

## Earlier VRAM use at 256k context

Measured after `READY 262144`, before a prompt, on an RTX 5090 at 400 W. The full context capacity was allocated at startup; expert placement used `--expert-cache 3072` (3,070 GPU slots and 2 in RAM). Idle `nvidia-smi` use was 1,895 MiB.

| KV format | Total VRAM used | Above idle | Saved vs FP16 |
| --- | ---: | ---: | ---: |
| FP16 | 30,192 MiB (29.48 GiB) | 28,297 MiB | — |
| FP8 | 25,840 MiB (25.23 GiB) | 23,945 MiB | 4,352 MiB (4.25 GiB) |
| INT8 | 26,366 MiB (25.75 GiB) | 24,471 MiB | 3,826 MiB (3.74 GiB) |

The previous INT8 row-writer parity matched the CPU reference for 257 rows of width 128. Dequantized max absolute error was 0.00559; the microbench took 2.81 µs for 4,096 rows at 957 GB/s.

## Drafting

The DSpark drafter only proposes; the target checks every draft, and a decode row rounds the same in any window. Greedy output
with drafting equals plain greedy output (prose, struct and conversation, after short and 16K-token prompts, and a
2,405-token prompt checked).
