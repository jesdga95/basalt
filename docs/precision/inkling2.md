# Inkling 2 Mini precision

Where Inkling 2 Mini rounds and how its output compares. Guide: [Inkling 2 Mini](../models/inkling2.md).
Method: the [quality suite](quality-suite.md) (preferred). The KLD, PPL and top-1 figures on this page were measured with [wikitext KLD](quality.md#wikitext-kld).

## Summary

- **Against the reference:** the default pack (`NVFP4-MX8`, the release's MXFP8 experts native) scores mean KLD 0.00297 and top-1 97.83% over 100 chunks, and 0.00303 and 97.72% with its RAM experts on the CPU; the `NVFP4-DQ8` pack 0.00305 and 97.76% ([below](#against-the-reference)).
- **INT8 KV:** on the DQ8 pack, the 100-chunk run scored mean KLD 0.00343 with 97.47% top-1; its row is 144 bytes versus 256 for FP16.
- **At 256k context:** FP8 saves 4.25 GiB and INT8 saves 3.74 GiB of total VRAM versus FP16 with the same expert placement ([measurements](#vram-use-at-256k-context)).
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
| DSpark drafter | BF16 in the file; projections rounded to Q8_0 as they load; its full layer's K/V fp16, int8 with `--kv int8` (the target's int8 quantization rule, its own dense layout), its window layers' rings fp16 | drafting only; the target verifies every token. Its cache size feeds the expert placement ([Drafting](#drafting)) |

The release marks NVFP4 activation quantization as required (its own serving runs W4A4); transformers and llama.cpp run
dequantized weights, and so does the reference.

Formats measured for layers 20 to 23 over 20 chunks: native MXFP8 0.00265 KLD, Q8_0 of the MXFP8 values 0.00256 KLD, NVFP4 0.00431. Quantizing `w.proj`/`w.out` to MXFP8 raised 20-chunk KLD to 0.00421 (97.16% top-1, 4.9x weight L2 error vs Q8_0), so `w.proj`/`w.out` remain Q8_0. Keeping attention in BF16 instead of Q8_0 scored 0.00218 for 1.2 GB more reads a token; it was not taken.

## Where each stage rounds

| Stage | Format | Arithmetic |
| --- | --- | --- |
| Routed experts | NVFP4, MXFP8 (Q8_0 in the DQ8 pack) | decode: q8_1 activations, fp32 sums (MXFP8 decoded to fp32 per block, matching CPU AVX2/AVX-512); experts in RAM on the CPU: Q8_0 activations, fp32 sums (MXFP8 codes decoded to fp32 by F16C, a weight block once for all the tokens of a window, every token's dot the same bits as the table-lookup scalar row on AVX2 and AVX-512), the card's softcapped SwiGLU; prompt chunks: NVFP4 layers on FP4 block-scaled tensor cores (two FP4 terms per activation row, fp16 SwiGLU/down intermediates), MXFP8 layers on block-scaled MXFP8 tensor cores (E4M3 activations + UE8M0 scales, fp32 accumulation, fp16 SwiGLU intermediate); Q8_0 layers via MMQ |
| Q8_0 projections (`w.proj`, `w.out`), head | Q8_0 (loaded into `q8_planes` for `w.proj`/`w.out`) | q8_1 activations via `native_q8_planes_mmvq` (decode, bitwise `native_q8_0_mmvq`); block-32 Q8_0 2-CTA TMA int8 tensor cores (`prefill::q8`, prompt); fp32 logits, softcap 30 |
| Router | F32 | fp32 sigmoid, top 8 |
| Attention-residual mixes, controllers, norms | F32, BF16 weights | fp32 |
| Attention caches | fp16, int8 (`--kv int8`: signed codes with an FP16 scale per 64 channels), or fp8 (E4M3, unit scale) | decode: fp32 scores and softmax (a full int8 cache at 32,768 keys and more: fp16 tensor-core tiles, [below](#attention-caches)); prompt: values dequantized to fp16 tiles, fp32 accumulation |
| Short-convolution state | fp32 | fp32 |

A decode row's arithmetic does not depend on how many rows share its window. Prompt chunks round differently from decoded rows;
a read of up to 16 tokens (64 with experts in RAM) runs as decode windows and rounds as decoded rows.

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
  keys in fp32 from code × scale (exact), values go to fp16 tiles rounded once from the same product. Decode rows of a
  full-attention cache at 32,768 keys and more read it on the tensor cores instead: the key codes are staged to fp16
  exactly and each 64-channel group's fp32 QK partial takes its scale after the accumulation; queries and the dequantized
  values round to fp16; the probabilities are scaled by 2^15 before they round to fp16 for the value product, and the fp32
  numerator drops that scale before the split merge, while the softmax denominator sums the unscaled fp32 probabilities. A
  row's arithmetic is the same alone and in a verify window at the same key count, but not bit-identical to the scalar
  reader, which shallower rows and the sliding-window layers keep.
- **Slices:** both INT8 readers split a full-cache row's keys into slices of whole 128-key passes, sized by the row's own
  key count, and merge them in fp32 in slice order. The most slices an INT8 row takes follows the card's multiprocessor
  count (the count whose launches end in whole waves, 63 on an 84-SM card, logged at load as `attention slices`). A row
  of more than 128 keys a slice at that count can take larger slices than at 64: at 63, exactly the rows that would take
  64 slices; at a smaller count, more rows. Moving the slice boundaries changes the fp32 merge order and, in the
  tensor-core reader, the running maximum a tile's probabilities are taken against before they round to fp16, so such a
  row can round differently from a card with another count. On one card, a row's bits do not depend on its window. FP16
  and FP8 caches take at most 64 slices on every card.

### Cost of the tensor-core INT8 reader

The tensor-core reader (full-attention rows of an INT8 cache at 32,768 keys and more) rounds the query and the softmax probabilities to fp16 where the scalar reader keeps fp32, so it is a distinct rounding and costs a little quality. It acts only under `--kv int8`; `--kv fp16` and `--kv fp8` keep their own tensor-core reader and shallower INT8 rows the scalar reader, unchanged.

Two measurements against the scalar INT8 reader of the same build, with the same weights and forced tokens (and the same expert placement, except where a row says otherwise).

The BF16-reference gate cannot reach the shipped path (its rows are 512 keys, below the cutoff), so it runs as a stress probe: a build with the cutoff at 1 sends every row through the tensor-core reader. The deltas are paired per scored row.

| Probe (DQ8 pack, RTX 5080, 360 W, 100 chunks of 512, T=3) | Mean KLD | Top-1 | Paired dNLL | PPL |
| --- | ---: | ---: | ---: | ---: |
| `--kv int8`, scalar reader | 0.00343 | 97.51% | | 7.0762 |
| tensor-core reader on every row, change | +0.00011 ± 0.00006 | -0.12 pt | +0.00036 ± 0.00046 | +0.036% |

The depth the probe cannot reach is measured by scoring 1,000 forced tokens after each of 8 distinct 131,072-token prompts (a code corpus under a chat framing: the absolute NLL is high and only the paired differences carry meaning). The paired dNLL is the 95% interval over the 8,000 tokens; the scalar reader run twice scores byte-identically, so a difference is the reader and not run-to-run noise. A flip is a token whose NLL differs by more than 5 nats.

| Deep comparison (DQ8 pack, RTX 5080, 360 W, 131,072 keys, 8 × 1,000 tokens) | Paired dNLL | Same top token | p90 \|dNLL\| | Flips (of 8,000) |
| --- | ---: | ---: | ---: | ---: |
| tensor-core reader against scalar INT8 | +0.0024 ± 0.025 | 93.92% | 0.37 | 102 |
| `--kv fp16` against scalar INT8 (fp16 plans 2,033 experts in RAM, INT8 plans 1,614) | -0.0055 ± 0.028 | 93.08% | 0.48 | 130 |

56 of the flips are at the same token in both comparisons. Flips go both ways at about equal rates, and the mean shift is inside its interval; the reader moves the deep output no more than changing the cache format does, though the fp16 row also changes the expert placement, so it is a yardstick and not a pure cache-format cost.

## Against the reference

The base: the official release with its routed experts dequantized to BF16 (llama.cpp 29b52d398, `convert_hf_to_gguf
--outtype bf16`), through `llama-perplexity`, wikitext-2, 100 chunks of 512, `-b 8192 -ub 8192`; `inkling2-bf16-base.kld`, base
PPL 7.0714. transformers' implementation on the CPU agrees with it (chunks 0 and 1: KLD 0.0017 and 0.0019, top-1 98.4% and
97.7%).

| Run | Mean KLD | Top-1 | PPL |
| --- | ---: | ---: | ---: |
| DQ8 pack, 100 chunks, `--kv fp16` (475 W) | 0.00304 | 97.81% | 7.0737 (base 7.0644 on these rows) |
| DQ8 pack, 100 chunks, `--kv fp8` (475 W) | 0.00452 | 97.04% | 7.0766 (base 7.0644 on these rows) |
| DQ8 pack, 100 chunks, `--kv int8` (475 W) | 0.00343 | 97.47% | 7.0788 (base 7.0644 on these rows) |
| DQ8 pack, 20 chunks (the quick gate) | 0.00264 | 97.80% | 7.6642 (base 7.6534) |
| DQ8 pack, 90 chunks, RTX 5060 Ti alone, 1,047 experts in RAM on the CPU | 0.00298 | 97.70% | |
| DQ8 pack, the same 90 chunks, the whole model on the card | 0.00304 | 97.76% | |
| DQ8 pack, 100 chunks, `gpu-mode split`: layers 0 to 11 on the RTX 5090, 12 to 23 on an RTX 5060 Ti | 0.00304 | 97.70% | 7.0718 (base 7.0633) |
| MX8 pack, 100 chunks, the whole model on the card | 0.00297 | 97.83% | |
| MX8 pack, 20 chunks | 0.00265 | | |
| MX8 pack, 100 chunks, experts in RAM on the CPU | 0.00303 | 97.72% | |
| MX8 pack, 100 chunks, `--expert-cache 1500`, one card with the spilled experts on the CPU | 0.00444 | 97.15% | |
| MX8 pack, 100 chunks, `--expert-cache 1500`, `gpu-mode peer` (the spilled experts on the second card) | 0.00440 | 97.25% | |
| MX8 pack, 100 chunks, the sm_120 prompt and decode kernels | 0.00441 | 97.22% | 7.0647 |

Which RAM experts have a card copy changes their arithmetic (the card's q8_1 activations against the CPU's Q8_0), so
placement is part of the result. With copies ranked by lookups per token and
`--vram-reserve-mib 300` at `--max-context 8192` (1,049 experts planned into RAM, 250 copied back to the card, 90.17% of
decode lookups on the card), 100 chunks against the base build (no copy-back):
PPL 7.0708 to 7.0725, paired NLL difference +0.00025 +- 0.00042, top-1 98.22% of rows the same (RTX 5080 at 360 W).
The copies then follow the session's own lookups and the prompt loan adds more while decode runs, so which pairs the card
computes depends on the conversation; it changes only at positions the requests fix, so a greedy request's placement, and
its tokens, are the same on every run. The ranking that picks those pairs (lookups discounted by the visits of each expert's
layer, [placement](../models/inkling2.md)) changes which pairs the card computes, not how: at equal placement the
arithmetic is the same.

The 475 W rows use the DQ8 pack and the BF16-derived reference `inkling2-bf16-base.kld`, with the same 100 wikitext-2 chunks (25,500 scored rows), T=3, and an RTX 5090 power limit of 475 W. The scored-row base PPL was 7.0644 for each run. The 256K VRAM readings were measured at 400 W unless noted. On the RTX 5060 Ti, the CPU computes experts in RAM (Ryzen 9 9950X, AVX-512).

## Decode: INT8 vs FP8

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

A kernel-only timing from `inkling2_kernels_parity`, one row and 2,998 keys on the same RTX 5090 at 475 W: FP16 12.5 µs, FP8 10.1 µs, INT8 11.7 µs.

## 16K prefill memory check

RTX 5090 at 475 W. The standard prefill fixture requested with `-p 16384` contains 15,810 tokens. Each run used `--max-context 32768`, `--max-new 1`, and `--spec 0`. These are single-run tok/s figures, so a small FP8 difference is not resolved without repeated pinned-placement `basalt-bench` window latency and passes/s.

| KV format | Prefill tok/s | Result |
| --- | ---: | --- |
| FP16 | 17,639 | Pass |
| FP8 | 17,647 | Pass |
| INT8 | 17,118 | Pass |

Staging is sized for the widest row format (FP16's 256 bytes, not INT8's 144), so every format fits.

## Fully resident 512K capacity

These runs allocated the full native 512K cache (`--max-context 524288`) on the RTX 5090 at 475 W, with the same q8 model and `--expert-cache 3072`; `--kv-resident` stayed at its default, so the cache remained fully resident. The 12-token `conv.ids` prompt started generation near position 12, and each run generated 1,200 tokens (ending near position 1,212). They demonstrate resident capacity, not decoding at position 524,288.

| KV format | Decode tok/s | Accept % | Draft len | Result |
| --- | ---: | ---: | ---: | --- |
| FP8 | 353.8 | 60.9% | 1.93 | Pass |
| INT8 | 313.0 | 67.2% | 1.51 | Pass |

## VRAM use at 256k context

Measured after `READY 262144`, before a prompt, on an RTX 5090 at 400 W. The full context capacity was allocated at startup; expert placement used `--expert-cache 3072` (3,070 GPU slots and 2 in RAM). Idle `nvidia-smi` use was 1,895 MiB.

| KV format | Total VRAM used | Above idle | Saved vs FP16 |
| --- | ---: | ---: | ---: |
| FP16 | 30,192 MiB (29.48 GiB) | 28,297 MiB | — |
| FP8 | 25,840 MiB (25.23 GiB) | 23,945 MiB | 4,352 MiB (4.25 GiB) |
| INT8 | 26,366 MiB (25.75 GiB) | 24,471 MiB | 3,826 MiB (3.74 GiB) |

The INT8 row-writer parity matched the CPU reference for 257 rows of width 128. Dequantized max absolute error was 0.00559; the microbench took 2.81 µs for 4,096 rows at 957 GB/s.

## Drafting

The DSpark drafter only proposes; the target checks every draft, and a decode row rounds the same in any window. In the checks
run with the drafter's K/V in fp16 (prose, struct and conversation, after short and 16K-token prompts, and a 2,405-token
prompt), greedy output with drafting equalled plain greedy output: an observation for those configurations, not a
guarantee.

With `--kv int8` the drafter's full layer keeps its K/V in int8 too (2.06 KiB a position instead of 4 KiB). That changes
which tokens it drafts, and the VRAM it frees goes to experts, so the expert placement moves (at `--max-context 262144` on a
16 GB card, 2,124 experts in RAM instead of 2,236). The target still verifies every token, but served output is not that of
an fp16 drafter: greedy decodes after 32K and 131K prompts part from the fp16 drafter's at tokens 71 and 40, and the BF16
gate over 20 chunks moves mean KLD 0.00300 -> 0.00357 (+0.00056 +/- 0.00039), PPL +0.05% and top-1 +0.02 points, with 5
rows carrying almost all of the KLD change. These runs differ in placement and drafter cache at once, so they do not say
which of the two moves the output.

The prompt attention and the FP4 experts round a row the same at any chunk width (the parity test checks both, the experts
in an 8,192-row chunk against a 256-row one); Q8_0 experts through MMQ do not, since MMQ splits K by the chunk. The
drafter's context of a chunk's rows comes from MMQ products over the whole chunk, so its last bits follow the chunk's width:
a read that runs 8,192-row chunks (more than 4,096 tokens, experts in RAM) gives the drafter other bits than 4,096-row
chunks would, and its drafts can differ (the target still checks every one).
