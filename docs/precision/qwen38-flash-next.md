# Qwen3.8-Flash-Next precision

Where Flash-Next rounds and what each rounding costs. Method and reference: [quality](quality.md). Guide:
[Flash-Next](../models/qwen38-flash-next.md).

Test setup: RTX 5090 at 400 W with an RTX 5060 Ti at 180 W as the peer expert tier, 128 GB of system RAM. Each number names its
build. Geometry: 2560 hidden, 4 hyper-connection streams, 48 layers (36 GDN, 12 QSA), 512 experts, top-10.

## Summary

- **Drafting does not change output** unless `--relaxed-drafting` is set. With the tiers pinned, 275 of 275 greedy prompts gave
  identical tokens on both packs tested (IQ3_XXS and IQ3_S), and every full-vocabulary logit row matched bit for bit.
- **Lossy flags are opt-in:** `--relaxed-drafting`, `--route-prior`, `--route-tail-skip`. Each is named in one start warning,
  which `basalt-bench` shows too; it marks such a run LOSSY, and a community run refuses them.
- **Closer to BF16 than llama.cpp.** Same Q8_0 weights, against the BF16 reference: llama.cpp scores KLD 0.0325, top-1 93.98%;
  the Basalt Q8_0 pack scores KLD 0.0200, top-1 95.34%.
- **Quantized once.** GGUF packs keep their blocks. The NVIDIA NVFP4 pack keeps NVIDIA's expert codes and scales bit for bit and
  quantizes each dense tensor once from BF16.
- **Only one rounding costs measurable quality by default:** the Q6_K dense projections, about 0.006 KLD and 0.3 to 0.6 points of
  top-1 against NVIDIA's precision. Everything else costs about 0.001 or less.

## Where each stage rounds

| Stage | Format | Arithmetic |
|---|---|---|
| Routed experts (GGUF packs) | GGUF blocks, byte for byte; mixed types across layers, and within one (UD-IQ1_S: IQ2_S gate, IQ2_XXS up) | q8_1 activations, dp4a or int8 mma, fp32 scales |
| Routed experts (NVFP4) | NVIDIA's E2M1 codes and UE4M3 scales, 4.5 bits/weight; FP32 tail `{s_gate, s_up, s_down}` | q8_1 activations, exact int32 per 16-value sub-block, fp32 scales; tails applied in fp32 |
| Dense projections (attention, GDN qkv/z/out, QSA) | Q8_0 (or Q6_K in the NVFP4 pack; Q4_K/Q5_K/Q6_K in others) | q8_1 activations, int8 tensor-core dot, exact int32 per 32 block |
| Router, GDN alpha/beta, indexer, PLE key/value | BF16 as stored | fp32 activations and sums |
| Hyper-connections | int8 copy, one fp16 scale per 32 inputs (`hc8`); the hc read takes fp32 activations | fp32 sums; the up product uses fp16 hi/lo mma |
| Token embedding | GGUF type; one row dequantized to fp32 per token | fp32 |
| Output head | Q8_0 (NVFP4 pack) or Q6_K (IQ3_S pack) | two-stage in verify, see [head](#two-stage-verify-head) |
| Drafter | Q8_0 and BF16 dense; Q2_0 experts in VRAM | proposes only; verify decides |
| KV cache | fp16 (default), int8 (fp16 scale per 64), q4_0, k8v4 | K/V dequantized to fp32; scores and softmax fp32 |
| GDN state and conv history | fp32 | fp32 |

Decode activations are int8 (q8_1) into every quantized matmul. The hc read stays fp32 (llama.cpp quantizes it to Q8_0).

### Expert tiers

GPU experts run the same kernels on every tier, so an expert gives the same bits on either card. CPU experts quantize activations with
ggml's `vec_dot_type` (Q8_K per 256 for i-quant gate/up, Q8_0 for down). They round about 1% of a layer's routed output differently, so
which experts sit where changes the last bits. Pin `--adapt-swaps 0 --peer-adapt-swaps 0` for identity checks.

### Prefill

Prefill is a different computation from decode: fp16 and BF16 GEMMs, int8 MMQ, and FP4 tensor cores for NVFP4 experts. The gate
scores the decode path.

| Path | Format | Rounding |
|---|---|---|
| Dense projections | fp16 weights and activations, cuBLAS, fp32 accumulate | fp16 activations |
| Routed experts, MMQ | q8_1 activations, int8 mma | int8 activation rounding (every MMQ format) |
| Routed experts, FP4 tensor cores (NVFP4, chunks of 1,024 or more) | E2M1 activations with a UE4M3 scale per 16 values and one fp32 scale per token; `mma kind::mxf4nvf4` | W4A4; numerics change against MMQ |
| Routed experts, CPU pool (chunks under 6,144 tokens) | ggml dot products, the same as decode's RAM tier | the CPU rounding above |
| Attention | int8 KV: q and p in fp16, one fp16 mma, fp32 accumulate | output within 1.5e-3 of an FP64 reference |
| GDN | chunked delta rule, 16 tokens per chunk; state and gates fp32; BF16 and TF32 tensor cores | output depends on where the 16-token grid falls (about 1e-3 relative) |

Rounding moves with the chunking: `--prefill auto` reads at most 16,384 tokens per chunk (32,768 for NVFP4), and the K/V rounding follows
the chunk boundaries. Prompts above the cap read a first chunk of 16,384 tokens. The CPU computes a fixed 0.6 of streamed prompt experts.
Split reads across cards match a main-card read: KLD mean 0.0008 (65,535 tokens) and 0.0005 (122,879), top-1 64 of 64 in both.

Measured prefill numbers (NVIDIA-checkpoint NVFP4 pack, 400 W + 180 W, T 3, int8 KV, 100-chunk gate unless noted):

| Measurement | Result |
|---|---|
| FP4 tensor cores vs MMQ, 2,049-token prompts, 24 chunks | KLD 0.0455 against 0.0437 (+0.0018 [+0.0002, +0.0035]); top-1 -0.19 points |
| FP4 vs MMQ, 8,193-token prompts, 6 chunks | KLD +0.0008 [-0.0015, +0.0030]; not resolved at 6 chunks |
| FP4 vs MMQ speed, 400 W + 180 W, cold | MMQ 17 to 19% slower: 16K 6,456 against 7,807 tok/s; 64K 8,201 against 10,126 tok/s |
| CPU share 0 (all GPU) against 1 (all CPU), NVFP4 DQ6_K, 20 chunks | PPL 2.3046 / 2.3068, KLD 0.07337 / 0.07085, top-1 93.04% / 93.12%; paired +0.0025 [-0.0005, +0.0055] |
| K-quant MMQ against the FP16 path | rel RMS 1.08e-2 against 5e-4 (MMQ carries int8 activation rounding); 100-chunk gate within run noise |

## Measured costs

Gate: 100 chunks, 400 W + 180 W, tiers pinned. Paired against the default (Q6_K dense, int8 KV),
the mean over chunks with a 95% interval. **Lossless** means the interval contains 0 or the output is bitwise equal. **Measurable**
means it excludes 0.

| Switch | Paired dKLD [95% CI] | Top-1 points | Verdict |
|---|---|---|---|
| Q6_K dense projections, against BF16 dense | -0.0072 [-0.0086, -0.0058] | +0.82 | measurable |
| Q6_K dense, against Q8_0 dense | -0.0059 [-0.0074, -0.0043] | +0.55 | measurable |
| Q8_0 dense, against BF16 dense | -0.0014 [-0.0025, -0.0003] | +0.27 | measurable |
| int8 hc copy (`hc8`) in decode | -0.0002 [-0.0014, +0.0011] | +0.02 | lossless |
| int8 activations into GDN and QSA projections | -0.0004 [-0.0014, +0.0006] | -0.04 | lossless |
| int8 activations into the shared expert | +0.0005 [-0.0004, +0.0014] | -0.02 | lossless |
| int8 activations into the head | -0.00009 [-0.00012, -0.00005] | 0.00 | measurable, 0.1% of KLD |
| int8 activations into routed experts (W4A8) | +0.0002 [-0.0008, +0.0012] | -0.15 | lossless |
| fast math and approximate intrinsics | -0.0003 [-0.0012, +0.0006] | +0.11 | lossless |
| int8 KV against fp16 KV | +0.0000 [-0.0010, +0.0010] | +0.15 | lossless |
| k8v4 KV | +0.0056 [+0.0038, +0.0073] | -0.20 | measurable |
| q4_0 KV | +0.0093 [+0.0074, +0.0112] | -0.38 | measurable |
| CPU tier against GPU tiers (98.8% of rows on the CPU) | -0.0005 [-0.0015, +0.0005] | +0.28 | lossless |
| prompt path against decode | -0.0000 [-0.0010, +0.0009] | +0.13 | lossless |
| verify window width, T 1 against T 3 | 0 | 0 | lossless, 5,100 of 5,100 rows bitwise equal |
| fp16 against fp32 indexer keys (below 2,048 tokens) | 0 | 0 | no effect, 25,500 of 25,500 rows equal |
| Q5_0 down layers on MMQ prefill and the device-planned decode tail, against the dequant GEMM and the host-planned tail: UD-Q3_K_XL (20 such layers) | -0.0005 [-0.0014, +0.0005] | +0.17 | lossless |
| the same on UD-Q2_K_XL (one such layer) | +0.0004 [-0.0006, +0.0013] | -0.03 | lossless |

All roundings off at once (BF16 dense, fp32 activations everywhere, fp16 KV, exact math): PPL 3.2866 against 3.2975 default,
KLD 0.05308 against 0.06213, top-1 92.55% against 91.59%. Removing every rounding lowers KLD by about 0.009 (15%). The Q8_0-dense
pack alone gives -0.0059 of that. The Q8_0-dense pack is about 7% slower on prose and 3% on struct than the default.

UD-IQ1_S, converted as shipped (layer 18: IQ2_S gate, IQ2_XXS up; each half on its own format's kernels, the same bits as
a uniform layer's rows): PPL 3.4049, KLD 0.2676, top-1 82.76% (100 chunks, T 3, int8 KV, resident experts).

The Q5_0 rows: KLD 0.0943 against 0.0948 on UD-Q3_K_XL, 0.1520 against 0.1517 on UD-Q2_K_XL, tiers auto-sized. The
device-planned tail runs only when every layer's expert formats are in it, so on these packs the decode rounding moves as well as
the prompt's.

## 6-bit dense projections

The NVFP4 pack's default `DQ6_K` stores attention q/k/v/o, GDN qkv/z/out and the shared experts' gate/up as Q6_K. The `DQ8` variant
stores them as Q8_0. Each tensor is quantized once from NVIDIA's BF16, never from the Q8_0 copy (which would round twice).

Speed, 400 W + 180 W, ABBA, 6 fresh processes per arm:

| Test | 8-bit dense tok/s | 6-bit dense tok/s | Change |
|---|---:|---:|---:|
| prose | 213.4 | 229.9 | +7.8% |
| conversation | 251.0 | 265.1 | +5.6% |
| structured | 420.4 | 463.5 | +10.2% |
| prefill 8K and 32K | 3,806 and 7,791 | 3,845 and 7,856 | +1% |

Quality: the Q6_K against Q8_0 dense row of [measured costs](#measured-costs).

## Drafting does not change output

The drafter proposes; verify decides. A window runs T rows (default 8) through the same kernels. Each row's logits equal plain decode's
at any width, and a rejected row leaves no trace.

- **Exactness:** 275 prompts, 256 greedy tokens, `--spec 0` against `--spec 7`, tiers pinned. IQ3_XXS: 275 identical, 62,271 of 62,271
  logit rows bit-identical, 69.9% of drafts accepted, 183.9 to 319.8 tok/s. IQ3_S: 275 identical, 60,878 of 60,878 rows, 70.1% accepted,
  145.1 to 239.5 tok/s. Prompts are short (19 to 395 tokens); this is the decode path, not long context.
- **Forced windows:** teacher-forced at T 1, 3 and 7, the 5,100 per-row lines (IQ3_S, 20 chunks, base PPL 2.2883) are byte identical.
- **Sampled requests** are drawn from the model's distribution, but the window length sets which random draw each position uses, so a
  seeded sampled reply can differ between window widths.
- **Logit bias** is applied on every verified row, and a draft is accepted only if it equals the biased pick. A -100 bias never yields
  the token. A biased request accepts fewer drafts.

## Request grammar

- **Grammar** is checked after each window. The first refused token is redrawn under the grammar mask, and the window ends there. The
  prompt is not constrained. Tool-call grammars start at the call opener. Relaxed drafting is off for a request with a grammar.

## Relaxed drafting

Under sampling, a draft with target probability P or more is kept even when the target's own draw would differ. At that row the output
is then greedy: total-variation distance from the model's distribution is 1 - p(d), and 0 elsewhere. Greedy, `logit_bias`, JSON
formats, the quality gate, batch slots and tool-name tokens stay exact.

Replay, per generated token (TV = mean total-variation per token; gain = extra tokens per window):

| Model | Temperature | Exact accept | P = 0.5 gain / TV | P = 0.3 gain / TV | P = 0.2 gain / TV |
|---|---|---|---|---|---|
| IQ3_S, natural prompts | 1.0 | 62.3% | +7.1% / 0.042 | +11.9% / 0.072 | +14.7% / 0.089 |
| IQ3_XXS, natural prompts | 1.0 | 62.9% | +6.6% / 0.043 | +12.1% / 0.073 | +15.2% / 0.089 |

Speed, 400 W + 180 W, IQ3_XXS, prose and natural prompts at temperature 1.0, two seeds: exact 332.5 tok/s, P = 0.5 366.9 (+10.3%), P = 0.3 385.5 (+15.9%),
P = 0.2 399.1 (+20.0%). Greedy struct is unchanged.

Quality, IQ3_S served with temperature 1.0, top_p 0.95, top_k 20, thinking off, paired per item: gsm8k (300 items) exact 0.947, P = 0.5
0.973 (+0.027 ± 0.020), P = 0.3 0.980 (+0.033 ± 0.022); MMLU-Pro (280) exact 0.682, P = 0.5 0.714 (+0.032 ± 0.040), P = 0.3 0.725.
The KLD gate cannot see this change, because it scores logits and relaxed drafting changes which token is emitted. Drafts kept by the
rule: 3.0% of generated tokens at P = 0.5, 5.0% at P = 0.3. Diversity is what this costs; these tasks do not measure it.

`--relaxed-drafting` alone means 0.5: the smallest cost, +5% at temperature 0.6 and +6 to 10% at 1.0.

## Two-stage verify head

The verify window reads all 248,320 rows of the head for every column. Two stages avoid the full read:

- **Q8_0 head (NVFP4 pack), exact.** Each row is held as bit planes. A screen reads each row's block scales and bits 2 to 7, and
  bounds every logit with an interval that provably holds it. Rows whose upper bound reaches the column's K-th largest lower bound are
  scored exactly. Every consumer reads the exact head's values, so the result is lossless by proof. K is 1 for greedy, `top_k` for a
  sampled list of 64 or fewer, and 0 (all rows exact) otherwise. Exact rows per column on real inputs: K=1 2 on average; K=20 301;
  K=64 1,214.
- **Q6_K head (IQ3_S pack), approximate tail.** Every row is scored from a 2-bit copy. Rows within 4 of the 64th-best approximate logit
  are rescored exactly. The softmax tail outside them carries the copy's error (about 0.7 logit rms). Top-64 recall 99.97% to 99.98%
  at margin 4 on 3,614 real inputs.

Costs: the Q6_K head adds KLD 0.0022 [0.0018, 0.0025] (1.5%) and PPL 0.25% (IQ3_S, 100 chunks, 400 W + 180 W), with no argmax change on
25,500 rows. The Q8_0 two-stage head is 350 us per window against 439 us for the Q8_0 head (NVFP4, cold prose, 400 W + 180 W).
Greedy output with the tiers pinned matches the exact head token for token.

## GDN split across two cards

With a peer card faster than the main card, each GDN mixer may be split by key head. The split changes reduction order, so output
changes in the last bits. Quality against the BF16 reference, swapped card order (peer at 400 W), NVFP4, 20 chunks, whole mixer against
split: PPL 2.3129 / 2.2999, KLD 0.07294 / 0.07262, top-1 92.94% / 92.96%. The paired KLD difference is -0.00032 ± 0.0014. With a
slower peer the split is not taken and output is bitwise the same as without it.

## Video input

A video reaches the encoder as transformers' Qwen3VLVideoProcessor prepares it. On four clips the sampled frames, canvas, groups and
timestamps are identical. The pixel values after normalisation are within one uint8 step: 0.016% and 0.007% of values are one step
off on 320x240 and 640x360 sources, 0.1% on a 720x358 one (40 of 14.9 M two steps). Frames convert to RGB with their own colour matrix, as PyAV's do.
What differs is the resize: the image path's Pillow-style bicubic against torchvision's float antialiased bicubic.

A group's two frames go through the patch embedding's two temporal taps apart (W0 a + W1 b, rel L2 6e-7 to 9e-7 against double
precision). A picture keeps the summed taps, so image output is bitwise as before. In the prompt, the template's whole video marker
becomes per-group `<t seconds>` text and pads, as the processor at the model's release (transformers 4.57.1) writes it.

## Lossy routing flags

Off by default, never picked automatically, and the two flags need the device-planned MoE tail. Both apply to experts held by any GPU tier.

- `--route-prior LAMBDA` (bare 0.5): picks experts as if held experts had LAMBDA added to their router logit.
- `--route-tail-skip R` (bare 9): skips an expert no GPU holds when every entry naming it ranks R or lower by weight.

Quality gate (100 chunks, 400 W + 180 W, int8 KV, both flags together) and prose decode:

| Pack | Flags | PPL | KLD mean | Same top-1 | Prose tok/s |
|---|---|---:|---:|---:|---:|
| IQ3_S | off | 3.382 | 0.1499 | 87.54% | 341.1 |
| IQ3_S | 1.5 / 7 | 3.437 | 0.1828 | 86.55% | 314.1 |
| NVFP4 | off | 3.304 | 0.0614 | 91.81% | 281.6 |
| NVFP4 | 1.5 / 7 | 3.616 | 0.2240 | 85.00% | 289.0 |
| NVFP4 | 16 / 1 | 4.422 | 0.4800 | 78.72% | 329.0 |
| Q8_0 | off | 3.282 | 0.0197 | 95.39% | 112.4 |
| Q8_0 | 16 / 1 | 8.873 | 1.2338 | 62.26% | 274.9 |

The flags' speed comes from fewer misses, so it grows with the share
of experts outside VRAM, and so does the error. The bare defaults are not measured.
