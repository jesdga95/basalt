# Isolated precision costs

The decode-path tables are in [index.md](index.md); the quality method is in [quality.md](quality.md). This page
measures every place where the NVIDIA NVFP4 pack's decode path rounds below the model's own precision, each switched
off on its own, and everything switched off at once. The model's own precision is NVIDIA's checkpoint: the NVFP4
experts as published, BF16 dense weights, and fp32 activations and sums.

## Method

- **Gate:** `tools/ppl_gate.py` (`basalt-bench --quality`), 100 chunks of wikitext-2, the 256-token prompt half read
  by the prompt path and rows 256 to 510 scored through decode at T 3 against the llama.cpp BF16 base of the original
  model ([quality.md](quality.md#the-bf16-reference)). RTX 5090 at
  400 W with an RTX 5060 Ti at 180 W as the expert tier, `--kv int8 --resident-experts --expert-cache 7000`, tiers
  pinned. The default is the official pack (Q6_K dense) with these flags.
- **Paired difference:** per chunk, the arm's mean KLD (and same-top-1 rate) minus the default's on the same rows;
  the table gives the mean over 100 chunks with a t-based 95% interval. **Lossless** means the interval contains 0, or
  the output is bitwise equal, or a proof makes it exact; **measurable** means the interval excludes 0.
- **Noise:** the gate is deterministic (two default runs: 25,500 of 25,500 rows bitwise equal). A change that only
  reorders sums moves the paired mean by up to about 0.001 (the reorder rows below), and the interval half-width is
  about 0.001, so shifts smaller than that are not resolved.
- **Variants:** each switch is a throwaway build or a converter option, not part of the engine. An activation switch
  feeds the projection the fp32 values its int8 rows were made from, through a plain fp32 kernel (weights decoded to
  fp32, fp32 sums); it also turns off the fused kernels that write int8 rows directly, so its control is the same
  build with those kernels off and int8 rows kept. The BF16-dense pack is `convert.py --checkpoint` with every
  projection stored as NVIDIA's BF16 instead of Q6_K.
- **Gate length:** the scored rows sit at positions 256 to 510. The QSA indexer budget is 2,048 tokens, so at this
  length every cell is selected and the indexer's own rounding never reaches the output; long-context effects of the
  KV formats are not covered either.

## Everything off against the shipped default

"All off" is the BF16-dense pack with fp32 activations into every dense projection, the shared expert and the head,
BF16 hyper-connection matrices (no 8-bit copy, fp32 sums), fp16 KV, no fast math and exact `expf`, division and
square root everywhere, and fp32 indexer keys. The routed experts stay NVIDIA's NVFP4 codes in every arm.

| arm | PPL | KLD mean | KLD p99 | same top-1 | paired dKLD [95% CI] | paired top-1, points [95% CI] |
|---|---:|---:|---:|---:|---|---|
| default (Q6_K dense, int8 KV) | 3.2975 | 0.06213 | 0.711 | 91.59% | | |
| all off | 3.2866 | 0.05308 | 0.631 | 92.55% | -0.0091 [-0.0110, -0.0072] | +0.95 [+0.68, +1.22] |
| all off, expert activations too ¹ | 3.2850 | 0.05382 | 0.633 | 92.30% | -0.0078 [-0.0093, -0.0064] | +0.43 [+0.15, +0.70] |
| Q8_0-dense pack, nothing else changed | 3.2863 | 0.05629 | 0.669 | 92.14% | -0.0059 [-0.0074, -0.0043] | +0.55 [+0.28, +0.82] |

¹ One card with a 256-slot expert cache, so the CPU tier computes 98.8% of the expert rows, with each activation held
as an int8 value plus an int8 residual (two dot products per row, error at most 1.6e-5 of the 32-value block's largest
value); paired against the default flags at the same placement.

Removing every reducible rounding lowers the KLD by about 0.009 (15%) and raises top-1 agreement by about one point.
The Q6_K dense projections are most of it: the Q8_0-dense pack alone lowers the KLD by 0.0059. All off against the
Q8_0-dense pack is a further -0.0032 [-0.0047, -0.0018] and +0.40 [+0.18, +0.63] points of top-1, the remaining
roundings together (BF16 against Q8_0 dense, -0.0014, and the items below).

Against the earlier llama.cpp Q8_0 base all off and the Q8_0-dense pack read the same (0.0624 against 0.0623): that
base carries its own rounding, which hid the last 0.003.

Speed, one cold run per row (`bench.sh` fixtures through `basalt-bench`, the default serving flags, 400 W + 180 W, or
the 5090 alone at 400 W). The all-off decode uses throwaway kernels (a warp-per-row fp32 matrix-vector product and the
unfused paths), so its cost is an upper bound of what dedicated kernels would cost:

| config | prose, 2 cards | struct, 2 cards | prose, 5090 alone | struct, 5090 alone | prefill 16K, 2 cards |
|---|---:|---:|---:|---:|---:|
| default | 326.7 | 586.8 | 272.7 | 450.5 | 7,760 |
| all off ² | 162.9 | 293.3 | 137.4 | 225.0 | 6,017 |
| Q8_0-dense pack | 304.1 | 566.6 | | | |

² With fast math and fp16 indexer keys kept; the full all-off build reads 156.0 tok/s on prose with two cards, and
the exact-math build alone 336.0 against 326.7 (one run each, inside single-run spread). Where the all-off time goes,
prose with two cards: BF16 hyper-connection matrices through the unfused read 238.0 tok/s, the unfused kernels the
activation switches need 294.6, BF16 dense with fp32 activations through the throwaway kernel 192.2.

The Q8_0-dense pack is the production-kernel way to get the all-off quality: 7% slower on prose and 3% on struct
than the default in the same session (ABBA numbers:
[6-bit dense projections](features.md#6-bit-dense-projections-nvidia-nvfp4-pack)). Both NVFP4 packs are on Hugging Face:
`Qwen3.8-FN-NVFP4-DQ6_K.basalt` (the default above) and `Qwen3.8-FN-NVFP4-DQ8.basalt` (`D` names the dense format).

## Lossless and measurable

Paired against the default unless a control is named; a negative dKLD means switching the rounding off lowers the KLD
by that much. Items that change no output bit are listed with how that is known.

| item | switch | paired dKLD [95% CI] | top-1, points | verdict |
|---|---|---|---|---|
| Q6_K dense projections (vs BF16) | BF16-dense pack | -0.0072 [-0.0086, -0.0058] | +0.82 | measurable |
| Q6_K dense projections (vs Q8_0) | Q8_0-dense pack | -0.0059 [-0.0074, -0.0043] | +0.55 | measurable |
| Q8_0 dense (vs BF16) | BF16-dense against Q8_0-dense | -0.0014 [-0.0025, -0.0003] | +0.27 | measurable |
| int8 hyper-connection copy (hc8), decode | BF16 matrices, fp32 sums | -0.0002 [-0.0014, +0.0011] | +0.02 | lossless |
| hc8 in decode and the prompt path | no 8-bit copy at all | -0.0014 [-0.0027, -0.0001] | +0.15 | measurable, at the edge of noise |
| int8 activations into GDN and QSA projections | fp32 activations (control: fused kernels off) | -0.0004 [-0.0014, +0.0006] | -0.04 | lossless |
| int8 activations into the shared expert | fp32 activations (same control) | +0.0005 [-0.0004, +0.0014] | -0.02 | lossless |
| int8 activations into the head | fp32 activations (same control) | -0.00009 [-0.00012, -0.00005] | 0.00 | measurable (0.1% of the KLD) |
| int8 activations into all three | fp32 activations (same control) | -0.0010 [-0.0019, -0.0002] | +0.15 | measurable, at the edge of noise |
| int8 activations into the routed experts (W4A8) | two-term split (control: same placement) | +0.0002 [-0.0008, +0.0012] | -0.15 | lossless |
| fast math in 18 kernel files, `__expf`, `__fdividef`, `rsqrtf`, approximate `ex2`/`rcp` | exact-math build (control: fp32 indexer keys build) | -0.0003 [-0.0012, +0.0006] | +0.11 | lossless |
| int8 KV (sample config) against fp16 KV | `--kv fp16` | +0.0000 [-0.0010, +0.0010] | +0.15 | lossless |
| k8v4 KV | `--kv k8v4` | +0.0056 [+0.0038, +0.0073] | -0.20 | measurable |
| q4_0 KV | `--kv q4_0` | +0.0093 [+0.0074, +0.0112] | -0.38 | measurable |
| prompt path (fp16 GEMMs, BF16 hc and router inputs, fp16 shared expert, MMQ experts, fp16-MMA attention, chunked GDN on BF16/TF32 tensor cores) | prompt half teacher-forced through decode | -0.0000 [-0.0010, +0.0009] | +0.13 | lossless |
| CPU tier against GPU tiers (Q8_0 vs q8_1 activations, kernel order) | 98.8% of expert rows on the CPU, one card | -0.0005 [-0.0015, +0.0005] | +0.28 | lossless |
| dense_tc against native_mmvq reduction order | dense_tc off | +0.0003 [-0.0004, +0.0010] | +0.14 | lossless |
| fused against unfused kernels (hc read int8 rows, GDN recurrence, mid chain, attention int8 output, split query projection, glue gate/up) | the activation switches' control | +0.0000 [-0.0008, +0.0008] | +0.10 | lossless (not bitwise: 0.3% of rows equal) |
| verify window width | T 1 against T 3, 20 chunks | 0 | 0 | lossless: 5,100 of 5,100 rows bitwise equal |
| fp16-rounded indexer keys | fp32 keys | 0 | 0 | no effect below 2,048 tokens: 25,500 of 25,500 rows bitwise equal |
| two-stage Q8_0 head (screen on six bits) | none needed | | | lossless by proof: every consumer reads the exact head ([two-stage head](two-stage-head.md#q8_0-head-bit-planes-exact)); the gate scores the exact pass |
| drafter (Q8_0/Q2_0 weights, fp16 draft attention, 2-bit draft-head screen) | none needed | | | no output effect: verify decides every token ([features](features.md#speculative-decoding-does-not-change-output)) |

## Inventory

Every rounding below the checkpoint's precision on the decode path, and where the prompt path feeds the scored rows.

| item | format | source | output effect | measured |
|---|---|---|---|---|
| routed experts | NVIDIA's NVFP4 codes and scales, bit for bit | tools/convert/checkpoint.py | none (the model's own) | |
| dense projections | Q6_K from BF16 (shared-expert down Q8_0) | checkpoint.py (CheckpointDense) | yes | measurable |
| output head | Q8_0 from BF16 | checkpoint.py (GLOBAL) | yes | not isolated ³ |
| token embedding | Q8_0 from BF16 | checkpoint.py (GLOBAL) | yes | not isolated ³ |
| PLE n-gram table | NVIDIA's FP8 values re-blocked to Q8_0 | model_sections.py (PleFp8) | yes | not isolated ⁴ |
| hyper-connection matrices | int8 copy, fp16 scale per 32 | tools/convert/hc8.py; hc_fused.cu | yes | lossless |
| hc up product | fp32 input as fp16 hi + lo, fp16 MMA, fp32 sums | hc_fused.cu | yes | lossless (with hc8) |
| router, ssm_alpha/beta, indexer projections, PLE key/value, shared-expert gate | BF16 as stored, fp32 activations and sums | native_bf16.cu, dec_gdn_rec.cu | none | |
| norms, conv1d, ssm_dt | F32 (norm weights BF16 + 1, exact) | checkpoint.py | none | |
| ssm_a | -exp(A_log) in f64, rounded once to F32 | checkpoint.py | below the BF16 source's precision | |
| PLE conv1d | F16 from BF16 (exact in range; the converter checks) | checkpoint.py | none | |
| dense projection activations | q8_1 (int8, fp16 scale per 32) | native_mmvq.cu, dense_tc.cu, record_gdn.cpp, record_qsa.cpp | yes | lossless |
| shared-expert activations | q8_1 into gate/up, q8_1 SwiGLU into down | shared_expert.cu | yes | lossless |
| head activations | q8_1 | verify/record_head.cpp, q80_vhead.cu | yes | measurable, 0.0001 |
| routed-expert activations | q8_1 (GPU), Q8_0 per 32 (CPU tier) | iq_kernels.cu, dec_moe.cuh, native_expert.cpp | yes | lossless |
| KV cache | fp16 default, int8 in the sample config, k8v4, q4_0 | kv_q8.cu, kv_q4.cu | yes | int8 lossless, k8v4 and q4_0 measurable; fp16 has no fp32 control ⁵ |
| indexer keys | fp16-rounded, held fp32 | native_qsa_indexer.cu, qsa_dec_fused.cu | above 2,048 tokens only | not exercised by the gate |
| indexer scores (prompt path) | 3xTF32 | qsa_select.cu | above 2,048 tokens only | not exercised by the gate |
| fast-math kernel files | approximate division and square root, flush to zero | CMakeLists.txt (`--use_fast_math` list) | yes | lossless |
| approximate intrinsics in SwiGLU, sigmoid gates, softmax, norms | `__expf`, `__fdividef`, `rsqrtf`, `ex2.approx`, `rcp.approx` | hc_fused.cu, qsa_dec_attn.cu, dec_gdn_rec.cu, shared_expert.cu, iq_kernels.cu, sampler.cu and others | yes | lossless (with fast math) |
| reduction order (dense_tc, fused kernels, CPU tier) | fp32 sums in another order | dense_tc.cu, mid_chain.cu, pool.cpp | yes | lossless |
| prompt path | fp16 GEMMs, BF16 hc and router inputs, int8 MMQ experts, fp16-MMA attention, BF16/TF32 chunked GDN | src/prefill | yes, through the prompt's KV and state | lossless |
| two-stage Q8_0 head | six-bit screen with a proven bound | q80_vhead.cu | none | by proof |
| drafter | Q8_0/Q2_0 weights, fp16 draft attention, 2-bit head screen | src/core/mtp | none | by construction |
| GDN mixer split | split-K over two cards | dec_moe.cuh | yes, only with a peer faster than the main card | [gdn-split.md](gdn-split.md); off with the test cards (the peer is slower) |

³ The head kernels (bit planes, two-stage screen, draft head rows) and the embedding gather read Q8_0; a BF16 head or
embedding has no path to switch to. ⁴ The engine has no FP8 table path. Against the exact FP8 values the Q8_0 rows
differ by 5.4e-3 relative RMS (65,536 sampled rows); the FP8 values themselves carry three mantissa bits. ⁵ There is
no fp32 KV; fp16 has three more mantissa bits than the BF16 the checkpoint computes in.

## Reading the result

- **Dense projections:** the largest default rounding. Q6_K costs 0.0059 KLD and 0.55 points of top-1 against Q8_0
  dense, 0.0072 and 0.82 against BF16; Q8_0 against BF16 dense is 0.0014. The Q8_0-dense pack recovers most of it at
  the cost in the speed table; every other rounding together, BF16 dense included, is a further 0.003.
- **hc8:** within noise in decode; with the prompt path too, 0.0014 at the edge of the interval.
- **8-bit activations:** within noise for the dense projections, the shared expert and the NVFP4 experts (W4A8 against a
  two-term int8 split). The head's int8 input costs 0.0001, resolved only because its paired differences are very
  consistent; all three dense consumers together cost 0.0010, at the edge of the interval.
- **KV:** int8 is within noise of fp16 at this length; k8v4 and q4_0 cost 0.006 and 0.009.
- **Fast math and approximate intrinsics, reduction order, the prompt path, CPU against GPU tiers:** within noise.
