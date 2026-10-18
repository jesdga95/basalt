# Precision and quality

This page shows what precision Basalt computes in, that speculative decoding (MTP) does not change its output, and how
its quality compares with llama.cpp on the same GGUFs against the same Q8_0 reference.

Test setup: one RTX 5090 at a 400 W power limit, an RTX 5060 Ti as the peer expert tier, 128 GB of system RAM
(121 GiB visible to the OS). Each section names the build it measured.

## Summary

- **MTP does not change output** (unless `--relaxed-drafting` is set: [see below](#relaxed-drafting)). With the expert tiers pinned, 275 of 275 greedy prompts produced identical tokens on
  both packs tested, and every full-vocabulary fp32 logit row (62,271 and 60,878) was bit-identical between `--spec 0`
  and `--spec 7`.
- **KLD matches llama.cpp.** On the same GGUFs against the same base, Basalt and llama.cpp's own `llama-perplexity`
  agree: IQ3_S 0.15264 vs 0.1526, IQ3_XXS 0.2433 vs 0.2428 at 100 chunks, with the exact head. The two-stage
  verify head of a Q6_K head (IQ3_S) adds about 0.0017 ([see below](#two-stage-verify-head)).
- **Quantized once, never twice.** GGUF packs keep the GGUF blocks as shipped. The NVIDIA NVFP4 pack keeps NVIDIA's
  NVFP4 expert blobs bit for bit and quantizes each dense tensor once from NVIDIA's BF16: Q6_K for the projections
  ([below](#6-bit-dense-projections-nvidia-nvfp4-pack)), Q8_0 for the head, token embedding and shared-expert down. Its
  n-gram table is NVIDIA's FP8 values re-blocked to Q8_0 (one more rounding). Decode also reads the hyper-connection
  matrices from an int8 copy (one fp16 scale per 32 inputs), and a Q6_K output head is screened through a 2-bit copy in
  verify: the rows near the top are the exact Q6_K logits, the far tail keeps the copy's approximate logits
  ([details](#two-stage-verify-head)).
- **fp32 where it matters.** Activations are fp32 in the residual stream, norms, router, softmax and logits, and int8
  (q8_1) into the quantized matmuls. GDN state is fp32. KV is fp16 by default, with int8, q4_0 and k8v4 options.
- **Quality is scored through decode.** Prefill is a different computation (fp16/BF16 GEMMs, int8 MMQ, and FP4 tensor cores for NVFP4 experts); the quality gate scores
  the decode path.
- **The reference has its own noise.** The reference is a llama.cpp Q8_0 base, which carries about 0.01 to 0.02 KLD of
  its own noise ([see below](#why-q8_0-against-q8_0-is-not-zero)).

## Decode path precision

Sources are `file:line` under `src/`, `include/basalt/` and `tools/convert/`. Model geometry: n_embd 2560, 4
hyper-connection streams, 48 layers (36 GDN, 12 QSA), 512 experts, top-10, head_dim 256.

### Weights

| Tensor class | Format | Source |
|---|---|---|
| Routed experts | GGUF blocks copied byte for byte, no requantization. Gate and up share one type per layer; types are mixed across layers (IQ3_XXS pack: gate/up IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S; down Q2_0 on 30 layers, IQ4_NL on 18. IQ3_S pack: gate/up mostly IQ3_XXS, IQ3_S, IQ2_S, one IQ4_XS; down IQ4_NL on 39, Q2_0 on 9) | docs/model-file.md; iq_kernels.cu:541-542; tests/data/native_experts/ |
| Routed experts, NVFP4 format | NVFP4 (type 40): per 16 weights, 8 bytes of E2M1 codes and one UE4M3 scale; per expert a 16-byte FP32 tail {s_gate, s_up, s_down, 0}. 4.5 bits per weight, all 48 layers | tools/convert/checkpoint.py; native_experts.txt; iq_kernels.cu (nvfp4_scale_dev) |
| NVIDIA-checkpoint pack (`--checkpoint`) | Built straight from NVIDIA's safetensors. The experts are NVIDIA's NVFP4 codes and UE4M3 scales, bit for bit, moved into Basalt's 64-value blocks; each expert's tail is its three `weight_scale_2` values (the `input_scale` entries are not used). The dense tensors are converted once from BF16: attention, GDN qkv/z/out and the shared experts' gate/up Q6_K (Q8_0 with `--dense-q8-0`), shared-expert down, head and token embedding Q8_0 (ggml's rounding); router, hyper-connection matrices, indexer, ssm_alpha/beta and the PLE key/value BF16 as stored, so nothing is requantized for the small matrices; norm weights BF16 + 1 in F32 (exact); ssm_a = -exp(A_log) computed in f64 and rounded once to F32; the GDN value heads reordered into the GGUF order. The n-gram table is the checkpoint's FP8 E4M3 codes times its one BF16 scale, re-blocked to Q8_0 (blocks of 32, fp16 scale = absmax/127, ggml's rounding): at most half a Q8_0 step plus the scale's fp16 rounding from the exact FP8 value, against the 4.5-bit IQ4_NL table of the other packs | tools/convert/checkpoint.py; convert.py (from_checkpoint); model_sections.py (PleFp8); test_checkpoint.py; test_convert.py (PleFp8Test) |
| Shared expert gate/up/down | GGUF blocks, any native_mmvq type | native_mmvq.hpp:55-58 |
| Shared-expert scalar gate | BF16 | model_sections.py:48-51 |
| QSA q/k/v/o, GDN qkv, z (attn_gate), ssm_out | GGUF blocks (dense_tc takes Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, Q3_K, Q2_0, Q8_0; native_mmvq otherwise). Q8_0 on dense_tc is not bitwise equal to native_mmvq: the int8 tensor-core dot is an exact int32 per 32-value block, then the fp32 scale terms are summed in a fixed order, where mmvq rounds `d_w * d_x * sumi` per 8-value quarter block in a warp tree. Against a float64 reference both sit at about 1e-7 RMS (dense_tc_parity, real projection geometry, T = 1..8, every column bitwise independent of T). Quality gate on the Q4_K_XL pack (20 chunks, decode path, 400 W): base PPL 2.3272 / KLD 0.0687 / p99 1.054 / same top-1 92.80%, with Q8_0 on dense_tc 2.3269 / 0.0685 / 1.044 / 93.08%; all deltas are inside the run-to-run noise of a 20-chunk run. The official NVIDIA NVFP4 pack stores these (and the shared-expert gate/up) as Q6_K from NVIDIA's BF16 weights, so they take dense_tc ([below](#6-bit-dense-projections-nvidia-nvfp4-pack)) | dense_tc.cu:1063-1075; dense_tc.hpp:2-12 |
| GDN ssm_alpha, ssm_beta; QSA indexer q/k projections | BF16 | model_sections.py:52-53 |
| Router | BF16, not quantized | model_sections.py:48 |
| Norm weights, ssm_a, ssm_dt, conv1d, ssm_norm | F32 | model_sections.py:55-60 |
| Hyper-connection (hc) matrices (down, up, inject; per layer and for the output head) | Stored in the pack as the source has them (BF16 in the GSQ-RCO and NVIDIA packs, Q8_0 in UD-Q4_K_XL); decode reads an int8 copy: int8 codes with one fp16 scale per 32 consecutive inputs, scale = f16(absmax/127), codes clamped to ±127 | hc_fused.cu:343, 1173-1191, 1215-1232; verify.cpp:561-572 |
| Token embedding | GGUF type kept in mapped host memory, one row dequantized to fp32 per token | native_head.hpp:42-45; verify.cpp:948-961 |
| Output head | A native_mmvq type with dedicated Q5_K and Q6_K kernels; the IQ3_S pack's head is Q6_K. A Q6_K head runs in two stages in verify: every row is scored from a 2-bit copy (the draft head's fit below: four levels and an fp16 scale per 32 weights, the scale raised by sqrt(\|w\|²/⟨w, ŵ⟩)), on int8 tensor cores; the rows within 4 of a lower bound on the column's 64th-best approximate logit are rescored exactly (the bits of the plain Q6_K kernel), and every other row keeps its approximate logit. The argmax and the sampler's top-64 list come from exact rows; the tail of the softmax is approximate (see [Two-stage verify head](#two-stage-verify-head)). The copy (199 MB for 248,320 rows) is packed at load, before the expert cache is sized | native_head.cpp; verify.cpp (the head); q6k_head.cu (q6k_vhead) |
| MTP drafter dense | Q8_0 for fc_embedding, fc_hidden, q/k/v/o, indexer projection and the shared-expert projections; BF16 for the rest; 1-D tensors F32 | model_sections.py:334-338, 444-470 |
| MTP drafter experts | Q2_0, all in VRAM | model_sections.py:482-487 |
| MTP draft head | Rows gathered from the main head (same type), optionally a vocabulary subset; for a Q5_K or Q6_K head the leading rows are first scored from a 2-bit copy (four levels and an fp16 scale per 32 weights) and the rows within 5 of the best approximate score (at most 64) rescored exactly, so a draft differs from the exact argmax only in rare cases, and the draft probability is approximate: the other rows' terms of its softmax sum come from the 2-bit scores. The copy's scale (Q5_K and Q6_K alike) is raised by sqrt(\|w\|²/⟨w, ŵ⟩) per group, since the least-squares copy scores the near-best rows low and inflates the probability (on the Q5_K parity test's synthetic head over 12 seeds the probability's mean relative error halves, from 0.028-0.043 to 0.010-0.020, and the worst trial from up to 1.04 to under 0.45); on 1500 real draft-head inputs of the IQ3_S pack the probability's mean error is +0.005 (+0.017 with the least-squares scale) and the pick matches the exact argmax in 99.93% | mtp.cpp, head_proxy.cuh, q5k_head.cu, q6k_head.cu |
| MTP hc matrices | The same int8 copies | mtp.cpp:218-263 |
| MTP draft length | The cumulative-acceptance threshold that ends a draft is the base value times a factor of 1.0 to 1.5, set once from the mean CPU-served expert bytes per expert slot of the pack's final tiers (1.0 at about 1.1 MB, 1.5 at 3.5 MB or more); it is a function of the pack and its placement, never of timing. Accepted tokens still come from verify alone, but the window width differs between packs and builds, and verify kernels are not batch-invariant across widths, so a changed width can change a token-for-token output wherever two candidates sit within rounding of each other | draft_cum.hpp; mtp.cpp; generate.cpp |

The drafter only proposes tokens; verify decides every token, so the drafter's precision affects speed, not output.

### Activations at each matmul

| Matmul | Activation format | Source |
|---|---|---|
| hc down and inject (hc read) | fp32, no activation quantization; int8 weights converted exactly, fp32 fmaf per group | hc_fused.cu:343-370, 492 |
| hc up | fp32 input split into an fp16 high and low part (about 22-bit mantissa), fp16 tensor-core mma with fp32 accumulate, per-group fp16 scale applied in fp32 | hc_fused.cu:635-698 |
| Router | fp32 times BF16 weight, fp32 FMA | hc_fused.cu:740-785; native_bf16.cu:34-125 |
| GDN qkv, QSA q, GDN ssm_out, QSA o | q8_1: int8 codes with an fp16 scale per 32 values, int8 tensor-core dot (dense_tc) | native_mmvq.cu:147-162; dense_tc.hpp:2-12 |
| GDN z, QSA k/v | z through dense_tc together with qkv when eligible; otherwise native_mmvq (dp4a on q8_1) | dense_tc.cu:1155; verify.cpp:1145-1170 |
| Routed experts gate/up | q8_1 of the fp32 FFN input; dp4a with exact int32 per block | hc_fused.cu:63-77; iq_kernels.cu:202-228 |
| Routed experts down | fp32 SwiGLU, then q8_1 per 32 | iq_kernels.cu:1909-1920 |
| Shared expert | q8_1 into gate/up; fp32 SwiGLU, q8_1 into down; sigmoid gate fp32 | shared_expert.cu:178-215 |
| GDN alpha/beta, indexer projections | fp32 activation times BF16 weight, fp32 FMA | dec_gdn_rec.cu:88-112; verify.cpp:1315-1327 |
| Output head | q8_1 of the fp32 final stream, mmvq or the Q5_K/Q6_K kernels; the two-stage Q6_K head's first stage takes the same q8_1 codes as int8 mma inputs (exact int32 per 32-group), fp32 scale terms | verify.cpp (the head); q6k_head.cu |
| CPU expert tier (experts not resident on a GPU) | The fp32 input quantized with ggml's vec_dot_type (Q8_K per 256 for i-quant gate/up; Q8_0 or Q8_K for down), so it rounds differently from the GPU tiers | native_expert.cpp:52-62, 84-139 |
| CPU expert tier, Q6_K gate/up | Q8_K per 256 of the fp32 input; ggml's AVX2 Q6_K x Q8_K dot (also run by AVX-512 CPUs): 6-bit codes minus 32, signed per-16 scales, integer sums per block in int32, times the fp16 block scale and the Q8_K scale in fp32. Checked against a double reference over the dequantized operands (worst error 1.3e-8 of the sum of absolute terms) | ggml_cpu_dots.c, native_expert_parity --cpu-synthetic |
| CPU expert tier, NVFP4 experts | Q8_0 per 32 of the fp32 input and of the fp32 SwiGLU output; integer sums per 16-value sub-block, times the UE4M3 sub-block scale and the Q8_0 scale in fp32. The expert's FP32 tail scales go on in fp32: s_gate and s_up on the gate/up dots before SwiGLU, s_down on the down output (folding s_down into up would put h near 1e-5, where the activation block scale goes subnormal). Against a double-precision expert on the same weights: rel 1.2e-2 from activation rounding, 7e-8 with the CPU's own rounded activations | native_expert.cpp (native_gu_rows, native_down_rows); ggml_cpu_dots.c (ggml_vec_dot_nvfp4_q8_0); native_expert_parity --synthetic nvfp4/nvfp4 |
| GPU experts, NVFP4 (grouped kernels: per-entry and decode-once, and the device-planned MoE tail on both cards; all bitwise equal) | q8_1 of the fp32 input and of the fp32 SwiGLU output (the same rounding as the other GPU formats); E2M1 codes through a 16-entry int8 table, dp4a with an exact int32 sum per 16-value sub-block, times the q8_1 scale and the UE4M3 sub-block scale in fp32. s_gate and s_up multiply the gate/up dots before SwiGLU and s_down the down output, in fp32, as on the CPU tier. Against a double-precision expert on the kernels' own q8_1 activations: rel 5-8e-8 (gate, up, down); on the unrounded input: rel 1.2e-2; GPU vs CPU tier: rel 8e-8 (real experts, layers 0-3, 20, 47) | iq_kernels.cu (vec_dot_nvfp4_q8_1, Split<40>, native_gu_kernel, native_down_kernel and their multi variants); dec_moe.cuh (dm_gu_row, dm_down_item, dm_gu_kernel); native_expert_parity --experts PACK; iq_multi_parity, native_grouped_parity, dec_moe_parity --nvfp4 |
| GPU experts, Q8_0 gate/up with Q8_0 down (decode-once kernels and the device-planned MoE tail, bitwise equal to the per-entry kernels; no numeric change) | q8_1 of the fp32 input and of the fp32 SwiGLU output; int8 weights, dp4a with an exact int32 sum per 32-value block, times the fp16 block scale and the q8_1 scale in fp32, the same expression in every kernel | iq_kernels.cu (Split<8>); dec_moe.cuh (DM_PAIRS); iq_multi_parity, dec_moe_parity --kq |
| GPU experts, Q6_K gate/up with Q8_0 down (decode-once grouped/verify kernels, the device-planned MoE tail and the peer kernel; bitwise equal to the per-entry kernels; no numeric change to existing formats) | q8_1 of the fp32 input and of the fp32 SwiGLU output; 6-bit codes minus 32 through dp4a against the q8_1 words, each 16-value sub-block's int32 sum times its signed int8 scale, the 32-value products summed in fp32 times each block's q8_1 scale, then the fp16 block scale. Decode-once and per-entry kernels share this arithmetic. Prompt path (MMQ) is a separate row once it lands | iq_kernels.cu (Split<14>), dec_moe_parity --kq |
| Prompt path, NVFP4 experts (MMQ, the default) | q8_1 of the fp32 input and of the fp32 SwiGLU output, as for the other MMQ formats; E2M1 codes through the 16-entry int8 table into the tile, int8 mma with an exact int32 sum per 16-value sub-block, times the UE4M3 sub-block scale and the q8_1 scale in fp32. The tail goes on in fp32 in the SwiGLU step: h = silu(s_gate g) * (s_up u) * s_down, so s_down rides on H into the down product (q8_1 keeps an fp32 scale per 32 values, so H rounds to the same codes up to fp32 rounding of that scale). **Numerics change against the FP16 fallback it replaces**: the error is now the int8 activation rounding every MMQ format has. Synthetic experts at real geometry: gate/up product vs a double reference on the same q8_1 input rel 1.7e-7; layer output vs that reference 6.8e-5 (H codes at rounding ties); vs the unrounded double reference 1.6e-2 (the FP16 fallback: 5.2e-4); MMQ vs FP16 fallback 1.6e-2 | mmq_load_tiles.cuh (ggml_cuda_mmq_load_tiles_nvfp4); mmq_vec_dot.cuh (q8_0_16); moe_mmq.cu (swiglu_kernel, gather tails); prefill.cpp; prefill_mmq_nvfp4_test |
| Prompt path, NVFP4 experts (FP4 tensor cores, chunks of 1,024 tokens or more; shorter chunks take MMQ) | **W4A4. Numerics change against the MMQ path it replaces.** The fp32 input and the fp32 SwiGLU output are rounded to E2M1 codes with a UE4M3 scale per 16 values and one fp32 scale per token (amax / (6 x 448)); the weight blocks are read in place, and `mma kind::mxf4nvf4` (block scale, 4 scales per k64) accumulates in fp32. The tail scales go on in the SwiGLU step as in MMQ. Synthetic experts at real geometry: gate/up rel rms vs a double reference 9.4e-2 (MMQ: 1.6e-2). Against a CPU reference with the same E2M1/UE4M3 activation rounding the kernel agrees to rel rms 1.4e-7 (gate/up) and 3.7e-7 (layer output), and a token's output is bitwise independent of its tile and neighbours. The quality gate's 256-token prompt halves are below the 1,024-token minimum, so the gate runs MMQ prompt experts and does not exercise this path. Long prompts, NVIDIA-checkpoint pack, 400 W, T 3, int8 KV, resident experts: 12 chunks of 4,096 tokens (8 consecutive 512-token wikitext chunks of the Q8_0 base each, so a BOS sits every 512 tokens), the 2,049-token prompt half through this path, 2,047 teacher-forced rows per chunk (24,564 rows): PPL 2.9556 against 2.9547 for MMQ (standard error 0.037 each); paired per-row NLL difference +0.0003 (+-0.0014, ratio 1.0003). No Q8_0 reference exists for that layout, so KLD and top-1 are not measured there | moe_fp4.cu; prefill.cpp (fp4_flush); prefill_fp4_test |
| Prompt path, NVFP4 experts (FP16 fallback, only where MMQ has no tile) | Weights dequantized to fp32 (exact), times the tail scale in fp32, rounded once to fp16; fp16 GEMM. Weights below fp16's normal range (about 6e-5) lose relative precision there | iq_kernels.cu (dq_nvfp4); prefill.cpp |

Unlike llama.cpp, Basalt does not quantize the activations of the hc read to Q8_0.

### Accumulation, state, softmax and logits

| Item | Precision | Source |
|---|---|---|
| int8 dot products | Exact int32 per 32-value block, then fp32 scale terms summed in fp32 | dense_tc.cu:60-68, 1007-1010 |
| Residual stream and hc buffers | fp32 | verify.hpp:468-469; hc_fused.hpp:22-38 |
| RMSNorm | fp32, eps 1e-6 | verify.cpp:116 |
| fp16/BF16 accumulators | None; hc up uses fp16 mma inputs with fp32 accumulate | hc_fused.cu:635-698 |
| KV cache (`--kv`) | fp16, int8, q4_0 (alias q4), k8v4 (int8 K, rotated Q4_0 V). Engine default fp16; the sample config in the README uses int8 | generate.cpp (`--kv` parse; qsa_set_kv_int8, qsa_set_kv_q4); docs/configuration.md |
| int8 KV layout | int8 codes with an fp16 scale per 64 values (4 per 256-wide head, separate for K and V), scale = f16(amax/127) | kv_q8.hpp:3-5, 26-28; kv_q8.cu:46-52 |
| Attention | K/V dequantized to fp32; scores in fp32; online softmax per 64-cell chunk in fp32 (`__expf`), fp32 merge; output gate sigmoid, then q8_1 for the o-projection | qsa_dec_attn.cu:114-138, 255-420 |
| Draft-only attention | fp16 mma kernel; it changes drafts only, never accepted tokens | qsa_dec_attn.cu:424-427 |
| Indexer | Keys fp16-rounded, held fp32; block scores fp32; radix top-k, ties to the lowest cell | native_qsa_indexer.hpp:14-18; qsa_select.hpp:6-9 |
| GDN recurrent state and conv history | fp32 | dec_gdn.hpp:21; verify.cpp:1134, 1158-1160 |
| Router softmax and top-10 | fp32, ties to the lower expert index, weights renormalized over the ten | native_router.cu:39-91 |
| Router, n_expert other than 512 (pruned packs) | Generic router kernel instead of the fused native one. Max in fp32; per-expert exp in double of (logit - max); the sum in double, ascending on one thread; probabilities rounded to fp32 as `(float)(exp * (float)(1/sum))`. Top-10 by ten block-wide argmax passes over the fp32 probabilities, ties to the lower expert index. The ten weights are summed in double, floored at 2^-14, divided in double and stored fp32. The native path takes exp, sum and reciprocal in fp32 with a warp-ordered sum, so the probabilities differ in the last bits and the chosen experts or weights can differ from it in rare near-tie cases | router_top10.cu:56-173 (max 74-88, exp 92, sum 96-102, probabilities 114, selection 131-162, renormalization 166-172); native_router.cu:46-91 |
| Expert combine | fp32, in router rank order, plus shared expert | native_moe.cu:35-50 |
| Final logits | fp32 `[T][n_vocab]` | verify.hpp:489 |
| Greedy sampling | fp32 argmax, lowest index wins ties | sampler.cu:79-90 |
| Request `logit_bias` | Only when a request sends one: fp32 bias added to the fp32 logits before the penalties and filters; -100 is -inf (the token cannot be picked), other values are added as given. Without one the sampler reads the same logits and takes the same code path as before | sampler.cu; logit_bias.hpp |
| KLD scoring (host) | Log-sum-exp and KLD accumulated in double from the fp32 logits | ppl_gate.hpp (the KLD function) |

### Prefill

Prefill and decode are different computations that read and write the same KV and state formats.

| Item | Prefill | Source |
|---|---|---|
| Dense projections | Weights dequantized to fp16, activations rounded to fp16, cuBLAS with fp32 accumulate; BF16 tensors BF16 x BF16 with fp32 compute; no q8_1 | gemm.cu:125-185; prefill.cpp:1122-1131 |
| hc matrices | Activations rounded to BF16; down/up use the int8 copy dequantized to BF16; inject uses the BF16 tensor | prefill.cpp:1500-1506; gemm.cu:136-159 |
| Router | BF16 activation (decode: fp32) | prefill.cpp:1631 |
| Shared expert | fp16 activation, fp16 SwiGLU | prefill.cpp:1634-1637 |
| Routed experts | Fused int8 mma kernels, llama.cpp MMQ (q8_1) or dequant-to-fp16 cuBLAS (IQ1_M only). NVFP4 takes the FP4 tensor-core kernels from 1,024-token chunks and MMQ below (see the Activations table). The K-quants (Q4_K, Q5_K, Q5_1: UD-Q4_K_XL) run on MMQ with fp16 activation scales and per-32 partial sums taken from the rounded int8 values (llama.cpp sums the floats), so the product is exactly the dequantized weights times the rounded activations. Q6_K gate/up (UD-Q5_K_XL, UD-Q6_K_XL) runs on MMQ only, with no FP16 path: the 6-bit codes minus 32 as int8 and a float scale (d times the int8 sub-scale) per 16 values, against the same int8 activations with one fp32 scale per 32 (no partial sums, as there is no minimum) | moe_fused_iq.hpp:1-12; moe_mmq.hpp:1-6; mmq/quantize.cu:59-69 |
| Attention | int8 KV: q and p rounded to fp16 (q prescaled by a power of two), V scale folded into the fp16 B operand, one fp16 mma, fp32 accumulate, per-warp online softmax merged once at the end. Output within 1.5e-3 of the output scale of an FP64 reference (the previous hi/lo split: 1e-6). fp16 and Q4_0 KV keep the hi/lo split of q and p | qsa_prompt_attn.hpp:1-22 |
| Routed experts on the CPU (chunks below 6,144 tokens) | Non-resident experts with a host copy and at most 8 rows run on the CPU pool with ggml's dot products, as decode's RAM tier does: the activation in the weight type's `vec_dot` format (Q8_K per 256 values for the IQ types, not q8_1 per 32), H quantized to the down type's format. Their rows are written unweighted and the combine weights them like the GPU's. Which experts go is set per layer from the measured CPU and GPU cost per expert, so it can differ from run to run | prefill.cpp:1862-1900,1928-1985,2339-2352 |
| Indexer scores | 3xTF32 tensor cores | qsa_select.hpp:36-38 |
| GDN | Chunked delta rule, 16 tokens per chunk: q/k L2-normalized in fp32 then rounded to BF16; v, gates and the state fp32. K S on TF32 tensor cores, Q S and the state update K^T Delta on BF16 tensor cores (Delta rounded to BF16 for that product only), fp32 accumulate; the within-chunk solve in fp32. A token's output depends on where the 16-token chunk grid falls (about 1e-3 relative). Out-projection input fp16. Against the previous kernel: output relative RMS 2.9e-3. Quality gate on the NVIDIA-checkpoint NVFP4 pack (dense Q8_0), 20 chunks through the decode path against the Q8_0 reference, same pinned tiers, 400 W: PPL 2.2913 to 2.2916, mean KLD 0.0687 to 0.0706, top-1 agreement 92.98% to 93.31% (previous build to this one) | gdn_chunked.cu |

**CPU share of the experts.** `tests/cuda/prefill_cpu_share_test.cpp` runs the pool and the MMQ path on random native
blobs at the model's geometry, up to 8 rows per expert, against a double-precision reference. Relative RMS error of the
expert outputs, MMQ / CPU: 1.0e-2 / 1.2e-2 with plain activations (all four format pairs), 1.4e-2 to 1.6e-2 / 3.1e-2 to
3.4e-2 when 0.5% of the activations are 12 times larger (the CPU's Q8_K block is 256 values wide, MMQ's 32). A row's output
does not depend on how many tokens share its expert's job (bitwise, every pair). Quality gate, IQ3_S, 100 chunks, 400 W,
the 256-token prompt half through this path: PPL 3.3727, KLD 0.15267, top-1 87.22% against 3.3769, 0.15351, 87.19%
for the build without it. The share is a fixed 0.6 of the streamed experts by default (`--prefill-cpu-share`), not a timing-driven value, so which experts the CPU computes depends
only on the routing and the tiers, and prefill is reproducible bit for bit between runs of the same configuration
(six identical greedy outputs on the NVFP4 pack and on Q4_K_XL). A different tier layout (cache size, peer slots) moves the
CPU's experts and with them the rounding, so identity checks compare sessions of one configuration.

**Expert path by chunk size.** A prompt chunk of fewer than 6,144 tokens stages only the experts its tokens route to and
runs them on MMQ, or on the FP4 tensor cores for NVFP4 experts from 1,024 tokens (the FP4 row above), with the CPU pool taking the low-row non-resident ones; a chunk of 6,144 tokens
or more streams every non-resident expert through the ring and runs the fused kernels (NVFP4 has none: it stays on the FP4 tensor cores, or MMQ below 1,024 tokens, so the cutoff there only changes which experts are staged). Staging everything costs the same
whatever the chunk, so it only pays once most experts are hit anyway. Chunks of 1,024
to 6,143 tokens (a whole prompt of that length on a bare `--prefill auto`) take the MMQ path, where a lower cutoff would
send them to the fused kernels (packs with fused kernels; an NVFP4 pack runs the same FP4 or MMQ products either way); the
two paths' rounding differs as the tables above describe. The quality gate's 256-token prompt half is below both
cutoffs, so the cutoff does not move its numbers. On 12 chunks of 4,096 tokens (2,048-token prompt halves, PPL only),
PPL is 2.9581 against 2.9547 with the lower cutoff, a paired per-token NLL difference of +0.0011 +- 0.0011 (400 W, one
5090). With a peer card without direct
access the cutoff stays at 2,560 (4,096 with `--peer-exclusive`).

**Prompt chunk size.** A bare `--prefill auto` reads a prompt in equal chunks of at most 16384 tokens, or 32768 for packs
whose expert blobs exceed 1.5 MB (NVFP4). A chunk's GEMM tiles set the rounding of its K/V, so a prompt longer than 16384
tokens on such a pack is chunked differently and its K/V can differ in the last bits; prompts up to 16384 tokens and the
256-token prompt half of the quality gate chunk as before, so the gate's numbers do not move.

**K-quant experts on MMQ.** `tests/cuda/prefill_mmq_kq_test.cpp` runs MMQ and the FP16 dequant + cuBLAS path it
replaced on random zero-centred experts at the model's geometry against a double-precision reference. Relative RMS
error of the expert outputs, MMQ / FP16 path: 1.08e-2 / 5e-4 with plain activations (Q4_K/Q5_1, Q4_K/Q8_0, Q5_K/Q8_0),
1.48e-2 to 1.51e-2 / 5e-4 with 0.5% of the activations 12 times larger (Q6_K/Q8_0, MMQ only: 1.17e-2 plain, 1.55e-2 with outliers); with llama.cpp's float partial sums MMQ read
1.21e-2 to 1.76e-2 and 1.77e-2 to 2.19e-2. A row re-run alone or in another batch differs from its row in the full batch
by at most 7e-4 of the row's largest output: MMQ's stream-k split and tile width follow the launch's size, and H's int8
rounding can turn a reordered sum into one int8 step. The i-quant MMQ path does the same (6.5e-4, IQ4_XS/IQ4_NL). Quality
gate, UD-Q4_K_XL, 100 chunks, 400 W, the 256-token prompt half through this path: PPL 3.3297, KLD 0.05653, top-1 92.06%
against 3.3223, 0.05695, 92.22% with the FP16 path.

## 6-bit dense projections (NVIDIA NVFP4 pack)

The official NVIDIA NVFP4 pack stores its dense projections as Q6_K where the Q8_0 variant stores them as Q8_0. This is
a weight-format change: `convert.py --checkpoint` quantizes each projection once from NVIDIA's BF16.
`--requant-dense-q6k PACK --checkpoint DIR` gives the same bytes. Without `--checkpoint` it quantizes from the pack's Q8_0
values, which rounds twice; the official file is not made that way.

- **What moves:** attention q/k/v/o, GDN qkv, z and out, and the shared experts' gate and up: 252 tensors, 2,948 MiB of
  dense weights down by 654 MiB. The shared experts' down (640-value rows are not whole 256-value blocks), the head and
  the token embedding stay Q8_0; the router, hyper-connection matrices, indexer, ssm_alpha/beta and PLE key/value stay
  BF16; the experts, the PLE table, the drafter and vision are untouched.
- **Source and rounding:** each tensor is quantized once from NVIDIA's BF16 weight, never from a Q8_0 copy (which
  would round twice). The GGUF lists the 48 GDN value heads in a different order from the checkpoint; the converter
  reorders them. `--requant-dense-q6k --checkpoint` compares each BF16 tensor with the pack's Q8_0 copy first (the
  largest distance is 0.62 Q8_0 steps; it stops above 1.05, which catches a wrong tensor or head order); `--checkpoint`
  with `--sections-from` reads the BF16 tensors by name and gives the same Q6_K bytes.
- **Method:** per 256-value block, least-squares scales per 16 values from a signed-max starting grid, 8-bit scales
  against one fp16 super-scale, values rounded against the scales as stored (ggml's `block_q6_K` layout). The error is
  about 1.9% of sigma on Gaussian groups, and the converter tests bound it by the 6-bit step.
- **Why it is faster:** 6-bit dense weights take the tensor-core dense path in decode, while Q8_0 dense weights fall back to
  the mmvq kernel, and they are about 22% fewer bytes (654 of 2,948 MiB).

Speed, RTX 5090 at 400 W with a 5060 Ti peer card, the two packs alternated in one session (ABBA, 6 fresh processes per
arm), development measurement; mean over 6:

| test | 8-bit dense tok/s | 6-bit dense tok/s | change | ms per window |
|---|---:|---:|---:|---|
| prose | 213.4 | 229.9 | +7.8% | 10.00 to 9.24 |
| conversation | 251.0 | 265.1 | +5.6% | 10.34 to 9.35 |
| structured | 420.4 | 463.5 | +10.2% | 15.88 to 14.39 |
| prefill, 8K and 32K | 3,806 and 7,791 | 3,845 and 7,856 | +1% | |

Draft acceptance is unchanged within the run-to-run spread, so the gain is window time.

Quality, against the Q8_0 reference through the decode path, 100 chunks, 400 W (three runs of the Q8_0-dense pack on three builds, KLD spread
0.0009; two of them are in the quality table below):

| dense format | PPL | KLD mean | KLD p99 | same top-1 |
|---|---:|---:|---:|---:|
| Q8_0 | 3.287-3.294 | 0.0612-0.0621 | 0.692-0.699 | 91.80-91.87% |
| Q6_K | 3.3081 | 0.0680 | 0.775 | 91.46% |

The paired per-chunk KLD difference is +0.0063 (95% CI +0.0048 to +0.0077), so the loss is outside the run-to-run noise:
about 10% of the KLD and 0.4 points of top-1. On long prompts (12 chunks of 4,096 tokens, no Q8_0 reference, so PPL only)
PPL goes from 2.9547 to 2.9816; paired NLL +0.0091 (95% CI +0.0049 to +0.0132). Q6_K dense runs through dense_tc and
dense_tc_pair; the same arithmetic applies as for any Q6_K dense tensor (q8_1 activations, int8 tensor-core dot).

## Two-stage verify head

A Q6_K output head (the IQ3_S pack's) runs in two stages in the verify window (`q6k_vhead`, q6k_head.cu). Every one of the
248,320 rows is scored from a 2-bit copy (the draft head's fit, head_proxy.cuh) on int8 tensor cores against the same
q8_1 codes the exact kernel reads. Each block of rows adds its per-column maximum to a histogram; the lower edge of the bin
holding the 64th largest block maximum is at or below the column's 64th largest approximate logit, and every row within 4
of it is rescored exactly with the plain Q6_K kernel's arithmetic (the same bits). The other rows keep their approximate
logits, so log-probabilities stay finite. The block partition does not depend on the window width, so a token's logits
are the same bits at every T.

What it changes: the argmax and the sampler's top-64 list come from exact rows unless the copy's error exceeds the margin;
the softmax tail outside the rescored rows carries the copy's error (rms about 0.7 logit).

Offline, on 3,614 real verify-head inputs dumped from the engine (conv, struct, prose fixtures), all 248,320 rows, exact =
Q6_K x q8_1 in float:

| Fixture | Inputs | Top-64 recall, margin 4 | Margin 6 | Margin 8 | Rows rescored per column, margin 4 (mean / p99) | Greedy pick unchanged | KLD exact -> two-stage (mean) |
|---|---|---|---|---|---|---|---|
| conv | 1,232 | 99.971% | 100% | 100% | 1,333 / 6,286 | 100% | 4.7e-4 |
| prose | 1,124 | 99.979% | 100% | 100% | 1,098 / 2,988 | 100% | 5.1e-4 |
| struct | 1,258 | 99.968% | 100% | 100% | 1,981 / 12,790 | 100% | 2.5e-5 |

Margins 6 and 8 rescore 5,569-18,649 and 27,962-78,806 rows per column. Neither a least-squares scale for the copy nor a
per-column correction of the tail's bias lowers the KLD (the tail's mass is unbiased; the cost is its noise).

Quality gate (IQ3_S, 100 chunks, against the Q8_0 base, tiers pinned to `--expert-cache 9000 --peer-slots 7562`, 400 W;
the gate leaves the PCIe share dynamic, so two runs of one build differ):

| Head | Run | PPL | KLD mean | KLD median | KLD p99 | Same top-1 |
|---|---|---|---|---|---|---|
| exact Q6_K | 1 | 3.3778 | 0.15355 | 0.03400 | 1.7316 | 87.36% |
| exact Q6_K | 2 | 3.3728 | 0.15288 | 0.03397 | 1.7082 | 87.18% |
| two-stage | 1 | 3.3852 | 0.15507 | 0.03484 | 1.7681 | 87.24% |
| two-stage | 2 | 3.3827 | 0.15472 | 0.03497 | 1.7381 | 87.40% |

The two-stage head raises KLD by about 0.0017 (1.1%) and PPL by 0.2%, more than the 0.0007 between two runs of one build:
the gate scores the whole softmax, including the approximate tail. Same top-1 is unchanged. Greedy decoding with the tiers
pinned (`tools/greedy_identity.py`, the three bench prompts, 900 tokens) is token-identical to the exact head.

## Request logit bias

`logit_bias` (Chat Completions) is applied by the verifier's own sampling, on every row of every window, so a bias is
part of the distribution speculation must reproduce: the host-side sampling that already follows each window (the same
one a temperature or a penalty uses) runs with the bias, and a draft is accepted only if it equals that pick. The draft
head ignores the bias; it only proposes tokens, so a banned token it proposes is rejected, and a biased request
accepts fewer drafts. A request without a bias takes the unchanged captured greedy sample.

Checked on IQ3_S with the tiers pinned as in the section below: requests with a bias (greedy and seeded, a ban, a
boost, a mixed map, with and without a presence penalty) return the same text with `--spec 7 --spec-min-p 0.5` as with
`--spec 0`, and a request with an empty map matches one without the field. A -100 bias never yields the token and
a +100 bias yields it at every step.

## Speculative decoding does not change output

Verify guarantees that a token's logits are bit for bit those of plain decode, whatever the window width
(verify.hpp:3-8). This section tests that end to end.

**Method.** 275 prompts: 272 chat-templated prompts with thinking off (152 prose, 60 chat, 60 code) plus the 3
`data/bench` decode prompts (conv, struct, prose). 256 new tokens each, temperature 0, one engine session per
configuration, same prompt order. A logit-hash diagnostic prints an FNV-1a hash of the full-vocabulary fp32 head logits
of every accepted row. Token ids and the hash at each generated position are compared between `--spec 0` and
`--spec 7 --spec-min-p 0.5`.

Expert tiers are pinned (`--adapt-swaps 0 --peer-adapt-swaps 0 --pcie-frac 0`, with `--expert-cache` and `--peer-slots`
fixed to the sizes auto-sizing picks), so the same experts sit on the same device in both runs.

| Pack | Prompts | Identical tokens | Logit rows compared | Bit-identical rows | First divergence | Accepted drafts | Decode tok/s, spec 0 → spec 7 |
|---|---|---|---|---|---|---|---|
| IQ3_XXS | 275 | 275 (100%) | 62,271 | 62,271 | none | 37,892 / 54,206 (69.9%) | 183.9 → 319.8 |
| IQ3_S | 275 | 275 (100%) | 60,878 | 60,878 | none | 36,650 / 52,312 (70.1%) | 145.1 → 239.5 |

Throughput is decode-only over these 275 requests, at 400 W. Verify windows in the spec 7 run have widths 1 to 7, so the
compared rows include logits computed at every window width T, all equal to the T=1 logits of the spec 0 run. On an
earlier build, a repeated IQ3_XXS spec 0 run was bit-identical to the first (62,491 of 62,491 rows), so the setup is
deterministic run to run.

### Forced-window KLD

The quality gate is teacher-forced: the window tokens are the reference text, forced to width `--t`, and the drafter
output is not used (`src/program/generate.cpp`, the `ppl != nullptr` branches). Forcing the window widths the drafter
produces exercises the same verify kernels as real drafting.

IQ3_XXS, 20 chunks of wikitext-2, against the llama.cpp Q8_0 base (base PPL 2.2815), int8 KV, tiers pinned, 400 W:

| Forced window T | PPL | KLD mean | KLD median | KLD p99 | Same top-1 | rows.tsv md5 |
|---|---|---|---|---|---|---|
| 1 | 2.5330 | 0.27827 | 0.02674 | 3.5108 | 86.80% | 179c7f42f1e6d89459bf27f780e99b35 |
| 3 | 2.5330 | 0.27827 | 0.02674 | 3.5108 | 86.80% | 179c7f42f1e6d89459bf27f780e99b35 |
| 7 | 2.5330 | 0.27827 | 0.02674 | 3.5108 | 86.80% | 179c7f42f1e6d89459bf27f780e99b35 |

All 5,100 per-row lines (NLL and KLD) are byte identical across T=1, 3 and 7 (`ppl_gate.py --diff`: max |dKLD|
0.000000). Window width does not change the logits, so KLD with MTP on equals KLD with MTP off; the 0.278 is the quant
against the base.

### Scope

- Identity is bitwise with the expert tiers pinned. With adaptation on, experts move between the CPU pool and the GPU,
  which round differently: a rounding difference, not a quality difference.
- The GPU tiers (VRAM and peer card) share one arithmetic; the CPU tier quantizes with ggml's vec_dot_type (Q8_K per 256
  for i-quant gate/up, against q8_1 per 32 on the GPU; native_expert.cpp:52-62). In the quality gate about 0.8% of
  routed experts were computed on the CPU (IQ3_XXS, default placement). The split is fixed per placement, so it is the
  same for spec 0 and spec 7.
- Prompts are 19 to 395 tokens with 256 new tokens: this tests the decode path, not long context.

## Relaxed drafting

`--relaxed-drafting P` (opt-in; off by default) gives up the guarantee above for sampled requests. In exact mode a draft
is kept only when it equals the target's own draw at that row (`src/program/generate.cpp`, the acceptance loop after
`ver.run`). In relaxed mode a draft is also kept when the target gives it probability P or more under the request's
final distribution (top_k, top_p, min_p, temperature and penalties applied). The sampler reports that probability next to
its draw (`sample_tokens_probe`, src/kernels/cuda/sampler.cu); the logits and the draws themselves do not change.

**What changes.** The drafts are the draft head's argmax, fixed by the context. At a row where the draft d has
p(d) ≥ P, the output is d every time instead of a draw from p: that row is greedy. Its total-variation distance from the
model's distribution is exactly 1 − p(d), and 0 at every other row. Rows after a kept draft are conditioned on it,
as plain decoding would be. What stays exact: greedy requests (nothing is drawn), requests with `logit_bias`, JSON
`response_format` (the server sends `relax=0`), the quality gate, `--parallel` batch slots, any drafted added token
(ids from `<|endoftext|>` on: control tokens, think and tool tags, end of turn), and every token from `<tool_call>` to
`</tool_call>`, so a tool name and its arguments are sampled exactly.

**Exact mode is unchanged.** With the flag off and the tiers pinned, IQ3_XXS, the three bench prompts greedy plus conv
and prose at temperature 0.6 and 1.0 with two seeds each (600 tokens): all 11 token streams are identical to the build
before the flag. With the flag on, the three greedy streams are identical too.

**Distribution cost per token (offline replay).** Real verify windows were logged (per row: the draft, the target's
draw, the target's final probabilities) for the bench prose and conv prompts (three seeds) and five natural chat
prompts, at temperature 0.6 and 1.0 (top_p 0.95, top_k 20), on IQ3_S and IQ3_XXS, about 3,500 tokens per cell. The
replay keeps a draft when the rule would, and averages 1 − p(d) over the relaxed rows per generated token
(TV/token; 0 for exact). Token gain is the replayed extra tokens per window at the logged window widths; the live
gain below is larger because the window controller lengthens windows when more drafts are kept.

| Model | Prompts | T | Exact accept | P = 0.5 gain / TV | P = 0.3 gain / TV | P = 0.2 gain / TV |
|---|---|---|---|---|---|---|
| IQ3_XXS | prose | 0.6 | 64.1% | +4.3% / 0.026 | +6.9% / 0.042 | +9.2% / 0.052 |
| IQ3_XXS | natural | 0.6 | 64.5% | +5.3% / 0.030 | +8.4% / 0.050 | +10.7% / 0.063 |
| IQ3_XXS | prose | 1.0 | 61.6% | +5.6% / 0.035 | +11.1% / 0.064 | +13.9% / 0.080 |
| IQ3_XXS | natural | 1.0 | 62.9% | +6.6% / 0.043 | +12.1% / 0.073 | +15.2% / 0.089 |
| IQ3_S | prose | 0.6 | 63.7% | +3.0% / 0.022 | +5.9% / 0.037 | +7.2% / 0.046 |
| IQ3_S | natural | 0.6 | 64.5% | +4.9% / 0.030 | +8.0% / 0.049 | +9.8% / 0.060 |
| IQ3_S | prose | 1.0 | 61.1% | +5.1% / 0.035 | +9.3% / 0.061 | +12.4% / 0.079 |
| IQ3_S | natural | 1.0 | 62.3% | +7.1% / 0.042 | +11.9% / 0.072 | +14.7% / 0.089 |

Every relaxed keep moves 1 − p(d) of probability mass, so tokens gained per unit of TV is nearly the same at every
threshold: about +1.5% per 0.01 TV/token. Medusa-style typical acceptance (a threshold that falls with the entropy of p)
was replayed too and lands on the same line; it is not a better trade, so the flag is the plain probability threshold.

**Speed (400 W).** `basalt-bench` (conv and prose at temperature 0.6, struct greedy), with `--spec 7
--spec-min-p 0.5 --kv int8 --vram-reserve-mib 500 --resident-experts --peer-exclusive`, each value in three rounds
interleaved with the others, the cold repetition dropped:

| Model | Fixture | Exact | P = 0.5 | P = 0.3 | P = 0.2 |
|---|---|---|---|---|---|
| IQ3_XXS | prose | 348.7 | 366.8 (+5.2%) | 373.8 (+7.2%) | 382.4 (+9.7%) |
| IQ3_XXS | conv | 383.8 | 421.5 (+9.8%) | 431.4 (+12.4%) | 423.3 (+10.3%) |
| IQ3_XXS | struct (greedy) | 660.2 | 658.5 | 659.4 | 658.0 |
| IQ3_S | prose | 323.8 | – | 347.2 (+7.2%) | – |
| IQ3_S | conv | 354.2 | – | 392.9 (+10.9%) | – |

In one engine session per model, the prose prompt and the five natural prompts, two seeds, the four settings
interleaved per prompt (decode tok/s, 400 W):

| Model | T | Exact | P = 0.5 | P = 0.3 | P = 0.2 |
|---|---|---|---|---|---|
| IQ3_XXS | 1.0 | 332.5 | 366.9 (+10.3%) | 385.5 (+15.9%) | 399.1 (+20.0%) |
| IQ3_XXS | 0.6 | 351.2 | 370.1 (+5.4%) | 383.1 (+9.1%) | 380.9 (+8.5%) |
| IQ3_S | 1.0 | 336.3 | 356.8 (+6.1%) | 369.7 (+9.9%) | 392.3 (+16.7%) |
| IQ3_S | 0.6 | 347.0 | 359.2 (+3.5%) | 370.9 (+6.9%) | 382.4 (+10.2%) |

**Quality.** The teacher-forced KLD gate cannot see this change: it scores logits, and relaxed drafting changes which
token is emitted, not the logits. So the evidence is the replayed TV above plus sampled tasks. IQ3_S through the server
with the same flags, temperature 1.0, top_p 0.95, top_k 20, thinking off, one seed per item so the settings are
paired, the four settings in one server session (`relaxed_drafting` per request), 400 W:

| Task | Exact | P = 0.5 | P = 0.3 | P = 0.2 |
|---|---|---|---|---|
| gsm8k, 300 items (flexible extract) | 0.947 ± 0.025 | 0.973 (+0.027 ± 0.020) | 0.980 (+0.033 ± 0.022) | 0.977 (+0.030 ± 0.023) |
| MMLU-Pro, 20 per subject (280) | 0.682 ± 0.055 | 0.714 (+0.032 ± 0.040) | 0.725 (+0.043 ± 0.045) | 0.721 (+0.039 ± 0.046) |
| Drafts kept by the rule, per generated token | 0 | 3.0% | 5.0% | 5.6% (with the tool requests) |

The ± are 95% intervals (paired difference against exact in brackets). An earlier session in which every arm ran exact
(the per-request key had not reached the engine) scored gsm8k 0.960 to 0.970 and MMLU-Pro 0.696 to 0.714 across four
exact runs, with 6 to 9 and 24 to 35 items flipping between runs (the adaptive expert tiers round differently run to
run): this session's exact arm is at the low end of that spread. No loss is measurable at this size; the relaxed arms
sit at or above the exact runs, as expected of a change that makes confident rows greedy (a lower effective
temperature). The cost is diversity, which these tasks do not measure; TV/token above is the direct measure.

Tool calls, IQ3_S, the same sampling, 50 requests over 8 tools (similar names such as `replace_in_file` and
`read_file`), thinking on for half: exact 53 calls and P = 0.2 52 calls, every name a declared tool, no leaked call
text, all arguments valid JSON. The engine log shows the rule kept almost no drafts in these requests (the calls
themselves are exact).

**Choosing P.** `--relaxed-drafting` alone means 0.5: the smallest cost (TV/token 0.03 to 0.04), roughly +5% at
temperature 0.6 and +6 to 10% at 1.0. 0.3 doubles both the gain and the cost; 0.2 is the aggressive end. Values above
0.5 keep few drafts (0.7: about +2.5% at TV 0.015).

## How quality is measured

`basalt-bench --quality` (`tools/ppl_gate.py`) scores wikitext-2 (`wiki.test.raw`) in 512-token chunks. The first half
of each chunk is prefilled and the second half (rows 256 to 510) is scored through the decode path, against a llama.cpp
Q8_0 base file written by `llama-perplexity --kl-divergence-base`, read in llama.cpp's own `.kld` format.

The harness (`include/basalt/program/ppl_gate.hpp` (the KLD function)) computes KL(base ‖ test) over the full vocabulary from the
fp32 log-softmax of the head logits; the base is decoded as scale*u16+min, and terms with base log-prob ≤ -16 are
skipped (a tiny underestimate). The prompt half is checked equal to the base tokens, the window tokens are the base
tokens, and KLD is flat across positions, so the two are aligned.

### Against llama.cpp

llama.cpp (commit 436f6f89e, CUDA) scores the same second half of each chunk, with the whole chunk in one batched
forward pass. 100 chunks, base PPL 3.286070, 400 W:

| Pack | Engine | PPL | KLD mean | KLD median | KLD p99 | Same top-1 |
|---|---|---|---|---|---|---|
| GSQ-RCO IQ3_S | llama.cpp | 3.3728 | 0.1526 | 0.0340 | 1.750 | 87.20% |
| GSQ-RCO IQ3_S | Basalt | 3.3723 | 0.15264 | – | – | 87.16% |
| GSQ-RCO IQ3_XXS | llama.cpp | 3.4675 | 0.2428 | 0.0596 | 2.607 | 84.05% |
| GSQ-RCO IQ3_XXS | Basalt | 3.4678 | 0.2433 | – | – | – |

At 20 chunks (Basalt: the README table; llama.cpp: its running value at chunk 20):

| Pack | Basalt KLD / top-1 | llama.cpp KLD / top-1 |
|---|---|---|
| GSQ-RCO IQ3_S | 0.177 / 89.4% | 0.1753 / 89.22% |
| GSQ-RCO IQ3_XXS | 0.277 / 86.8% | 0.2780 / 86.65% |
| UD-Q4_K_XL | 0.068 / 93.0% | 0.0579 / 93.96% |

NVFP4 and UD-Q4_K_XL in Basalt only (same harness and base, T 3, int8 KV, `--resident-experts`, expert tiers pinned (`--adapt-swaps 0 --peer-adapt-swaps 0`), 400 W; a
development measurement):

| Pack | Chunks | PPL | KLD mean | KLD median | KLD p99 | Same top-1 |
|---|---|---|---|---|---|---|
| UD-Q4_K_XL | 100 | 3.3201 | 0.05656 | 0.01213 | 0.6586 | 92.27% |
| NVIDIA-checkpoint NVFP4 (dense Q8_0) | 100 | 3.2873 | 0.06167 | 0.01345 | 0.6923 | 91.80% |
| NVIDIA-checkpoint NVFP4 (dense Q8_0), combined build | 100 | 3.2870 | 0.06115 | 0.01365 | 0.6992 | 91.87% |
| NVIDIA-checkpoint NVFP4, dense Q6_K (official) | 100 | 3.3081 | 0.06799 | 0.0150 | 0.775 | 91.46% |
| UD-Q4_K_XL | 20 | 2.3223 | 0.0685 | 0.00514 | 0.9928 | 93.02% |

The NVIDIA-checkpoint (dense Q8_0) row was run on the build before the FP4 prompt path, T 3, int8 KV, resident experts, 400 W, base PPL 3.28607.
Against UD-Q4_K_XL it has 0.0051 higher KLD and 0.47 points lower top-1, and a lower PPL (3.2873 against 3.3201), which
is not a quality signal.

The combined-build row (FP4 prompt experts, chunked delta rule, FP16-MMA prompt attention, AVX2 CPU experts, 6,144-token
stream cutoff) used the same flags as the row above, 400 W. The 256-token prompt half is below the 1,024-token FP4 minimum,
so the gate exercises MMQ prompt experts, and the change from that row (KLD -0.0005, top-1 +0.07 points, PPL -0.0003) is
inside run noise. Long prompts (2,048-token halves, 12 chunks, PPL only) on the same build: 2.9579 against 2.9547 for
MMQ (standard error 0.037 each), paired per-row NLL difference +0.0011 (+-0.0014).

IQ3_S and IQ3_XXS agree between the two engines to within 0.002 KLD and 0.2 points of top-1 at both chunk counts. The
first 20 chunks read higher than the 100-chunk set (higher KLD and higher top-1 for the IQ3 packs), so the README table,
which uses 20 chunks, reads higher than the 100-chunk numbers; the 100-chunk numbers are the ones to compare. The one gap
is UD-Q4_K_XL at 20 chunks (0.068 vs 0.0579), explained in the next section.

## Why Q8_0 against Q8_0 is not zero

The Basalt Q8_0 pack scores KLD 0.0456 and top-1 95.6% (4 chunks, fp16 KV) against the stock llama.cpp Q8_0 base. This
is mostly the base's own noise. llama.cpp quantizes the activations of its Q8_0 matmuls. In the hyper-connection read
(down projection 10240 → 320, then up) the sum has cancellation, so Q8_0 quantization of the input (0.2 to 0.4% per
element) becomes about 1% in the mixed stream at layer 0. Basalt's hc read uses fp32 activations and matches the exact
math to 1e-5.

At layer 0, computed in numpy from llama.cpp's own hc_norm output and the GGUF Q8_0 hc weights, against an fp64 result
with no activation quantization: Basalt's mixed stream has relative error 0.00000 and llama.cpp's 1.06%. llama.cpp
against an emulation of Q8_0 activation quantization differs by 0.009%.

With the hc read cast to F32 in llama.cpp (a local patch), 4 chunks, 400 W:

| Comparison | KLD mean | Same top-1 |
|---|---|---|
| Basalt Q8_0 vs stock base | 0.0456 | 95.6% |
| Basalt Q8_0 vs F32-hc base | 0.0233 | 97.16% |
| llama.cpp Q8_0 with F32 hc vs stock base | 0.0436 | 96.0% |
| llama.cpp with ssm_alpha/beta activation quantization also removed vs F32-hc base | 0.0241 | 96.86% |

Basalt's 0.0233 is the level of two independent implementations of this model (0.0241 between two llama.cpp variants).
The stock base's hc noise lifts every pack's KLD by about 0.01 to 0.02 (Basalt, 4 chunks):

| Pack | Stock base | F32-hc base |
|---|---|---|
| UD-Q4_K_XL (hc Q8_0) | 0.0825 | 0.0748 |
| GSQ-RCO IQ3_S (hc BF16) | 0.2100 | 0.1950 |

The noise cancels only when the tested llama.cpp pack also has Q8_0 hc and quantizes the same way, which is why
llama.cpp against its own base scores better on UD-Q4_K_XL (0.0579 vs 0.068 at 20 chunks). Reported KLD numbers
therefore include about 0.01 to 0.02 of the reference's own noise. They are kept against the stock base because anyone
can regenerate it.

This is not a CPU-tier effect: the 4-chunk Q8_0 result holds whatever share of experts sits on the CPU pool (0.0456 to
0.0488 at the default placement, 0.0466 with 79% on the CPU). Other causes tested and excluded: KV mode, window width T,
`--gr-fp32-activations`, dense_tc against native_mmvq (not bit-identical: dense_tc takes exact int32 dots per 32-value
block, while mmvq rounds per 8-value quarter block, and the summation orders differ; switching it off made no
measurable difference to this gap), and refitting the hc8 scales.

## Compiler parity

The CPU expert kernel and the expert pool are compiled by whatever compiler the platform uses, so the two supported
platforms do not run the same object code: MSVC on Windows, GCC on Linux. What that costs in accuracy is measured, not
assumed. Both builds of one commit run the same two programs against the same extracted expert section — `expert_parity`
for the Q2_0 expert kernel and its AVX-512 VNNI path, `pool_test` for the pool that dispatches them:

| check | MSVC 19.44 (Windows) | GCC 13.3 (Linux) |
| --- | --- | --- |
| `expert_parity` | 0 failures | 0 failures |
| blob decodes to live scales | 40/40 non-zero, \|s\| in [0.0179, 0.03564] | same |
| the next expert is different data | 96.9% of bytes differ | same |
| VNNI vs scalar, INT8 activations | rel 1.111e-06 | rel 9.412e-07 |
| VNNI vs scalar, FP32 activations | rel 1.209e-02 | rel 1.139e-02 |
| `pool_test` | 0 failures | 0 failures |
| pool output vs the serial run | 0 of 25600 differ, worst \|d\| 0.000e+00 | same |

The structural checks and the pool output are identical, the pool to the bit. The two cross-compiler figures agree to
the third significant figure, which is the range FMA contraction and reduction order can move, and each sits three to
four orders of magnitude inside the 3e-3 tolerance the kernel asserts. The FP32 row is reported by the test, not
asserted: it is the cost of the Q8_1 activation contract, and it is the same cost on both compilers. No kernel in this
port changed, so the Linux numbers are unchanged from before it.

Scope: these two programs cover the Q2_0 expert kernel and the expert pool. The quantized kernels (`iq_avx512`,
`iq_avx2`, `kq_avx2`, `kq_avx1`, `native_expert`, `ggml_cpu_dots`) are compiled by both toolchains as well but no
program here exercises them with real weights, so this table is not evidence about them.

### Greedy tokens are not identical across the two builds

Agreement to a tolerance is not the same as bitwise identity, and the token stream shows it. The same pack, the same
commit, temperature 0, expert tiers pinned (`--expert-cache 4096 --resident-experts`), adaptive swaps off, the two
builds place experts identically (VRAM 10.2 GiB, RAM 36.7 GiB, 5341 slots) and generate the three bench prompts for
300 tokens each: `struct` is token-identical, `conv` diverges at token 23, `prose` at token 44. Run twice, one build
against itself under the same flags, all three prompts are token-identical, so the difference follows the build rather
than the run.

Two things follow. The port does not promise byte identity with Linux, and this is the measurement of what that means.
And it is the host compiler that is being measured, not the GPU toolchain: the Linux side was then rebuilt with CUDA
13.0 — the same toolkit release the Windows build used — and it diverges at the same two tokens with the same ids, so
the toolkit version accounts for none of it. What remains between the two builds is GCC against MSVC on the CPU, where
78% of the expert rows are computed, and the figures in the table above say how far apart they are.

Adaptive swaps must be **off** for any of this to mean anything: with them on, one build does not reproduce its own
tokens between runs, because the tier exchanges experts while it decodes.

The quality gate is the check that decides whether such a divergence is acceptable, and it needs a KL-divergence base
built by llama.cpp from a Q8_0 copy of the model. The Q8_0 pack is published, but no such base has been built on the
machine these numbers were taken from, so the gate has not been run on the Windows build.

Both programs read a flat expert section: 512 blobs of 1,382,400 bytes, the Q2_0 layout they were written for. A pack
carries one in its `mtp/experts` section, which copies out unmodified, so the same bytes run on either build:

```sh
python3 -c "import sys; sys.path.insert(0,'tools'); from model_file import ModelFile; \
  open('experts.bin','wb').write(ModelFile('PACK.basalt').read('mtp/experts'))"
./expert_parity --file experts.bin
./pool_test --file experts.bin
```

## Reproduce

Build llama.cpp, then the Q8_0 base from the Q8_0 GGUF (100 chunks):

```sh
llama-perplexity -m MODEL-Q8_0-00001-of-0000N.gguf -f wikitext-2-raw/wiki.test.raw \
    --kl-divergence-base flashnext-q8-base.kld -c 512 -b 2048 -ngl 99 --n-cpu-moe 99 -t 16
```

llama.cpp's numbers for a pack:

```sh
llama-perplexity -m PACK.gguf -f wikitext-2-raw/wiki.test.raw --kl-divergence-base flashnext-q8-base.kld \
    --kl-divergence -c 512 -b 2048 -ngl 99 --n-cpu-moe 99 -t 16
```

Basalt's numbers, and the forced-window identity check (run with `--t` 1, 3 and 7, then diff the rows):

```sh
python3 tools/ppl_gate.py --model PACK.basalt --engine-bin bin/basalt --base flashnext-q8-base.kld \
    --chunks 20 --t {1|3|7} --out DIR --expert-cache N --peer-slots M --kv int8 --resident-experts \
    --peer-exclusive --prefill auto
python3 tools/ppl_gate.py --diff DIR1/rows.tsv DIR7/rows.tsv
```

MTP on/off identity: one engine session per configuration, driven over `--serve`:

```sh
bin/basalt --serve --model MODEL.basalt {--spec 0 | --spec 7 --spec-min-p 0.5} \
    --expert-cache N --peer-slots M --prefill auto --kv int8 --vram-reserve-mib 700 --resident-experts \
    --peer-exclusive --peer-device 1 --max-context 8192 --adapt-swaps 0 --peer-adapt-swaps 0 --pcie-frac 0 \
    --exp concur9_logit_hash=1
```

Wait for a `READY` line on stdout, then write one request per line on stdin: `GEN 256 temperature=0 ID,ID,...` with the
prompt token ids. Each generated token comes back on stdout as `T ID`, and `DONE` ends the request; `QUIT` ends the
session. The logit hashes are printed on stderr as `LH` lines, one per accepted row with its position and hash.

Compare the token ids and the per-position hashes between the two sessions. With `--spec 0` there is one row per token;
the `--spec 7` run may print extra rows past the end of a request, so match rows by position. The pinned sizes used here
were N = 11582, M = 8876 for IQ3_XXS and N = 9979, M = 7562 for IQ3_S.
