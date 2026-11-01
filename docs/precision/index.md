# Precision and quality

This page shows what precision Basalt computes in, that speculative decoding (MTP) does not change its output, and how
its quality compares with llama.cpp on the same GGUFs against the same Q8_0 reference.

Test setup: one RTX 5090 at a 400 W power limit, an RTX 5060 Ti as the peer expert tier, 128 GB of system RAM
(121 GiB visible to the OS). Each section names the build it measured.

## Summary

- **MTP does not change output** (unless `--relaxed-drafting` is set: [details](features.md#relaxed-drafting)). With the expert tiers pinned, 275 of 275 greedy prompts produced identical tokens on
  both packs tested, and every full-vocabulary fp32 logit row (62,271 and 60,878) was bit-identical between `--spec 0`
  and `--spec 7`.
- **Lossy flags are opt-in.** `--relaxed-drafting`, `--route-prior` and `--route-tail-skip` change output; each is off by default and listed in one start warning ([relaxed drafting](features.md#relaxed-drafting), [routing](routing.md)).
- **KLD matches llama.cpp.** On the same GGUFs against the same base, Basalt and llama.cpp's own `llama-perplexity`
  agree: IQ3_S 0.15264 vs 0.1526, IQ3_XXS 0.2433 vs 0.2428 at 100 chunks, with the exact head. The two-stage
  verify head of a Q6_K head (IQ3_S) adds about 0.0017 ([details](features.md#two-stage-verify-head)).
- **Quantized once, never twice.** GGUF packs keep the GGUF blocks as shipped. The NVIDIA NVFP4 pack keeps NVIDIA's
  NVFP4 expert blobs bit for bit and quantizes each dense tensor once from NVIDIA's BF16: Q6_K for the projections
  ([details](features.md#6-bit-dense-projections-nvidia-nvfp4-pack)), Q8_0 for the head, token embedding and shared-expert down. Its
  n-gram table is NVIDIA's FP8 values re-blocked to Q8_0 (one more rounding). Decode also reads the hyper-connection
  matrices from an int8 copy (one fp16 scale per 32 inputs), and a Q6_K output head is screened through a 2-bit copy in
  verify: the rows near the top are the exact Q6_K logits, the far tail keeps the copy's approximate logits
  ([details](features.md#two-stage-verify-head)).
- **fp32 where it matters.** Activations are fp32 in the residual stream, norms, router, softmax and logits, and int8
  (q8_1) into the quantized matmuls. GDN state is fp32. KV is fp16 by default, with int8, q4_0 and k8v4 options.
- **Quality is scored through decode.** Prefill is a different computation (fp16/BF16 GEMMs, int8 MMQ, and FP4 tensor cores for NVFP4 experts); the quality gate scores
  the decode path.
- **The reference has its own noise.** The reference is a llama.cpp Q8_0 base, which carries about 0.01 to 0.02 KLD of
  its own noise ([details](quality.md#why-q8_0-against-q8_0-is-not-zero)).

## Decode path precision

Sources are `file:line` under `src/`, `include/basalt/` and `tools/convert/`. Model geometry: n_embd 2560, 4
hyper-connection streams, 48 layers (36 GDN, 12 QSA), 512 experts, top-10, head_dim 256.

### Weights

| Tensor class | Format | Source |
|---|---|---|
| Routed experts | GGUF blocks copied byte for byte, no requantization. Gate and up share one type per layer; types are mixed across layers (IQ3_XXS pack: gate/up IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S; down Q2_0 on 30 layers, IQ4_NL on 18. IQ3_S pack: gate/up mostly IQ3_XXS, IQ3_S, IQ2_S, one IQ4_XS; down IQ4_NL on 39, Q2_0 on 9) | docs/model-file.md; iq_kernels.cu:536-567; tests/data/native_experts/ |
| Routed experts, NVFP4 format | NVFP4 (type 40): per 16 weights, 8 bytes of E2M1 codes and one UE4M3 scale; per expert a 16-byte FP32 tail {s_gate, s_up, s_down, 0}. 4.5 bits per weight, all 48 layers | tools/convert/checkpoint.py; native_experts.txt; iq_kernels.cu (nvfp4_scale_dev) |
| NVIDIA-checkpoint pack (`--checkpoint`) | Built straight from NVIDIA's safetensors. The experts are NVIDIA's NVFP4 codes and UE4M3 scales, bit for bit, moved into Basalt's 64-value blocks; each expert's tail is its three `weight_scale_2` values (the `input_scale` entries are not used). The dense tensors are converted once from BF16: attention, GDN qkv/z/out and the shared experts' gate/up Q6_K (Q8_0 with `--dense-q8-0`), shared-expert down, head and token embedding Q8_0 (ggml's rounding); router, hyper-connection matrices, indexer, ssm_alpha/beta and the PLE key/value BF16 as stored, so nothing is requantized for the small matrices; norm weights BF16 + 1 in F32 (exact); ssm_a = -exp(A_log) computed in f64 and rounded once to F32; the GDN value heads reordered into the GGUF order. The n-gram table is the checkpoint's FP8 E4M3 codes times its one BF16 scale, re-blocked to Q8_0 (blocks of 32, fp16 scale = absmax/127, ggml's rounding): at most half a Q8_0 step plus the scale's fp16 rounding from the exact FP8 value, against the 4.5-bit IQ4_NL table of the other packs | tools/convert/checkpoint.py; convert.py (from_checkpoint); model_sections.py (PleFp8); test_checkpoint.py; test_convert.py (PleFp8Test) |
| Shared expert gate/up/down | GGUF blocks, any native_mmvq type | native_mmvq.hpp:120 |
| Shared-expert scalar gate | BF16 | model_sections.py:48 |
| QSA q/k/v/o, GDN qkv, z (attn_gate), ssm_out | GGUF blocks (dense_tc takes Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, Q3_K, Q2_0, Q8_0; native_mmvq otherwise). Q8_0 on dense_tc is not bitwise equal to native_mmvq: the int8 tensor-core dot is an exact int32 per 32-value block, then the fp32 scale terms are summed in a fixed order, where mmvq rounds `d_w * d_x * sumi` per 8-value quarter block in a warp tree. Against a float64 reference both sit at about 1e-7 RMS (dense_tc_parity, real projection geometry, T = 1..8, every column bitwise independent of T). Quality gate on the Q4_K_XL pack (20 chunks, decode path, 400 W): base PPL 2.3272 / KLD 0.0687 / p99 1.054 / same top-1 92.80%, with Q8_0 on dense_tc 2.3269 / 0.0685 / 1.044 / 93.08%; all deltas are inside the run-to-run noise of a 20-chunk run. The official NVIDIA NVFP4 pack stores these (and the shared-expert gate/up) as Q6_K from NVIDIA's BF16 weights, so they take dense_tc ([details](features.md#6-bit-dense-projections-nvidia-nvfp4-pack)) | dense_tc.cu:1109-1115; dense_tc.hpp:2-12 |
| GDN ssm_alpha, ssm_beta; QSA indexer q/k projections | BF16 | model_sections.py:52-53 |
| Router | BF16, not quantized | model_sections.py:48 |
| Norm weights, ssm_a, ssm_dt, conv1d, ssm_norm | F32 | model_sections.py:55-59 |
| Hyper-connection (hc) matrices (down, up, inject; per layer and for the output head) | Stored in the pack as the source has them (BF16 in the GSQ-RCO and NVIDIA packs, Q8_0 in UD-Q4_K_XL); decode reads an int8 copy: int8 codes with one fp16 scale per 32 consecutive inputs, scale = f16(absmax/127), codes clamped to ±127 | hc_fused.cu:1200-1232; verify/hc8.cpp |
| Token embedding | GGUF type kept in mapped host memory, one row dequantized to fp32 per token. The first 8192 rows (the lowest ids) also have a VRAM copy that the gather reads when a token is among them; the same bytes through the same dequantizer, so the same numerics | native_head.hpp:46-49; native_head.cpp (iq_embed_rows call); iq_kernels.cu (iq_embed_rows); verify/record_inputs.cpp; native_expert_parity --q8-embd-hot |
| Output head | A native_mmvq type with dedicated Q5_K and Q6_K kernels, and for a Q8_0 head a one-wave kernel at windows of four tokens or more (the same bits as native_mmvq); the IQ3_S pack's head is Q6_K. A Q6_K head runs in two stages in verify: every row is scored from a 2-bit copy (the draft head's fit below: four levels and an fp16 scale per 32 weights, the scale raised by sqrt(\|w\|²/⟨w, ŵ⟩)), on int8 tensor cores; the rows within 4 of a lower bound on the column's 64th-best approximate logit are rescored exactly (the bits of the plain Q6_K kernel), and every other row keeps its approximate logit. The argmax and the sampler's top-64 list come from exact rows; the tail of the softmax is approximate (see [Two-stage verify head](features.md#two-stage-verify-head)); a request sampling a longer list has its windows' head run again exactly. The copy (199 MB for 248,320 rows) is packed at load, before the expert cache is sized | native_head.cpp; verify/record_head.cpp; q6k_head.cu (q6k_vhead); q80_head.cu |
| MTP drafter dense | Q8_0 for fc_embedding, fc_hidden, q/k/v/o, indexer projection and the shared-expert projections; BF16 for the rest; 1-D tensors F32 | model_sections.py:447-492 |
| MTP drafter experts | Q2_0, all in VRAM | model_sections.py:496-507 |
| MTP draft head | Rows of the main head (same type), optionally a vocabulary subset: a Q6_K head reads them from the main head through the subset's row map, others gather a copy (the same bits either way); for a Q5_K or Q6_K head the leading rows are first scored from a 2-bit copy (four levels and an fp16 scale per 32 weights) and the rows within 5 of the best approximate score (at most 64) rescored exactly, so a draft differs from the exact argmax only in rare cases, and the draft probability is approximate: the other rows' terms of its softmax sum come from the 2-bit scores. The copy's scale (Q5_K and Q6_K alike) is raised by sqrt(\|w\|²/⟨w, ŵ⟩) per group, since the least-squares copy scores the near-best rows low and inflates the probability (on the Q5_K parity test's synthetic head over 12 seeds the probability's mean relative error halves, from 0.028-0.043 to 0.010-0.020, and the worst trial from up to 1.04 to under 0.45); on 1500 real draft-head inputs of the IQ3_S pack the probability's mean error is +0.005 (+0.017 with the least-squares scale) and the pick matches the exact argmax in 99.93% | mtp.cpp, head_proxy.cuh, q5k_head.cu, q6k_head.cu |
| MTP hc matrices | The same int8 copies | core/mtp/load.cpp:273-287 |
| MTP draft length | The cumulative-acceptance threshold that ends a draft is the base value times a factor of 1.0 to 1.5, set once from the mean CPU-served expert bytes per expert slot of the pack's final tiers (1.0 at about 1.1 MB, 1.5 at 3.5 MB or more); it is a function of the pack and its placement, never of timing. Accepted tokens still come from verify alone, but the window width differs between packs and builds, and verify kernels are not batch-invariant across widths, so a changed width can change a token-for-token output wherever two candidates sit within rounding of each other | draft_cum.hpp; mtp.cpp; generate.cpp |

The drafter only proposes tokens; verify decides every token, so the drafter's precision affects speed, not output.

### Activations at each matmul

| Matmul | Activation format | Source |
|---|---|---|
| hc down and inject (hc read) | fp32, no activation quantization; int8 weights converted exactly, fp32 fmaf per group | hc_fused.cu:378-388, 510 |
| hc up | fp32 input split into an fp16 high and low part (about 22-bit mantissa), fp16 tensor-core mma with fp32 accumulate, per-group fp16 scale applied in fp32 | hc_fused.cu:635-698 |
| Router | fp32 times BF16 weight, fp32 FMA | hc_fused.cu:762-833; native_bf16.cu:45-125 |
| GDN qkv, QSA q, GDN ssm_out, QSA o | q8_1: int8 codes with an fp16 scale per 32 values, int8 tensor-core dot (dense_tc) | native_mmvq.cu:147-162; dense_tc.hpp:2-12 |
| GDN z, QSA k/v | z through dense_tc together with qkv when eligible; otherwise native_mmvq (dp4a on q8_1) | dense_tc.cu:1148-1154; verify/record_gdn.cpp, verify/record_qsa.cpp |
| Routed experts gate/up | q8_1 of the fp32 FFN input; dp4a with exact int32 per block | hc_fused.cu:63-77; iq_kernels.cu:202-228 |
| Routed experts down | fp32 SwiGLU, then q8_1 per 32 | iq_kernels.cu:1909-1920 |
| Shared expert | q8_1 into gate/up; fp32 SwiGLU, q8_1 into down; sigmoid gate fp32 | shared_expert.cu:189-234 |
| GDN alpha/beta, indexer projections | fp32 activation times BF16 weight, fp32 FMA | dec_gdn_rec.cu:88-112; verify/record_gdn.cpp, verify/record_qsa.cpp |
| Output head | q8_1 of the fp32 final stream, mmvq or the Q5_K/Q6_K kernels; the two-stage Q6_K head's first stage takes the same q8_1 codes as int8 mma inputs (exact int32 per 32-group), fp32 scale terms | verify/record_head.cpp; q6k_head.cu |
| CPU expert tier (experts not resident on a GPU) | The fp32 input quantized with ggml's vec_dot_type (Q8_K per 256 for i-quant gate/up; Q8_0 or Q8_K for down), so it rounds differently from the GPU tiers. Which experts start on a GPU is the model's profile, or the `--expert-profile-save` file an earlier run wrote when there is one; the adaptive tier moves them from there, so two starts on different profiles can differ in a token's last bits. A sleep and wake (`--idle-sleep`) puts every expert back in the slot it held as far as the VRAM after the driver's restore reaches (the last few slots can stay empty, so the next request can round differently from one without the sleep), and a conversation read back from disk is the bytes that were saved (one stored as of its last checkpoint, a reply short of its live end, reads that reply again through the prompt path) | native_expert.cpp:48-49, 76-86; generate.cpp (profile, SLEEP/WAKE); conversation_store.cpp |
| CPU expert tier, Q6_K gate/up | Q8_K per 256 of the fp32 input; ggml's AVX2 Q6_K x Q8_K dot (also run by AVX-512 CPUs): 6-bit codes minus 32, signed per-16 scales, integer sums per block in int32, times the fp16 block scale and the Q8_K scale in fp32. Checked against a double reference over the dequantized operands (worst error 1.3e-8 of the sum of absolute terms) | ggml_cpu_dots.c, native_expert_parity --cpu-synthetic |
| CPU expert tier, AVX2 i-quant rows | Gate/up of IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS against Q8_K, down of IQ4_NL against Q8_0: each 32-value chunk decoded once, every token's int8 products summed in int32 per block, then the fp32 block scales. Where the CPU has AVX-VNNI the int16 products are accumulated by one `vpdpwssd` (IQ4_NL: `vpdpbusd`) instead of `madd` + `add`, which gives the same integers (no saturation in either), so every output bit is unchanged. The pool cuts a layer into row ranges for its threads (and quantizes an expert's intermediate once its own gate/up rows are done): each row is computed by the same code from the same bytes, so the output does not depend on the cut or the worker count (bitwise). Against a double reference over the dequantized operands: worst relative error 5.7e-5 (IQ2_S rows), 3.6e-5 (IQ4_NL down) | iq_avx2_rows.inc, iq_avx2_vnni.cpp, pool.cpp; iq_avx2_test, pool_native_test |
| CPU expert tier, NVFP4 experts | Q8_0 per 32 of the fp32 input and of the fp32 SwiGLU output; integer sums per 16-value sub-block, times the UE4M3 sub-block scale and the Q8_0 scale in fp32. The expert's FP32 tail scales go on in fp32: s_gate and s_up on the gate/up dots before SwiGLU, s_down on the down output (folding s_down into up would put h near 1e-5, where the activation block scale goes subnormal). Against a double-precision expert on the same weights: rel 1.2e-2 from activation rounding, 7e-8 with the CPU's own rounded activations | native_expert.cpp (native_gu_rows, native_down_rows); ggml_cpu_dots.c (ggml_vec_dot_nvfp4_q8_0); native_expert_parity --synthetic nvfp4/nvfp4 |
| CPU misses staged over PCIe (batch windows on one card) | In a batch window, a layer with enough CPU misses has its last few (a count set from the CPU pool's and the link's cost per expert) copied from their RAM copies into VRAM staging and computed by CUDA0's grouped kernels: the GPU tiers' arithmetic instead of the CPU tier's, bitwise the rows the same experts give from VRAM. Which experts go depends on the window's routing and on where the RAM tier holds them, so batch output rounds as with adaptation; a request decoding alone runs the single-stream path and stages none | dec_moe.cuh (dm_stage_split, dm_stage_kernel); verify/expert_resources.cpp (dm_init), verify/batch_execution.cpp (dm_stage_places), verify/record_moe.cpp; dec_moe_parity |
| GPU experts, NVFP4 (grouped kernels: per-entry and decode-once, and the device-planned MoE tail on both cards; NVFP4 gate/up takes the glue path, with the plan and the tier ring inside the gate/up launch instead of a separate route launch; all bitwise equal) | q8_1 of the fp32 input and of the fp32 SwiGLU output (the same rounding as the other GPU formats); E2M1 codes through a 16-entry int8 table, dp4a with an exact int32 sum per 16-value sub-block, times the q8_1 scale and the UE4M3 sub-block scale in fp32. s_gate and s_up multiply the gate/up dots before SwiGLU and s_down the down output, in fp32, as on the CPU tier. Against a double-precision expert on the kernels' own q8_1 activations: rel 5-8e-8 (gate, up, down); on the unrounded input: rel 1.2e-2; GPU vs CPU tier: rel 8e-8 (real experts, layers 0-3, 20, 47) | iq_kernels.cu (vec_dot_nvfp4_q8_1, Split<40>, native_gu_kernel, native_down_kernel and their multi variants); dec_moe.cuh (dm_gu_row, dm_down_item, dm_gu_kernel, dm_glue_gu1_kernel); native_expert_parity --experts PACK; iq_multi_parity, native_grouped_parity, dec_moe_parity --nvfp4 |
| GPU experts, Q8_0 gate/up with Q8_0 down (decode-once kernels and the device-planned MoE tail, bitwise equal to the per-entry kernels; no numeric change) | q8_1 of the fp32 input and of the fp32 SwiGLU output; int8 weights, dp4a with an exact int32 sum per 32-value block, times the fp16 block scale and the q8_1 scale in fp32, the same expression in every kernel | iq_kernels.cu (Split<8>); dec_moe.cuh (DM_PAIRS); iq_multi_parity, dec_moe_parity --kq |
| GPU experts, Q6_K gate/up with Q8_0 down (decode-once grouped/verify kernels, the device-planned MoE tail and the peer kernel; bitwise equal to the per-entry kernels; no numeric change to existing formats) | q8_1 of the fp32 input and of the fp32 SwiGLU output; 6-bit codes minus 32 through dp4a against the q8_1 words, each 16-value sub-block's int32 sum times its signed int8 scale, the 32-value products summed in fp32 times each block's q8_1 scale, then the fp16 block scale. Decode-once and per-entry kernels share this arithmetic. Prompt path (MMQ) is a separate row once it lands | iq_kernels.cu (Split<14>), dec_moe_parity --kq |
| Prompt path, NVFP4 experts (MMQ, the default) | q8_1 of the fp32 input and of the fp32 SwiGLU output, as for the other MMQ formats; E2M1 codes through the 16-entry int8 table into the tile, int8 mma with an exact int32 sum per 16-value sub-block, times the UE4M3 sub-block scale and the q8_1 scale in fp32. The tail goes on in fp32 in the SwiGLU step: h = silu(s_gate g) * (s_up u) * s_down, so s_down rides on H into the down product (q8_1 keeps an fp32 scale per 32 values, so H rounds to the same codes up to fp32 rounding of that scale). **Numerics change against the FP16 fallback it replaces**: the error is now the int8 activation rounding every MMQ format has. Synthetic experts at real geometry: gate/up product vs a double reference on the same q8_1 input rel 1.7e-7; layer output vs that reference 6.8e-5 (H codes at rounding ties); vs the unrounded double reference 1.6e-2 (the FP16 fallback: 5.2e-4); MMQ vs FP16 fallback 1.6e-2 | mmq_load_tiles.cuh (ggml_cuda_mmq_load_tiles_nvfp4); mmq_vec_dot.cuh (q8_0_16); moe_mmq.cu (swiglu_kernel, gather tails); prefill/expert_mmq.cpp; prefill_mmq_nvfp4_test |
| Prompt path, NVFP4 experts (FP4 tensor cores, chunks of 1,024 tokens or more; shorter chunks take MMQ) | **W4A4. Numerics change against the MMQ path it replaces.** The fp32 input and the fp32 SwiGLU output are rounded to E2M1 codes with a UE4M3 scale per 16 values and one fp32 scale per token (amax / (6 x 448)); the weight blocks are read in place, and `mma kind::mxf4nvf4` (block scale, 4 scales per k64) accumulates in fp32. The tail scales go on in the SwiGLU step as in MMQ. Synthetic experts at real geometry: gate/up rel rms vs a double reference 9.4e-2 (MMQ: 1.6e-2). Against a CPU reference with the same E2M1/UE4M3 activation rounding the kernel agrees to rel rms 1.4e-7 (gate/up) and 3.7e-7 (layer output), and a token's output is bitwise independent of its tile and neighbours. The quality gate's 256-token prompt halves are below the 1,024-token minimum, so the gate runs MMQ prompt experts and does not exercise this path. Long prompts, NVIDIA-checkpoint pack, 400 W, T 3, int8 KV, resident experts: 12 chunks of 4,096 tokens (8 consecutive 512-token wikitext chunks of the Q8_0 base each, so a BOS sits every 512 tokens), the 2,049-token prompt half through this path, 2,047 teacher-forced rows per chunk (24,564 rows): PPL 2.9556 against 2.9547 for MMQ (standard error 0.037 each); paired per-row NLL difference +0.0003 (+-0.0014, ratio 1.0003). No Q8_0 reference exists for that layout, so KLD and top-1 are not measured there. A peer card as fast as the primary computes the rows of the experts it holds for streamed chunks of up to 5,120 tokens with these kernels, from its own copy of the activations and in whole-tile pieces: bitwise equal to the primary's rows, so no numerics change; a slower peer card takes no streamed chunk (the chunks of 2,560 to 6,143 tokens then read as with one card, so their text can differ from before in the last bits) (prefill_fp4_test peer share: 512 experts, top 10, four routings incl. hot experts) | moe_fp4.cu; prefill/expert_fp4.cpp (fp4_flush); prefill/peer_experts.cpp (the peer's FP4 share); prefill_fp4_test |
| Prompt path, NVFP4 experts (FP16 fallback, only where MMQ has no tile) | Weights dequantized to fp32 (exact), times the tail scale in fp32, rounded once to fp16; fp16 GEMM. Weights below fp16's normal range (about 6e-5) lose relative precision there | iq_kernels.cu (dq_nvfp4); prefill/expert_fp16.cpp:13-30 |

Unlike llama.cpp, Basalt does not quantize the activations of the hc read to Q8_0.

### Accumulation, state, softmax and logits

| Item | Precision | Source |
|---|---|---|
| int8 dot products | Exact int32 per 32-value block, then fp32 scale terms summed in fp32 | dense_tc.cu:60-68; dense_tc.hpp:2-12 |
| Residual stream and hc buffers | fp32 | verify.hpp:491; hc_fused.hpp:28-40 |
| RMSNorm | fp32, eps 1e-6 | verify/common.hpp (EPS) |
| fp16/BF16 accumulators | None; hc up uses fp16 mma inputs with fp32 accumulate | hc_fused.cu:635-698 |
| KV cache (`--kv`) | fp16, int8, q4_0 (alias q4), k8v4 (int8 K, rotated Q4_0 V). Engine default fp16; the sample config in the README uses int8 | program/cli/arguments.cpp:67 (`--kv` parse); program/generate/configure.cpp:185-187 (qsa_set_kv_int8, qsa_set_kv_q4, qsa_set_kv_hybrid); docs/configuration/ |
| int8 KV layout | int8 codes with an fp16 scale per 64 values (4 per 256-wide head, separate for K and V), scale = f16(amax/127) | kv_q8.hpp:3-5, 21-23; kv_q8.cu:46-52 |
| Elastic KV (serve) | A session backs its floor's cells at start and its prompt plus a step per request, and gives back, on the floor's grid, what a later, shorter request does not use; the rest of the KV's VRAM holds expert slots. The K/V values do not change, but how many slots a request decodes with moves placement (which experts the GPU or the CPU computes), so free-running text after a deeper request can differ from a build that gives back less | generate.cpp (kKvElasticFloor, kKvGrowCells, kv_fit, kv_release), kv_arena.cpp (the grid) |
| Attention | K/V dequantized to fp32; scores in fp32; online softmax per 64-cell chunk in fp32 (`__expf`), fp32 merge; output gate sigmoid, then q8_1 for the o-projection | qsa_dec_attn.cu:138, 255-427 |
| Draft-only attention | fp16 mma kernel; it changes drafts only, never accepted tokens | qsa_dec_attn.cu:429-431 |
| Indexer | Keys fp16-rounded, held fp32; block scores fp32; radix top-k, ties to the lowest cell | native_qsa_indexer.hpp:11-16; qsa_select.hpp:13-14 |
| GDN recurrent state and conv history | fp32 | dec_gdn.hpp:19; verify/record_gdn.cpp |
| GDN middle of a window up to four rows (mid chain), async commit finisher | The qkv + z pair, the recurrence and the output projection as one persistent kernel with the same arithmetic in the same order as the three kernels it replaces: bitwise identical outputs and state, no numerics change. On a device with fewer than 160 SMs the output projection takes the one-tile kernel's reduction order, which is not the order of that device's unfused kernels (a numerics change there only; 20-chunk gate on such a card, PPL 2.3076 to 2.3024, mean KLD 0.0772 to 0.0769, top-1 93.10 to 92.69 %); the larger devices stay bitwise. The commit claims its items from a counter and the next window starts with a finisher kernel that takes the rest; the GDN layers after the first are committed inside the next window, ahead while it waits for the CPU tier; each item is computed as before. It needs a translation unit built without fast math, so it stays out of the fast-math list in CMakeLists.txt; `mid_chain_parity`, `dec_commit_parity --finisher` and `--lazy` check the bits | mid_chain.hpp; mid_chain.cu; verify_kernels.cu |
| GDN mixer split by key head across two cards (peer card faster than the main card; decided at start) | Off with a slower peer: bitwise the whole mixer on the main card. On: the out projection is a split-K sum of two fp32 partials, the peer runs its projections as dp4a over q8_1 rows, and a q8_1 block scale can land one fp16 step away at a rounding tie, so reduction order changes the last bits ([details](gdn-split.md)) | dec_moe.cuh (dm_mix_rows, dm_mix_rows_m); verify/gdn_split.cpp (gs_plan) |
| Router softmax and top-10 | fp32, ties to the lower expert index, weights renormalized over the ten | native_router.cu:44-97 |
| Router, n_expert other than 512 (pruned packs) | Generic router kernel instead of the fused native one. Max in fp32; per-expert exp in double of (logit - max); the sum in double, ascending on one thread; probabilities rounded to fp32 as `(float)(exp * (float)(1/sum))`. Top-10 by ten block-wide argmax passes over the fp32 probabilities, ties to the lower expert index. The ten weights are summed in double, floored at 2^-14, divided in double and stored fp32. The native path takes exp, sum and reciprocal in fp32 with a warp-ordered sum, so the probabilities differ in the last bits and the chosen experts or weights can differ from it in rare near-tie cases | router_top10.cu:56-173 (max 74-88, exp 92, sum 96-102, probabilities 114, selection 131-162, renormalization 166-172); native_router.cu:44-97 |
| Expert combine | fp32, in router rank order, plus shared expert | native_moe.cu:36-50 |
| Final logits | fp32 `[T][n_vocab]` | verify.hpp:501 |
| Greedy sampling | fp32 argmax, lowest index wins ties | sampler.cu:84-95 |
| Sampled pick (top_k, top_p, min_p, temperature, draw) | One launch for top_k <= 32 (`sampler_sort_kernel`, sampler.cu:1046-1134; the tail in `sampled_tail_pair`, 978): each 4,096-logit block finds its first k with bitonic networks, and the row's last block merges the block lists and runs the tail with the list in shared memory. A wider top_k takes the one-launch rounds split (`sampler_fused_kernel`, 695). Same top-k set, same exp arguments, same summation order and quotients as the two-launch split (`sampler_split_part_kernel` 586, `sampler_split_merge_kernel` 669, the parity reference), so the same picks and probe probabilities bit for bit. A list longer than 64 (top_k 0, the whole vocabulary, or above 64) takes the wide path: the row's penalised logits radix-sorted in the selection order (ties to the lower id, -0 = +0), then the same chain over the sorted list with the same exp arguments, its sums chunked over 1,024 threads on every SM (each chunk's terms in order, the chunk sums by a fixed shuffle tree) and each crossing tested as cum >= top_p S or u S < cum, so a seed replays the same draws. Against the narrow kernels at 64 entries or fewer (sampler_parity --split, split 3): the same picks, the probe's probabilities within a few float ulps (the sums' order). **Numerics change for top_k 0 or above 64**: such a list was cut to 64 before; it is now the list asked for. top_k 1..64: no change | sampler.cu; sampler_parity --split |
| Request `logit_bias` | Only when a request sends one: fp32 bias added to the fp32 logits before the penalties and filters; -100 is -inf (the token cannot be picked), other values are added as given. Without one the sampler reads the same logits and takes the same code path as before | sampler.cu; logit_bias.hpp |
| KLD scoring (host) | Log-sum-exp and KLD accumulated in double from the fp32 logits | ppl_gate.hpp (the KLD function) |

### Prefill

Prefill and decode are different computations that read and write the same KV and state formats.

| Item | Prefill | Source |
|---|---|---|
| Dense projections | Weights dequantized to fp16, activations rounded to fp16, cuBLAS with fp32 accumulate; BF16 tensors BF16 x BF16 with fp32 compute; no q8_1 | gemm.cu:122-184; prefill/projection.cpp:29-40 |
| hc matrices | Activations rounded to BF16; down/up use the int8 copy dequantized to BF16; inject uses the BF16 tensor. From 33 tokens the up product and the mix run as one kernel (the FP32 gate rows stay in registers): each gate value is one m16n8k16 chain over k from zero, the chain cuBLAS runs for that product, and the mix keeps gr_mix_r's arithmetic, so the outputs are the same bits as the two steps (`prefill_hc_up_mix_test`) | prefill/projection.cpp:16-22 (up_mix); kernels.cu (gr_up_mix_kernel); gemm.cu (bf16_w, bf16_rows) |
| Router | BF16 activation (decode: fp32) | prefill/routing.cpp:26 |
| Shared expert | fp16 activation, fp16 SwiGLU | prefill/shared_expert.cpp:10-20 |
| Routed experts | Fused int8 mma kernels, llama.cpp MMQ (q8_1) or dequant-to-fp16 cuBLAS (IQ1_M only). NVFP4 takes the FP4 tensor-core kernels from 1,024-token chunks and MMQ below (see the Activations table above). The K-quants (Q4_K, Q5_K, Q5_1: UD-Q4_K_XL) run on MMQ with fp16 activation scales and per-32 partial sums taken from the rounded int8 values (llama.cpp sums the floats), so the product is exactly the dequantized weights times the rounded activations. Q6_K gate/up (UD-Q5_K_XL, UD-Q6_K_XL) runs on MMQ only, with no FP16 path: the 6-bit codes minus 32 as int8 and a float scale (d times the int8 sub-scale) per 16 values, against the same int8 activations with one fp32 scale per 32 (no partial sums, as there is no minimum) | moe_fused_iq.hpp:1-12; moe_mmq.hpp:1-6; mmq/quantize.cu:59-69 |
| Attention | int8 KV: q and p rounded to fp16 (q prescaled by a power of two), V scale folded into the fp16 B operand, one fp16 mma, fp32 accumulate, per-warp online softmax merged once at the end. Output within 1.5e-3 of the output scale of an FP64 reference (the previous hi/lo split: 1e-6). fp16 and Q4_0 KV keep the hi/lo split of q and p | qsa_prompt_attn.hpp:1-22 |
| Routed experts on the CPU (chunks below 6,144 tokens) | Non-resident experts with a host copy and at most 8 rows run on the CPU pool with ggml's dot products, as decode's RAM tier does: the activation in the weight type's `vec_dot` format (Q8_K per 256 values for the IQ types, not q8_1 per 32), H quantized to the down type's format. Their rows are written unweighted and the combine weights them like the GPU's. Which experts go is a fixed share of the streamed ones, fewest rows first (the CPU-share paragraph below) | prefill/cpu_experts.cpp:48-76, 119 |
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
CPU's experts and with them the rounding, so identity checks compare sessions of one configuration. With `--parallel`
a prompt read that runs alone (up to 16384 tokens, or any read while no slot decodes) takes the same share; a longer
read beside decoding slots runs every expert it streams on the GPU. The experts the decoding slots routed lately stay in
VRAM while such a read takes cache slots (they trade places with less-routed ones), so which of the slots' experts the
CPU computes meanwhile follows their routing, as with any tier layout.

**Expert path by chunk size.** A prompt chunk of fewer than 6,144 tokens stages only the experts its tokens route to and
runs them on MMQ, or on the FP4 tensor cores for NVFP4 experts from 1,024 tokens (the FP4 row above), with the CPU pool taking the low-row non-resident ones; a chunk of 6,144 tokens
or more streams every non-resident expert through the ring and runs the fused kernels (NVFP4 has none: it stays on the FP4 tensor cores, or MMQ below 1,024 tokens, so the cutoff there only changes which experts are staged). Staging everything costs the same
whatever the chunk, so it only pays once most experts are hit anyway. In a chunk below the cutoff the FP4 experts run in groups of 16, as in the streamed walk,
bitwise equal to one at a time (`prefill_fp4_test` checks every split of a group into runs); its 32 staging slots come
out of the prompt's loan like its other buffers, so the experts the GPU holds during the read, and with them the CPU's
share, follow the ring's size as they follow any tier layout. Each layer of such a chunk also runs the next layer's router on its own FFN rows, and the
experts that guess gives more than 8 rows are copied ahead into slots of their own while the layer ends: that changes
where an expert's bytes are read from, never what is computed. Chunks of 1,024
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
256-token prompt half of the quality gate chunk as before, so the gate's numbers do not move. With `--parallel` a
read that runs alone chunks as the single-stream path does; a read beside decoding slots takes the smaller chunk its
capped loan allows, so the same prompt's K/V can differ in the last bits between the two. A read beside decoding slots
that gives way to a shorter request reads its rest later in the same chunks, beside the slots, with no CPU share.

**Short first chunk.** A prompt above the chunk cap (and at least 32,768 tokens) reads its first 16,384 tokens as one
chunk and the rest in equal chunks of at most the cap, so the first chunk waits for fewer PLE rows. This moves the chunk
boundaries (GEMM tiles, so the rounding of the K/V) and nothing else: prompts up to the cap and the quality gate's
256-token prompt half chunk as before (the 20-chunk gate is identical to the previous build: PPL 2.2924, KLD 0.06989,
top-1 92.86%, 400 W, NVIDIA-checkpoint pack). Long-context check, 65,515-token prompt, the first 64 greedy tokens
teacher-forced with the previous chunking's tokens, spec 1, int8 KV, 400 W: KLD of the next-token distribution against
the previous chunking has mean 0.0032, median 0.00013, max 0.038, and the same top-1 at 64 of 64 positions. The 16-token
greedy text of the 64K benchmark prompt differs at one near-tie word.

**Prompt split across the cards.** With expert tier cards, a long fresh part of a prompt can be read across the cards
([GPUs](../configuration/gpus.md)): the tier cards read the later layers of every chunk but the last, each a block of
them. The chunks are the ones the main card alone reads, so no chunk boundary moves. On a tier card those layers read
the hyper-connection down/up matrices as the pack's BF16 tensors (the main card reads its 8-bit copies, dequantized to
BF16), and its cuBLAS products may run other kernels; the routed experts and the other products keep the main card's
arithmetic. A layer gives the same bits on any stage of the same card model: cutting one tier card's layers into two
stages gives the same tokens and checkpoints. The draft layer builds its K/V on the main card from the last stage's
rows, read back bit for bit. With KV streaming (`kv-resident`) the main card's K/V of the tier layers comes from and
goes back to its RAM copy byte for byte, so the read is the same: a 65,536-token prompt gives the same 400 greedy
tokens with and without streaming. Parts the price keeps on the main card, parts of one chunk and the quality gate's 256-token
prompt halves read as before. Long-context check over two cards, NVIDIA NVFP4 pack, int8 KV, spec 1, 400 W, the first 64
greedy tokens of a main-card read teacher-forced into the split read, full-vocabulary KLD: 65,535-token prompt mean
0.0008, median 0.0003, max 0.012, top-1 64 of 64; 122,879-token prompt mean 0.0005, median 0.0003, max 0.003, top-1 64
of 64. With `--parallel` slots all idle the read is the same: the 64 greedy tokens after a 122,879-token prompt are
identical with one slot and with four (same expert cache). A read that gives way to a shorter request ends with the main
card reading its kept chunk's later layers, as a main-card read does; the rest of the prompt is read later from there in
its own chunks, as after a main-card read gives way.

**K-quant experts on MMQ.** `tests/cuda/prefill_mmq_kq_test.cpp` runs MMQ and the FP16 dequant + cuBLAS path it
replaced on random zero-centred experts at the model's geometry against a double-precision reference. Relative RMS
error of the expert outputs, MMQ / FP16 path: 1.08e-2 / 5e-4 with plain activations (Q4_K/Q5_1, Q4_K/Q8_0, Q5_K/Q8_0),
1.48e-2 to 1.51e-2 / 5e-4 with 0.5% of the activations 12 times larger (Q6_K/Q8_0, MMQ only: 1.17e-2 plain, 1.55e-2 with outliers); with llama.cpp's float partial sums MMQ read
1.21e-2 to 1.76e-2 and 1.77e-2 to 2.19e-2. A row re-run alone or in another batch differs from its row in the full batch
by at most 7e-4 of the row's largest output: MMQ's stream-k split and tile width follow the launch's size, and H's int8
rounding can turn a reordered sum into one int8 step. The i-quant MMQ path does the same (6.5e-4, IQ4_XS/IQ4_NL). Quality
gate, UD-Q4_K_XL, 100 chunks, 400 W, the 256-token prompt half through this path: PPL 3.3297, KLD 0.05653, top-1 92.06%
against 3.3223, 0.05695, 92.22% with the FP16 path.
