# DeepSeek-V4.1-Flash precision

What precision the `deepseek41` model computes in, stage by stage, and how it differs from the llama.cpp support it was
ported from. The Flash-Next tables are in [index.md](index.md). Sources are `file:function` under
`src/models/deepseek41/` unless a path says otherwise.

Geometry: n_embd 5120, 40 layers, 4 hyper-connection streams, 384 routed experts per layer (top-6) plus one shared
expert, 64 query heads of dimension 512 over one shared latent, a 128-token window, compressed caches at ratio 1 or 2
with a two-level top-512 indexer, and an Engram table on two layers.

## Weights as stored

The pack keeps every GGUF tensor as published ([model file](../model-file.md#deepseek-v41-pack)); nothing is
requantized. Types in the IQ1_S, IQ1_M and GSQ-RCO packs (a GSQ-RCO build assigns gate/up and down per layer):

| Tensor class | Stored as | Read by |
|---|---|---|
| Routed experts, gate/up | IQ1_S (19), IQ1_M (29), IQ2_XXS (16) or IQ2_XS (17), per layer | GPU grouped kernels, MMQ (not IQ1_M), CPU dots |
| Routed experts, down | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS or IQ2_S (22) | the same |
| Attention wq_a, wq_b, wkv, wo_a, wo_b | Q6_K, Q5_K or Q8_0, as the pack assigns them | native_mmvq (decode), MMQ (prompt) |
| Shared expert gate/up/down | Q6_K, Q5_K or Q8_0 | dense gated MLP, clamped |
| Token embedding, output head, Engram `engram_kv` | Q6_K or Q5_K | per-row lookup, matvec by type |
| Engram table | IQ4_NL, rows of 256 values (144 bytes) | host row read, `dequantize_iq4_nl` |
| Compressor kv/gate, indexer q/k/weight projections | BF16 | `gemv_bf16` (decode), `gemm_bf16` (prompt) (attn_index.cu) |
| Router gate (F32, or BF16 widened exactly at load), router bias (and the picture bias), hyper-connection fn/scale/base, all norm weights, attention sinks, Engram gate | F32 | fp32 kernels |
| Vision projector (optional `vision/mmproj` section) | BF16 | vision/encoder.cu (cuBLAS) |

## Activations and accumulation

All activations between stages are fp32: the four-stream residual, norms, router logits and weights, hyper-connection
mixes, attention softmax and the output logits. Quantized weights read q8_1 (GPU) or Q8_K (CPU) copies of fp32 inputs,
as in llama.cpp.

| Stage | Computation | Where |
|---|---|---|
| Norms | fp32 sum of squares, `rsqrtf(mean + eps)`, eps from `meta.geometry` (1e-20); hyper-connection pre-mix fused with the norm in decode and prompt | kernels/norm.cu rms_norm_rows, hc_pre_rms_norm; pack.cpp |
| Hyper-connections | fp32 mixes from the flattened streams, sigmoid splits, Sinkhorn row/column normalisation in fp32 for the configured iteration count (a serial chain in one warp), pre/post weighted sums in fp32 | kernels/mhc.cu hc_mixes, hc_pre, hc_post |
| Router | fp32 dot per gate row (one block per row), `sqrt(softplus(logit))`, selection on score plus bias with ties to the lower index, weights normalised over the six picked (+1e-20) and scaled by `expert_weights_scale`. | kernels/router.cu router_logits_kernel, router_select_token |
| Engram | n-gram hash on the host, IQ4_NL rows dequantized to fp32 on the host, Q6_K `engram_kv` matvec over the fp32 rows, gated add into the stream in fp32. Picture rows skip it | kernels/engram_hash.cpp, engram.cu engram_apply; session.cpp (native_q6_k_f32) |
| KV and window | `wkv` Q6_K/Q5_K matvec, kv_norm and partial interleaved RoPE (angle `float(pos) * step`, last 64 dims) in fp32, then stored as **fp16**: the 132-slot window ring (position `p % 132`) | kernels/attn_index.cu kv_window |
| Compressor | BF16 weights, fp32 activations and accumulation, softmax-pooled over the group in fp32 with fp32 partial-group state; the finished row is stored **fp16** (compressed cache, K = V) | attn_index.cu compress, gemv_bf16 |
| Indexer | q and weights from BF16 projections in fp32; keys stored fp16; scores accumulated fp32 in a thread per row; level-1 candidate blocks (8 rows) then level-2 top-512, ids ascending | attn_index.cu idx_score_kernel, indexer_select |
| Query path and output | q_a, q_b Q6_K/Q5_K matvecs over q8_1, RMS norm of the low-rank query in fp32, RoPE; output through eight low-rank group products (wo_a) then wo_b, each a quantized matvec over q8_1 | kernels/q_path.cu, attn_out.cu |
| Sparse attention, decode | Window and selected compressed rows read as fp16 and widened, scores and `expf` softmax with the attention sink in fp32, fp32 accumulation, inverse RoPE on the output | kernels/attn_mqa.cu sparse_mqa |
| Output head | final hyper-connection collapse, norm, Q6_K matvec to fp32 logits | session.cpp; kernels/norm.cu |

A decode kernel's output does not depend on how many tokens share a launch: the parity tests compare T=4 against four
T=1 calls and require bitwise equality (attn_index_parity, attn_mqa_parity, router_parity, hc_parity).

### Routed and shared experts

| Part | Precision | Where |
|---|---|---|
| GPU experts, IQ1_S gate/up or down (IQ2_XXS, IQ2_XS and IQ2_S likewise; the same q8_1 and dp4a scheme with each format's grid; decode-once grouped kernels) | q8_1 of the fp32 input and of the fp32 SwiGLU output; per 32 values the grid codes (0..2) against the int8 activation with dp4a, plus the per-32 shift (+-1/8) times the activation's integer sum, times the 3-bit scale (2s+1), the fp16 block scale and the q8_1 scale in fp32. Against a double reference on the dequantized weights and the rounded activation: rel RMS 8e-8 for the dot, 4e-3 for a whole expert (the q8_1 rounding of the SwiGLU output) | src/kernels/cuda/iq_kernels.cu vec_dot_iq1_s_q8_1, Split<19>; src/kernels/iq1s_parity.cpp |
| Clamped SwiGLU (routed and shared experts) | gate capped at the limit, up clamped to +-limit, then `silu(gate) * up` in fp32 with exact exp. A separate compile-time instantiation of the grouped launches; the unclamped launches are unchanged. The limit rides in the expert layout, so every tier applies it | iq_kernels.cu grouped_impl, swiglu_clamp_q8_1_entries_kernel; native_expert.cpp gu_rows_ggml |
| CPU experts, IQ1_S | Q8_K of the fp32 input and of the SwiGLU output; integer sums per 32 with the 3-bit scale, the shift term from the activation's group sums. The AVX-512 row kernel is bit for bit ggml's single-token dot; rel RMS 1e-7 against the double reference | src/kernels/cpu/iq_avx512.cpp iq1s_512_gu_rows; ggml_cpu_dots.c ggml_vec_dot_iq1_s_q8_K |
| Decode, by tier | Each routed expert runs where it sits: from a VRAM slot on the GPU (q8_1, native grouped kernels), or on the CPU pool from the pinned RAM copy or the mapped pack (Q8_K, ggml arithmetic). **The two differ in the last bits**, so a token's output depends on which experts were resident. Residency changes only at token boundaries and follows the routing history, so a fixed prompt on a fixed memory budget repeats exactly; a different RAM-tier size (free memory at start) can change the text. A repeat miss within 64 tokens is promoted to VRAM, half of a layer's misses go to the CPU and half over the bus, and a file-backed expert that misses twice swaps into the pinned tier for the least recently used pinned one (taking effect one token later), so the placement, and with it the last bits of later tokens, differs from the first-miss-promotes policy; the first 16 tokens of the fixture prompts are unchanged. The CPU's rows are read in place from mapped host memory by the weighted sum (same summation order) | tier.cpp lookup, fetch, run_cpu; ops_kernels.cpp routed_experts |
| Prompt chunks (up to 8,192 tokens) | Routed experts on MMQ: q8_1 (per 32, MMQ layout) of the fp32 input and of the fp32 clamped SwiGLU output, int8 mma against the IQ1_S / IQ2_XXS codes (the IQ1_S grid shift rides in the activation block sums), fp32 accumulation, rows summed per token in slot order. **Numerics change against the single-token kernels** (a different int8 reduction order; both round activations to q8_1). Experts are staged from the VRAM and pinned RAM tiers, so the prompt path never mixes in CPU arithmetic | prefill_ffn.cpp routed; prefill_ffn.cu; src/kernels/cuda/mmq_load_tiles.cuh load_tiles_iq1_s |
| Shared expert in prompts | MMQ GEMMs over 1,024-token sub-chunks (gate, up, clamped SwiGLU, down; q8_1 activations with an fp16 block scale, int8 mma, fp32 accumulation). **Numerics change** against the decode kernel: about 1e-3 per layer. The per-token weighted sum of the routed rows keeps the slot order, bitwise equal to the per-token launch | prefill_ffn.cpp shared; weighted_sum_tokens |

A prompt layer with an IQ1_M side has no MMQ product: it runs the grouped decode kernels (q8_1 activations, fp32
accumulation), exact to decode's numerics for that layer (prefill_ffn.cpp grouped_launch).

## Prompt attention

The prompt path batches a chunk per layer call.

- **Dense projections:** wkv, wq_a, wq_b, wo_a and wo_b run as MMQ GEMMs over 1,024-token sub-chunks (q8_1 activations
  with an fp16 block scale, int8 mma, fp32 accumulation; wo_a as eight MMQ groups). **Numerics change**: about 2e-3
  relative RMS per layer against the decode matvec (attn_batch.cpp dense_in, dense_out; attn_dense.cu).
- **Sparse attention:** fp16 tensor cores, 64 keys by 16 heads per block, queries and softmax weights rounded to fp16,
  fp16 latents, fp32 accumulation and softmax. 2e-4 relative RMS against the fp32 kernel, bitwise independent of the
  number of tokens in the launch (kernels/prefill_attn.cu sparse_mqa_tc; ds41_prefill_attn_parity).
- **Window, compressor and indexer:** one launch set per 1,024-token sub-chunk. The window rows are staged in a
  1,152-slot fp16 ring (rows identical to the decode ring's) and the decode ring receives its last 132 rows, so decode
  starts from the same state. The BF16 projections (compressor kv/gate, indexer q and weights) are tensor-core GEMMs with
  the fp32 activations split into three bf16 parts and a fresh accumulator per 16 products: 3e-7 relative RMS against a
  double reference (the decode matvec: about 1e-7). **Numerics change**: compressed rows differ from the decode
  kernels' by 1e-5 (fp16 storage rounding included), selected ids are identical on random inputs, and on real prompts
  the first compressed layer's `attn_out` differs by 1e-3 relative RMS from the previous build (everything else in that
  path is bitwise equal when the compressor runs the old kernels). The distance of the first fifteen positions to the
  llama.cpp fixture is unchanged at layers 2 to 4 (kernels/attn_index.cu; ds41_prefill_index_parity).

## Pictures

The projector (BF16 weights, fp32 activations, cuBLAS) turns the picture into a block of embedding rows that replace the
token-embedding lookup; the rows run through the batched prompt path as text does. **Numerics change against the
reference encoder**: 8.7e-3 relative RMS on the embedding block (different accumulation order and attention kernel).
Picture rows select experts with the image routing bias (`exp_probs_b_vl`) when the pack has it (else `exp_probs_b`),
take no Engram contribution, and enter the n-gram history as dead ids, as the reference does (vision/encoder.cu;
prefill.cpp run_embeddings; session.cpp).

## What differs from llama.cpp's support, and why

The kernels are Basalt's own ports and rewrites of the same graph (`build_attention41`, hyper-connections, router,
Engram); the layer order, RoPE tables, window and compressor definitions follow it.

| Difference | Why |
|---|---|
| Experts: GPU decode kernels quantize activations to q8_1 like llama.cpp's CUDA path, but residency splits the work over GPU and CPU. llama.cpp's own CPU path matches the CPU tier to 1e-7; its GPU path differs from ours by about 1e-2 on IQ1_S layers | Expert tiers (VRAM, pinned RAM, mapped pack) are how a 107 GiB expert set runs on one card |
| Prompt attention and experts use MMQ and fp16 tensor-core kernels in sub-chunks; llama.cpp batches whole ubatches through its own MMQ and flash kernels | Prompt speed; both sides round activations, in different reduction orders |
| Decode kernels are independent of the token count per launch (bitwise); llama.cpp's ubatch size changes its rounding | Greedy repeatability across prompt chunking |
| Router logits: decode ids equal llama.cpp's; prompt logits differ at 3.5e-4 relative RMS (its batched path) | Different reduction order; the selection is unchanged on the fixtures |
| Latents (window, compressed) and index keys are stored fp16 | Half the cache traffic; the fixtures agree at this storage |
| The compressed rotation (YaRN factor 16 over 65,536 trained positions) is read from the pack; angles are `float(pos) * step`, so at positions near 1M the product carries fp32 rounding (about 6e-8 relative, up to 0.03 rad on the fastest pair at 1M) | The context runs to 1,048,576 positions with the same storage and kernels, and the indexer reads only the rows a token can see (bitwise equal to scoring the whole buffer) |
| Vision encoder is Basalt's own | 8.7e-3 relative RMS on the embedding block against the reference encoder |

Measured against llama.cpp on the same weights (per-layer tensors from recorded runs, relative RMS): router logits
1e-7 (decode), hyper-connection chain 1e-7 (decode) to 5e-5 (prompt), attention inverse-RoPE output 3e-4 to 2e-3,
`attn_out` 4e-3 to 1.1e-2 (the reference's prompt MMQ noise is the same size), top-512 selection overlap 0.998 on
average over a 6,250-token prompt. Greedy decoding reproduces llama.cpp's first 16 tokens on two of three fixture
prompts; on the third it can diverge after a few tokens, because 1.5-bit weights make a near-tie flip under
last-bit changes.

## Quality

GSQ-RCO IQ2_XXS, 5090 only, against llama.cpp b11870 on the same GGUF, llama's tokens fed one decode step at a time on
three prompts: Basalt's greedy choice differs at 5 to 8 of 688 generated positions, depending on which tier each expert
sits in during the run, with 1.0 at most of log-probability between the two tokens at llama's logits. KL against llama's logits on 512 positions of real text: mean
0.0025, median 0.0007, top-1 agreement 98.2%, position 1 left out: llama's own row at position 1 moves by KL 4.2 between
a 2-token and a 512-token batch, and Basalt is within 0.07 of its 2-token row. There is no PPL/KLD run against the
native model on this pack yet.

Quality is the quant's, not the engine's. The IQ1_S pack scores, against the native model's logits (teacher: the
native FP4 checkpoint through the reference code; Lucebox's 8 x 1,024 held-out tokens, KL over the teacher's top-256 ids,
llama.cpp with the same GGUF): KL 0.362, top-1 agreement 75.3%, PPL 10.75 (teacher 8.56). Lucebox's own files score KL
0.19 to 0.27. Basalt is held to the same weights by the per-layer agreement above; there is no Basalt PPL/KLD run for this
model yet.
