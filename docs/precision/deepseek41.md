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
| Routed experts, gate/up | IQ1_S (19), IQ1_M (29), IQ2_XXS (16) or IQ2_XS (17), per layer | GPU grouped kernels, MMQ, CPU dots |
| Routed experts, down | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS or IQ2_S (22) | the same |
| Attention wq_a, wq_b, wkv, wo_a, wo_b | Q6_K, Q5_K or Q8_0, as the pack assigns them | native_mmvq (decode), MMQ (prompt) |
| Shared expert gate/up/down | Q6_K, Q5_K or Q8_0 | dense gated MLP, clamped |
| Token embedding, output head, Engram `engram_kv` | Q6_K or Q5_K | per-row lookup, matvec by type |
| Engram table | IQ4_NL, rows of 256 values (144 bytes) | host row read, `dequantize_iq4_nl` |
| Compressor kv/gate, indexer q/k/weight projections | BF16 | `gemv_bf16` (decode), `gemm_bf16` (prompt) (attn_index.cu) |
| Router gate (F32, or BF16 widened exactly at load), router bias (and the picture bias), hyper-connection fn/scale/base, all norm weights, attention sinks, Engram gate | F32 | fp32 kernels |
| Vision projector (optional `vision/mmproj` section) | BF16 | vision/encoder.cu (cuBLAS) |

## Paths

A prompt reads through the MMQ products in chunks; a prompt of up to 12 tokens reads through the decode kernels, one step at a
time, as the single-token tail always did. In a chunk under 8,334 tokens the second card's experts of up to 16 rows use the decode kernels and RAM experts of up to 8 rows the CPU dots.
Both round as their kernels do, inside the placement spread the quality section gives: against MMQ alone, on reads of 2,500 and
4,300 tokens, the last prompt row's KL is 3.3e-5 and 8.9e-4, its top-1 the same; on 1,000 tokens against up to 2 CPU rows, 4.6e-3.

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
| Decode, by tier | Each routed expert runs where it sits: from a slot on the main or the second card (q8_1, native grouped kernels, the same bits on either card; ds41_peer_parity), or on the CPU pool from the pinned RAM copy or the mapped pack (Q8_K, ggml arithmetic). **The two round activations differently, about 1% of a layer's routed output**, so a token's output depends on which experts were resident. The tiers start from the model's usual routing order (expert_rank.inc): on one card the RAM tier holds the most routed experts and the VRAM slots start with the head of them; on two cards the main card's and the second card's homes hold them, nowhere else, and the main card's copies start with the head of the RAM tier. Residency changes only at token boundaries and follows the routing history, so a fixed prompt on a fixed memory budget repeats exactly; a different RAM-tier size (free memory at start) can change the text. A repeat miss within 64 tokens is promoted to VRAM, half of a layer's misses go to the CPU and half over the bus, and a file-backed expert that misses twice swaps into the pinned tier for the least recently used pinned one (taking effect one token later), so the placement, and with it the rounding of later tokens, differs from the first-miss-promotes policy; the first 16 tokens of the fixture prompts are unchanged. The CPU's rows are read in place from mapped host memory by the weighted sum (same summation order) | tier.cpp lookup, fetch, run_cpu; ops_kernels.cpp routed_experts |
| Prompt chunks (up to 16,384 tokens) | Routed experts on MMQ: q8_1 (per 32, MMQ layout) of the fp32 input and of the fp32 clamped SwiGLU output, int8 mma against the IQ1_S, IQ1_M, IQ2_XXS or IQ2_XS codes (the IQ1_S grid shift rides in the activation block sums), fp32 accumulation, rows summed per token in slot order. **Numerics change against the single-token kernels** (a different int8 reduction order; both round activations to q8_1). Experts are staged from the cards' slots and the pinned RAM tier, so the prompt path never mixes in CPU arithmetic | prefill_ffn.cpp routed; prefill_ffn.cu; src/kernels/cuda/mmq_load_tiles.cuh load_tiles_iq1_s |
| Shared expert in prompts | MMQ GEMMs over 1,024-token sub-chunks (gate, up, clamped SwiGLU, down; q8_1 activations with an fp16 block scale, int8 mma, fp32 accumulation). **Numerics change** against the decode kernel: about 1e-3 per layer. The per-token weighted sum of the routed rows keeps the slot order, bitwise equal to the per-token launch | prefill_ffn.cpp shared; weighted_sum_tokens |

IQ1_M on MMQ: the int8 weight is 8 * (code - 1) +- 1, which holds the grid code and the +-1/8 shift per eight values
exactly, and the scale per 16 values is d * (2s + 1) / 8 in fp32 (src/prefill/mmq/mmq_load_tiles.cuh load_tiles_iq1_m).
Against a double reference on the dequantized weights and the fp32 activations: rel RMS 5.4e-3, as the other formats
(the activation's q8_1 rounding); against the dp4a matvec 2.2e-4. A row moves by 1.6e-7 relative RMS with the launch or
the row order, not bitwise, as on the other MMQ formats (prefill_mmq_iq1s_test).

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

## Prompt skip

Past layer 20 a row reads its own layer's window, 127 positions back, and layer 20's compressed rows and index keys. A row
further than 20 * 127 + 128 = 2,668 positions from the read's end therefore stops at layer 20's compressor: no token from
that end on can read what it would compute past it. The compressor and every layer before it run for every row, so the
compressed rows are all there. `--no-prompt-skip` runs every row through every layer. On when no Engram layer sits at or past
layer 20 (prefill.cpp `skip_layer`).

- **The math is the same.** An 8K prompt with the near rows one decode step each and the tier frozen, so no batch or
  placement ties two rows together; 111,788 hashes of layer outputs and logits. Two full reads: none differ. Full against
  skip: layers 0 to 19 none; from layer 20 on only rows inside the skipped rows' reach, 127 more per layer, and none past it.
  The last prompt token and 60 generated tokens are bit-identical.
- **Numerics change.** In a real read the near rows share batches with the rows inside the reach, whose routing differs, so
  the expert groups and their MMQ rounding move (inferred from the test above: without the batch, the differences are gone).
  The last prompt row against a full read: KL 2.4e-6, 1.7e-4, 7.7e-5 at 8K, 32K, 64K, top-1 the same, greedy 31, 210, 204
  of 210 tokens. Another chunk size on a full read moves the same row by KL 3.2e-5, 8.4e-4, 1.2e-3 and the tokens part at 50,
  158, 72 of 210. The BF16 reference scores 512-token chunks, which a skipped read never reaches.
- **What holds wrong values.** After a skipped read, rows inside the reach hold wrong values past layer 20 until the window
  rings move past them, and the rings keep only clean rows by the read's end. So state is saved or resumed only at a read's
  end, and one window back at most (`prefillp::may_save`, `may_resume`; prefill_plan_test): the session refuses anything else.
  Requests that diverge inside a skipped read read again from the last checkpoint on their path. With the skipped rows
  overwritten with NaN, 44 server requests (edits at seven depths, a system prompt edited inside, cancels, pictures, a restart
  from the disk tier, two slots) read none; moving the cut 300 rows towards the end turns the last row all NaN.

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
| Experts: GPU decode kernels quantize activations to q8_1 like llama.cpp's CUDA path, but residency splits the work over GPU and CPU. llama.cpp's own CPU path matches the CPU tier to 1e-7; its GPU path differs from ours by about 1e-2 on IQ1_S layers | Expert tiers (VRAM on one or two cards, pinned RAM, mapped pack) are how a 107 GiB expert set runs on these cards |
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
prompts; on the third it can diverge after a few tokens (see [Quality](#quality)).

## Quality

GSQ-RCO IQ2_XXS, 5090 only, against llama.cpp b11870 on the same GGUF, llama's tokens fed one decode step at a time on
three prompts: Basalt's greedy choice differs from llama's at 8 of 688 generated positions at the default tier state
(5, 2 and 1 on the three prompts), 10 with the RAM tier one expert larger and 11 with a VRAM tier of 6 experts per layer.
The count moves with which tier each expert sits in. KL against llama's logits on 512 positions of real text: mean
0.0025, median 0.0007, top-1 agreement 98.0%, position 1 left out: llama's own row at position 1 moves by KL 4.2
between a 2-token and a 512-token batch, and Basalt is within 0.04 of its 2-token row. On two cards as a server runs them, with slots that follow the routing and a context of
8,192: 8 of 688 positions (7, 1 and 0), KL mean 0.0030, median 0.0007, top-1 agreement 97.5%, position 1 left out.

Neither engine has one answer to compare. On the same tokens of the first prompt (319 generated positions; each row
names the two runs compared, KL over the full vocabulary):

| Pair | Mean KL | Top-1 differs at |
|---|---|---|
| Basalt default tier vs llama batched | 0.0032 | 9 of 319 |
| Basalt RAM tier 305 vs llama batched | 0.0032 | 10 of 319 |
| Basalt VRAM tier of 6 vs llama batched | 0.0031 | 10 of 319 |
| Basalt default tier vs Basalt RAM tier 305 | 0.0016 | 3 of 319 |
| Basalt default tier vs Basalt VRAM tier of 6 | 0.0020 | 3 of 319 |
| llama serial vs llama batched | top-5 log-probabilities move by 0.32 (median of per-row maxima), 2.08 (max) | 4 of 319 |

Over all three prompts the default tier differs from the RAM tier 305 run at 4 of 688 positions (mean KL 0.0019) and
from the VRAM-6 run at 7 of 688 (0.0021).

The largest gap between the two engines' tokens shows it. Log-probability of the first token minus the second, same row
(first prompt, generated position 124): llama serial +1.01, llama batched +0.39, Basalt -0.58 at the default tier, -0.21
at RAM tier 305 and -0.48 at VRAM 6, so Basalt takes the second token at all three. The mechanism: with input and
routing bit-equal, placement alone changes a layer's routed output by 0.9% (median; 1.9% at most), since q8_1 and Q8_K
round the activations differently. That is enough to move a top-6 choice in a later layer, and in the one position
traced a changed choice moved the layer's output by about 10%; that is a single observation, not a spread. Reruns at
one tier setting are bitwise equal, and asking for logits does not change the tokens.

### Against the BF16 reference

`basalt-bench --quality` scores the model against a BF16 reference made by @giveen: llama.cpp 330f0189, a plain
`convert_hf_to_gguf --outtype bf16`, wikitext-2, 100 chunks of 512 tokens, base PPL 2.7394 (2.7314 from the stored rows).
Basalt reads the first half of each chunk and scores the rest one row per window through the plain decode path. Two cards,
400 W + 180 W, 25,500 rows each:

| Pack | PPL | KLD mean | KLD median | KLD p99 | Rows above 2 | Top-1 |
|---|---|---|---|---|---|---|
| GSQ-RCO IQ2_XXS (the published pack) | 3.6353 | 0.4311 | 0.0894 | 4.39 | 5.06% | 79.97% |
| UD-IQ1_M (Unsloth's GGUF, converted here) | 3.0334 | 0.2779 | 0.0550 | 2.86 | 2.26% | 83.88% |

Both carry the same IQ4_NL Engram table. On the first 5 chunks of the IQ2_XXS pack, the same rows are worst whatever the
engine does: in reverse chunk order 59 of the 64 worst rows are the same (KLD correlation 0.996), on one card 61, with the
second card's and the CPU's prompt rows off 62 (KLD 0.4080 against 0.4088). The Engram add moves the stream by 64% of its
norm at layer 1 and 19% at layer 14 (medians), and the worst rows move no more than the rest. Prompt skip cannot be scored
this way (512-token chunks): on a 64K prompt, on against `--no-prompt-skip`, the 200 tokens after it are the same, KL mean
6.2e-4 between the two runs.

Earlier, the IQ1_S pack against the native model's logits (teacher: the native FP4 checkpoint through the reference
code; Lucebox's 8 x 1,024 held-out tokens, KL over the teacher's top-256 ids, llama.cpp with the same GGUF): KL 0.362,
top-1 agreement 75.3%, PPL 10.75 (teacher 8.56).

## Verify windows

`--spec` with a drafter, or `--spec-oracle`, runs T rows (at most 8) through the decode kernels at once. A kernel's output does
not depend on T or on a row's place, and each distinct expert of a layer runs once, on the card that holds it or on the CPU,
over the rows that routed to it. The compressor keeps its carried state until the window commits: the rows past the first
rejected one write nothing a later row reads. A row's pick is the draw of its own index, seed and position as in plain decode.

With placement fixed (`--diag-tier-frozen --diag-tier-ram 304`: no bus copy, promotion or swap), 400 W + 180 W, 300 tokens, prose
and code, short context:

- **Windows of the expected text**, 2, 3, 5 and 8 rows, every draft accepted: every row's logits equal plain decode's, bit for bit.
- **The drafter**, on either card, against plain decode with the same memory reserved so that the expert homes match: every
  accepted row's logits equal plain's, bit for bit.
- **Sampled**, temperature 1, two seeds, the drafter on the second card: the same text as plain decode.

With the tiers moving, which experts are resident follows the windows, whose rows route to more of them than one token does. A
row then rounds as under any other placement: forced through the plain run's text in windows of 2, 3, 4, 5 and 8 rows, of 800 tokens,
15 to 25 prose picks and 1 to 6 code picks differ from the plain run's. The BF16 reference was not run with windows wider than one row.

## Sampling

Draws run on the shared `sample_tokens` kernel, the same chain as the other models: a seed gives other text than the earlier host sampler did. Greedy decoding keeps its argmax, so its tokens do not change.
