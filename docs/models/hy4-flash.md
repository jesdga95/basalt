# Hy4-flash (design)

Full Basalt support for `tencent/Hy4-flash` (`HYV4FlashForConditionalGeneration`): architecture id `hy4flash`, code in
`src/models/hy4flash`, `include/basalt/models/hy4flash`, `serve/models/hy4flash.py`. Supported from 0.5.0. Pack and converter (the
Q4_K_M test pack first): [pack](hy4-flash/pack.md). References read: llama.cpp PR 32517 (`hy-v4-flash.cpp`,
`llama-memory-idx-iswa.cpp`, `conversion/hy_v4_flash.py`), the HF config, chat template and processor config, vLLM's `hy_v4` model
package, AngelSpec's DFly drafter, a routing trace of 57K prefill and 3.6K decode tokens. Numbers marked *est* are not measured.

## The model, as verified against config.json and the GGUF

| | Value |
|---|---|
| Trunk | 42 layers, hidden 3072, vocab 120,832, untied FP32-logit head, 4 identity hyper-connection (iHC) streams |
| Attention | MLA over one shared latent: q_lora 1024, kv_lora 512, qk 192 nope + 64 rope, v 256; K row = latent 512 (RMS-normed) + k_pe 64 (roped), V = the latent; query absorbed through `k_b` (192 -> 512 per head), output through `v_b` (512 -> 256 per head); learnable per-head sink in the softmax denominator; sigmoid output gate `attn_gate(x)`; sandwich post-norm after `o_proj` |
| Layer types | 12 sparse (0, 1, 5, 9, ..., 41): 32 heads, RoPE 1e7, indexer, top 2,048 positions; 30 SWA: 64 heads, RoPE 1e4, window 512 (itself plus 511) |
| Indexer (sparse layers) | per token `ik = RMS(attn_k x)` (128), a value and a gate projection of `ik` (128x128 each); every 4 tokens one pooled key `RMS(sum softmax(gate + ape) * value)`, roped at the group's first position; query `q_b_idx(qr)` 16 heads x 128 (roped on the last 64 dims); score `sum_h w_h relu(q_h . pool)`, `w = proj(x) / sqrt(2048)`; the best 511 pools plus the current group (4 cells) = 2,048 positions. Up to 2,048 visible positions it selects everything, so the indexer changes nothing there |
| MoE | layer 0 dense (13,824); layers 1-41: 512 routed experts, top 16, sigmoid scores + `exp_probs_b` for the choice, weights normalised, x 3.998; routed experts run in a 1536 latent (`routed_down` 3072 -> 1536, experts 1536 -> 1536 -> 1536, `routed_up` 1536 -> 3072); one shared expert 1536; SwiGLU clamp 10 on routed only; post-norm after the FFN |
| iHC | per sublayer 8 mixes (pre 4, post 4): `pre = sigmoid(m * s0 + b) + eps`, `post = 2 * sigmoid(m * s1 + b) + eps`, `out[i] = residual[i] + post[i] * f`; **no cross-stream mix and no Sinkhorn** (DeepSeek and GLM have both); RMS eps 1e-5; head: 4 sigmoid weights, weighted sum |
| Engram | layers 1, 11, 21, 31 (0-based); trigram: 2 orders x 8 hash heads = 16 rows of 96 FP8 values per token and module; one table of 660,006,416 rows (63.4 GB); rows RMS-normed per head (eps 1.19e-7), fused projection with bias to 4 key copies + 1 value, gate as DeepSeek's, `delta = norm(value) * gamma * (gate + hc_scale)` added to each stream; masked (valid 0) at image positions and at the first token |
| Vision | SigLIP2-So400m-derived ViT, 27 layers, 1152, 16 heads, MLP 4304, patch 16, learned 128 x 128 position table (bilinear), 2 x 2 merge, RMS + 2-layer MLP projector -> 3072, learned `image_newline` row; up to 4,096 merged tokens (max_pixels 2048^2); mean = std = 0.5; resized with Pillow's bicubic (the configured filter 1, Lanczos, is unused) |
| DFly2 | 5 dense layers (MLA as the target's sparse layer, no indexer, MLP 9,216), initialised from target layers 25, 29, 33, 37, 41 and reading their KV and index selection; `mtp_fusion` (gated fusion of last token embedding and last-layer hidden state), `hidden_correction` (SwiGLU over `norm(h) :: norm(e_prev)`, applied sequentially), `confidence_head`, mask token 120009, 5 drafts per block |

Differences that matter next to DeepSeek-V4.1: a K row is 576 wide with V = its first 512 (DeepSeek's key and value are the same 512
latent); two layer types over one cache geometry; no compressed (CSA/HCA) rows, the sparse layers keep every position's row and an
indexer picks among them; iHC without comb; LatentMoE; 512 experts x 41 layers at top 16 (DeepSeek: 384 x 40, top 6) so a decode
token routes 656 experts and reads 2.61 GB of NVFP4.

## Pack

Test pack (Q4_K_M GGUF): [pack](hy4-flash/pack.md). Release pack, from the BF16 checkpoint (`tencent/Hy4-flash`, 148 shards):

- **Routed experts: NVFP4**, RTN from BF16 with the 27B pack's rounding (tensor scale amax / (6 x 448), per 16 values the UE4M3 scale
  with the least squared error among half to four times amax / 6). Layout as DeepSeek's and Flash-Next's NVFP4 experts: ggml type 40
  blobs `[gate | up | down]` plus a 16-byte FP32 tail `{s_gate, s_up, s_down, 0}`: 3,981,328 B per expert (the same size as a Q4_K
  blob: both are 4.5 bits), 83.6 GB for 21,012 experts. The checkpoint stores `gate_up_proj` fused: split it.
- **Dense: Q8_0** of the BF16 values (attention, latent projections, shared experts, dense FFN, `engram_kv`, indexer projections,
  embedding); BF16 head (742 MB) first, `F8_E4M3` rows with a scale (type 44) later if it pays; norms, hc, router, biases F32.
  Quantising from BF16 reproduces AngelSlim's Q8_0 bytes (checked: blocks 0 to 5, the embedding, the head).
- **Engram table: E4M3, rounded once** from the BF16 values (not E4M3: 9 % exact, 2.7 % RMS error; the test pack's rows are the same
  rounding scaled to a largest code of 448, scale dropped), 96 values per row, no scale (type 43), own `engram` section, read by row.
- **Side sections**, from the checkpoint, no weight copies: `dfly` (the 0.85B drafter: Q8_0 projections, F32 norms), `vision/mmproj`
  (ViT and projector BF16, plus `image_newline`), `dfly/draft_vocab` later. Tokenizer, chat template, generation defaults.
- Whether NVFP4 RTN beats Q4_K_M quality is not known (Q4_K_M used an imatrix; RTN has no calibration): the KLD gate decides the
  default pack (risks). The converter needs only the BF16 shards, streamed (`--stream-shards`) like DeepSeek's.

## Memory (RTX 5090 counted as 34 GB, RTX 5060 Ti, 121 GiB RAM), GB

| Main card | Bytes |
|---|---|
| Dense Q8_0 + BF16 head + F32 (a token reads 7.6 of it: the embedding is a row lookup) | 8.0 |
| Context, graphs, scratch, reserve | 1.3 |
| Prefill workspace at chunk 16,384 (4 streams x 16K x 3072 FP32 in and out, qkv, router, expert stage) | 2.5 |
| DFly2 (Q8_0) / BF16 | 0.9 / 1.7 |
| Vision ViT 0.86 BF16 | lent from the expert slots while an image is encoded (Flash-Next's `attach`/`detach`), none standing |
| KV, sparse layers, BF16 rows of 576 (1,152 B x 12 = 13,824 B per position) + pooled keys 768 B per position | 128K: 1.81 + 0.10; 256K: 3.62 + 0.21; 1M: 14.5 + 0.81 |
| SWA rings, 30 x 512 x 1,152 B | 0.018 per conversation |
| Left for expert slots (3.98 MB each) | 128K: 19.4 = 4,870 slots; 256K: 17.4 = 4,370; 1M BF16 KV: 6.0 = 1,500; 1M FP8 KV (656 B rows: 8.25 + 0.41): 12.6 = 3,170 |

RTX 5060 Ti (16 GiB): about 14.4 GB of slots (context 0.8, prompt buffers 1.5, reserve 0.4) = 3,620 slots. Each further 16 GiB card
adds the same. Slots as a share of all 21,012 experts, 128K context: two cards 8,490 (40 %), three 12,100 (58 %), four 15,700 (75 %).

Hit rate from the routing trace (decode streams, per-layer LRU, start contents from the prefill profile; k = 1 reproduces the
0.82 to 0.85 of the trace analysis; a miss is admitted after its k-th CPU use; admitted MB are host-to-card copies per
token):

| Slots (GB) | k = 1 | k = 2 | k = 3 |
|---|---|---|---|
| 30 (183 per layer) | hit 0.830, 443 MB | 0.810, 225 | 0.797, 147 |
| 33 (202) | 0.851, 388 | 0.833, 197 | 0.820, 129 |
| 45 (275) | 0.914, 226 | 0.901, 113 | 0.892, 74 |
| 60 (367) | 0.964, 93 | 0.958, 46 | 0.953, 30 |

The traces are 94 to 1,500 tokens, so these are indicative. Plain decode, two cards, 33 GB, k = 2 (*est*): dense 5.1 to 5.5 ms on
the main card (7.6 GB at 1.4 to 1.5 TB/s); CPU and DRAM: misses 0.44 GB plus admission copies 0.20 GB at the 52.9 GB/s the analysis
measured = 12 ms; the cards' expert reads overlap. About 13 to 14 ms per token, **65 to 75 tok/s**, inside the 60 to 80 target;
the analysis's ceiling at hit 0.83 is about 120. **The DRAM, not the GPUs, sets the speed**: admission copies and CPU products
read the same RAM, so the admission threshold is a tuned default (risk 1). Three cards (hit 0.90): CPU 7 ms, the main card's 5.5 ms of dense reads next, so 100 tok/s or a little more *est*. RAM holds every expert (NVFP4 77.9 GiB; Q4_K test pack 83.4 GiB) plus the checkpoint blocks and pinned buffers; the
Engram table (59 GiB) stays on the SSD, with the rest of RAM as the OS cache of its hot rows.

## Decode and prefill, kernel by kernel

R = reused as is from DeepSeek (DS) / the shared tree, S = moved to shared code (a commit of its own, DS output bit-identical),
N = new in `src/models/hy4flash`. T-invariant (a row's result does not depend on the window width) is the rule for all.

| Stage | Kernel | |
|---|---|---|
| Dense Q8_0 / Q4_K projections, q8_1 activations | `native_mmvq`, `quantize_act` | R |
| Embedding row, BF16 head, FP32 logits | `iq_kernels` embed, `bf16_gemv` | R |
| iHC mixes (8), pre, post (no comb), head | `kernels/mhc.cu` from glm53's `a62b2de5` (already moved to shared code) plus a variant with 8 mixes, no Sinkhorn, post eps | S + N variant |
| RMS norm rows, hc_pre + norm | DS `norm.cu` (`rms_norm_rows`, `hc_pre_rms_norm`) | S (they sit in DS's kernels lib) |
| Router: sigmoid, bias for the choice, normalise, scale | DS `router.cu` with a gate-function parameter | S + param |
| Rope tables, interleaved MRoPE (12/10/10 over 32 pairs) | DS `attn.hpp` step tables; MRoPE reads 3 position planes | R + N (MRoPE) |
| q path, kv path, latent RMS, cache write (BF16 rows 576; ring or full) | DS `q_path.cu` shape, `cache_rows.cuh` | N (shapes differ) |
| `k_b` absorb, `v_b` expand + sigmoid gate epilogue (batched per head Q8_0 GEMV, 32/64 heads) | | N |
| Sparse and SWA attention: 576-wide K, 512-wide V, sink, window ring or selected rows | DS `attn_mqa.cu` generalised (dK != dV, head count, one row source); its prefill twin `prefill_attn.cu` | S + N |
| Indexer: key norm, value/gate, pool (softmax over 4), pooled rope, score, top-511 + tail | DS `attn_index.cu` top-k and scoring skeleton; glm53's `indexer.cu` is a naive FP32 single-query version: **not taken** | N |
| Prefill indexer: fused `relu(q.k)` x `w` head-sum epilogue over key tiles, streaming top-k (no 16K x 262K score matrix) | DS `prefill_index` structure | N |
| Latent MoE: `routed_down`, experts (clamped SwiGLU), `routed_up`, shared expert, post-norm | `native_mmvq`, grouped expert kernels (`iq_kernels`, `kq_avx2`, NVFP4 `native_expert`), DS clamp | R (+ CPU `q6k_dot` for the test pack) |
| Engram: hash (same family as DS: rolling XOR of token x multiplier, `% prime + offset`), SSD row reads, apply | DS `engram_hash.cpp`, `file_reads`, `engram_rows.hpp`, `prompt_host`; apply kernel differs (bias, per-head norm, `norm(value) * gamma`, valid mask) | S (hash, reads) + N (apply) |
| Prefill dense and expert GEMMs | `window_gemm*`, `prefill/mmq`, `moe_mmq`, `moe_fp4`, DS `prefill_ffn` | R |
| Sampling, logprobs | shared | R |

glm53 branch: take `a62b2de5` (the hc kernels in shared code) only, by cherry-pick or by waiting for its landing; its converter hooks
(`model_sections.py`, `convert.py`) overlap with the Hy4 converter changes, so whichever lands second rebases. Its MLA and indexer
kernels are single-query FP32 decode kernels with a 4,096-cell cap.

Decode attention (`kernels/mla.cu`), up to 8 rows: `mla_query` (the `k_b` product into the query rows, their rope part, the cache rows), then
`mla_attend` on DeepSeek's MQA chunk stage (`basalt/kernels/mqa_chunk.cuh`: a block per 64 cells and 8 or 16 heads reads each cell once, the
last block of a row's head group merges the chunks with the sink), then the `v_b` product. The router and the row norms are shared already (`basalt/kernels/router.hpp`, `row_norm.hpp`).

Prompt path (`session.cpp`): a prompt opens row buffers for chunks of up to 16,384 rows (halved to fit) and the VRAM tier lends
them only the slots they and the expert ring need (`ExpertTier::lend`: the base arena is mapped in 256 MiB segments laid out by depth,
so its tail holds every layer's last slots; each layer's most recently used experts move into the slots it keeps), all of them only
when that is not enough; the latent caches borrow their growth the same way. Each chunk runs the 42 layers; the attention runs in steps of 2,048 rows (the rings are sized for a step), each step
with its own row spans. Sparse layers score every row against the pooled keys before it on tensor cores (`scores_mma_kernel`) and
select 64 rows a launch; the attention over the selected or windowed cells runs on tensor cores with fp16 operands
(`attend_mma_kernel`, 32 heads a block sharing each tile of cells). The heads products and the router are one launch a step, the dense products MMQ launches of a step's rows. From 1,024 rows every
expert is used: a ring of two layer slots copies the next layer's experts while a layer computes, and the chunk's small host inputs
(spans, launch tables, Engram rows, embeddings) go through mapped memory and a copy kernel so they never wait on the copy engine
behind the ring. Every Engram layer's rows are asked for when the chunk starts. With a second card, its prompt share (`prefill::PeerShare`) computes its own experts and the RAM experts
its link prices lower (a ringed chunk plans before routing and the ring leaves them out). A chunk below 1,024 rows keeps the card's
expert slots, and the CPU pool takes RAM experts of up to 8 rows: on one card as many as `prefill::cpu_take` prices to end with the
bus (fixed prices, the same split every run), beside a second card by DeepSeek's rule. The prompt log counts the experts by source.
The slots a long prompt leaves resident are staged card to card by the ring, and the next turn and its decode find them.

Measured (1x RTX 5090 at 400 W, Q4_K_M test pack, greedy, real text): a 3,519-token summary prompt 3.27 s (1,076 tok/s), a
16,191-token prompt 5.13 s (3,156 tok/s); hy4-a before this path: 18.7 s and 90.8 s. A chunk of 1,024 rows or more is bound by the
bus: 89.5 GB of experts at 29 GB/s (the card's x8 PCIe 5 link) = 3.1 s a chunk, so short prompts sit near that floor.

## 1M context

Sparse layers keep every position (fp16 rows of 1,152 B x 12 and the pooled keys: 14.6 KB a position, 14.3 GiB at 1M; `--kv fp8`,
vLLM's `fp8_ds_mla` row of 656 B: 8.4 GiB). The caches are elastic as Flash-Next's: the first 32,768 positions are backed at start, the rest
come out of the expert tier when a prompt needs them and go back when a later prompt needs fewer; a prompt that would leave the
tier below a prompt chunk's room is refused and the engine serves on. Formats, rounding, selection cost and the 1M measurements:
[KV cache](hy4-flash/kv.md).

## Expert tier and what becomes shared

DeepSeek's `tier.cpp` (1,169 lines), `tier_policy.hpp` (232, CUDA-free), `file_reads`, `elastic`, `prompt_host` are DeepSeek's only
by their pack accessors. Extract, as its own commits with DS output hash-identical and DS speed unchanged (ABBA, tiers pinned):

1. `core/tier_policy`: homes by routing rank, multi-card layer blocks, RAM ceiling, admission rule (+ `kPromoteUses`), already free of CUDA.
2. `core/expert_tier`: slots, per-layer LRU, RAM tier with CPU compute, file reads of experts left on the SSD, copies on a side
   stream, peer cards. The model supplies a descriptor: first MoE layer, experts per layer, top-k, per-layer blob offsets and
   sizes, formats, and the hooks `clamp` and `latent` (the expert input is the latent for Hy4, the layer input for DS).
   Hy4 also gives per-layer weights for the main card's copies (`layer_weights`; DeepSeek gives none and keeps the same
   copies in every layer): its early layers route more widely, so layers 0-3 hold about 1.4x the mean and layers 15, 27 and
   31-33 about 0.73x, the same total (135 a layer on average beside the Q8_0 drafter). On serve bench.sh against the same
   copies everywhere: struct +4.6%, prose +2.3%, conv and prefill within noise. The table and its traces are in `model.cpp`.
   The start places copies, second-card homes and the slots beside the base arena by `data/profiles/hy4flash-42x512.prof`
   (each layer's experts by decode routing over prose, code, conversation, structured output and agent turns): the first
   request after a start decodes about 17% faster than with the experts in id order; after a few hundred tokens the LRU's own
   placement takes over. The misses left are mostly reuse at one to two times the slots per layer (recency beats frequency:
   decayed-count eviction, a pinned hot set and count-gated admission all simulated no better than the LRU).
3. `core/engram_table`: row reader (O_DIRECT, row-aligned, prefetch, hot-row LRU in pinned RAM); the row decode (IQ4_NL, MXFP4,
   E4M3) is the model's.
4. `core/elastic_array` (DS `elastic.hpp`, beside `core/vmm.hpp`) for the KV backing.
5. Qwen's adaptive rounds (count-based swaps, `docs/configuration/adaptive.md`) and DS's miss-driven LRU are two policies of one
   tier; Hy4 takes the DS policy. Unifying them is out of scope here.

Until step 2 lands, `hy4flash` runs on the shared primitives it has (`core::ExpertCache`, `ExpertPool`, `PeerExperts`: RAM tier,
CPU products, static main-card slots) behind one small interface (`lookup`, `compute`, `admit`); that adapter is deleted when the
shared tier replaces it. Multi-GPU: second card and further cards hold layer blocks as DS (second-card homes by contiguous layer
blocks, up to four after the main card); a prompt's layers split by the shared prefill split. Three to four cards at 14 GB each
reach 75 % of experts.

## Vision

The SigLIP2-derived ViT runs on the shared CUDA encoder (`src/vision/encoder_cuda.cu`) from the side file's projector,
staged once in pinned RAM; a request's pictures borrow the expert tier's room on the main card while they are encoded. Pillow's
bicubic resize (as transformers' processor), the `<image_start>` pad run with newline slots, interleaved MRoPE [12, 10, 10] (the
indexer's pooled key turns by its group's first cell), Engram masked at picture rows: [vision](hy4-flash/vision.md).

## DFly2

- **Weights**: 5 layers, 0.85B, Q8_0 (BF16 first), as a `dfly` section; shares the token embedding and head with the target.
- **Block pass**: input is `[last accepted token, mask x 5]` (`num_speculative_tokens` 5; the block size is confirmed from the drafter's own config). Gated fusion: `embed(last token)` and the target's
  last-layer hidden state (post `output_norm`, 3072 floats) go through `enorm`, `hnorm`, `eh_proj`, scaled by `gate`.
  Layer i runs MLA with its own q path over the block rows and attends **target layer 25 + 4i's cache**: the context rows named
  by the index selection of that layer **at the anchor position** (Index Share; at most 2,048 rows) plus its own 5 block rows (K/V
  from its own `kv_a`), sink, then the dense MLP. Cost is constant in context length.
- **Sequential head**: for draft j, `h'_j = h_j + SwiGLU(norm(h_j) :: norm(embed(draft_{j-1})))`, logits, argmax or sample, and
  the confidence head's chance. Five sequential heads read a 742 MB BF16 head each (0.5 ms); the pack's `draft_vocab` restricted
  head (DS has the mechanism) or a Q8_0 head cuts it to 0.1 to 0.27 ms. Block pass about 1 ms, the heads 1 to 2.5 ms *est*
  (DS's DSpark block takes 1.4 ms).
- **Plugs into** `include/basalt/spec/draft_policy.hpp`: a block drafter that states each draft's chance picks the window by
  expected tokens per millisecond from measured round costs, as DSpark does; `--spec` default 6 (anchor + 5), 0 off. The window
  is verified as one T <= 6 pass.
- **Exact verify**: output is identical to plain decode with placement fixed because every verify row equals plain decode's (T
  invariance), including the indexer: the pooled key of a group is built from the same 4 raw `(value, gate)` rows by the same
  code whichever window completes it, the raw rows live in a ring of the last >= 4 + window tokens, pooled rows written for rejected
  positions are overwritten, and the SWA ring is sized 512 + window. The drafter reads the target's caches read-only and writes
  only its own block rows (scratch), so a rejected block leaves no trace. Tests: greedy identity with and without `--spec`, tiers
  pinned; per-kernel T = 1..8 identity.
- **Source and forward**: vLLM PR 68296 has the inference code (the drafter files under `vllm/models/hy_v4/nvidia/`); the forward, the Basalt mapping and the
  assumptions that remain are in [dfly2](hy4-flash/dfly2.md). Acceptance only affects speed, never output.

## Serve

- Registry: the id in `BASALT_BUILTIN_MODELS`; `dialect_for("hy_v4_flash")` in `serve/models/hy4flash.py`; one router config entry.
- Prompt, effort, tools, images, sampling: [serving](hy4-flash/serve.md). The template is the pack's (the special-token suffix is read
  from it: the AngelSlim GGUF names them `:6124c78e`, the HF release `:opensource`); `reasoning_effort` `none` is `no_think`,
  `low`/`medium`/`high` are `high`, `max` is `max`, numbers are refused; `preserve_reasoning` is the template's `preserved_thinking`.
- Tools: `<tool_calls><tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call></tool_calls>`; the parser is the
  dialect's, `reasoning_close` is the think end token and the stop text is the `hy_end` token (120025).
- Engine: `session` calls (`decode_window`, `prefill_chunk`, `save_cells`, `restore_cells`, `running_state`: the SWA ring, pooled
  keys, the raw indexer ring, the last two token ids for Engram, DFly's last hidden state). Sleep frees the slots and the dense
  weights (re-read from the pack in seconds) and keeps RAM; wake opens the base slots again empty (the slots beside them are
  seeded by the profile) and decode refills them. With every expert in RAM, two-card sleep needs the KV in free RAM (DS's rule: 8 GiB spare), so
  it may be refused at long contexts. Park and swap use the conversation cache; `POST /v1/vram` and `--vram-reserve-mib` as DS.
- Load: `plan` sizes weights, then the KV, tier and workspace from what is left; the announced total is the dense bytes + the RAM
  tier + the drafter; one placement pass; `attach` packs forms and times nothing else.

## Precision gates

1. Parity (Q4_K_M test pack, same weights as llama.cpp): per-layer hidden states against llama.cpp's eval callback, a fixed
   prompt; relative RMS per layer <= 2e-3 (DS prompt-path level), selected ids identical, router ids equal except near ties.
2. Against llama.cpp end to end on the same GGUF: KLD mean < 0.005, top-1 >= 99 %. Against the BF16 reference
   (`hy4flash-bf16-base.kld`) through `tools/ppl_gate.py`, 100 chunks: Q4_K_M test pack, NVFP4 pack, both next to
   llama.cpp's Q4_K_M; docs/precision/hy4-flash.md gets the rounding table (as DS's).
3. Bitwise: every new kernel T = 1..8 identical, frozen hashes; the drafting identity; one card vs two for decode with pinned
   tiers; the DS and Flash-Next outputs unchanged by every shared extraction (`tools/greedy_identity.py`).
4. PyTorch: a CPU layer-level reference from the HF modelling code (transformers 5.23) on single layers, once available, for the
   BF16 tensors the GGUF cannot give (the indexer pooling in BF16 versus FP32).
5. Long context: selected ids against llama.cpp at 8K, 32K, 128K on the test pack; needle retrieval at 128K, 512K, 1M.
6. Before any of it: the test GGUF is the released checkpoint (dequantise `blk.0.attn_q_a` Q8_0 and compare with the BF16 shard;
   the token suffix differs from the release, so it may be a pre-release revision).

## Milestones

| | What | Lands when | Agent |
|---|---|---|---|
| M0 | Converter for the Q4_K_M GGUF, tests, `Hy4-flash-Q4_K_M.basalt`; revision check (gate 6) | day 1 | C (converter) |
| M1 | Engine bring-up: adapter, pack, all decode kernels with parity tests, **contexts <= 2,048 (no indexer)**, one card, all experts from RAM by CPU plus optional static slots, Engram from SSD, greedy 128 tokens equal to llama.cpp's on 5 prompts, gates 1 and 2 | about 5 to 6 days | A |
| M2 | Indexer and pooled keys, prefill path (chunks, sparse prefill attention + prefill indexer, MoE prefill), BF16 KV with elastic backing, contexts to 128K then 256K and 1M, gate 5 | about 5 days | A |
| M3 | Shared extraction (steps 1 to 4 above) then the multi-card, RAM-LRU tier on it; admission threshold by measurement | about 6 days, parallel with M1 and M2 | B (`core/`, `src/models/deepseek41`; never `hy4flash`) |
| M4 | Serve: dialect, template corpus, effort, tools, router entry, sleep/park/swap, load line, docs | parallel from M1, about 3 days | D (`serve/`, docs) |
| M5 | NVFP4 pack from BF16 (quantiser, split of fused experts, streaming), KLD gates, default decision | parallel from day 2, about 3 days | C |
| M6 | DFly2: `dfly` section, block pass, sequential head, draft policy, exact-verify tests; needs M2 | after M2, about 5 days | A or E |
| M7 | Vision: encoder changes, Lanczos, prompt layout, MRoPE; needs M2's embedding prefill | after M2, parallel with M6, about 4 days | F |
| M8 | Precision doc, `basalt-bench` rows (power limits stated), community and README entries, 1M numbers | last | all |

Files each agent owns: A `src/models/hy4flash`, `include/basalt/models/hy4flash`; B `core/`, `src/models/deepseek41`; C `tools/convert`
(`arch/hy4flash.py`, `model_sections.py`, NVFP4 quantiser); D `serve/`, `docs/`; F `src/vision` plus hy4flash's vision file.
Shared kernel moves (hc, norm, router, attention generalisation) are A's, each its own commit with the DS parity and speed numbers.

**First milestone, precisely (start tomorrow).** C: implement the six converter changes in [pack](hy4-flash/pack.md), run the dry
run, produce the pack, run the byte checks, report the revision check. A, the same morning, without waiting for the pack: (1) rebase
`a62b2de5`; (2) skeleton (`adapter`, `pack` reading `meta.geometry`, `meta.engram`, tensor table; CMake id; refuses unknown
layouts); (3) kernels in this order, each with a parity test against CPU double reference and llama.cpp's graph (`llama-eval-callback`
dumps of `blk.0`, `blk.1`, `blk.2` for one fixed prompt on the Q4_K_M GGUF, stored under `data/`): iHC variant, router (sigmoid),
q path with absorb, K/V write, SWA/sparse attention with sink (<= 2,048 rows), `v_b` + gate, latent MoE on the CPU pool, Engram.
Exit: layer 0 to 2 hidden states within gate 1, then all 42. GPU tests are queued on the box's GPU lock.

## Risks and open questions, with the measurement that settles each

1. **DRAM-bound decode.** CPU products and admission copies share about 53 GB/s. Measure: decode bench with k = 1, 2, 3 and 30 to
   33 GB, tok/s and RAM bandwidth; replay the sim with a longer trace (agent sessions, 20K tokens).
2. **Prefill speed.** Every chunk streams the experts not in slots. Measure DS and Flash-Next prefill tok/s on this box first; then
   Hy4's at 4K, 16K, 128K. A 1M prompt is minutes.
3. **Indexer cost and numerics at 1M.** Measure the fused score + streaming top-k kernel at 262K pooled keys (TFLOP/s against the
   55 est.); FP8 pooled keys only if the selected-id overlap with FP16 is above 99.9 % and KLD is unchanged.
4. **DFly2 acceptance** (the forward is in dfly2.md). Measure acceptance per position on code and prose; compare with README's 150+ tok/s claim.
5. **Test pack revision / tokenizer suffix.** Gate 6, a day-1 check. If it is a pre-release revision, its KLD against the BF16 file
   is not a quantisation figure; use llama.cpp's Q4_K_M on the same file as the parity base and the BF16 file for NVFP4 only.
6. **Engram SSD.** 64 random 96-byte rows per token (4 KB reads), about 1M reads per 16K chunk. Measure: 4K random read IOPS at
   QD64 on `/mnt/nvme`; hot-row hit rate by replaying the trace's token streams through the hash (CPU only); prefetch overlap.
7. **NVFP4 RTN quality** without calibration. Measure KLD against Q4_K_M and BF16; fall back: per-layer scale search, a higher-bit
   `down`, or Q4_K as the default pack. Weight RMS error: NVFP4 0.081; Q4_K 0.071 (gate, up), 0.077 (down); Q6_K down 0.018 (19 layers).
8. **KV precision.** fp16 default (the reference run's cache, Basalt's convention); `--kv fp8` (vLLM's rows) is opt-in, measured in [KV cache](hy4-flash/kv.md).
9. **Selection ties and T invariance.** FP16 tensor-core scoring versus the reference's BF16 can flip near-tied pools; measure the
   id overlap against llama.cpp per layer (decode and prefill) and the logit effect; keep the scoring order fixed across windows.
10. **Q6_K down in the grouped expert kernels (GPU) and no `q6k_dot` on the CPU** for the test pack: parity tests first (pack doc).
11. **GPU time.** One shared GPU limits kernel and bench throughput; kernel parity runs on short slots, benches batched.

## Open decisions

1. **License.** Tencent Hy License 1.0 is Apache-derived with a MaaS clause (section 10). Packs are derivative works; read it
   before any pack goes on Hugging Face. The converter embeds its text.
2. **Default pack**: NVFP4 (fast path: tensor-core prefill, shared kernels) versus the Q4_K_M-style pack if RTN NVFP4's KLD is worse.
   Settled by gate 2 after M5.
3. **KV default**: BF16 (exact, 14.5 GB at 1M) versus FP8 (8.25 GB); and whether 1M is advertised at first release or 256K.
4. **Parallel plan**: tier extraction (M3) alongside M1/M2 with a throwaway adapter, versus extraction first (cleaner, delays Hy4 by
   about a week). Recommended: parallel.
5. **Names**: `hy4flash` for the id and directories, `Hy4-flash` in docs and pack names.
6. **Video** left out of the first cut (the model supports it); DFly2 forward may need the vLLM image.
