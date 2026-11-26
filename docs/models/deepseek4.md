# DeepSeek-V4-Flash-Vision-Exp

## Architecture decision

Implement V4 as its own registered Basalt model (`deepseek4`) with a dedicated pack reader and session. Reuse the proven DeepSeek-V4.1 infrastructure only where the source geometry and semantics match: adapter lifecycle, expert tier, hyper-connection (mHC) execution, checkpoint transfer, peer hooks, batching scaffolding, and the vision service integration points. Keep V4 cache construction and index selection model-specific because their compression schedules and cache semantics differ. The source uses the same 64-head, 512-wide sparse MQA shape, but its `act_quant(..., inplace=True)` path simulates FP8 and rounds back into BF16, its query and KV tensors are BF16, and the attention result is BF16. The existing V4.1 MQA path accepts FP8/FP4/FP16/FP32 cache rows and emits FP32; no BF16 row mode is present. Its sink and inverse-RoPE structure matches the source, but sharing it remains unproven until a BF16-input/output parity test covers rounding and accumulation.

The target checkpoint is `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, converted from its source GGUF/safetensors into a Basalt pack. Its 12 sections and all 1,371 source tensors were verified against the source GGUF; every section hash and tensor hash matched. It has the 43 trunk layers, token-to-expert tensors for the first three layers, and an embedded `deepseek4v` projector. Its manifest has no MTP tensors or drafter section, so an MTP drafter is not available in this artifact. Preserving the source tensors and projector does not constitute runtime support.

## Runtime layout

- `Pack` owns the validated V4 geometry, named dense tensor references, per-layer expert ranges, and the embedded projector file range. `plan` places the dense working set on the primary GPU and queues the source ranges once through `LoadPlan`; the token embedding stays file-backed for row reads, and routed experts stay in the `ExpertTier` file/RAM/VRAM placement rather than being duplicated.
- A V4 `BatchSession` owns per-slot attention state. Its `ElasticArray` cache rows retain their source position and layer schedule: sliding rows, ratio-4 CSA rows, ratio-128 HCA rows, and separate indexer rows. Chunk continuation, rewind, save/restore, and batch slots must preserve those identities. The cache builder and selected-position attention stay V4-specific; mHC mixing and compatible router/expert primitives are shared.
- A text step expands the mHC stream, applies the layer's attention and cache update, runs V4's token/hash routing plus shared expert, then collapses the stream. Prompt chunks initially use the same single-token path for identity; a batched prefill path is admitted only after matching it. Peer mode distributes expert tiers only, with one-card identity as the reference.
- Vision follows the V4.1 loan lifecycle but uses V4 preprocessing and N-layout. At rest, the session owns no image bytes, patch buffers, or encoder weights. On an image request it borrows expert-tier room, decodes and grows per-slot buffers, loads the embedded `deepseek4v` tower, encodes and assembles rows, then releases buffers and encoder weights in `images_done` before the tier regrows. The image token positions must reach V4's positional path; text-only hash lookup must not treat image rows as text IDs.
- The existing `serve/models/deepseek4.py` uses the DeepSeek message encoder and shared incremental DSML parser with V4's tool-tag spelling, EOS, reasoning marker, and image placeholder. Keep that dialect model-local; C++ metadata supplies EOS, image, and pad token IDs and the joyai-llm pre-tokenizer.

The staged path is registration/adapter and one-card serial decode, then token identity and KLD, then chunked prefill and persistent state, then expert-tier and peer/batch behavior, and finally request-level Vision and zero-idle-memory checks. Do not register an adapter that cannot execute the selected pack.

## Model path

1. Validate and describe `general.architecture=deepseek4` metadata, including geometry, compression schedule, MoE routing, tokenizer, context and multimodal fields.
2. Parse the Basalt pack into a V4-specific `Pack`/`Geometry`; validate tensor presence, shapes, storage types and per-layer schedule before allocating runtime state. `ds4_pack_test` opens the converted checkpoint through the C++ reader and passes the 43-layer geometry, all tensor shape/type checks, hash tables, expert layout and vision section checks.
3. Load embeddings, attention projections, mHC matrices, indexer, router/hash tensors, shared expert and routed experts. Reuse DeepSeek-V4.1 expert-tier placement and file/RAM/VRAM read mechanisms where tensor formats match; reject unsupported storage explicitly.
4. Implement text prefill and decode with V4's attention and cache contracts. Integrate V4-specific compression states and attention indexing. Share only shape-compatible projection and mHC primitives with V4.1.
5. Add the model's tokenizer/chat dialect and serving registration. Keep checkpoints, batch execution and peer execution semantically equivalent to single-session execution.

## V4 cache and attention

V4 uses a per-layer compression schedule, including sliding attention, ratio-4 CSA, and ratio-128 HCA layers. CSA has overlapping candidate windows and separate key/value and score lanes; HCA uses a non-overlapping long window. Both require the learned APE contribution and their own normalization/RoPE behavior. The indexer has a separate compressor and supplies the selected positions used by attention. Cache state, row identity, sequence boundaries and chunk continuation must be explicit in the V4 session.

`compress.cu` implements projected CSA/HCA pooling, BF16/RoPE rounding, and the source's FP8 cache simulation. `indexer_compress.cu` implements the separate 128-wide ratio-4 path with BF16, Hadamard, and FP4 round trips. `indexer_project.cu` builds the BF16 query/head-weight projections and applies the indexer's RoPE, Hadamard, and FP4 path; `indexer_select.cu` scores visible compressed keys and selects the top 512 with per-token causal limits. All four sm_120 translation units and parity executables build; all four GPU parity tests pass on an RTX 5090 (compute capability 12.0), including double-reference output and chunk-independence checks. The CPU vision-plan and C++ pack/projector tests also pass against the actual converted artifact. The projected attention compressor is not yet fed from pack weights, and none of these paths is connected to cache allocation, MLA attention, or a V4 session.

Three repeated standalone sm_120 microbench runs measured HCA compression at 55.46–57.83 µs/call, indexer compression at 9.12–9.13 µs for T=16/chunk 8 and 5.16–5.18 µs for chunk 4, indexer projection at 14.65–14.98 µs, and index selection at 148.74–148.78 µs (T=8, 702 visible rows, top-512). These are kernel harness timings, not end-to-end model throughput.

## MoE routing

The source computes gate logits in FP32, applies `sqrt(softplus(logit))`, and scales normalized selected weights by 1.5. Layers 0–2 use `tid2eid` to choose the fixed top-6 experts for ordinary text tokens, but still derive their weights from those experts' current gate scores. Later layers choose top-6 from the gate scores plus the learned selection bias. Selection bias affects expert choice only; it is excluded from the routed weights. Image-token IDs (IDs at or above the text vocabulary size) use score-based top-6 with `bias_vl`, including in the hash-routed layers. The shared expert runs alongside these six routed experts.

This routing contract is distinct from DeepSeek-V4.1's generic top-k router. The V4 route-row helper marks ordinary text rows for hash lookup in layers 0–2 and image-token rows for alternate-bias selection in every layer. The shared mixed router gathers fixed IDs from the token-indexed table and normalizes the selected unbiased scores. This route path is not yet connected to a V4 session; validate token IDs and table bounds before dispatch.

## Vision path

The source uses 14-pixel patches, 3x3 downsampling, a 384-token image budget, 147,456 minimum pixels and an 8:1 landscape cap. Its N-layout interleaves adjacent rows, adds newline and pad token types, and inserts up to three leading pad tokens to align image positions to four. These rules differ from V4.1's ordinary merged-grid image blocks.

`include/basalt/models/deepseek4/vision.hpp` and `src/models/deepseek4/vision/plan.cpp` implement CPU resize planning, BF16-rounded patch extraction, N-layout type/permutation and assembly of encoded patch rows with the learned start/end/pad/newline rows. The shared CUDA transformer loads the embedded projector and returns projected grid rows; V4 assembles those rows in its own N-layout. The CPU planner test and real-pack GPU test pass. Against the official PyTorch Vision/Aligner using deterministic nonzero 3x3 patches, the shared CUDA tower's projected grid has 8.004e-3 relative RMS; the test also checks finite output and V4 special-row assembly. This is a small-input encoder parity check, not full-image preprocessing or request-path parity. Recreate the source dump with `python3 tools/vision/deepseek4_reference.py --source /mnt/storage/models/deepseek/source-vision --output /tmp/ds4-vision-reference.bin`, then configure CMake with `-DBASALT_DEEPSEEK4_VISION_REFERENCE=/tmp/ds4-vision-reference.bin` alongside `BASALT_DEEPSEEK4_PACK` and run `ctest -R '^ds4_vision_tower_test$' --verbose`. Spatial token embeddings, language position assignment, prompt integration and dynamic memory lifecycle remain unimplemented. The model is not runnable.

The V4 reference tower matches V4.1's encoder geometry: 32 layers, width 1024, 16 heads, FFN 2816, patch 14, 2D RoPE theta 10000, and a 3x3 GELU aligner to width 4096. The ViT and aligner math now live in the model-neutral `basalt::vision::TransformerEncoder`; it can return projected grid rows or the V4.1 START/NEWLINE/END block. DeepSeek-V4 applies its N-layout permutation and learned START/END/PAD/NEWLINE assembly separately.

## Verification gates

- Converter output: all source tensors accounted for; projector and tokenizer metadata retained; pack shape/type checks pass.
- Model registration: `deepseek4` builds independently under the model option and is discoverable by Basalt's adapter registry.
- Kernels: CPU/reference parity and chunk-boundary consistency for compression, indexer and attention; each new kernel has a microbench entry.
- Runtime: source-precision identity and KLD against a trusted reference, prompt-chunk consistency, decode, save/restore, multi-slot batching and supported peer mode.
- Vision: preprocessing/encoder parity, dynamic mode idle-memory check, image request endpoint test, and text-only behavior with vision enabled.
- Quality and performance: perplexity/KLD gates from the precision policy, representative decode/prefill benchmark, and reported run-to-run variance.
- Repository: model-layering checks, clean Release builds for this model and `all`, relevant test suite, updated model and precision docs (each under 300 lines).

Progress is incomplete until these gates are evidenced on the current Basalt runtime.
