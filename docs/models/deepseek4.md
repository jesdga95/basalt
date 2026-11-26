# DeepSeek-V4-Flash-Vision-Exp

## Architecture decision

Implement V4 as its own registered Basalt model (`deepseek4`) with a dedicated pack reader and session. Reuse the proven DeepSeek-V4.1 infrastructure only where the source geometry and semantics match: adapter lifecycle, expert tier, hyper-connection (mHC) execution, checkpoint transfer, peer hooks, batching scaffolding, and the vision service integration points. Keep V4 cache construction and attention separate from V4.1; their compression schedules and cache semantics differ.

The target checkpoint is `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, converted from its source GGUF/safetensors into a Basalt pack. The current pack has the 43 trunk layers, token-to-expert tensors for the first three layers, and an embedded projector. Its manifest has no MTP tensors or drafter section, so an MTP drafter is not available in this artifact. Preserving the source tensors and projector does not constitute runtime support.

## Model path

1. Validate and describe `general.architecture=deepseek4` metadata, including geometry, compression schedule, MoE routing, tokenizer, context and multimodal fields.
2. Parse the Basalt pack into a V4-specific `Pack`/`Geometry`; validate tensor presence, shapes, storage types and per-layer schedule before allocating runtime state.
3. Load embeddings, attention projections, mHC matrices, indexer, router/hash tensors, shared expert and routed experts. Reuse DeepSeek-V4.1 expert-tier placement and file/RAM/VRAM read mechanisms where tensor formats match; reject unsupported storage explicitly.
4. Implement text prefill and decode with V4's attention and cache contracts. Integrate V4-specific compression states and attention indexing. Share only shape-compatible projection and mHC primitives with V4.1.
5. Add the model's tokenizer/chat dialect and serving registration. Keep checkpoints, batch execution and peer execution semantically equivalent to single-session execution.

## V4 cache and attention

V4 uses a per-layer compression schedule, including sliding attention, ratio-4 CSA, and ratio-128 HCA layers. CSA has overlapping candidate windows and separate key/value and score lanes; HCA uses a non-overlapping long window. Both require the learned APE contribution and their own normalization/RoPE behavior. The indexer has a separate compressor and supplies the selected positions used by attention. Cache state, row identity, sequence boundaries and chunk continuation must be explicit in the V4 session.

`compress.cu` implements projected CSA/HCA pooling, BF16/RoPE rounding, and the source's FP8 cache simulation. `indexer_compress.cu` implements the separate 128-wide ratio-4 path with BF16, Hadamard, and FP4 round trips. `indexer_select.cu` scores visible compressed keys and selects the top 512 with per-token causal limits. Their parity/benchmark harnesses compile and link, but have not run on the GPU. None of these kernels is connected to pack loading, attention projections, cache allocation, or a V4 session, so the model is not runnable yet.

## MoE routing

The source computes gate logits in FP32, applies `sqrt(softplus(logit))`, and scales normalized selected weights by 1.5. Layers 0–2 use `tid2eid` to choose the fixed top-6 experts for ordinary text tokens, but still derive their weights from those experts' current gate scores. Later layers choose top-6 from the gate scores plus the learned selection bias. Selection bias affects expert choice only; it is excluded from the routed weights. Image-token IDs (IDs at or above the text vocabulary size) use score-based top-6 with `bias_vl`, including in the hash-routed layers. The shared expert runs alongside these six routed experts.

This routing contract is distinct from DeepSeek-V4.1's generic top-k router. The V4 route-row helper marks ordinary text rows for hash lookup in layers 0–2 and image-token rows for alternate-bias selection in every layer. The shared mixed router gathers fixed IDs from the token-indexed table and normalizes the selected unbiased scores. This route path is not yet connected to a V4 session; validate token IDs and table bounds before dispatch.

## Vision path

Reuse the existing Basalt image request, image decoding, projector loading and dynamic vision lifecycle where tensor layout and preprocessing match. Add a V4 projector/vision adapter for its actual source metadata, patch geometry, positional handling and image-to-token output. Dynamic mode must return temporary weights/workspaces after each image request and leave no vision-specific VRAM or RAM resident at idle. Verify image inputs through the model's serving endpoint as well as direct encoder parity.

## Verification gates

- Converter output: all source tensors accounted for; projector and tokenizer metadata retained; pack shape/type checks pass.
- Model registration: `deepseek4` builds independently under the model option and is discoverable by Basalt's adapter registry.
- Kernels: CPU/reference parity and chunk-boundary consistency for compression, indexer and attention; each new kernel has a microbench entry.
- Runtime: source-precision identity and KLD against a trusted reference, prompt-chunk consistency, decode, save/restore, multi-slot batching and supported peer mode.
- Vision: preprocessing/encoder parity, dynamic mode idle-memory check, image request endpoint test, and text-only behavior with vision enabled.
- Quality and performance: perplexity/KLD gates from the precision policy, representative decode/prefill benchmark, and reported run-to-run variance.
- Repository: model-layering checks, clean Release builds for this model and `all`, relevant test suite, updated model and precision docs (each under 300 lines).

Progress is incomplete until these gates are evidenced on the current Basalt runtime.
