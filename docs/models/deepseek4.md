# DeepSeek V4 Vision in Basalt

## Target and current state

The target pack is the DeepSeek-V4-Flash-Vision-Exp checkpoint derived from V4-Flash-0731. It has text architecture
`deepseek4` and a `deepseek4v` vision projector GGUF. Its text reference is checkpoint-specific: the converted source
uses MXFP4 routed experts and Q8_0 for source FP8 linears, so it is not a BF16 0731 reference.

The converter, pack validation, serving dialect, and model-local CUDA primitives exist. The pack reader can fetch validated
BF16 token-embedding rows from the GGUF section on demand. The engine does not yet register
`deepseek4` as a loadable model: there is no `ModelAdapter`, `LoadedModel`, `ModelSession`, or full transformer forward.
The final BF16 RMSNorm and logits projection stage is implemented and verified against the real output head, but is not yet
connected to a session.
The CPU image planner and patch formatter exist, but the vision transformer/projector execution is not integrated into a
Basalt session. The attention stage now accepts decode windows of up to eight rows and matches sequential one-row output and
cache state bitwise. Do not describe this pack as runnable until text decode, image prefill, and serving tests pass.

## Architecture and ownership

- `tools/convert/arch/deepseek4.py` maps GGUF metadata and tensors into the Basalt pack; conversion also takes the matching
  Hugging Face checkpoint through `--hf-dir` to preserve the final `hc_head_*` weights, which the source GGUF omits. The
  vision sidecar stores the projector section separately and validates its projector type; the engine pack reader checks
  section bounds.
- `src/models/deepseek4/pack.*` owns V4 geometry and tensor validation. The future adapter and model/session stay under
  `src/models/deepseek4/` and `include/basalt/models/deepseek4/`; shared code must not name this architecture.
- Text execution owns V4-specific hyper-connections, hash routing, indexer/compression scheduling, attention and cache
  layout. Reuse shared quantized matrix, router, expert-tier, load-plan, batch, elastic-cache, and serving APIs where their
  contracts match. Do not copy V4.1's forward loop: V4 has distinct hash routing and compression semantics.
- `src/models/deepseek4/vision/` owns V4 image-grid planning and patch ordering. Shared `src/vision/` owns image decode
  and reusable preprocessing primitives. Model-specific CUDA tower/projector operations belong under
  `src/models/deepseek4/` unless another model can use the exact same tensor contract.
- `src/models/deepseek4/tokenizer/pretok_deepseek4.cpp` implements the checkpoint tokenizer's numeric, CJK, and text
  boundaries in a model-local target; `ds4_tokenizer_test` covers those boundaries. The adapter still needs to connect it to
  the pack tokenizer metadata.
- `serve/models/deepseek4.py` owns prompt rendering, image placeholders, stop/reasoning/tool parsing. Shared `serve/`
  code remains architecture-agnostic.

## Memory and execution plan

Text weights are read from the pack into the GPU or RAM/file-backed expert tiers according to `core::LoadPlan` and
`core::ExpertTier`. Token embeddings remain file-backed where the pack loader supports row reads. Per-session state owns
KV/sliding-window rows, compressed attention rows, index keys, mHC state, prefill/decode scratch, and position state;
`core::ElasticArray` should grow those caches on demand and participate in cell save/restore. Model-shared tier and batch
state must not be duplicated per session.

Implement one correct token path first, then chunked prefill through the same layer semantics. Each layer must follow the
checkpoint's ordering: hyper-connection expansion, indexer/compressor work where scheduled, latent attention and output
projection, feed-forward hyper-connections, hash/expert routing, expert and shared-expert accumulation, then collapse.
Reuse existing kernels; add a device stage only when no shared kernel implements the exact contract. Keep CUDA wrappers
thin and translation units independently compilable.

Image input is resized and tiled by the V4 planner, normalized to the checkpoint's RGB range with BF16 boundary rounding,
then encoded by the vision tower and projector. Preserve the planner's patch permutation and image token layout when
building prompt positions. Add the encoder's CPU reference before CUDA parity. Image weights and patch buffers must be
staged only while encoding; `--vision dynamic` must return encoder memory to the expert tier after each image batch.

## Milestones and gates

1. Finish converter identity/metadata checks and document pack tensor coverage.
2. Register the model behind `BASALT_MODEL_DEEPSEEK4`; implement adapter, strict pack loader, expert-tier planning, and
   session lifecycle.
3. Implement decode and chunked prefill; compare token, chunk, and window execution against CPU-double references and
   the converted checkpoint. Add greedy/batch identity and the Wikitext KLD gate.
4. Implement vision CPU reference, CUDA tower/projector, dynamic memory give-back, and image prompt integration. Verify
   CUDA/CPU parity, low-VRAM behavior, and OpenAI/Anthropic image requests.
5. Add save/restore, sleep/wake, context refusal, peer hooks, and batch behavior. Add benchmark fixtures and report cold
   decode, prefill, and power-limit data.

The definition of done is the complete verification list in the implementation objective: both clean model-only and
`all` builds, repository static checks, converter and serving suites, CPU/GPU parity, identity gates, KLD against the
matching converted checkpoint, vision end-to-end tests, and benchmarks. Until those pass, this document describes a
work-in-progress implementation rather than a supported model.
