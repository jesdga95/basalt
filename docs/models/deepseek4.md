# DeepSeek V4 Vision in Basalt

## Target and current state

The target pack is the DeepSeek-V4-Flash-Vision-Exp checkpoint derived from V4-Flash-0731. It has text architecture
`deepseek4` and a `deepseek4v` vision projector GGUF. Its text reference is checkpoint-specific: the converted source
uses MXFP4 routed experts and Q8_0 for source FP8 linears, so it is not a BF16 0731 reference.

The converter, pack validation, serving dialect, model-local CUDA primitives, and C++ metadata adapter exist. The adapter
selects the pack's `joyai-llm` tokenizer, V4 token names, image pad, DSML grammar, and one-GPU mode. The regular
`DeepSeek-V4-Flash-Vision-Exp.basalt` fails pack validation because it lacks `hc_head_fn.weight`; the separately generated
`-hchead.basalt` includes the three checkpoint-sourced head tensors and passes Basalt's pack test and source-GGUF byte
verification. Keep the original pack unchanged. `ModelWeights` queues dense tensors in the start's `LoadPlan`, widens the
BF16 router gate to F32 and leaves token embeddings and hash routes file-backed. Its layer views bind validated attention,
router, shared-expert, and output-head tensors.
MXFP4 grouped experts now accept gate/up as well as down tensors, with Basalt's grouped parity suite covering synthetic
and double-reference V4-shaped cases. The pack reader can fetch validated BF16 token-embedding rows from the GGUF section
on demand. The indexer query projection now consumes the converted Q8_0 matrix. Its V4-shaped Q8_0/Q8_1
matvec matches the CPU double reference with max absolute error 2.24e-8 and relative RMS 1.13e-7; batched
rows match single-row calls bitwise. On the RTX 5090 at a 475 W power limit, the V4-shaped synthetic two-row
projection takes 17.52 us and moves 9.48 MB (541.1 GB/s). A `LoadedModel` and single-session text forward are wired into the
adapter. On the RTX 5090 at 475 W, Basalt loaded the `-hchead` pack with `--resident-budget-gib 32` and generated token 67
from token ID 1. The default all-RAM tier exhausted VRAM during its post-pin allocation check; the 32 GiB resident budget
left 160 experts per layer file-backed. This is a bring-up smoke only, not a quality or performance result.

The session now maintains per-layer sliding, compressed-attention, and indexer caches and runs sparse MQA in the V4
layer schedule. A one-token decode from token ID 1 succeeds on the derived `-hchead` pack. A 130-ID prompt also passes the
former 128-row boundary: Basalt prefetched 129 prompt tokens in 15.4 seconds and generated one token. This is a smoke test,
not a quality or long-context result; chunk/token identity, images, and resource release after an image request remain
unverified. Image preprocessing previously produced 197 rows for the sample logo, but that run happened before compressed
cache prefill was connected and does not establish image understanding. Keep the model marked work-in-progress until the
vision, serving, precision, lifecycle, and performance gates below pass.

Conversation checkpoints now snapshot the sliding-window rings and rolling attention/indexer compressor state; compressed
attention and CSA index rows use the shared positional-cell save/restore path. A two-chunk Wikitext gate exercised save,
restore, and restarting at position zero. `BatchSession` is registered with the shared batch driver, but V4's host-steered
expert routing currently selects its sequential fallback; this verifies API identity semantics, not parallel batch speed.
The single available RTX 5090 leaves multi-GPU peer identity untested.

## Drafting

When the pack carries the DSpark drafter, decode drafts by default in windows of up to 6 rows (the last accepted token
plus five drafts; `--spec 0` turns it off). The drafter's three stages keep their 32 most routed MXFP4 experts on the card
(`--draft-experts`, 32 to 256 in steps of 32, or 0 for all; ranked by `data/profiles/deepseek4-43x256-dspark.prof`), and
their block and Markov head run as two CUDA graphs captured once per row count. A verify window's target features become
the stages' latents beside the window on a low-priority stream; only the accepted rows are committed into its rings.
The expert placement itself starts from `data/profiles/deepseek4-43x256.prof`. Both shipped files rank each expert by the
L2 norm of its router gate row — a placement prior built from the checkpoint, not measured routing heat; a run that
measures heat can replace them. Greedy output is unchanged by drafting: the same 12 tokens came from `--spec 0` and
`--spec 6` on a tokenizer-derived prompt, and the verifier's picks matched the plain draws.

## Architecture and ownership

- `tools/convert/arch/deepseek4.py` maps GGUF metadata and tensors into the Basalt pack; conversion also takes the matching
  Hugging Face checkpoint through `--hf-dir` to preserve the final `hc_head_*` weights, which the source GGUF omits. The
  vision sidecar stores the projector section separately and validates its projector type; the engine pack reader checks
  section bounds.
- `src/models/deepseek4/adapter.*` owns model metadata, tokenizer, control-token, and DSML grammar choices. Pack geometry
  stays in `pack.*`; the model/session implementation also belongs under `src/models/deepseek4/` and
  `include/basalt/models/deepseek4/`, while shared code must not name this architecture.
- Text execution owns V4-specific hyper-connections, hash routing, indexer/compression scheduling, attention and cache
  layout. Reuse shared quantized matrix, router, expert-tier, load-plan, batch, elastic-cache, and serving APIs where their
  contracts match. Do not copy V4.1's forward loop: V4 has distinct hash routing and compression semantics.
- `src/models/deepseek4/vision/` owns V4 image-grid planning and patch ordering. Shared `src/vision/` owns image decode
  and reusable preprocessing primitives. Model-specific CUDA tower/projector operations belong under
  `src/models/deepseek4/` unless another model can use the exact same tensor contract.
- `src/program/tokenizer/pretok_joyai.cpp` implements the pack's `tokenizer.ggml.pre=joyai-llm` boundaries. The
  DeepSeek4 wrapper and the existing DeepSeek4.1 wrapper share this tokenizer; `ds4_tokenizer_test` covers its numeric,
  CJK, and text splits. The adapter exposes the JoyAI pretokenizer and `tokenizer.ggml.tokens` metadata to the shared loader.
- `serve/models/deepseek4.py` owns prompt rendering, image placeholders, stop/reasoning/tool parsing. Shared `serve/`
  code remains architecture-agnostic.

## Memory and execution plan

Text weights are read from the pack into the GPU or RAM/file-backed expert tiers according to `core::LoadPlan` and
`core::ExpertTier`. Token embeddings remain file-backed where the pack loader supports row reads. Per-session state owns
KV/sliding-window rows, compressed attention rows, index keys, mHC state, prefill/decode scratch, and position state;
`core::ElasticArray` should grow those caches on demand and participate in cell save/restore. Model-shared tier and batch
state must not be duplicated per session.

The per-layer RoPE builder follows the source model: ratio-zero layers use the plain `rope_base`, while ratio-4 and
ratio-128 layers use `comp_rope_base` with the checkpoint's YaRN frequency interpolation. The source applies unit-magnitude
complex phases, so the table has no additional YaRN magnitude correction. `ds4_rope_test` checks both schedules against the
source formula.

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
