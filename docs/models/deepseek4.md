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
layer schedule. Prefill uses tensor-core sparse MQA (`sparse_mqa_tc`), batched indexer scoring, grouped `wo_a` MMQ,
and 6-slot cross-layer expert staging. Decode uses per-layer CUDA graphs, multi-stream attention, compute-kernel
staging, and the 3-stage DSpark drafter. The expert tier admits an expert to the card on two CPU uses with a two-lookup
margin over its victim, halves the counts that pick victims every 256 tokens, budgets 128 copy claims per window row, and
lends its prompt arena in 256 MiB segments, so promoted experts stop thrashing a 256-slot layer. Measured on a single
RTX 5090 (32 GiB VRAM, 475 W power limit) + 247 GiB DDR5 RAM (`--resident-budget-gib 145`, 29/256 MXFP4 experts/layer in
VRAM, drafting at its default):

| Benchmark | Rate (tok/s) | Notes |
| --- | ---: | --- |
| `pp16384` | 2,330 | Engine (`2,309` wall) |
| `tg conv / struct / prose` (lossless) | 34-36 / 38-43 / 35-36 | 59.2% VRAM, 12.1% PCIe, 28.7% CPU |
| `tg conv / struct / prose` (`--route-prior 0.5 --route-tail-skip 4`) | 34 / 43 / 35 | Opt-in lossy routing |

100-chunk `--kv fp8` Wikitext-2 KLD gate against `dsv4visionexp-source-base.kld`: PPL `3.9836 +/- 0.0596` (base
`4.3379`), mean KLD `0.15429`, top-1 `87.92%`. Vision (`ds4_vision_tower_test`, `ds4_vision_reserve_test`) and
DSpark attach (`tools/convert/test_dspark_deepseek4_attach.py`) pass.

Conversation checkpoints now snapshot the sliding-window rings and rolling attention/indexer compressor state; compressed
attention and CSA index rows use the shared positional-cell save/restore path. A two-chunk Wikitext gate exercised save,
restore, and restarting at position zero. `BatchSession` is registered with the shared batch driver, but V4's host-steered
expert routing currently selects its sequential fallback; this verifies API identity semantics, not parallel batch speed.
The single available RTX 5090 leaves multi-GPU peer identity untested.

## Decode pass breakdown

Where a decode pass's 26.3 ms/token goes, on the tree and configuration of the table above (145 GiB RAM budget,
1,000-expert cache, drafting at its default, 475 W). It is an Nsight Systems 2026.3.2 trace of 96 generated tokens in
40 windows of 2.42 rows (`nsys profile --trace=cuda`, no CPU sampling; the engine emits no NVTX ranges). Each row is a
group of kernel names over the decode window's activities, the copies are its Host-to-Device activities of 12 MB and up,
and the card's busy time is the union of the two. The host's `lookup`, claim and publish poll have no timer of their own,
so they are inside the idle row rather than split out.

| stage | ms/token | of the wall | what it is |
| --- | ---: | ---: | --- |
| expert blobs over PCIe | 14.86 | 57% | 499 MB a token: 3,584 copies of 12.75 MB at 33.6 GB/s |
| CPU-share wait | 4.15 | 16% | `wait_word` spins while the pool computes its experts (96 us a layer) |
| routed expert matmuls | 2.71 | 10% | gate/up, down and swiglu of the resident and copy-resident experts |
| other card work | 0.75 | 3% | attention, indexer, mHC, router, staged copies, publish, weighted sum, LM head, sampler |
| idle between launches | 5.47 | 21% | 14,737 gaps, p90 184 us: the host's poll, lookup, claim and submission |
| copies that overlap kernels | -1.71 | -7% | inside the kernels' intervals |
| **total** | **26.23** | **100%** | the engine reported 2,527 ms for those 96 tokens, 26.32 ms/token |

The kernels alone are 7.61 ms/token and the copies 14.86, and with the idle gaps they reconcile the pass to within 0.4%
of what the engine measured. Nearly three quarters of it is expert traffic and the waits it forces, so a pass waits on
the placement the tier chose, not on the token path that joins it. The profiled run's own 38.0 tok/s is not a throughput
figure: different fixture, and a trace is not the benchmark.

## KV caches and `--kv`

The session keeps three compressed cache families: sliding-window rows, the compressors' output rows, and the CSA
indexer's keys. `--kv` chooses their row layouts at start (it is a model flag, validated by the `deepseek4` adapter):

| `--kv` | window row | compressed row | index key | 65,536-token reserve | 1,048,576-token reserve |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp8` (default) | 528 B | 288 B | 68 B | 0.13 GiB | 1.88 GiB |
| `fp16` | 1,024 B | 1,024 B | 256 B | 0.44 GiB | 6.74 GiB |

The packed layouts are the reference's own roundings (`act_quant` for window rows and `fp4_act_quant` for compressed
rows and index keys, as imported from Project Maya/Strata): E4M3 codes with a power-of-two (UE8M0) scale per 32,
E2M1 codes with an E4M3 scale per 16, and E2M1 codes with a UE8M0 scale per 32. The indexer rounds its queries the same
way before scoring. Sparse MQA decodes the packed rows exactly: score keys use each group's scale, PV values multiply
their row's per-group scale, and masked window or selected rows decode as zeros. `ds4_cache_rows_parity` checks the
writers and readers against a CPU packing byte for byte, and `ds4_attention_step_parity` runs the window path with
Fp8 rows against the F32 ring (max abs 0.10 on its random fixture).

Saved conversation cells carry the layout in their format code (500+ratio for packed rows, 600+ratio for fp16) and the
row geometry, so a snapshot taken with one `--kv` is refused by the other. Cache sizes are planned by
`src/models/deepseek4/context_policy.hpp` from those row bytes; the table's reserves are the printed cache reserve at
`--max-context 65536` and `--max-context 1048576` on the `-hchead` pack (the 1M fp8 run used
`--resident-budget-gib 120`). Decode is 21-22 tok/s in both layouts at 64k and 1M contexts on that SSD-bound tier (within the run-to-run spread),
so this change buys memory, not speed.

## Drafting

When the pack carries the DSpark drafter, decode drafts by default in windows of up to 6 rows (the last accepted token
plus five drafts; `--spec 0` turns it off). The drafter's three stages keep their 32 most routed MXFP4 experts on the card
(`--draft-experts`, 32 to 256 in steps of 32, or 0 for all; ranked by `data/profiles/deepseek4-43x256-dspark.prof`), and
their block and Markov head run as two CUDA graphs captured once per row count. A verify window's target features become
the stages' latents beside the window on a low-priority stream; only the accepted rows are committed into its rings.
The expert placement itself starts from `data/profiles/deepseek4-43x256.prof` (measured routing profile). Greedy output
is unchanged by drafting: the same 12 tokens came from `--spec 0` and `--spec 6` on a tokenizer-derived prompt, and
the verifier's picks matched the plain draws.

What it buys depends on where the experts sit. At a frozen placement (36% VRAM, 64% CPU, no bus copies) drafting is
near break-even: `--spec 0` against `--spec 6` ran A1/B1/B2/A2 in one session at 475 W, 145 GiB RAM budget and a
1,000-expert cache, one repetition an arm. Paired means were conv 23.70 against 24.30, struct 19.95 against 25.35 and
prose 26.25 against 25.20 tok/s (off against on); token-weighted over the three fixtures those are 24.73 against 24.71
tok/s, -0.1%. A window multiplies expert traffic and the frozen tier answers every miss from the CPU pool, so the rows it
accepts pay for the rows it adds: conv gains 2.5% and struct 27%, prose loses 4%.

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
