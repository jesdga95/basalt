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
RTX 5090 (32 GiB VRAM, 475 W power limit) + 247 GiB DDR5 RAM (`--resident-budget-gib 145`, drafting at its default, no
`--expert-cache` so the tier fills the card with 27 of 256 MXFP4 experts a layer), two arms a row:

| Benchmark | Rate (tok/s) | Notes |
| --- | ---: | --- |
| `pp16384` | 2,290-2,300 | Engine (`2,271` / `2,281` wall) |
| `tg conv / struct / prose` (lossless) | 36-38 / 42-44 / 38 | 70% VRAM, 11% PCIe, 19% CPU |
| `tg conv / struct / prose` (`--expert-cache 1000`) | 35-36 / 41-42 / 36 | 23/256 a layer, 2.9 GiB of VRAM left idle: 67% VRAM, 12% PCIe, 21% CPU |
| `tg conv / struct / prose` (`--route-tail-skip 3`, the lossy preset) | 55-60 / 70-76 / 57 | Opt-in: 41% fewer routed experts, mean KLD `0.29460` against `0.15465`, top-1 `82.11%` against `88.04%` |

The 100-chunk `--kv fp8` Wikitext-2 KLD gate against `dsv4visionexp-source-base.kld` ran on this tree at the table's
default (`--resident-budget-gib 145`, no `--expert-cache`): 25,500 rows, PPL `3.9807 +/- 0.0595` (base `4.3379`,
`ln(PPL/base) -0.08595`), mean KLD `0.15465`, median `0.03152`, p99 `2.0187`, max `7.8189`, top-1 `88.04%`, 61.0% VRAM /
10.3% PCIe / 28.7% CPU lookups. The pinned 1,000-expert cache scored PPL `3.9965 +/- 0.0599`, mean KLD `0.15568`, top-1
`87.92%` (57.4% / 11.7% / 31.0%), and the earlier gate (`3.9836` / `0.15429` / `87.92%`) held 34 experts a layer: the three
sit within the placement spread this gate gives, which is why the per-flag comparisons below are the ones to read.
Vision (`ds4_vision_tower_test`, `ds4_vision_reserve_test`) and DSpark attach
(`tools/convert/test_dspark_deepseek4_attach.py`) pass.

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

Those copies already run at the link's own rate: `basalt-bench --hardware` measures this machine's pinned host-to-device
copy at 33.0 GB/s and the pass moves its blobs at 33.6 GB/s, so no scheduling change makes the same bytes arrive sooner.
Only fewer miss bytes shorten the pass, and the card's shares are the lever: a pinned 1,000-expert cache is 23 of 256
experts a layer and leaves 2.9 GiB of VRAM free, while the tier's own sizing (no `--expert-cache`) fills it to 27-28, the
free memory at start deciding which. That is the whole of it: 23 to 27 slots a layer measured conv `37.6` and `35.6` against `35.7` and `35.4` tok/s, struct `43.7`
and `41.9` against `41.7` and `40.7`, prose `38.2` and `38.1` against `35.9` and `35.8`, and pp16384 `2,290` and `2,300`
against `2,319` and `2,312`: +6% prose and +4% struct, inside the conv spread, 1% off on the prompt, which uses the room
the extra copies take. The default was already the fast path; the pinned size was the slower one. Fifty tokens a second
would need a 20 ms pass, and its 499 MB of miss traffic alone is 15 ms at this link, so that target needs fewer routed
experts a token (the opt-in lossy preset below) or a faster link, not another placement.

## Lossy routing

Two opt-in flags (`basalt/model/route_flags.hpp`; off by default, each one given named in the start's warning and the run's
LOSSY mark). `--route-tail-skip R` neither copies nor computes an expert no GPU holds when every entry of the decode
window that names it ranks R or lower (0-based of this model's six routes): its entries get weight 0, the others keep
theirs. `--route-prior LAMBDA` picks the window's six as if every expert a GPU holds scored LAMBDA higher; the weights stay
the router's own. Both pay only once the card holds the experts it keeps, which is why they were worth measuring again
after the admission policy and the card-filling cache: at 28 of 256 experts a layer they cut the experts routed to a token,
where before they mostly moved work to the CPU.

Measured this pass at the card's own cache (`--resident-budget-gib 145`, no `--expert-cache`), two arms a fixture and the
100-chunk Wikitext-2 gate on every arm that could ship, RTX 5090 at 475 W. `conv` and `prose` decide; `struct` is advisory because it swings
a few tok/s between arms on this machine:

| arm | conv tok/s | prose tok/s | struct tok/s | routed/token (conv) | PPL | mean KLD | top-1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| flags off | 37.6 / 35.6 | 38.2 / 38.1 | 43.7 / 41.9 | 298,002 | 3.9807 | 0.15465 | 88.04% |
| `--route-prior 0.5` | 37.8 / 36.9 | 38.7 / 37.7 | 44.7 / 42.2 | 298,310 | 3.9807 | 0.15465 | 88.04% |
| `--route-tail-skip 5` | 42.4 / 41.5 | 42.7 / 44.3 | 49.0 / 47.1 | 258,504 | - | - | - |
| `--route-tail-skip 4` | 46.6 / 48.2 | 49.4 / 50.1 | 57.6 / 58.2 | 217,299 | 4.1404 | 0.20600 | 85.58% |
| **`--route-tail-skip 3`** (preset) | **54.9 / 59.7** | **57.2 / 57.1** | 69.7 / 75.8 | 175,408 | 4.6074 | 0.29460 | 82.11% |
| prior 0.5 + tail-skip 4 | 49.5 / 46.8 | 50.0 / 48.8 | 58.0 / 56.3 | 217,605 | 4.1431 | 0.20570 | 85.50% |

**The prior carries nothing.** At 0.5 it does not change the pick at all: its gate reproduced all 25,500 flags-off rows
bitwise (max |dNLL| and |dKLD| 0.000000, the same 6,579,000 routed experts), its speed is the flags-off speed, and adding
it to `--route-tail-skip 4` measured the same routed count (4,539,000), the same KLD to within 0.0003 and the same speed.
With 28 of 256 experts held, the score gap inside the router's top six is wider than the 0.5 the flag adds, so it never
binds. It is not part of the preset, and yesterday's pair result was `--route-tail-skip 4` with a flag that did nothing.

**The tail-skip carries the speed and the cost, and its depth is the knob:** R=5 (the mildest, which drops only a
sixth-ranked miss) measures 42 tok/s, R=4 47, R=3 55. Only R=3 holds both deciding fixtures at or above 50 tok/s in both
arms, so `--route-tail-skip 3` is the preset. Its price is the steepest row on the table: mean KLD `0.29460` against
`0.15465`, top-1 `82.11%` against `88.04%`, p99 `3.0290`, max `10.5874`, for 41% fewer experts routed a token and a third
shorter decode half in the gate's own timing (5,144 against 7,700 ms a chunk).

**R=4 is the cheaper step and misses the target.** It costs half the divergence (KLD `0.20600`, top-1 `85.58%`, p99
`2.3935`, 5,898 ms a chunk) but measures conv `46.6`/`48.2` - 2 to 3 tok/s short in both arms - and prose `49.4`/`50.1`,
so it cannot be called 50 tok/s. R=5 is milder still and 5 tok/s further off. The pair above is R=4's routing (the same
4,539,000 routed experts) and is not kept. With no flag nothing changed: this pass altered no code, and the flags-off rows
are the lossless numbers this page already carried.

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

## Open items

- **Per-layer slot weights (`core::ExpertTierSpec::layer_weights`) were derived from the measured profile and rejected on
  measurement.** `data/profiles/deepseek4-43x256.prof` ranks all 11,008 (layer, expert) pairs by measured routing
  frequency, so its first 1,204 pairs are the most-routed experts the card can hold at 28 slots a layer; those pairs give
  layers 19 (layer 0) to 36 (layer 42) slots instead of 28 each. At `--expert-cache 1000` that is 16-30 slots: the ABBA
  measured conv `34.3` and `34.3` against `35.7` and `35.4` tok/s, struct `39.6` and `41.8` against `41.7` and `40.7`,
  prose `36.6` and `35.6` against `35.9` and `35.8`, and pp16384 `2,305` and `2,306` against `2,319` and `2,312`. Both conv
  arms sit below both uniform arms, and struct's PCIe share rose from 16.4% to 19.5%. A corpus-wide routing ranking is not
  a decode miss trace: the admission policy finds its own equilibrium on the traffic a request actually sends, and a layer
  short of slots pays it back in copies. The table was removed rather than retuned.
- **What is left for decode speed.** The pass's expert copies are at the measured link rate (33.6 against 33.0 GB/s), so
  the remaining lever is residency, and the tier already takes it: four more slots a layer (23 to 27) buys 3-6% (the
  table), and past that the card is full at 27-28 of 256 experts a layer. Two more levers are measured and closed:
  per-layer slot weights (above) and the bus/CPU split of the misses (`bus_share` at 75% measured neutral). Past that,
  fewer routed experts a token is the only lever left: the opt-in preset `--route-tail-skip 3` reaches 55-60 conv and 57
  prose for the quality priced above, and R=4 is the cheaper step that falls short of 50. Nothing here is a default.

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
