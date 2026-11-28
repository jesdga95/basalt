# DeepSeek-V4-Flash-Vision-Exp precision

Vision-Exp is an experimental multimodal release further trained from V4-Flash-0731. Its quality
reference is specific to this checkpoint, not a BF16 reference for 0731.

## Source conversion

Checkpoint: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, snapshot
`6821d6ad3681a4b137b066b76094fa82ebd0a380`. GGUF conversion used llama.cpp `abca69200` and
`convert_hf_to_gguf.py --outtype bf16`.

The output is mixed precision, not an all-BF16 model. The converter repacks routed experts as MXFP4
and maps the checkpoint's FP8 linear weights to Q8_0. Other tensors follow llama.cpp's DeepSeek-V4
conversion rules.

| Artifact | Size | Contents |
| --- | ---: | --- |
| `DeepSeek-V4-Flash-Vision-Exp-source-precision.gguf` | 156.4 GB (1,371 tensors) | Text model |
| `mmproj-DeepSeek-V4-Flash-Vision-Exp.gguf` | 934.4 MB (299 tensors) | Vision encoder and projector |
| `DeepSeek-V4-Flash-Vision-Exp.basalt` | 149.297 GiB | Basalt pack; all 1,371 source tensors verified byte-for-byte |

The packer preserves the source tensor encodings and adds the `deepseek4v` vision projector. This
container conversion does not add DeepSeek-V4 runtime support to Basalt.

## Reference base

The text reference is llama.cpp's source-converted Vision-Exp output, scored on Wikitext-2. The
`.kld` file stores reference logits; it is not a model-versus-reference KLD score. Image understanding
is not covered by this text-only gate.

| Base | Size | Chunks | Base PPL | Run |
| --- | ---: | ---: | ---: | --- |
| `dsv4visionexp-source-base.kld` | 6.2 GiB | 100 x 512 | 4.3495 +/- 0.06634 | RTX 5090, 475 W; 16 CPU threads |

This base is not BF16: routed experts are MXFP4 and FP8 linears are Q8_0 in the GGUF. Use it to
measure Basalt against this same converted Vision-Exp reference. Its KLD is not comparable with a
true-BF16 reference or a 0731 base.

## Basalt comparison

The original 100-chunk result below is superseded: DeepSeek4 `copy_logits(row)` returned the final logits
row for every row of a multi-token window, which corrupted its PPL comparison. The session now retains each
row when scoring asks for it. A second issue was the synchronous default-stream token upload racing work on
the session's nonblocking stream; `forward_one` now queues the input copy on that same stream. The routed-expert
path synchronizes before the pinned input row is reused.

The corrected 100-chunk Wikitext-2 gate used the derived `-hchead.basalt` pack, `--resident-budget-gib 145`,
`--expert-cache 1480`, `--diag-tier-frozen`, and an RTX 5090 at 475 W. It scored 25,500 rows: Basalt PPL
3.9694 +/- 0.0591, matched reference PPL 4.3379, mean KLD 0.15216, median KLD 0.03090, p99 KLD 1.9484,
max KLD 7.4493, and top-1 agreement 88.04%. The reference table above reports 4.3495 +/- 0.06634 from
base generation; 4.3379 is the matched-sample value for this comparison. The gate loaded the full 137.1 GiB
RAM expert tier, with 34/256 experts per layer in an 18.20 GiB GPU cache.

The original extremes are cleared: chunk 55 position 279 now has KLD 0.0014176 with matching top-1, and
chunk 82 position 285 has KLD 0.1256984 with matching top-1. Chunk 65 position 279 remains a reproducible
KLD 3.999452; chunk 98 position 433 remains 7.449312. Other large tails include chunk 75 position 256
(6.8149), chunk 5 position 371 (6.231941), and chunk 99 position 257 (6.239594).

A targeted `t=1` replay compared all 255 rows in each of 14 selected chunks (5, 7, 12, 16, 47, 55, 65,
68, 75, 82, 88, 91, 98, 99) with the full `t=3` run. All 3,570 rows were bitwise identical, including the
remaining high-KLD positions. This rules out window size and run-to-run variation as causes of those tails;
they are reproducible Basalt/reference divergences and form the corrected parity baseline.

A later 100-chunk gate on the tree that carries the admission policy and 6-slot prompt staging scored, with the card's own
default cache size (`--resident-budget-gib 145`), PPL `3.9807 +/- 0.0595`, mean KLD `0.15465`, p99 `2.0187`, max `7.8189`
and top-1 `88.04%` (61.0% VRAM / 10.3% PCIe / 28.7% CPU), and at a pinned `--expert-cache 1000` PPL `3.9965 +/- 0.0599`,
mean KLD `0.15568`, top-1 `87.92%` (57.4% / 11.7% / 31.0%). The opt-in lossy flags on the same tree and cache scored PPL
`4.1431 +/- 0.0626`, mean KLD `0.20570` and top-1 `85.50%`; the model page carries the speeds beside them.

The run loaded the full 137.1 GiB RAM expert tier (256/256 experts per layer), with zero experts in
the file tier; the GPU held 34/256 experts per layer in an 18.20 GiB cache. The final 8-chunk continuation
reported 63.6% VRAM lookups, 14.1% bus copies, 22.3% CPU lookups, and 0.0% file reads. Prompt reads
averaged about 13 tok/s and decode about 13.5 tok/s. A one-chunk Nsight Systems trace recorded 375,707
host-to-device copies (505,091 MB), 11.1 s cumulative `cudaStreamSynchronize`, and 13.6 s cumulative
`cudaMemcpyAsync` API time; these totals include startup and overlap across threads.

An `--expert-cache auto` trial selected 39 experts per layer (20.88 GiB) but failed to grow the compressed
cache at position 3 with only 104 MiB free. An exact 37-per-layer cache (`--expert-cache 1591`, 19.81 GiB)
completed one chunk with 760 MiB free; it changed lookup shares only slightly (64.1% VRAM, 13.9% bus,
21.9% CPU) and did not improve observed decode speed (13.3 tok/s). These are bring-up measurements,
not tuned performance results.

## Decode staging milestone

The routed path now copies route IDs and the normalized expert input into mapped pinned staging memory, polls the IDs on the host, and falls back to a stream sync if staging does not finish within the bounded poll. This removes the per-layer stack-buffer ID copy and usually avoids the initial stream synchronization; CPU/GPU expert work and the final weighted sum keep the established ordering. A trial that made the weighted sum wait on a mapped device join word stalled on chunk 55, even after matching DeepSeek41's 64-byte initialized mapped allocation, and was removed. A second trial reused the expert-publish sequence word for the join and removed the per-layer stream sync; the chunk-55 prompt stalled with the GPU spinning while CPU workers were parked, so that path was also backed out. The safe ordered route remains active.

Chunk 55 was replayed with the old and new routed paths at T=3 using the same 145 GiB RAM budget, 1,480-expert GPU cache, and frozen placement. All 255 scored rows were identical; max absolute NLL and KLD deltas were zero.

The decode ABBA used `basalt-bench --no-windows` because its per-window draft-log flag is unsupported by the DeepSeek4 engine, which has no drafter. It used three tokenizer-derived DeepSeek4 prompts, a fixed seed, one token per pass, the 145 GiB RAM budget, 1,480-expert GPU cache, frozen placement, and an RTX 5090 power limit of 475 W. A1/B1/B2/A2 aggregate decode rates were 13.55/13.52/13.69/13.66 tok/s; paired A averaged 13.603 tok/s (73.513 ms/pass) and paired B averaged 13.602 tok/s (73.521 ms/pass). The 0.01% difference is below the observed run spread. Per-fixture rates moved in both directions, so this staging change has no established speed gain. Passes/s equals tok/s here because each decode pass emits one token.

A separate prefill ABBA at requested depth 1,024 (1,038 prompt tokens including framing) measured 13.728 tok/s for A and 13.749 tok/s for B. This short diagnostic is also neutral and does not establish a prefill gain. Power limit was 475 W for all arms. These measurements do not replace the standard 16k/64k prefill results.

## Automatic expert-cache reserve milestone

For `--expert-cache auto`, DeepSeek4 now reserves the full compressed-cache footprint plus module room for both home and copy allocation, and adds `pin_mapping_bytes` for the maximum RAM expert tier before selecting GPU copies. The mapping estimate includes the one-extra-expert-per-layer spare used by RAM loading and is capped by an explicit resident-memory budget plus those spares. Explicit cache sizes skip the auto-only mapping reserve.

This fixes an observed startup failure at `--max-context 65536 --resident-budget-gib 145 --expert-cache auto`: the previous build failed at position 3 while growing the compressed cache with out-of-device-memory. The updated build completed the same one-chunk gate with the full 137.1 GiB RAM expert tier (256/256 per layer), a 35-expert-per-layer initial GPU copy allocation and 37/256 final tier, and about 945 MiB VRAM free. At fixed cache size 1,480, the old and updated builds produced identical logits for all 255 chunk-55 rows (zero NLL/KLD delta).

A fixed-placement decode ABBA used one tokenizer-derived 1,200-token conversation prompt, seed 1, `--no-windows`, a 145 GiB RAM budget, a 1,480-expert cache, and 475 W. Rates were A1 13.7, B1 13.6421, B2 13.6988, A2 13.7016 tok/s (one repetition per arm). Paired means were approximately 13.701 and 13.670 tok/s respectively, or 73.0 and 73.2 ms/decode pass. The roughly 0.2% difference is within run spread and does not establish a decode regression or gain. The prefill rate for the short prompt was about 14 tok/s; passes/s equals tok/s. The single-prompt comparison is less representative than the earlier three-prompt ABBA.

## Single-kernel expert handoff milestone

DeepSeek4 now publishes the normalized routed row and expert IDs from device memory into mapped host staging with the shared `doorbell_publish` kernel. The kernel fences the payload before incrementing a mapped sequence word; the CPU polls that one word instead of observing per-ID sentinels. This replaces two per-layer device-to-host copy operations and retains a bounded-poll stream-sync fallback.

The fixed-placement chunk-55 T=3 gate matched the pre-change output for all 255 rows (zero NLL/KLD delta). Decode ABBA used one tokenizer-derived 1,200-token conversation prompt, seed 1, `--no-windows`, frozen placement, 1,480-expert cache, 145 GiB RAM budget, and 475 W. A1/B1/B2/A2 measured 13.7209/13.6677/13.7024/13.7011 tok/s (73.0/73.2/73.0/73.0 ms/pass). Paired A/B were 13.7110/13.6851 tok/s, about 72.93/73.07 ms/pass; the 0.2% gap is within single-prompt run spread and establishes no speed gain. The prompt prefill rate was about 14 tok/s. Passes/s equals tok/s because each pass emits one token. Nsight was not rerun for this step, so the reduction in per-layer copies is inferred from the replaced calls rather than a new trace count.

## Batched attention metadata milestone

The three per-token attention fields (position, sliding-window start, and visible window length) now share one 12-byte host-to-device copy instead of three 4-byte copies. Existing consumers keep the same device addresses through offsets into the combined allocation. The fixed-placement chunk-55 T=3 gate matched all 255 pre-change rows exactly, with zero NLL/KLD delta.

The decode ABBA used the same one-prompt, 1,200-token fixture, seed 1, frozen expert placement, 1,480-expert cache, 145 GiB RAM budget, and 475 W. A1/B1/B2/A2 measured 13.7193/13.6960/13.7135/13.6950 tok/s (72.89/73.01/72.92/73.02 ms/pass). Paired A/B means were 13.7072/13.7048 tok/s (72.95/72.97 ms/pass), a 0.02% difference within run spread. Prompt rates were 14.4/14.4/14.4/14.3 tok/s for A1/B1/B2/A2. This reduces two small copy submissions per token but shows no measurable decode gain on this fixture; passes/s equals tok/s.

## Queue-ordered routed sum milestone

Removed the per-layer stream synchronization immediately before `weighted_route_sum`. GPU expert kernels and the sum are submitted to the same stream, so stream order retains their dependency; the host still waits on the routed-row publication before reusing the row buffers. The mapped join-word variants described above remain backed out.

The fixed-placement chunk-55 T=3 gate matched the prior output on all 255 rows (zero NLL/KLD delta). Decode ABBA used the same one-prompt, 1,200-token fixture, seed 1, frozen expert placement, 1,480-expert cache, 145 GiB RAM budget, and 475 W. A1/B1/B2/A2 measured 13.6771/13.7144/13.7577/13.7513 tok/s (73.12/72.92/72.69/72.72 ms/pass). Paired means were 13.7142/13.7361 tok/s (72.92/72.80 ms/pass), a 0.16% B improvement, within the single-prompt run spread and not an established gain. Prompt rates were 14.3/14.4/14.4/14.4 tok/s. Passes/s equals tok/s because each pass emits one token. Nsight was not rerun, so synchronization-call reduction is inferred from the removed call; no multi-fixture or prefill comparison was run.

## Mapped route-metadata milestone

The per-layer route selector's fixed row and two masks now share one 8-byte mapped pinned record. The router reads those fields from the mapped record, replacing three separate tiny H2D copies per layer. This keeps the selector values and routing logic unchanged; the record is reused only after the preceding layer's route-selection kernel has published its IDs.

The fixed-placement chunk-55 T=3 gate matched the preceding output on all 255 rows (zero NLL/KLD delta). Decode ABBA used the same one-prompt 1,200-token fixture, seed 1, frozen placement, 1,480-expert cache, 145 GiB RAM budget, and 475 W. A1/B1/B2/A2 measured 13.6854/13.7786/13.7114/13.7460 tok/s (73.07/72.58/72.93/72.75 ms/pass). Paired A/B means were 13.7157/13.7450 tok/s (72.91/72.75 ms/pass), a 0.21% B increase versus about 0.5% within-arm run spread; this is neutral and does not establish a decode gain. Prompt rates were 14.3/14.4/14.3/14.4 tok/s. Passes/s equals tok/s; the one-prompt prefill means were both 14.35 tok/s.

Matched Nsight Systems traces of 1,200 generated tokens with the same cache and frozen placement counted 308,452 versus 151,330 CUDA H2D activity operations (pre-change/post-change), a 50.9% reduction; H2D bytes were effectively unchanged at 27,869.177 versus 27,868.863 MB. `cudaMemcpyAsync` calls fell from 309,778 to 152,656, and their cumulative API time fell from 1.529 to 1.329 s. These traces include startup and should be used for operation counts, not throughput; their profiled decode rates were 13.5863 and 13.5107 tok/s. The decode ABBA, not the profiled run, is the throughput comparison.

## Static decode graph experiment (rolled back)

Capture-once graphs were tried for each layer's attention mHC collapse/expansion, FFN collapse/router/shared expert, FFN expansion, and for the output head. Dynamic host expert lookup and the expert plan stayed outside capture. The chunk-55 T=3 gate was exact against the mapped-metadata build: all 255 rows matched with zero NLL/KLD delta.

The 475 W ABBA used the same one-prompt fixture, seed, frozen 1,480-expert GPU tier and 145 GiB RAM tier. A1/B1/B2/A2 were 13.7263/13.6784/13.6811/13.7437 tok/s (72.85/73.11/73.09/72.76 ms/pass). Paired means were 13.7350 tok/s (72.81 ms/pass) without graphs and 13.6798 tok/s (73.10 ms/pass) with graphs, a 0.40% decode regression. The B arms were both slower than their adjacent A arms, so this graph segmentation was removed. Prompt rates were 14.4 tok/s in A and 14.3 tok/s in B; passes/s equals tok/s. Nsight was not rerun for this experiment.

An FFN-only capture then grouped each layer's collapse, route preparation, and shared expert into one graph while leaving dynamic expert execution and expansion on the normal stream. Chunk-55 T=3 remained exact (255/255 rows; zero NLL/KLD delta). At 475 W, A1/B1/B2/A2 measured 13.7373/13.6730/13.7542/13.7084 tok/s (72.79/73.14/72.71/72.95 ms/pass); paired means were 13.7229/13.7136 tok/s (72.87/72.92 ms/pass), a 0.07% difference within run spread. Prompt rates were 14.4/14.1/14.4/14.4 tok/s. Nsight saw kernel-launch API calls fall from 2,673,219 to 2,097,578, but graph-launch calls added 52,374; combined launch API time rose from 5.403 to 5.767 s. Since it produced no decode gain and increased measured launch API time, this variant was also removed.

## Mapped expert-plan experiment (rolled back)

The per-layer 144-byte expert-plan H2D upload was replaced with one mapped pinned descriptor per layer. Initial runs omitted frozen placement and were discarded. With `--expert-cache 1480 --resident-budget-gib 145 --diag-tier-frozen`, chunk 55 at T=3 matched all 255 rows bitwise (zero NLL/KLD delta).

The fixed-placement 475 W ABBA used one tokenizer-derived 1,200-token conversation prompt, seed 1, `--no-windows`, the 1,480-expert GPU cache, and 145 GiB RAM tier. A1/B1/B2/A2 decode rates were 13.7129/13.6298/13.5959/13.7072 tok/s (72.92/73.37/73.55/72.95 ms/pass); paired means were 13.7101/13.6129 tok/s (72.94/73.46 ms/pass). The mapped plan was 0.71% slower across both B arms. Prompt rates were 14.4/14.2 tok/s for A/B. Passes/s equals tok/s because each decode pass emits one token. The mapped-plan change was removed to preserve decode throughput.

## Packed indexer metadata experiment (rolled back)

For CSA indexer layers, the host combined the 4-byte compressed-attention limit and 128-byte rotary-angle vector into one 132-byte H2D upload. The separate HCA limit upload remained unchanged. The fixed-placement chunk-55 T=3 gate matched the prior output on all 255 rows (zero NLL/KLD delta).

At 475 W, the same frozen-placement 1,200-token fixture, seed, 1,480-expert cache and 145 GiB RAM budget measured A1/B1/B2/A2 at 13.6875/13.7121/13.6911/13.7158 tok/s (73.06/72.93/73.04/72.91 ms/pass). Paired A/B means were 13.7017/13.7016 tok/s (72.98/72.98 ms/pass), effectively identical. Prompt rates were 14.4/14.2/14.2/14.2 tok/s. Passes/s equals tok/s. Since the one-prompt ABBA showed no decode gain, the packed upload was removed; no new Nsight trace was collected for this candidate.

## HCA contiguous-row selection experiment (rolled back)

HCA compressed-attention rows are sequential, so a candidate represented them as a contiguous range in sparse MQA and removed the per-token host ID-vector construction and H2D upload. The fixed-placement chunk-55 T=3 output matched the previous rows file byte for byte (255/255 rows).

The 475 W ABBA used the same frozen-placement 1,200-token fixture, seed, 1,480-expert GPU cache, and 145 GiB RAM tier. A1/B1/B2/A2 measured 13.7588/13.6984/13.7267/13.7256 tok/s (72.68/72.99/72.85/72.86 ms/pass). Paired A/B means were 13.7422/13.7126 tok/s (72.77/72.93 ms/pass), a 0.22% B regression within observed run spread. The two B arms moved in opposite directions relative to their adjacent A arms. Prompt rates and Nsight transfer counts were not collected; this candidate was removed because it did not establish a decode gain. Passes/s equals tok/s.

## Single-block indexer top-k

For up to 12,288 visible indexer rows, score generation now writes FP32 scores and one block selects the threshold winners by four radix-byte passes. It orders only the selected 512 rows by score and row id, preserving the existing sparse-attention accumulation order. Larger selections keep the CUB full sort. The GPU parity test matches the double-reference ordering at 702, 12,288, and 12,289 rows; the 8-query chunking and empty/single-row cases also match. The fixed-placement chunk-55 T=3 gate matched the prior 255 output rows byte for byte; the gate reported mean KLD 0.10152, p99 1.4961, max 2.0715 and 87.06% top-1 against the broad base, unchanged from the preceding build.

The kernel microbench compared the one-block path with the previous CUB path on the same fixtures: at 702 rows, 112.19 versus 148.59 µs/call; at 12,288 rows, 79.30 versus 126.80 µs/call. The 12,289-row boundary remains on CUB. Matching 1,200-token Nsight traces showed the CUB single-tile sort plus ID writer at 25,515 calls each, totaling 318.3 ms; the new selector ran 25,515 times and totaled 152.7 ms. `cudaLaunchKernel` calls fell by 25,515 (2,673,219 to 2,647,704), while `cudaMemcpyAsync` stayed at 152,656 calls. The candidate trace had 1,954 `cudaStreamSynchronize` calls versus 1,346 in the earlier trace; that discrepancy is not attributed to the selector and is not used as a speed claim.

At 475 W, two frozen-placement ABBA sequences used the same tokenizer-derived 1,200-token conversation prompt, seed 1, 1,480-expert GPU cache and 145 GiB RAM tier. A1/B1/B2/A2/A3/B3/B4/A4 measured 13.7211/13.7489/13.7093/13.6776/13.6933/13.7172/13.7531/13.6944 tok/s. Paired means were A 13.6966 tok/s (73.011 ms/pass) and B 13.7321 tok/s (72.822 ms/pass), a 0.26% B gain. Each B arm exceeded its adjacent A arm. Prompt prefill averaged 14.35 tok/s for A and 14.38 tok/s for B. Passes/s equals tok/s because each pass emits one token.

## Routed expert window groundwork

The session's routed-expert bridge now accepts up to eight rows. It publishes the row-major activations and route IDs together, deduplicates expert IDs while retaining each token/route destination, and uses `ExpertTier::start_cpu_window` plus grouped GPU expert entries. The single-row decode plan keeps its original compact 144-byte upload; the larger plan is uploaded only for multi-row calls. The existing forward path still calls this with one row, so this is groundwork only and does not implement chunked prefill or establish a speed gain.

The build passed, and a fixed-placement chunk-55 T=3 PPL gate reproduced all 255 rows byte-for-byte against the previous indexer-top-k result (KLD mean 0.10152, p99 1.4961, max 2.0715, top-1 87.06%). No ABBA is claimed: the multi-row call path is not wired into prefill yet, and this change does not establish a speed improvement. The next required step is the row-batched layer loop that exercises this bridge; its decode ABBA must be measured before keeping any throughput candidate.

Reproduce the base from the model GGUF:

```sh
llama-perplexity -m DeepSeek-V4-Flash-Vision-Exp-source-precision.gguf -f wikitext-2-raw/wiki.test.raw \
    --kl-divergence-base dsv4visionexp-source-base.kld -c 512 -b 8192 -ub 8192 \
    --chunks 100 -ngl 99 --n-cpu-moe 99 -t 16
```

The completed run took 13.07 minutes.

## KV cache formats (2026-11-27)

The session caches now have a selectable row precision: `--kv fp8` (default) stores window rows as E4M3 codes with a
power-of-two (UE8M0) scale per 32, compressor output rows as E2M1 codes with an E4M3 scale per 16, and CSA index keys
as E2M1 codes with a UE8M0 scale per 32. These are the reference's `act_quant` and `fp4_act_quant` roundings from the
Maya/Strata import; the indexer rounds its queries with the same E2M1/2^k scheme before scoring. `--kv fp16` keeps plain
half rows. Sparse MQA decodes packed rows exactly (score keys by group scale, PV values by row group scale, masked rows as
zeros). `ds4_cache_rows_parity` checks both writers and readers against a CPU packing byte for byte, and
`ds4_attention_step_parity` compares an Fp8 window against the F32 ring (max abs 0.102 on its random fixture).

Planned cache reserve on the `-hchead` pack, from the printed `cache reserve` at start: 65,536 tokens = 0.13 GiB packed /
0.44 GiB fp16; 1,048,576 tokens = 1.88 GiB packed / 6.74 GiB fp16. Prose decode at 64k and 1M with
`--resident-budget-gib 120 --expert-cache 512 --spec 0` ran at 21-22 tok/s in both layouts, within the run-to-run spread.
The 100-chunk `--kv fp8` Wikitext-2 gate against `dsv4visionexp-source-base.kld` (25,500 rows, RTX 5090 @ 475 W,
`--resident-budget-gib 145`) completed with PPL `3.9836 +/- 0.0596` (base `4.3379`, `ln(PPL/base) -0.08522`), mean KLD
`0.15429`, median `0.03136`, p99 `1.9900`, max `7.2995`, and top-1 agreement `87.92%` (`prompt ms/chunk 1568`,
`decode ms/chunk 11596`, `60.9% VRAM / 15.4% PCIe / 23.7% CPU`). At 475 W, `basalt-bench` measured `pp16384` at `1,884`
tok/s, lossless `tg` at `23 / 24-27 / 24-25` tok/s (`conv / struct / prose`), and opt-in lossy `tg`
(`--route-prior 0.5 --route-tail-skip 4`) at `34 / 43 / 35` tok/s.
