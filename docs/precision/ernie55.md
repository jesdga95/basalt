# ERNIE-5.5-Flash precision

Where ERNIE-5.5-Flash rounds and how its output compares. Method: [quality](quality.md). Guide: [ERNIE-5.5-Flash](../models/ernie55.md).

## Summary

- **Pack source & BF16 reference:** The `.basalt` pack converts Baidu's official `baidu/ERNIE-5.5-Flash-Preview` release
  (native `MXFP4` routed experts + 128×128 blockwise `FP8` dense projections). Baidu also publishes
  `baidu/ERNIE-5.5-Flash-Preview-BF16` (~560 GB), so a true BF16 KLD reference is possible (pending a community run on a
  machine that fits the ~560 GB BF16 checkpoint).
- **Routed experts are bit for bit the release's bits.** All 42 MoE layers × 512 experts (top-10 routed + 1 shared) keep the
  release's MXFP4 E2M1 codes and UE8M0 scales unchanged (`GGML_TYPE_MXFP4 = 39`, 17 bytes per 32 values).
- **CPU and GPU MXFP4 parity:** The multi-row AVX2 / AVX-VNNI (`vpdpbusd`) CPU MXFP4 kernel is bitwise identical (`memcmp == 0`)
  to scalar `ggml_vec_dot_mxfp4_q8_0` across `nt = 1..8` (including SiTU-GLU `4.0 / 25.0`), verified by
  `native_expert_parity --mxfp4`.
- **HCA Q-RMSNorm (`8af95001`):** Every HCA layer (37/43) and every DSpark layer (3/3) applies weightless per-head Q RMSNorm
  (`head_dim = 512`, `eps = 1e-6`) before interleaved RoPE, matching upstream `vllm`.
- **KV cache (`--kv fp8` vs. `--kv fp16`):** At `65,536` context on a 16 GB RTX 5070 Ti, `fp8` saves `0.26 GiB` VRAM (+42
  expert slots on GPU 0); at `262,144` context, `fp8` leaves `3.40 GiB` for the VRAM expert tier (13/512 experts/layer), whereas
  `fp16` exceeds 16 GB GPU 0 VRAM.
- **Lossy routing flags:** `--route-tail-skip 9` lifts decode throughput by `+7%` to `+16%` (dropping `3.2%` of all routed
  entries), `--route-tail-skip 8` by `+18%` to `+27%` (`7.1%` dropped), and `--route-prior 0.05` by `+11%` to `+32%`.

## Weights

Every conversion rounding is deterministic and runs once in [`checkpoint_ernie55.py`](../../tools/convert/checkpoint_ernie55.py)
and [`dspark_ernie55_attach.py`](../../tools/convert/dspark_ernie55_attach.py) ([model file](../model-file.md#ernie-55-flash-pack)):

| Tensors | Stored as | From the release |
| --- | --- | --- |
| Routed experts (layers 1 to 42, 512/layer) | MXFP4 (`type 39`: 1 UE8M0 scale + 16 split-half E2M1 bytes per 32 values) | bit for bit |
| Dense projections (HCA, MLA-DSA, dense MLP layer 0, shared experts, latent down/up) | Q8_0 | 128×128 blockwise FP8 dequantized to F32, rounded once by ggml's reference Q8_0 rounding |
| `token_embd.weight`, `output.weight` | BF16 (`token_embd` mmap'd in RAM) | as stored (`output.weight` rounded to Q8_0 on GPU at load) |
| HCA compressor (`kv`, `gate`) and MLA-DSA indexer (`wk`, `weights_proj`, `flash_gate`) | BF16 | as stored |
| Norms, mHC mixers/biases/scales, folded VHA PostMix (`I + V @ U^T`), attention sinks, compressor APE, split-feature routers, router bias & scale, multimax head params | F32 | as stored (or BF16 widened to F32; routers packed to BF16 on GPU at load) |
| DSpark drafter (`model.mtp.*`) | MXFP4 experts + Q8_0 projections + BF16 `markov_w1` + F32 norms/mHC/routers | drafting only: cannot change verified output |

## Where each stage rounds

| Stage | Format | Arithmetic |
| --- | --- | --- |
| Routed experts (decode, GPU 0 & peer GPUs) | MXFP4 (`type 39`) | `q8_1` activations (`native_mmvq` / `iq_kernels`), F32 accumulation, SiTU-GLU softcap `4.0 / 25.0` |
| Routed experts (decode, RAM on CPU) | MXFP4 (`type 39`) | `q8_0` activations via AVX2 / AVX-VNNI `vpdpbusd` (`mxfp4_avx2*`, bitwise `ggml_vec_dot_mxfp4_q8_0`), F32 sums, SiTU-GLU `4.0 / 25.0` |
| Routed experts (prompt chunks, `R` up to `8192`) | MXFP4 (`type 39`) | `q8_1` activations through Tensor-Core MMQ (`prefill::mmq::Context`) on GPU 0 and 50/50 dual-GPU `PeerShare`, F32 accumulation |
| Q8_0 dense projections & output head | Q8_0 | decode: `q8_1` MMVQ (PDL fused); prompt: Q8_0 MMQ / cuBLAS, F32 logits |
| Router | BF16 weights on GPU (F32 in file) | F32 logits + bias, bitonic top-10 selection (tie-breaking by lower expert id, parity-checked) |
| mHC residuals, norms, VHA PostMix | F32 | F32 |
| HCA attention (37 layers) | `--kv fp8` (default) or `--kv fp16` | F32 QK scores (weightless per-head Q-RMSNorm + interleaved RoPE), softmax, and weighted sum; 128-token blocks compressed via learned compressor |
| MLA-DSA attention & Lightning Indexer (6 layers) | `--kv fp8` (default: FP8 latent + MXFP4 indexer K) or `--kv fp16` | prompt indexer: 64-head BF16 Tensor-Core scoring + deterministic warp-ballot 256-bin FP32 radix top-K; MLA attention: FP16 batched `W_UK` / `W_UV` + F32 softmax/accumulation |

## Attention caches

- **`--kv fp8` (default):**
  - HCA compressed KV (37 layers, 1 entry per 128 tokens): 448 E4M3 nope values + 7× F32 group scales (per 64 values) +
    64 BF16 RoPE values = 604 bytes padded to 608 bytes per compressed row (~`185 B / pos` across 37 layers).
  - MLA-DSA latent KV (6 layers): 512 E4M3 values + 4× F32 scales + 64 BF16 RoPE channels (656 bytes) plus 128-dim MXFP4
    indexer K (68 bytes) = `4,344 B / pos` across 6 layers.
  - Total: ~`4.5 KB / pos` (~`1.13 GiB` at `262,144` context).
- **`--kv fp16`:**
  - Unrounded FP16 HCA compressed rows (`296 B / pos` across 37 layers) and FP16 MLA-DSA latent + indexer K (`8,448 B / pos`
    across 6 layers).
  - Total: ~`8.7 KB / pos` (~`2.18 GiB` at `262,144` context).

### Measured VRAM and paired output (`--kv fp8` vs. `--kv fp16`)

Measured on 2× RTX 5070 Ti (16 GB each, `--gpu 0,1 --gpu-mode peer --ram-reserve-mib 24576 --pool-workers 20 --peer-reserve-mib 350`):

| Context | `--kv` | GPU 0 VRAM expert tier | Main-card copies | Notes |
| --- | --- | ---: | ---: | --- |
| `65,536` | `fp16` (`--spec 0`) | `4.71 GiB` | `18 / 512` per layer (`756` slots) | `8.50 GiB` dense |
| `65,536` | `fp8` (`--spec 0`) | `4.97 GiB` | `19 / 512` per layer (`798` slots) | `+0.26 GiB` (`+42` slots) vs. `fp16` |
| `65,536` | `fp8` (`--spec 8`) | `4.18 GiB` | `16 / 512` per layer (`672` slots) | includes `0.60 GiB` DSpark on GPU 0 |
| `262,144` | `fp8` (`--spec 8`) | `3.40 GiB` | `13 / 512` per layer (`546` slots) | fits 16 GB GPU 0 (`fp16` at 262K exceeds 16 GB GPU 0) |

Paired greedy output (`--max-context 65536 --expert-cache 672 --spec 0`, `256` tokens/fixture, `temperature=0`):

| Fixture | First token identical | Identical prefix | `\|dlogp\|` at token 0 | Mean `\|dlogp\|` (prefix) | Max `\|dlogp\|` (prefix) |
| --- | --- | ---: | ---: | ---: | ---: |
| `struct` | yes | `256 / 256` | `0.000000` | `0.000000` | `0.000010` |
| `prose` | yes | `58 / 256` | `0.048457` | `0.011537` | `0.437116` |
| `conv` | yes | `17 / 256` | `0.009635` | `0.003593` | `0.046308` |

## Drafting

DSpark (`--spec 8` default, up to 7 speculative tokens/window) proposes candidate tokens that are verified by the main model's
decode window path. At equal initial expert budget (`--expert-cache 672`, `--kv fp8`, `--max-context 65536`), greedy `struct`
matches `256 / 256` (`mean |dlogp| = 0.000000`, `max = 0.000009`) between `--spec 0` and `--spec 8`. On multi-turn runs with
dynamic tier promotions and per-window GPU-1 peer RAM fetches enabled, the GPU `q8_1` MMVQ and CPU `q8_0` AVX-VNNI paths
accumulate slightly differently (~`1e-3` to `1e-2` log-prob drift when an expert moves between GPU and CPU), which can flip
near-tie tokens on open-ended fixtures (`conv`: `17 / 256` prefix, `mean |dlogp| = 0.003445`).

## Lossy routing flags

ERNIE-5.5 wires both opt-in decode routing flags ([quality](quality.md#lossy-routing-flags)), off by default and inactive during
prompts and DSpark drafting:

- `--route-tail-skip [R]`: skips uncached RAM experts ranked `R` or lower (`R` in `1..9` out of top-10; bare flag = `9`).
- `--route-prior [LAMBDA]`: biases GPU-resident experts by `+LAMBDA` during top-10 selection while preserving the router's own
  weights (bare flag = `0.05`).

Measured with `basalt-bench -t conv,struct,prose` on 2× RTX 5070 Ti (`300 W`, `--gpu 0,1 --gpu-mode peer --kv fp8 --ram-reserve-mib 24576 --pool-workers 20 --peer-reserve-mib 350`):

| Setting | `tg conv` | `tg struct` | `tg prose` | CPU lookups | Dropped routed entries |
| --- | ---: | ---: | ---: | ---: | --- |
| Lossless (default) | `37 t/s` (`69.0%`, `1.51`) | `88 t/s` (`99.4%`, `6.20`) | `45 t/s` (`62.8%`, `1.50`) | `29–42%` | `0%` |
| `--route-tail-skip 9` | `43 t/s` (`+16%`) | `97 t/s` (`+10%`) | `48 t/s` (`+7%`) | `26.6%` | `14.6%` of uncached (`3.2%` of all) |
| `--route-tail-skip 8` | `47 t/s` (`+27%`) | `104 t/s` (`+18%`) | `53 t/s` (`+18%`) | `23.7%` | `30.6%` of uncached (`7.1%` of all) |
| `--route-prior 0.05` | `49 t/s` (`+32%`) | `110 t/s` (`+25%`) | `50 t/s` (`+11%`) | `20.6%` | none (biases GPU-held experts) |

## Prefill vs. decode and cached prompt resume

For prompt chunks (`> 8` rows), ERNIE-5.5 uses batched Tensor-Core paths (BF16 tiled Lightning Indexer, FP16 pre-absorbed MLA, and
widened F32 SGEMM router) rather than the decode-window paths (`<= 8` rows: `fast128` indexer, absorbed `Q8_0` MLA, BF16 GEMV
router). Kernel-level parity is checked directly in `ernie55_kernels_parity`:

- Prompt SGEMM router vs. decode BF16 GEMV router: `8.1e-8` relative L2 error.
- FP16 pre-absorbed MLA vs. decode `dsa_mla_attention`: `3.2e-4` relative L2 error (both `--kv fp8` and `--kv fp16`).
- BF16 tiled Lightning Indexer vs. `fast128` (64 heads, MXFP4): `4.2e-4` relative L2 error.

End-to-end cached prompt resume vs. full recompute (`tests/program/prompt_cache_resume_drift_test.py`, `2,765`-token shared prefix
exceeding both the 128-token HCA compression ratio and the `2,048 + 128` MLA-DSA top-K threshold, 5 suffixes covering both
`<= 8`-token decode resume and `> 8`-token prefill resume, greedy `temperature=0`, `--spec 0`):

| `--kv` | Reused prefix | First-token match | Matched greedy prefix | Mean `\|dlogp\|` (token 0) | Max `\|dlogp\|` (token 0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp8` (default) | `2,765` | `5 / 5` | `73 / 80` (4/5 full `16/16`) | `0.132` | `0.386` |
| `fp16` | `2,765` | `4 / 5` | `50 / 80` | `0.238` | `0.746` |

Because ERNIE-5.5 routes top-10 of 512 experts across 42 MoE layers, small floating-point differences between batched prefill and
resumed decode can occasionally flip a boundary expert on near-tie tokens while preserving high first-token agreement.

## Reference and verification

- **Kernel & CPU parity (`-DWERROR=ON`):**
  - `ernie55_kernels_parity`: **65/65 PASS** (HCA Q-RMSNorm, GQA HCA, parallel `hca_compress`, 64-head fast128 & BF16 Tensor-Core
    Lightning Indexer, deterministic radix top-K, MLA-DSA & FP8/FP16 pre-absorbed MLA, prompt SGEMM vs. decode BF16 router, mHC,
    SiTU-GLU, PDL fused decode kernels, bitonic router).
  - `ernie55_dspark_parity`: **5/5 PASS**.
  - `ernie55_pack_test` & `ernie55_pretok_test`: **0 failures** (strict required-key validation, hard-coded kernel dim checks,
    shared `cache_layout.hpp` strides, exact-BF16 router check, real pack open).
  - `native_expert_parity --mxfp4`: **0 failures** (bitwise `memcmp == 0` across `nt = 1..8`, AVX2 & AVX-VNNI).
  - `prompt_cache_resume_drift_test.py`: **PASS** (cached resume vs. full recompute across short and long suffixes).
  - `conversation_store_test` & `conversation_cache_test`: **PASS** (hybrid HCA `ratio=128` + DSA multi-block roundtrip; 15,307 checks).
- **BF16 KLD reference (wikitext-2, 100 chunks):**
  - Pending community run against `baidu/ERNIE-5.5-Flash-Preview-BF16` (~560 GB, exceeding our 192 GB workstation).
