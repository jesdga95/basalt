# ERNIE-5.5-Flash

Baidu's **ERNIE-5.5-Flash-Preview** ([Hugging Face](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview), Apache-2.0 preview
weights). Runs with the same command as any model: `basalt serve PACK`. What it supports next to the others:
[models](index.md). Rounding and quality: [precision](../precision/ernie55.md).

The preview release is text-only (no vision encoder).

## Pack

The ready-built pack `ERNIE-5.5-Flash-Preview-MXFP4.basalt` (`155.27 GiB` / `166.72 GB` with DSpark attached, SHA-256
`7af8cd7a90ce86c269c4fbe5a24f0b5f40266d5f72d054ba9999a7edda088728`) is available on
[Hugging Face](https://huggingface.co/florian862-art/ERNIE-5.5-Flash-Preview-Basalt):

```sh
hf download florian862-art/ERNIE-5.5-Flash-Preview-Basalt ERNIE-5.5-Flash-Preview-MXFP4.basalt --local-dir models
```

Or build it directly from Baidu's official Hugging Face checkpoint (`baidu/ERNIE-5.5-Flash-Preview`), which includes
the main weights and the bundled `model.mtp.*` DSpark drafter tensors (detailed section layout:
[model file](../model-file/ernie55.md)):

```sh
hf download baidu/ERNIE-5.5-Flash-Preview --local-dir ./ERNIE-5.5-Flash-Preview
python3 tools/convert/convert.py \
  --checkpoint ./ERNIE-5.5-Flash-Preview \
  --out models/ERNIE-5.5-Flash-Preview-MXFP4.basalt \
  --workers 16
python3 tools/convert/dspark_ernie55_attach.py \
  --checkpoint ./ERNIE-5.5-Flash-Preview \
  --into models/ERNIE-5.5-Flash-Preview-MXFP4.basalt \
  --workers 16
```

- Routed experts (42 MoE layers × 512 experts, `133.88 GiB`): packed bit for bit as ggml `MXFP4` (`type 39`).
- Dense projections (`gguf`, `11.08 GiB`): 128×128 blockwise FP8 projections rounded once to `Q8_0`; `token_embd`, `output` head,
  compressor projections and indexer `wk` / `weights_proj` / `flash_gate` as `BF16`; norms, mHC mixers, sinks and routers as `F32`.
- Bundled DSpark drafter (`dspark`, `10.28 GiB`): attached in ~15 s without reconverting the main model (`--undo-dir DIR` keeps
  rollback state for `--roll-back`).

## Architecture

The `ernie55` architecture (code in `src/models/ernie55/`): 43 layers (1 dense + 42 Latent MoE), mHC 4-stream Sinkhorn-Knopp
residuals, and hybrid attention:

- **Attention:** 37× HCA layers (with weightless per-head Q-RMSNorm before RoPE, 128:1 compressed KV cache, VHA PostMix and
  attention sinks) and 6× sparse MLA-DSA layers (with a 64-head Lightning Indexer and 256-bin deterministic radix top-K).
- **MoE:** 512 routed experts per MoE layer, top-10 active + shared expert, latent down/up projections, and SiTU-GLU softcap
  `4.0 / 25.0`.
- **DSpark:** 3-stage SWA-mHC speculative drafter tapping target layers `[40, 41, 42]`, proposing up to 7 tokens per verify window.
- **Chat, reasoning and tools:** `serve/models/ernie55.py` implements Baidu's prompt template, `<ernie:think>` reasoning with
  `reasoning_effort` (`none`, `low`, `medium`, `high`, `max`), and parallel `<ernie:tool_call>` XML tool calling.

## Memory

- **Loaded footprint:** ~`152.6 GiB` total (`~9.1 GiB` GPU 0 dense/KV/DSpark + remaining experts distributed across GPU 0, peer
  GPUs, and page-locked system RAM). For 16–32 GB GPUs, 192 GB system RAM is recommended (`--ram-reserve-mib` reserves host RAM
  for the OS and serving guard).
- **Context (`--max-context`):** `1,024` to `1,048,576` (default `8,192`, native YaRN).
- **KV cache per position (`--kv`):**
  - `fp8` (default): ~`4.5 KB / token` (128:1 compressed HCA + FP8 MLA-DSA + MXFP4 indexer keys) — ~`1.13 GiB` at `262,144`
    context.
  - `fp16`: ~`8.7 KB / token`.
- **Conversation cache:** full hybrid HCA (`disk_row_cells = 128` pooled) + DSA (`k` / `v`) `save_cells` / `restore_cells`
  support (~`169 MB` on disk/RAM for a 35K-token FP8 conversation); DSpark sliding rings reset/rewind cleanly on restore.
- **Shipped profile:** `data/profiles/ernie55-43x512.prof` (180-request decode routing profile) places the hottest experts on the
  GPUs at start.
- **Prefill scratch:** temporary `R = 8192` aliased `PromptBufs` are lent from `ExpertTier` via `lend()` / `reclaim()`,
  preserving warm decode VRAM experts after each prompt.

## Cards

`gpu` names 1 to 4 cards ([GPUs](../configuration/gpus.md)):

- **One card (`"gpu": "0"`):** dense weights, KV cache, and the hottest routed experts sit on GPU 0; the remaining experts live
  in page-locked RAM and run on the CPU pool (multi-row AVX2 / AVX-VNNI `vpdpbusd` MXFP4 kernels) in parallel with GPU 0.
- **Peer (`"gpu": "0,1", "gpu-mode": "peer"`):** up to 4 cards (main + 3 peers). Each peer holds a contiguous block of MoE
  layers' experts, computes its resident experts and a multi-token-prioritized share of RAM experts over PCIe during decode, and
  splits prompt MoE work 50/50 via `PeerShare` (with cross-layer `PeerShare::prepare()` read-ahead).
- **`split` and `tensor`:** refused.

Example `basalt_config.json` for two 16 GB Blackwell cards and 192 GB RAM:

```json
{
  "version": 2,
  "models": [
    {
      "model": "models/ERNIE-5.5-Flash-Preview-MXFP4.basalt",
      "alias": "ernie-5.5-flash-preview-mxfp4",
      "gpu": "0,1",
      "gpu-mode": "peer",
      "max-context": 262144,
      "kv": "fp8",
      "spec": 8,
      "ram-reserve-mib": 24576
    }
  ],
  "engine": {
    "pool-workers": 20,
    "peer-reserve-mib": 350,
    "conversation-cache-disk-mib": 16384
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8080
  }
}
```

## Flags

- `--kv fp8|fp16` — KV cache format (`fp8` default).
- `--spec N` — verify window size (default `8` when the `dspark` section is present, `0` to disable).
- `--draft-experts N` — per-stage GPU-0 expert cache slots for DSpark (default `0`: CPU-only with L3-grouped warming, freeing
  ~321 MiB VRAM on GPU 0 for main-model experts).
- `--route-tail-skip [R]` / `--route-prior [LAMBDA]` — optional lossy decode routing flags (default off; bare `--route-prior`
  uses `0.05`; see [precision](../precision/ernie55.md#lossy-routing-flags)).
- `--diag-rows N` / `--draft-log PATH` — diagnostic chunk cap and drafter acceptance log.

## Benchmarks

- **Hardware:** 2× NVIDIA RTX 5070 Ti (16 GB each, SM 12.0), Intel Core Ultra 7 270K Plus (AVX2 + AVX-VNNI), 192 GB DDR5-6200
- **Footprint:** `152.6 GiB` loaded in `37.9–46 s` (`GPU 0: 9.1 GiB`, `GPU 1: 14.6 GiB`, `RAM: 128.9 GiB`, `--gpu 0,1 --gpu-mode peer --kv fp8`)

### `basalt-bench` (`--kv fp8`, `power limit 300 W`, `--gpu 0,1 --gpu-mode peer`)

| Test | Throughput | Accept % | Draft Len |
|---|---:|---:|---:|
| `pp16384` | **751 t/s** | — | — |
| `pp65536` | **621 t/s** | — | — |
| `tg conv` | **36.9–40.5 t/s** | 69.0–72.4% | 1.51–1.71 |
| `tg struct` | **87.6–88.8 t/s** | 99.4–99.5% | 6.20–6.60 |
| `tg prose` | **45.3–46.4 t/s** | 62.8–65.7% | 1.50–1.60 |

### Long-Context Serve (`--gpu 0,1 --gpu-mode peer --kv fp8`)

| Workload | Throughput | Details |
|---|---:|---|
| **Prefill `0–8K` (`R=8192`)** | **836–844 t/s** | `pp=7,554` in `9.0 s`, `pp=8,192` in `10 s` |
| **Prefill `16K` (16,608 tok)** | **751 t/s** | `basalt-bench pp16384` (`749 t/s` wall) |
| **Prefill `64K` (66,363 tok)** | **621 t/s** | `basalt-bench pp65536` (BF16 Tensor-Core Lightning Indexer) |
| **Prefill `131K` (131,072 tok)** | **489 t/s** | `278 s` (`ctx=262144`) |
| **Prefill `170K` (170,328 tok)** | **433 t/s** | `393.3 s`, `tg=27.7 t/s`, `acc=82%`, `dl=2.1` (`ctx=262144`) |
| **Sustained Decode (`7.5K` ctx, 1,305 tok)** | **46.4 t/s** | `acc=74%`, `dl=2.7`, `hit=28.8%`, `pcie=44.8%` |
| **Decode @ `16K / 24K` ctx** | **40.9 / 42.0 t/s** | `ctx=65536` |
| **DSpark Draft Latency** | **4.27 ms** | `conv --spec 3` |
| **CPU MXFP4 Kernel (1-thread, 2-tok down)** | **700 µs (8.95 GB/s)** | **8.3×** vs. ggml (`5805 µs`), bitwise `memcmp == 0` |
