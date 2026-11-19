# Arm64 and compute capability 12.1

Basalt builds and runs on Arm64 with an NVIDIA GPU of compute capability 12.1: the integrated Blackwell GPU of the RTX Spark N1X
and of the DGX Spark (GB10). It has been built and tested on one machine, an RTX Spark N1X laptop (128 GB of unified memory, 96 GB
of it given to the GPU), both natively on Windows on Arm64 and on Linux Arm64 (WSL on the same laptop). A DGX Spark has the same
GPU but has not been tested here.

A build is for one CPU architecture and one compute capability. x86-64 with 12.0 is unchanged. Arm64 defaults to 12.1;
`-DCMAKE_CUDA_ARCHITECTURES=120` builds Arm64 for a 12.0 card instead (not tested). `BASALT_CUDA_CC` (120 or 121) follows the
architecture you build for and is what the start checks the GPU against.

| | Linux Arm64 | Windows on Arm64 |
| --- | --- | --- |
| Tested on | Ubuntu 26.04 aarch64 (WSL2), GCC 15.2 | Windows 11 on Arm, MSVC 19.52 (Visual Studio 2026 Build Tools) |
| CPU | Armv8.2 with the dot product (NEON CPU kernels) | the same |
| CUDA | 13.5 | 13.5 |
| Tests | 224 of 225 (below) | 213 of 219 (below) |

## Building

### The CUDA toolkit

Any CUDA 13 toolkit for the platform works. `tools/port/fetch_cuda.py` assembles one from NVIDIA's redistributable archives, without
the installer and without a driver component (the machine's own driver runs the binaries); each archive is checked against NVIDIA's
SHA-256:

```sh
python3 tools/port/fetch_cuda.py linux-sbsa ~/cuda-13.5          # Linux Arm64
python tools\port\fetch_cuda.py windows-arm64 C:\cuda-13.5       # Windows on Arm64
```

On Linux it makes `lib64` a link to `lib` (nvcc looks there for `libcudadevrt`). On Windows the runtime DLLs land in `bin\arm64`.

### Linux Arm64

```sh
export PATH=~/cuda-13.5/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON    # "basalt: Arm64, NEON CPU kernels; CUDA arch 121"
cmake --build build
LD_LIBRARY_PATH=~/cuda-13.5/lib ctest --test-dir build -j 4
```

Under WSL2, add `/usr/lib/wsl/lib` (the driver) to `LD_LIBRARY_PATH`, and keep the build out of `/tmp`, which WSL empties.

### Windows on Arm64

Visual Studio 2026 or its Build Tools with the ARM64 C++ tools; the CMake and Ninja that come with it are enough.

```bat
tools\port\build-win-arm64.cmd build C:\cuda-13.5
call tools\port\env-win-arm64.cmd C:\cuda-13.5
ctest --test-dir build -j 4
```

`env-win-arm64.cmd` finds Visual Studio with `vswhere`, enters its Arm64 environment and puts the toolkit's `bin` and `bin\arm64` on
`PATH`. Keep `bin\arm64` on `PATH` whenever the engine or a test runs: without it Windows stops each program that links cuBLAS with
a "cublas64_13.dll was not found" dialog, which waits for a click while ctest waits for the program.

## What changes on Arm64

- **CPU kernels.** The x86 kernels (AVX2, AVX-512, F16C) are replaced by NEON ones (`cmake/arm64.cmake`). The dot products and
  activation quantizers of every format the packs use are generated from llama.cpp's Arm kernels of the commit the x86 file came
  from (MIT) by `tools/port/extract_ggml_neon.py`; NVFP4, the Q2_0 rows of several tokens, the router dot and the vision GEMM are
  Basalt's own. The x86 kernels' own tests are not built. GCC and Clang build with `-ffp-contract=off`, so nothing fuses a
  multiply-add the source does not write (the x86 build has none outside its AVX files).
- **Memory ordering.** `include/basalt/platform/compiler.hpp` has an Arm64 body: `cpu_relax` is `isb`, and `store_fence`,
  `load_fence` and `prefetch` replace `_mm_sfence`, `_mm_prefetch` and the like. A host loop that polls a ring the GPU wrote takes a
  load fence after the poll: Arm64 does not order those loads.
- **Not written yet:** NEON kernels that compute several tokens of an NVFP4 or i-quant expert at once. Those rows run the
  one-token dot product per token, which matters only for experts computed on the CPU (none, when every expert fits the GPU).

## The integrated GPU

The GPU has no memory of its own: the firmware gives it a carve-out of the machine's RAM (96 GB of 128 on the N1X). Measured on the
N1X under Windows, and handled in `core/device_memory` and `core/device_copy`:

- **Free memory.** `cudaMemGetInfo` reports the carve-out plus a shared segment (110.5 GiB on the N1X), and its free figure misses
  memory mapped through the virtual-memory API, which is how the expert cache and the KV arena are made. Every sizing decision now
  reads `core::device_mem_info`: on a discrete card exactly `cudaMemGetInfo`; on an integrated GPU the carve-out as the total, and
  as used the larger of what every process has written (NVML) and what the others held at the first reading plus what this process
  holds now (DXGI on Windows).
- **The Windows commit limit.** Windows charges an integrated GPU's memory to the commit limit (RAM plus page file) when it is
  allocated, before a byte is written. The engine keeps the commit headroom less its reserve (`--ram-reserve-mib` plus the 2 GiB
  for what loads after the plan) free, in the device reading and for the page-locked RAM copy, and says so once in the log when the
  commit limit binds before the GPU's memory does. The first load of the Flash-Next NVFP4 pack without this took the commit charge to
  97.3 of 98.5 GiB. **On a 128 GB N1X, give Windows a page file of at least about 72 GiB** (custom size, 81,920 to 98,304 MB
  here): with the system-managed 68 GiB the guard kept 229 experts on the SSD and dropped the RAM copy of the prompt path's lent
  slots (refilled from the SSD in 0.9 s per request instead of 8 ms); with 80 GiB every expert sits on the GPU and the peak commit
  charge is 87.4 of 110.5 GiB.
- **Uploads from pageable memory land late.** A synchronous `cudaMemcpy` from pageable memory to the device returns before the
  bytes are visible to a kernel on another stream (seen on Linux and Windows). `core::device_copy` waits for the default stream after
  such a copy on an integrated GPU; every synchronous copy in the engine goes through it. On a discrete card it is `cudaMemcpy`.
- **Page-locked memory.** The pin budget comes from the adapter's local segment group (DXGI), since an integrated GPU has no
  non-local one.
- **The NPU.** NVML lists the laptop's NPU as a device with no compute capability; GPU selection skips it.
- **Modern Standby.** Such machines may move GPU memory out in standby, and a running engine stalls until the wake. The engine holds
  a power request while it runs (no idle sleep, `powercfg /requests` shows it); `SLEEP` withdraws it and `WAKE` or a request asks
  again. It changes no setting and does not keep the display on. Entering standby with the engine loaded has not been tested.

## Running

As on any platform: `basalt serve PACK`. On Windows pass `--python` with a Python that has `requirements.txt` installed (the
default looks for `.venv/bin/python`, a Linux layout). The server here ran on Python 3.13 for Windows on Arm64; installing
`requirements.txt` into a fresh Arm64 environment has not been tried.

`tools/serve_speed.py` times a running server: `long` sends one long chat prompt (a text file, or a filler of N tokens) and
reports the prompt read and the decode; `fixtures` sends basalt-bench's requests as token ids to another engine's
`/v1/completions` (vLLM, llama.cpp) for a like-for-like comparison. `tools/serve_soak.py ... topics` measures several streams at once.

## Tests

| | Linux Arm64 (WSL2) | Windows on Arm64 |
| --- | --- | --- |
| Pass | 224 of 225 | 213 of 219 |
| Disabled | the x86 kernels' own tests (`iq_avx2_test`, `iq_avx512_test`, `nvfp4_cpu_rows_test`, `router_dot_parity`) | the same |

What does not pass:

- `video_test` (Linux): the decoded frames' channel sums are about 3% under the reference. Not investigated.
- `ds41_router_parity`, `ds41_dspark_parity`, `gqa_attn_parity` (Windows): they compare against hashes frozen from inputs made by
  libstdc++'s `std::normal_distribution`, which MSVC's draws differently. A limit of the tests, not of the kernels.
- `gdn_head_split_parity` (Windows): every bitwise check passes; the maximum relative error on MSVC's inputs is slightly over the bound.
- `file_expert_repin_test` (Windows): fails in the Windows-only path that pins host memory mapped through the virtual-memory API
  ("pinned before its reads"). Not investigated.
- `ds41_attn_index_parity` (Windows): not investigated.
- `kv_arena_test` can fail when other GPU tests run beside it (`ctest -j`): it reads free memory while they allocate. Alone it passes.

## Speed on the RTX Spark N1X

Windows 11 on Arm, native build, one process, the page file above. vLLM is a community build for the RTX Spark (vLLM 0.32.0) with
NVIDIA's NVFP4 checkpoint of the same model, MTP with two drafts, FP8 KV and one sequence; it read the same requests through
`tools/serve_speed.py fixtures`.

Qwen3.8-Flash-Next, `NVFP4-DQ6_K` pack, basalt-bench (fp16 KV):

| | Basalt | accepted, tokens per window | vLLM |
| --- | ---: | --- | ---: |
| decode conv | 80.8 tok/s | 65.9%, 2.55 | 49.1 tok/s |
| decode struct | 140.3 tok/s | 98.4%, 6.91 | 57.9 tok/s |
| decode prose | 67.8 tok/s | 64.3%, 1.88 | 45.6 tok/s |
| decode, token-weighted | about 116 tok/s | | about 55 tok/s |
| prompt read 16k / 64k | 2,558 / 2,638 tok/s | | 1,753 / 1,780 tok/s |

Every expert fits the GPU (24,576 slots, 63.3 GiB). The server, `--kv int8`, one long Chinese text as one user message, thinking
off, temperature 0, 512 tokens out:

| Prompt | Read | Decode |
| --- | --- | --- |
| 4,128 tokens | 1,473 tok/s | 63.6 tok/s |
| 191,231 tokens | 2,387 tok/s (80 s) | 55.1 tok/s |
| 260,124 tokens | 2,224 tok/s (117 s) | 54.3 tok/s |

`--parallel 4 --slot-full-context` (four slots of 262,144): one stream 63.7 tok/s, two 78.4 together, four 80.1 together. A
mixture of experts routes each request to its own experts, so batching saves few weight reads, and the memory bandwidth (LPDDR5X)
sets the ceiling.

Qwen3.8-27B, `QUASAR-NVFP4` pack, basalt-bench (fp16 KV): decode conv 50 tok/s (44.8% accepted, 6.20 per window), struct 157
(87.0%, 14.38), prose 36 (34.3%, 4.83); `--spec 0` 16.5 tok/s, about the memory bandwidth for 16 GB of weights a token; prompt read
2,757 tok/s at 16k and 1,632 at 64k. The server (`--kv int8`, four slots): a 4,125-token prompt reads at 2,773 tok/s and decodes at
27.2; a 260,130-token one at 1,036 and 14.3 (each token reads 8.6 GB of K/V besides the weights); one, two and four streams decode at
28.6, 52.4 and 72.3 tok/s together.
