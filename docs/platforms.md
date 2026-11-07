# Platforms

Linux x64 and Windows x64, NVIDIA Blackwell (sm_120), and Windows on Arm64 with the RTX Spark's integrated Blackwell GPU (sm_121, the
N1X). Linux is where the engine is tested and benchmarked. CI builds Linux and Windows x64 and runs the CPU tests on each. The GPU
tests and the quality gate need a card.

| | Linux | Windows | Windows on Arm |
| --- | --- | --- | --- |
| OS | x86-64, AVX2 (AVX-512 used when present) | Windows 10 1903+ x64 | Windows 11 on Arm64, NEON with the dot product |
| Compiler, build | GCC 13+, CMake 3.24+ | MSVC 19.44 (VS 2022), CMake 3.24+, Ninja | MSVC 19.51 Arm64 (VS 2026 Build Tools), CMake 3.24+, Ninja |
| CUDA, Python | 13, 3.10+ | 13, 3.10+ | 13.4+ (the Arm64 toolkit), 3.10+ (Arm64) |

A build targets one compute capability: 12.0, or 12.1 when the target is Arm64 (`-DCMAKE_CUDA_ARCHITECTURES=120` or `121` picks
one). The engine refuses a card of any other.

## Platform code

OS code lives only in `src/platform/` and `include/basalt/platform/`, and CMake `if(WIN32)` blocks. Anywhere else there is no `_WIN32`,
`_MSC_VER`, `__linux__`, `<windows.h>`, `<sys/mman.h>`, `<unistd.h>` or raw OS call (mmap, madvise, pread, pthread/affinity,
fork/exec/pipe, dlopen, /proc or /sys reads). `tools/check_platform_boundary.sh` enforces it. Each seam has a Linux and a Windows body,
the Linux body being the code you would have written anyway, and a test in `tests/platform/` that runs on both. Paths are UTF-8 and go
through the platform file calls.

The CPU kernels are per architecture, chosen by CMake (`cmake/target_arch.cmake`): AVX2 / AVX-512 on x86-64, NEON with the dot product
on Arm64 (`expert_neon.cpp`, `nvfp4_neon.cpp`, `router_q4_neon.cpp`, `ggml/ggml_cpu_dots_neon.c`, `src/vision/gemm_cpu_neon.cpp`).
Shared code calls an x86 kernel only inside `if constexpr (platform::kX86)`.

## Windows limitations

- Pinned memory is capped under WDDM: the pinned slices are limited to `RAM/2 - 8 GiB` when the whole arena cannot be registered.
- Large pages need "Lock pages in memory" (Administrators). Without it the engine uses 4 KiB pages and says so at start.
- No shared arena files and no cgroup limits. `--ple-io direct` reads through the file cache.
- `basalt serve` runs the Python server as a child.
- A display-attached GPU can hit the driver's timeout (TDR). Raise `TdrDelay` under `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`.
- A card that drives the desktop has VRAM that moves between starts, so greedy output can differ run to run. Pin `expert-cache` and raise
  `vram-reserve-mib` for reproducible output.
- Quality is not measured on Windows, and there are no headline benchmark numbers for it. Report Windows problems as issues labelled `windows`.

## Building on Windows

From a Developer command prompt (or after `vcvarsall.bat x64`):

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build -L cpu -j 8 --output-on-failure
```

If tests fail to create fixtures, point `TEMP` and `TMP` into the build tree. CUDA 13.2+ headers need `/Zc:preprocessor`; the CMakeLists
passes it to nvcc's host compiler.

## Numerics across the two builds

The CPU expert kernel is compiled by MSVC on Windows and GCC on Linux. On one commit, `expert_parity` and `pool_test` give 0 failures
on both, and the pool output matches the serial run bit for bit.

Greedy tokens differ across the builds: with tiers pinned and adaptive swaps off, `struct` matches over 300 tokens, `conv` diverges at
token 23 and `prose` at token 44. One build matches itself run to run, so the difference is the host compiler's CPU code (78% of expert
rows). Quality of the Windows build has not been run ([quality](precision/quality.md)).

## Windows on Arm (RTX Spark)

The N1X's GPU is integrated: its memory is a carve-out of the machine's RAM that Windows does not see (the GPU memory size the NVIDIA
Control Panel sets), and past it a shared segment of the RAM Windows uses. `cudaMemGetInfo` reports both. Everything above applies, and:

- Every sizing decision reads `device_mem_info` (`include/basalt/core/device_memory.hpp`), which on an integrated GPU reports the
  carve-out as NVML gives it: the expert cache is sized like a discrete card's VRAM (start line
  `expert cache: the integrated GPU's dedicated memory`). Without a carve-out it is the RAM available less the reserve
  (`ram-reserve-mib`) and the host's late buffers.
- A carve-out that holds every expert with room to spare keeps the prompt buffers beside them ([`prefill`](configuration/index.md)),
  and nothing is copied to RAM. With less room the prompt path borrows the cache's last slots, and a server keeps page-locked copies of
  their experts in the RAM above the reserve ([memory floors](running.md#memory-floors)). An explicit `resident-budget-gib`
  keeps no such copies: the borrowed slots are refilled from the file after a prompt.
- `nvidia-smi` lists the NPU as a second device with no CUDA compute capability: `gpu auto` skips it, and an unset `gpu` does not
  count it.
- The resident RAM copy stays page-locked: the GPU reads it at addresses of its own.
- In Modern Standby (screen off or session locked) the driver can move the GPU's memory to RAM and the pagefile, and a running engine
  stalls until the machine wakes. Keep the machine awake while it serves.
- Only the Qwen3.8-Flash-Next NVFP4 pack has been run on Arm64, and its quality has not been measured.

The NEON kernels sum in their own order, so the CPU tier's rows differ from the x86 builds' within the same tolerances. On the NVFP4
pack's layer 0 (`native_expert_parity`, MSVC 19.51):

| check | result |
| --- | --- |
| NVFP4 NEON dot vs plain C (`--experts PACK 0`) | 3200 rows, rel 2.16e-07 |
| NVFP4 multi-token rows vs the NEON dot, 1 to 9 tokens | 0 of 90 token-rowsets differ in any bit |
| NVFP4 CPU expert vs the GPU's | rel 7.24e-08 |
| Q6_K NEON dot vs a double reference (`--cpu-synthetic q6_K/q8_0`) | worst error / sum\|terms\| 1.26e-08 |

A token rounds the same alone as in a window. The build reproduces its own greedy tokens with the expert cache pinned; no x86 build's
tokens have been compared with it.

## Building on Windows on Arm

From an Arm64 Developer command prompt (or after `vcvarsall.bat arm64`), with the Arm64 CUDA toolkit:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j 12
ctest --test-dir build -L cpu -j 8 --output-on-failure
```

CMake picks the NEON kernels and compute capability 12.1. The x86-only kernel tests (`avx_bench`, `iq_avx2_test`, `iq_avx512_test`,
`nvfp4_cpu_rows_test`, `router_dot_parity`) are not built. The server needs an Arm64 Python: every package in `requirements.txt` has a
`win_arm64` wheel.
