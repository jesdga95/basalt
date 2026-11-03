# Platforms

Basalt runs natively on Linux x64 and Windows x64 on NVIDIA Blackwell (sm_120), and on Windows on Arm64 with
the integrated Blackwell GPU of an NVIDIA RTX Spark (sm_121, the N1X). Linux is where the engine is tested and
benchmarked. CI builds Linux and Windows x64 and runs the CPU tests on each. The GPU tests and the quality gate
run on a machine with a card.

## Requirements

| | Linux | Windows | Windows on Arm |
| --- | --- | --- | --- |
| OS | x86-64, AVX2 (AVX-512 used when present) | Windows 10 1903 or later, x64 | Windows 11 on Arm64, NEON with the dot product |
| Compiler | C++20 (GCC or Clang) | Visual Studio 2022 (MSVC 19.44) | Visual Studio 2026 Build Tools, Arm64 (MSVC 19.51) |
| CUDA | 13 | 13 | 13.4 (the Arm64 toolkit) |
| Build | CMake 3.24+ | CMake 3.24+ and Ninja | CMake 3.24+ and Ninja |
| Python | 3.10+ | 3.10+ | 3.10+ (Arm64) |
| GPU | NVIDIA Blackwell (sm_120); optional second GPU as a peer expert tier (`gpu 0,1`) | NVIDIA Blackwell (sm_120); optional second GPU as a peer expert tier (`gpu 0,1`) | the RTX Spark's integrated GPU (sm_121) |

A build targets one compute capability: 12.0 by default, 12.1 when the target is Arm64. Pass
`-DCMAKE_CUDA_ARCHITECTURES=120` or `121` to choose; the engine refuses a card of any other.

## Platform code

OS-specific code lives in `src/platform/` (declared in `include/basalt/platform/`) and in CMake
`WIN32` blocks. Engine code outside them names no OS or compiler.
`tools/check_platform_boundary.sh` enforces this. Every seam has a Linux body and a Windows body
behind one declaration, and a test in `tests/platform/` that runs on both.

| seam | header | wraps |
| --- | --- | --- |
| file I/O | `file.hpp` | UTF-8 paths, open, positional reads, unbuffered descriptors |
| unbuffered reads | `direct_file.hpp` | asynchronous unbuffered file reads (the n-gram table) |
| file views | `mapped_file.hpp` | mapped files, prefetch and release hints, locking |
| memory | `memory.hpp` | RAM probes, page locking, aligned and huge-page allocation |
| CPU | `cpu.hpp` | topology, the Arm dot product probe |
| processes | `process.hpp` | spawn, wait, terminate, pipes, signals |
| system | `system.hpp`, `dynlib.hpp` | process memory, disk queries, full speed in the background, the NVML loader |
| compiler | `compiler.hpp`, `format_check.hpp` | CPUID, bit scan, spin hint, store fence, fp16 rounding, `kX86`, printf-format checking |
| Win32 | `win.hpp` | the only inclusion of `windows.h` |

The CPU kernels are per architecture, chosen by CMake (`cmake/target_arch.cmake`): AVX2 / AVX-512 on x86-64,
NEON with the dot product on Arm64 (`expert_neon.cpp`, `nvfp4_neon.cpp`, `router_q4_neon.cpp`, `ggml/ggml_cpu_dots_neon.c`,
`src/vision/gemm_cpu_neon.cpp`). Shared code calls an x86 kernel only inside `if constexpr (platform::kX86)`.

Programs that take paths carry a manifest (`src/platform/basalt.exe.manifest`) that makes the
process code page UTF-8. A path past the plain Win32 length limit is opened in its `\\?\` form, so it works with the
system's long-path policy on or off.

## Windows limitations

- Pinned memory is capped by the driver under WDDM. When the whole arena cannot be registered,
  the pinned slices are limited to the GPU's shared-memory budget (`RAM/2 - 8 GiB`), and the
  start-up note says what was capped.
- Large pages need the "Lock pages in memory" privilege, which is granted to Administrators by
  default; locking the resident expert copy needs "Increase a process working set", which is
  granted to Users, so it works without elevation. The engine enables whichever its token holds.
  Without large pages it falls back to 4 KiB pages and says so at start-up.
- Shared arena files and cgroup quotas do not exist. The engine refuses the first and reports no
  container limit.
- The n-gram table (`--ple-io direct`) is read through the file cache. NTFS completes the
  unbuffered reads of a file one at a time while the file is mapped or read through the cache,
  and the model file is both. Table pages the reads bring in are cached memory, which Windows
  hands back when a program needs it.
- `basalt serve` runs the Python server as a child and exits with its status; Linux replaces
  the process.
- A display-attached GPU is subject to the driver's timeout recovery (TDR). Raise `TdrDelay` under
  `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` if long kernels are reset.
- A card that also drives the desktop has free VRAM that moves by a few MiB between starts, so the
  auto-sized expert cache can land on a different slot count and greedy output can differ from run
  to run. For reproducible output (`tools/greedy_identity.py`), pin `expert-cache` and raise
  `vram-reserve-mib`.
- Quality has not been measured on Windows. No perplexity or KL-divergence run exists for it;
  [precision/builds.md](precision/builds.md) has the compiler-parity checks. There are no
  headline benchmark numbers for Windows.

Windows is supported with the help of the community. Report Windows problems as issues labelled
`windows`.

## Building on Windows

Run from a Developer command prompt (or call `vcvarsall.bat x64` first), so `cl.exe` is on `PATH`:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build -L cpu -j 8 --output-on-failure
```

If tests fail to create fixtures, point `TEMP` and `TMP` at a directory inside the build tree.
`cpu_quota_test` and `pool_affinity_test` exercise Linux-only surfaces and are not built.

CUDA 13.2 and later ship CCCL headers that `#error` under cl.exe's traditional preprocessor (C1189); the
CMakeLists hands `/Zc:preprocessor` to nvcc's host compiler, verified on CUDA 13.4 and MSVC 19.44.
Anything compiling these headers outside this CMakeLists needs the same flag.

## Windows on Arm (RTX Spark)

The N1X's GPU is integrated. Its memory is a carve-out of the machine's RAM that Windows does not see (the GPU memory
size the NVIDIA Control Panel sets), and past it a shared segment out of the RAM Windows uses; `cudaMemGetInfo`
reports both. Everything above applies, and:

- Every sizing decision reads `device_mem_info` (`include/basalt/core/device_memory.hpp`), which on an integrated GPU
  reports the carve-out as NVML gives it: the expert cache is sized like a discrete card's VRAM, and the RAM copy from
  Windows' RAM. A larger carve-out keeps more experts on the GPU; the start line
  `expert cache: the integrated GPU's dedicated memory` gives the figure. Where NVML reports no carve-out, the cache
  is sized from the RAM available, less the host's late memory and the file cache's floor.
- The carve-out is RAM Windows no longer has. With every expert on the GPU, a server still keeps page-locked copies of
  the experts in the cache slots its prompt path borrows, in the RAM left over beside the margin
  ([memory floors](running.md#memory-floors)), and on Windows a request stops once the RAM available falls below the
  guard. An explicit `resident-budget-gib` keeps no such copies: the borrowed slots are refilled from the file after a
  prompt, so a long prompt takes longer and Windows keeps that RAM.
- `nvidia-smi` lists the NPU as a second device with no CUDA compute capability: `gpu auto` skips it, and an unset
  `gpu` does not count it.
- The GPU reads page-locked host memory at addresses of its own, so the resident RAM copy is never unpinned.
- In Modern Standby (the screen off or the session locked, on a laptop that sleeps that way) the driver can move the
  GPU's memory to RAM and the pagefile, and a running engine stalls until the machine wakes. Keep it awake while it
  serves: no sleep and no screen timeout while plugged in.
- The CPU tier runs the NEON kernels, whose rows differ from the x86 ones by summation order
  ([precision/builds.md](precision/builds.md#the-arm64-build)). Quality has not been measured on Arm64.

## Building on Windows on Arm

Run from an Arm64 Developer command prompt (or call `vcvarsall.bat arm64` first), with the Arm64 CUDA toolkit:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j 12
ctest --test-dir build -L cpu -j 8 --output-on-failure
```

CMake picks the NEON kernels and compute capability 12.1. The x86-only kernel tests (`avx_bench`, `iq_avx2_test`,
`iq_avx512_test`, `nvfp4_cpu_rows_test`, `router_dot_parity`) are not built. Use an Arm64 Python for the server:
every package in `requirements.txt` has a `win_arm64` wheel.
