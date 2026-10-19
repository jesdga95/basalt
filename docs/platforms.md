# Platforms

Basalt runs natively on Linux x64 and Windows x64, on NVIDIA Blackwell (sm_120). Linux is where
the engine is tested and benchmarked. CI builds both and runs the CPU tests on each. The GPU
tests and the quality gate run on a machine with a card.

## Requirements

| | Linux | Windows |
| --- | --- | --- |
| OS | x86-64, AVX2 (AVX-512 used when present) | Windows 10 1903 or later, x64 |
| Compiler | C++20 (GCC or Clang) | Visual Studio 2022 (MSVC 19.44) |
| CUDA | 13 | 13 |
| Build | CMake 3.24+ | CMake 3.24+ and Ninja |
| Python | 3.10+ | 3.10+ |
| GPU | NVIDIA Blackwell (sm_120); optional second GPU as a peer expert tier | NVIDIA Blackwell (sm_120), one GPU |

## Platform code

OS-specific code lives in `src/platform/` (declared in `include/basalt/platform/`) and in CMake
`WIN32` blocks. Engine code outside them names no OS or compiler.
`tools/check_platform_boundary.sh` enforces this. Every seam has a Linux body and a Windows body
behind one declaration, and a test in `tests/platform/` that runs on both.

| seam | header | wraps |
| --- | --- | --- |
| file I/O | `file.hpp` | UTF-8 paths, open, positional reads, unbuffered descriptors |
| file views | `mapped_file.hpp` | mapped files, prefetch and release hints, locking |
| memory | `memory.hpp` | RAM probes, page locking, aligned and huge-page allocation |
| CPU | `cpu.hpp` | topology, thread affinity |
| processes | `process.hpp` | spawn, wait, terminate, pipes, signals |
| system | `system.hpp`, `dynlib.hpp` | process memory, disk queries, the NVML loader |
| compiler | `compiler.hpp` | CPUID, bit scan, spin hint, fp16 rounding |
| Win32 | `win.hpp` | the only inclusion of `windows.h` |

Programs that take paths carry a manifest (`src/platform/basalt.exe.manifest`) that makes the
process code page UTF-8 and lifts the path length limit.

## Windows limitations

- One GPU. The peer and multi-GPU flags stop the engine at start-up.
- Pinned memory is capped by the driver under WDDM. When the whole arena cannot be registered,
  the pinned slices are limited to the GPU's shared-memory budget (`RAM/2 - 8 GiB`), and the
  start-up note says what was capped.
- Large pages need the "Lock pages in memory" privilege, and locking the resident expert copy
  needs "Increase a process working set". Grant both under Local Security Policy, User Rights
  Assignment. Without them the engine falls back to 4 KiB pages and says so at start-up.
- Shared arena files and cgroup quotas do not exist. The engine refuses the first and reports no
  container limit.
- `basalt serve` runs the Python server as a child and exits with its status; Linux replaces
  the process.
- A display-attached GPU is subject to the driver's timeout recovery (TDR). Raise `TdrDelay` under
  `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` if long kernels are reset.
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
