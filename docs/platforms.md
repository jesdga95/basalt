# Platforms

Linux x64 and Windows x64, NVIDIA Blackwell (sm_120). Linux is where the engine is tested and benchmarked. CI builds both and runs
the CPU tests on each. The GPU tests and the quality gate need a card.

| | Linux | Windows |
| --- | --- | --- |
| OS | x86-64, AVX2 (AVX-512 used when present) | Windows 10 1903+ x64 |
| Compiler, build | GCC 13+, CMake 3.24+ | MSVC 19.44 (VS 2022), CMake 3.24+, Ninja |
| CUDA, Python | 13, 3.10+ | 13, 3.10+ |

## Release archives

Each release carries `basalt-vX.Y.Z-linux-x64.tar.gz` and `basalt-vX.Y.Z-windows-x64.zip`: `bin/` (`basalt`, `basalt-bench`,
`basalt-probe`, `basalt-vision`), `serve/`, `tools/` (what the server imports), `data/`, `docs/`, `requirements.txt` and the licences.
They hold no CUDA libraries and no Python environment. Needs:

- An NVIDIA driver of the R580 branch or newer. The archives are built with CUDA 13.0.2, whose Linux minimum is 580.95.05; NVIDIA
  publishes no Windows minimum since 13.0, so use a current driver there.
- The CUDA 13 runtime libraries (cudart, cuBLAS) on the library path (`PATH` on Windows).
- Python 3.10+ with the requirements in a `.venv` at the archive's top directory, which `bin/basalt serve` uses when it is there:

```sh
cd basalt-vX.Y.Z-linux-x64                   # tar xzf the archive first; on Windows unzip it
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

On Windows: `py -3 -m venv .venv` then `.venv\Scripts\pip install -r requirements.txt`. Then follow the README's quick start from the
model download on: `bin/basalt serve --config basalt_config.json`. Run `basalt-bench` from the top directory (it reads `data/bench`).

## Platform code

OS code lives only in `src/platform/` and `include/basalt/platform/`, and CMake `if(WIN32)` blocks. Anywhere else there is no `_WIN32`,
`_MSC_VER`, `__linux__`, `<windows.h>`, `<sys/mman.h>`, `<unistd.h>` or raw OS call (mmap, madvise, pread, pthread/affinity,
fork/exec/pipe, dlopen, /proc or /sys reads). `tools/check_platform_boundary.sh` enforces it. Each seam has a Linux and a Windows body,
the Linux body being the code you would have written anyway, and a test in `tests/platform/` that runs on both. Paths are UTF-8 and go
through the platform file calls.

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
