# Platforms

Linux x64 and Windows x64, NVIDIA Blackwell (sm_120). Linux is where the engine is tested and benchmarked. CI builds both and runs
the CPU tests on each. The GPU tests and the quality gate need a card.

The main card is always sm_120, with one exception in the multi-GPU tier: a second card of sm_89 (Ada) can hold and
compute the expert tier ([GPUs](configuration/gpus.md#an-older-peer-card)); every other path and every other
architecture is Blackwell-only.

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
model download on: `bin/basalt serve --config basalt_config.json`. `bin/basalt-bench` starts the `basalt` beside it and
reads the archive's `data/bench`.

## Container image

`Dockerfile` builds the engine, the server and the server's venv into one image on a CUDA 13 base: the release archive's tree,
with the venv already installed. Building it needs no GPU.

```sh
docker build -t basalt .
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -v "$PWD/appdata/basalt:/data" -v "$PWD/models:/models" -p 8080:8080 basalt
```

`/data` holds `basalt_config.json` (v2; `tools/migrations/config_v1_to_v2.py` converts an older file) and takes the server's
folder per model under `~/.cache/basalt`, where `HOME` points: its engine log and, when the conversation disk tier is on, that
tier. A pack is read where it lies, so the packs go on `/models` and the config names them by their mount path
(`/models/<pack>.basalt`); paths in the config are relative to the config file. The entrypoint is
`basalt serve --config /data/basalt_config.json --host 0.0.0.0` and arguments after the image name follow it (`--api-key KEY`).
The served port is `server.port` (8080) and the mapped host port has to match it; `--gpu` names the cards the container sees.

The container path is untested: no `basalt-bench` run and no PPL/KLD quality run has been measured inside the image, so the
benchmark numbers describe native runs.

## Platform code

OS code lives only in `src/platform/` and `include/basalt/platform/`, and CMake `if(WIN32)` blocks. Anywhere else there is no `_WIN32`,
`_MSC_VER`, `__linux__`, `<windows.h>`, `<sys/mman.h>`, `<unistd.h>` or raw OS call (mmap, madvise, pread, pthread/affinity,
fork/exec/pipe, dlopen, /proc or /sys reads). `tools/check_platform_boundary.sh` enforces it. Each seam has a Linux and a Windows body,
the Linux body being the code you would have written anyway, and a test in `tests/platform/` that runs on both. Paths are UTF-8 and go
through the platform file calls.

## FFmpeg for video

Video input loads FFmpeg's shared libraries at run time (libavformat, libavcodec, libavutil and libswscale of release 6, 7 or 8; nothing
is linked or shipped). Linux: the distribution's runtime packages (`libavformat62` and its siblings on Ubuntu 26.04, `libavformat60`
on 24.04), found by the system loader. Windows: a shared FFmpeg build (`avformat-62.dll` and the rest, from gyan.dev or BtbN) beside
`basalt.exe` or in a `PATH` directory; the current directory is never searched. Without them video is off and the server says why.
GPU decoding needs an FFmpeg build with NVDEC (the distribution packages and the gyan.dev and BtbN builds have it); without it
frames decode on the CPU.

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
