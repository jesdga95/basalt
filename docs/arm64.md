# Arm64 and the integrated Blackwell GPU

Basalt builds for Arm64 Linux with an integrated Blackwell GPU of compute capability 12.1: the GB10 of the DGX Spark and the
machines built on it. A build is for one CPU architecture and one compute capability. x86-64 with 12.0 is the default and is
described in [platforms](platforms.md). Windows on Arm is not supported.

## Building

Needs CUDA 13, GCC 13+ and CMake 3.24+, as on x86.

```sh
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build -j
ctest --test-dir build -L cpu -LE avx512 -j 8 --output-on-failure
```

`CUDACXX` names the toolkit's own `nvcc`: the `nvcc` of a distribution's CUDA package can stop on an SVE declaration in
`math-vector.h` from the system's glibc. CMake picks Arm64 from the machine and defaults the architecture to 121 there, so
`-DCMAKE_CUDA_ARCHITECTURES=121` only says it. `-DCMAKE_CUDA_ARCHITECTURES=120` on an Arm64 host builds for a discrete card.
A binary holds the code of one capability and refuses a card of another at start.

The tests that need a GPU run on the machine as they do on x86. The AVX tests and benches of the x86 kernels are not built on Arm64.

## What an Arm64 build changes

- **CPU kernels are NEON.** The dot products and activation quantizers of every format a pack uses are generated from llama.cpp's
  Arm code (`tools/port/extract_ggml_neon.py`), NVFP4, MXFP8 and Q2_0 rows are Basalt's own, and the vision encoder's CPU product folds its
  lanes in the AVX2 file's order. They need the Armv8.2 dot product (`-march=armv8.2-a+dotprod`).
- **No fused multiply-add the source did not ask for.** The build sets `-ffp-contract=off`, so a scalar reference rounds as it does
  on x86; the kernels ask for the FMAs they want.
- **Memory ordering.** x86 keeps loads and stores in order; Arm64 does not. The platform layer (`compiler.hpp`) has `store_fence`,
  `load_fence`, `prefetch` and `cpu_relax` (`isb`); a host thread that polls a word the GPU or another thread wrote takes a load
  fence before it reads what the word covers, and one that publishes takes a store fence first.
- **Synchronous copies.** On an integrated GPU a `cudaMemcpy` from pageable memory can return before the bytes reach the device, so
  every synchronous copy of the engine goes through `core/device_copy.hpp`, which synchronizes the default stream after it.

## Memory

The CPU and the GPU share one pool of memory. What `cudaMemGetInfo` reports for an integrated GPU follows the system's free
memory, which moves with the page cache and with whatever else runs, so the VRAM the planner sees is not a fixed card size. At
start Basalt logs one line per integrated card:

```
basalt: gpu 0 is integrated: cudaMemGetInfo <free> MiB free of <total> MiB; host RAM <avail> MiB available of <total> MiB; --vram-reserve-mib <n>
```

The planner sizes the weights and caches from the free figure and keeps the reserve clear of it. Anything the RAM tiers hold comes
out of the same pool, so `--vram-reserve-mib` and `--ram-reserve-mib` together are the room left for the rest of the system. The
reserve has no integrated-GPU default: set it by hand.

## Limits

- Linux only. The platform seams have no Windows-on-Arm body.
- The GPU kernels are the ones of the discrete card, compiled for 12.1; nothing is tuned for the integrated GPU's memory.
- Output of the NEON kernels is not bit-identical to the x86 kernels: greedy tokens can differ from an x86 build of the same pack.
