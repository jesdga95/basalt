# Contributing to Basalt

Basalt is a single-purpose engine: Qwen3.8-Flash-Next on Linux with NVIDIA Blackwell GPUs. Every rule below exists to keep it fast and correct on that one target.

## Scope

- **CUDA on Blackwell (sm_120) only.** No ROCm/HIP, no AMD, no Intel/SYCL/oneAPI, no Apple/Metal, no CPU-only main path (a CPU fallback is fine where the GPU path is the default, like the experts held in RAM or `--vision cpu`), and no older NVIDIA architectures (Hopper sm_90 and earlier). No compatibility layers or `#if` branches for them. Pull requests adding another backend or architecture will be closed.
- **Linux and Windows x64 are both supported; Linux is where we test.** CI compiles and CPU-tests Windows on every pull request. Windows runs on one GPU, with no performance claims, and the Windows limitations are in `docs/platforms.md`. Changes must not remove the platform seams (`src/platform/`, CMake `if(WIN32)`). Windows-only issues get a `windows` label. Headline benchmark numbers stay with the maintainer's Linux runs.
- **Engine code is platform-neutral.** Outside `src/platform/` and `include/basalt/platform/` there is no `_WIN32`, `_MSC_VER`, `__linux__`, `<windows.h>`, `<sys/mman.h>`, `<unistd.h>` or raw OS call (mmap, madvise, pread, pthread/affinity, fork/exec/pipe, dlopen, /proc or /sys reads). Call the platform layer; when a seam is missing, add it there with a Linux body and a Windows body, the Linux body being the code you would have written anyway, and a test in `tests/platform/` that runs on both. Paths are UTF-8 and go through the platform file calls. `tools/check_platform_boundary.sh` enforces this.
- **No `.bat` or PowerShell scripts** are added to the tree. A documented build recipe that calls the toolchain's own scripts (`vcvarsall.bat`, say) from a shell is not one.
- **One fast path.** No per-user tuning flags or alternative kernels selected at run time. A change that wins replaces the old path; a change that doesn't win isn't merged. A/B and diagnostic flags are off by default and are not documented as settings. User-facing settings are limited to model, context, multi-GPU and serving options.
- **No environment-variable knobs.** Everything the engine reads is a command-line flag or a config entry; diagnostics (profiling, timing, logs) are flags too, off by default.

## What every change must include

1. **A parity test** for every new or changed kernel: random inputs plus the real model geometry, compared against (a) the current Basalt kernel it replaces and (b) the plain CPU reference for that operation (dequantize, then accumulate in double precision). Report the max absolute and relative error against both. A kernel meant to be bitwise-neutral must match the frozen output hashes of the current kernels. The output for a token must not depend on the batch/window width or the token's position in it; the test must check that.
2. **A profile.** Give a microbench line for the kernel: µs per call and achieved GB/s (or TFLOP/s) against the bytes or FLOPs it must move. Give an end-to-end number from `basalt-bench`: ms per decode window and passes/s, not only tok/s, which drafting luck moves by ±3-5%.
3. **A quality check:**
   - Bitwise-identical output: show the greedy-identity check against the previous build. Pin the expert tiers, since changes in GPU module size move the auto-sized tiers and change rounding.
   - Changed numerics (new reduction order, precision, quantization): run the perplexity/KLD gate against the Q8_0 reference through the decode path. Report PPL, mean KLD and top-1 agreement next to the previous build's. Large changes need the full 100-chunk run, not just the 20-chunk quick gate.
4. **Benchmark hygiene:**
   - Every performance number states the GPU power limit it was measured at; the standard is 400 W.
   - A/B comparisons alternate the two builds in the same session. Numbers taken in different sessions drift and aren't comparable.
   - Keep warm-up and fixture order as `basalt-bench` defines them.

## Code style

- Kernels are written as `__device__` stage functions that take their block/warp index as an argument, with thin `__global__` wrappers, so they can be fused into persistent kernels later.
- Comments are brief and stateless: what the code does and why. No hardware names, measured numbers, dates or history in comments. Those belong in the commit message or the PR.
- Never use `cp.async.bulk.shared::cluster`. Use `cp.async.bulk.tensor.*.shared::cta` for TMA.
- Keep adapted third-party code's license headers intact (ggml/llama.cpp is MIT).

## Commits and pull requests

- Small, logical commits with the message style `area: what it does`. Each commit must build.
- Never amend or force-push shared branches; add new commits.
- No notes, scratch files or generated reports in the repo.
- Docs: no file over 300 lines; split a longer topic into a folder of files; keep docs short, stateless and free of duplication.
- A PR description lists: what changed, the parity results, the microbench and `basalt-bench` before/after (with the power limit), and the quality-check result.

## Other ways to help

- Cannot contribute code? Add a benchmark run from your hardware to the community table (details in [community runs](docs/community.md)):
  1. Run `basalt-bench --community --model PACK --submitter YOUR_NAME`. It prints one JSON row ([building](docs/building.md)).
  2. Add the row to the `runs` list in `docs/community/benchmarks.json`.
  3. Check it with `python3 tools/check_benchmarks.py docs/community/benchmarks.json`.
  4. Open a pull request; the `pages` workflow runs the same check.
  5. Once merged, the row appears on the runs page.
- Any Blackwell (sm_120) machine with spare hours helps; bigger setups help most: 96 GB+ VRAM (single large cards or multi-GPU), quad- or octa-channel DDR5.
- A few hours of remote access to a machine speeds up development.
- Or sponsor the project through GitHub Sponsors (the repository's Sponsor button): it pays for tokens for agentic development and for time on different hardware to test on.

## License

Basalt is MIT licensed and is based on Strata (MIT, Copyright (c) 2026 Niko1221 and the Strata contributors). By contributing you agree your contribution is licensed under the same terms. See `LICENSE` and `THIRD_PARTY_LICENSES`.
