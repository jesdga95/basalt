# Contributing to Basalt

Basalt is an engine for mixture-of-experts models on NVIDIA Blackwell GPUs. Every rule below exists to keep its models fast and correct on that one target.

## Scope

- **CUDA on Blackwell (sm_120) only.** No ROCm/HIP, no AMD, no Intel/SYCL/oneAPI, no Apple/Metal, no CPU-only main path (a CPU fallback is fine where the GPU path is the default, like the experts held in RAM or `--vision cpu`), and no older NVIDIA architectures (Hopper sm_90 and earlier). No compatibility layers or `#if` branches for them. Pull requests adding another backend or architecture will be closed.
- **Linux and Windows x64 are both supported.** CI compiles and CPU-tests Windows on every pull request. Windows-only issues get a `windows` label.
- **Engine code is platform-neutral.** No OS or compiler names and no raw OS calls outside the platform layer; a missing seam is added there with a Linux and a Windows body. See [platforms](docs/platforms.md).
- **Engine log lines carry their level.** A line to stderr goes through `include/basalt/log.hpp` (`basalt::log::info`, `warn` or `error`), which writes `basalt <area>: [warning: |error: ]<text>`; the server and the terminal read the level from that marker and never from the words. A warning is a fallback or a lost function, an error is a failure or a refusal, everything else is info. `tools/check_stderr_logging.sh` fails when engine code names stderr anywhere else. Lines are terse facts (what was decided, its value, at most the flag to change it), and the text of lines the server parses does not change.
- **No run scripts are shipped.** Running, serving, benchmarking and converting go through Basalt's own executables, never a shell, `.bat` or PowerShell wrapper. Repository checks under `tools/` may be scripts.
- **One fast path.** No per-user tuning flags or alternative kernels selected at run time. A change that wins replaces the old path; a change that doesn't win isn't merged. A/B and diagnostic flags are off by default and are not documented as settings. User-facing settings are limited to model, context, multi-GPU and serving options.
- **No shims and no dead code.** A removed or renamed flag, option, field or code path is simply gone: no alias, no fallback, no "was retired" message. The normal unknown-flag error covers it. Code nothing uses is deleted in its own commit.
- **Defaults are measured.** A default, or an automatic pick that replaces one, changes only after a measured run on hardware where it is picked.
- **No environment-variable knobs.** Everything the engine reads is a command-line flag or a config entry; diagnostics (profiling, timing, logs) are flags too, off by default.
- **Decode comes first.** A prefill gain never costs decode speed; every prefill change shows a decode number too.

## Code structure

- **Shared code is model-agnostic.** It sees a model only through the adapter, a model's own code stays in `src/models/<model>`, and a change for one model leaves the others' output and speed as they were. See [adding a model](docs/adding-a-model.md).
- **Logic exists once.** What two models need lives in shared code and takes each model's numbers as parameters. Follow the existing pattern before adding a new one.
- **One serving loop** runs every request of every model; a model plugs into it and never forks it. See [persistent serving](src/program/serve/README.md).
- **The engine does the work, the server serves.** Tokenizing and per-request work run in C++. `serve/` parses, routes and streams, names no model, and never repairs model output; constrained decoding (grammar) handles structure.
- **Build structure: no slow TU.** A clean Release build must finish in under 10 minutes on 32 threads, and nvcc compiles a translation unit on one core, so the build takes as long as its slowest `.cu`. Keep every `.cu` to about a minute or two alone: a kernel that is instantiated in several variants (like `dm_peer_kernel`) gets one `.cu` per variant, behind a getter that returns its address (no relocatable device code), and shared device code goes in a header with no kernel in it. A `.cu` does not include another `.cu` (`tools/check_cu_includes.sh` fails on a new one; `tools/cu_includes.allow` lists the few that must inline). Time a new or changed `.cu` alone (`nvcc --time`) and report it with the change.

## What every change must include

1. **A parity test** for every new or changed kernel: random inputs plus the real model geometry, compared against (a) the current Basalt kernel it replaces and (b) the plain CPU reference for that operation (dequantize, then accumulate in double precision). Report the max absolute and relative error against both. A kernel meant to be bitwise-neutral must match the frozen output hashes of the current kernels. The output for a token must not depend on the batch/window width or the token's position in it; the test must check that.
2. **A profile.** Give a microbench line for the kernel: µs per call and achieved GB/s (or TFLOP/s) against the bytes or FLOPs it must move. Give an end-to-end number from `basalt-bench`: ms per decode window and passes/s, not only tok/s, which drafting luck moves.
3. **A quality check:**
   - Bitwise-identical output: show the greedy-identity check against the previous build. Pin the expert tiers, since changes in GPU module size move the auto-sized tiers and change rounding.
   - Changed numerics (new reduction order, precision, quantization): run the perplexity/KLD gate against the BF16 reference ([docs/precision/quality.md](docs/precision/quality.md#the-bf16-reference)) through the decode path. Report PPL, mean KLD and top-1 agreement next to the previous build's. Large changes need the full 100-chunk run, not just the 20-chunk quick gate. The `docs/precision/` page that covers the change is updated in the same change, with fresh numbers where it quotes them.
4. **Benchmark hygiene:**
   - Every performance number states the GPU power limit it was measured at.
   - A/B comparisons alternate the two builds in the same session (ABBA), with the expert tiers pinned to the same placement in both. Numbers taken in different sessions drift and aren't comparable.
   - Numbers are cold first runs of the real engine, at a placement real users get. A gain that only exists with experts spilled to the SSD is not a headline.
   - A claim that something can't get faster comes with a per-step breakdown against the measured hardware limit.
   - Keep warm-up and fixture order as `basalt-bench` defines them.

## Code style

- Kernels are `__device__` stage functions that take their block/warp index as an argument, with thin `__global__` wrappers, so they can be fused into persistent kernels. Never use `cp.async.bulk.shared::cluster`; use `cp.async.bulk.tensor.*.shared::cta` for TMA.
- Comments are brief and stateless: what the code does and why. No hardware names, measured numbers, dates or history in comments. Those belong in the commit message or the PR.
- Keep adapted third-party code's license headers intact (ggml/llama.cpp is MIT).

## Commits and pull requests

- Small, logical commits with the message style `area: what it does`. Each commit must build. A small change across areas is one commit.
- No personal names or serving configurations in commit messages, comments or docs.
- Ported or adapted work keeps its original authors as `Co-Authored-By` lines.
- Never amend or force-push shared branches; add new commits.
- No notes, scratch files or generated reports in the repo.
- Docs: no file over 300 lines; split a longer topic into a folder of files; keep docs short, stateless and free of duplication. Web UI text uses proper casing ("Basalt") and no em dashes.
- A PR description lists: what changed, the parity results, the microbench and `basalt-bench` before/after (with the power limit), and the quality-check result.

## Other ways to help

- Cannot contribute code? Add a benchmark run from your hardware to the community table (details in [community runs](docs/community.md)):
  1. Run `basalt-bench --community --model PACK --submitter YourName`. It adds the row to `docs/community/benchmarks.json` ([building](docs/building.md)).
  2. Review the row. After a hand edit, format and check it with `python3 tools/check_benchmarks.py --fix docs/community/benchmarks.json`.
  3. Commit it.
  4. Open a pull request; the `community` workflow runs the same check.
  5. Once merged, the row appears on the [runs page](https://basalt.jgulfo.com/community/).
- Any Blackwell (sm_120) machine with spare hours helps; bigger setups help most: 96 GB+ VRAM (single large cards or multi-GPU), quad- or octa-channel DDR5.
- A few hours of remote access to a machine speeds up development.
- Or sponsor the project through GitHub Sponsors (the repository's Sponsor button): it pays for tokens for agentic development and for time on different hardware to test on.

## License

Basalt is MIT licensed and is based on Strata (MIT, Copyright (c) 2026 Niko1221 and the Strata contributors). By contributing you agree your contribution is licensed under the same terms. See `LICENSE` and `THIRD_PARTY_LICENSES`.
