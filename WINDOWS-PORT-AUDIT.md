# Basalt → Windows port: code audit & reusability report

**Date:** 2026-10-14 · **Basalt:** `92d9e2f` (452 files) at `E:\LM_SERV\basalt` · **Strata:** `E:\LM_SERV\strata-src` (v0.1.44-era, cross-platform) · **Infernix:** `E:\LM_SERV\infernix-src` (Windows-native)

## 1. Verdict

A native Windows build is a **~1.5–2.5 week project**, not 3–4 — because **Strata (Basalt's direct parent) already solved almost every Windows problem, file-for-file, in the same class and function shapes.** Basalt stripped those `#ifdef _WIN32` blocks out when it forked; the port is largely lifting them back and re-fitting them around Basalt's engine changes. Infernix is a second, independent production-grade reference for the same OS problems (it's Windows-first).

## 2. Basalt's Linux surface (complete grep inventory)

47 POSIX includes across 19 files. **Zero Windows awareness exists in Basalt's own code** (only vendored `stb_image.h`). The coupling is concentrated:

### Tier A — dedicated platform layer (2 files)
| File | POSIX surface | Strata's Windows solution (same file, same API) |
|---|---|---|
| `src/platform/direct_file.cpp` | `O_DIRECT` + `pread` + thread pool, `posix_memalign` | ✔ full IOCP impl: `FILE_FLAG_NO_BUFFERING \| FILE_FLAG_OVERLAPPED` + `CreateIoCompletionPort`, `VirtualAlloc` aligned buffers (12.8 KB vs Basalt's 4.9 KB — identical POSIX branch + added Windows branch) |
| `src/platform/memory.cpp` | `mlock`/`munlock`, `/proc/meminfo`, `madvise`/`posix_fadvise`, `sysconf` | ✔ `SetProcessWorkingSetSizeEx` + chunked `VirtualLock`, `GlobalMemoryStatusEx`, no-op readahead. Basalt's watchdog/ram-guard threads are pure `std::thread` — portable as-is |

### Tier B — mmap + file APIs in the core (same code family exists upstream)
| File | Use | Port path |
|---|---|---|
| `src/core/pinned.cu` | `mmap(MAP_HUGETLB)`/`MAP_HUGEPAGE` arena, `MAP_SHARED` shm file, `cudaHostRegister` | **Strata's `pinned.cu` already has the Windows branch**: `AdjustTokenPrivileges(SeLockMemoryPrivilege)` + `VirtualAlloc(MEM_LARGE_PAGES)` w/ fallback, per-layer pipelined `cudaHostRegister`. `cudaHostRegister` itself is OS-neutral. Mechanical merge around Basalt's slice logic |
| `include/basalt/artifact/gguf_reader.hpp` | `mmap` model file | **Strata's `gguf_reader.hpp` already `#ifdef _WIN32`** → `CreateFileMappingW`/`MapViewOfFile` (header even carries the `NOMINMAX` lesson learned) |
| `src/core/expert_source.cpp` | `mmap`/fd expert tiers | Strata's `expert_source.cpp` has 20+ Windows blocks to copy |
| `src/kernels/ngram.cpp` | `mmap` n-gram table | Strata's ngram is already dual-OS (and keeps `<windows.h>` out of headers via pimpl — a design note Basalt should steal) |
| `src/vision/mmproj.cpp`, `src/core/model_file.cpp` | stat/open/mmap | trivial Win32 swaps |

### Tier C — threads/affinity
| File | Use | Port path |
|---|---|---|
| `src/kernels/cpu/pool.cpp` + `pool_affinity_linux.hpp` | `pthread_setaffinity_np`/`sched_setaffinity`, cgroup quota probe | **Strata ships `pool_affinity_win.hpp`** (`SetThreadGroupAffinity`, hybrid-core detection on Windows) |
| cgroup quota (`cpu_quota.cpp`) | `/sys/fs/cgroup` | no-op on Windows (fine) |

### Tier D — loose ends
`src/program/generate.cpp` (only **4** POSIX includes: unistd/stat/pthread/sched — the platform discipline held even in the 590k-line monolith; Strata's generate.cpp has the same sites already guarded, ~25 `#ifdef _WIN32` blocks to mirror), `bench_main.cpp`, `prefill.cpp` (`sysconf`→`GetSystemInfo`), `ppl_gate.hpp`, test files (GNU `-Wl,--wrap` etc. → exclude on Windows).

**Python server:** `requirements.txt` is fully cross-platform (numpy/jinja2/regex/pyyaml/tqdm/pillow/psutil). Only path/subprocess polish expected.

## 3. Toolchain risk — downgraded from "hard" to "done before"

- My earlier concern (nvcc needs cl.exe; ggml's "GNU extensions" break MSVC) **largely evaporates**:
  - **Strata builds Windows/MSVC + CUDA today** — its CMake has MSVC runtime pinning (`/MT` static runtime for portable builds), per-language flag guards so nvcc never sees `/W4`, AVX-512/VNNI/VBMI wired for MSVC (`/arch:` + target pragmas). It's the proof this toolchain combination works on the same code family.
  - Basalt's extracted `ggml_quants.c` / `ggml_cpu_dots.c` grep **clean of statement-expressions and designated initializers** — ANSI-C-ish; modern MSVC (VS2019 16.10+) supports the AVX-512 intrinsics they use. Worst case: rename the two `.c` files to `.cpp` and build them under the C++ frontend.
  - Fallback toolchain (CUDA-clang device mode) if anything proves stubborn.
- CUDA 13 + sm_120 is OS-neutral; cuBLAS/cudart ship on Windows.

## 4. What Strata/Infernix contribute (the actual answer to "have these issues been addressed?")

| Problem | Strata (parent fork — highest reuse value) | Infernix (independent Windows-native engine) |
|---|---|---|
| Huge-page pinned arena | ✔ MEM_LARGE_PAGES + privilege enabling, honest fallback notes | ✔ `host_memory.cpp`: VirtualAlloc arena |
| Unbuffered async disk reads | ✔ IOCP DirectFile (same submit/wait API as Basalt's) | ✔ `direct_read_queue.cpp` (overlapped + NO_BUFFERING) |
| RAM-availability guards | ✔ GlobalMemoryStatusEx (+ DXGI VRAM budget query Basalt doesn't have yet) | ✔ `GlobalMemoryStatusEx` |
| mmap model/expert files | ✔ `_WIN32` in gguf_reader/expert_source/ngram | ✔ `read_only_file_win32.cpp` (CreateFileMappingW) |
| Core-affinity pool | ✔ `pool_affinity_win.hpp` | n/a |
| MSVC+CUDA CMake patterns | ✔ entire CMakeLists is the template | ✔ `CMakePresets.json`, `build_native.bat` |
| Proof the *architecture* runs fast on Win | ✔ community: 5090 @ 193.8 tok/s (Strata, Windows) | ✔ 5090 + **192 GB** box @ 125–137 tok/s (llamaperf reports) |

## 5. Realistic effort plan

| Step | Effort | Notes |
|---|---|---|
| Port Tier A (platform layer) from Strata | 1–2 d | rename `strata::`→`basalt::`, keep Basalt's watchdog additions |
| Tier B mmap ports | 2–3 d | expert_source + model_file + mmproj + ngram; pinned.cu merge is the trickiest 300 lines |
| Tier C pool affinity | 1–2 d | copy `pool_affinity_win.hpp`, wire `PoolAffinity` detection |
| Tier D + CMake + tests exclusion | 2–3 d | mirror Strata's guarded blocks in generate.cpp; CMake from Strata's MSVC patterns |
| First-token + bring-up | 1–2 d | expect dequant/AVX flag surprises, not architecture surprises |
| Performance parity + soak | 3–5 d | validate large-page arena + cudaHostRegister at 60–100 GB pinned; benchmark vs 585/316 single-5090 baseline |
| **Total** | **~10–17 working days** | someone who knows CUDA + Windows memory APIs; I'd budget 3 weeks calendar |

## 6. Risks & strategic note

1. **Upstream posture:** Basalt's author explicitly single-targets Linux+one-GPU; a PR is unlikely to merge — expect fork ownership. Mitigation: keep the port as a small guarded diff (all of it sits in ~15 files), which makes rebasing on Basalt commits cheap. Basalt is itself the faster-kernel fork of Strata (2.6× on Blackwell); the *platform layer* is Strata's solved problem, the *kernels* are Basalt's.
2. **API drift:** Basalt added functions Strata lacks (`available_memory` MemAvailable equivalent, `ram_budget`, watchdog/ram-guard, `watch_resident_ram`). Windows bodies are trivial (GlobalMemoryStatusEx — Strata already shows how) but they must be **written**, not copied.
3. **Windows pinned-arena scale:** 60–100 GB `cudaHostRegister` + large pages under WSL2 has known caps; native avoids them, but Strata's per-slice registration fallback is exactly why this risk is bounded — copy that fallback chain verbatim.
4. **Peer/multi-GPU** (`--gpu 0,1`): keep it behind a runtime check; single 5090 is the benchmark-matched path anyway (585/316 vs 665/354).
5. **Not covered by any published data:** Basalt + Q4_K_XL fully resident — this machine (5090 + 192 GB) would be the reference run.

## 7. Recommended sequencing on this machine

1. **Now:** run Basalt under WSL2 (hours, not weeks) → validates packs, gets your baseline tok/s, proves RAM-floor behavior.
2. **In parallel or after:** the native port above — you now have Strata's Windows branch as the answer key and Infernix-src as the cross-check.
3. Publish your single-5090+192 GB numbers (nobody's posted IQ3_S-resident or Q4_K_XL-resident yet).
