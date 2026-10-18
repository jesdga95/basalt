# Basalt on Windows

Native Windows build of the engine: the C++/CUDA engine and the Python server, one RTX 5090
(sm_120). This is a platform port - no kernel math, no weight format and no precision behaviour
changed. Windows support is community-supported: the engine logic is shared with Linux, which
stays the regression oracle, and OS-specific code belongs to `src/platform/` and CMake `WIN32`
blocks rather than to engine logic.

## Toolchain

The versions this port builds with:

| component | version |
| --- | --- |
| Windows | 11 (26100), x64 |
| Visual Studio | Community 2022 (17.14) + Build Tools |
| MSVC | 19.44.35222, toolset 14.44.35207 |
| CUDA Toolkit | 13.0 (V13.0.88) |
| NVIDIA driver | 616.92 |
| CMake | 4.1.0-rc4 |
| Ninja | 1.13.0 (kitware jobserver build) |
| Python | 3.12.10 for the server; 3.11 also works |

CUDA architecture is fixed at 120 (Blackwell), as elsewhere in the project.

## Building

Open a **Developer command prompt** (or wrap any command in a `vcvarsall.bat x64` call - the
compiler is not on `PATH` by itself):

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
```

The configure step must run inside that developer environment: CMake probes the compiler, and
`nvcc`'s host compiler has to be the `cl.exe` that `vcvarsall` put on `PATH`.

Ninja's worker pool uses a named pipe. Anything that confines a process so it cannot create one
(an antivirus filter driver, a sandboxed shell, a restricted token) makes the build stop at
"Detecting C compiler ABI info" with no error. A bare `cl.exe` still works there, so the symptom
is confusing: run the build from a normal shell if configure hangs.

### Running the tests

```bat
set TEMP=E:\path\to\basalt\build\tmp
set TMP=E:\path\to\basalt\build\tmp
ctest --test-dir build -j 8 --output-on-failure
```

Several tests write fixtures through `std::filesystem::temp_directory_path()`. If the machine's
temporary directory is denied to the test process - which a security product can arrange - the
fixtures fail to open and the tests report a creation error rather than a numeric one. Pointing
`TEMP`/`TMP` into the build tree avoids that.

Two tests are not built on Windows at all, because they exercise a Linux-only surface: the cgroup
CPU-quota probe (`cpu_quota_test`) and the Linux affinity backend (`pool_affinity_test`).

### When a failure is the machine and not the code

Most of what looks like a test failure on Windows is the environment. Two checks separate them
before any debugging:

- **Run the test binary from a directory outside the build tree.** A filtered token, a security
  product's folder rule or a working-directory restriction can deny file creation to a child
  process. The same binary in the same session then passes from another directory, which is the
  proof that the code is fine. `gguf_reader_test`, `gguf_split_test`, `model_file_test`,
  `expert_profile_save_test`, `native_dense_ple_key_test`, `file_expert_source_test`,
  `ple_reader_selftest` and `dec_gdn_parity` all behave this way.
- **Read the mechanism, not the verdict.** `platform_memory_test` prints its own diagnosis when it
  fails: `lock_resident: ok=0 ... SetProcessWorkingSetSizeEx failed (error 1314); VirtualLock
  stopped at 0 of 256 MiB (error 1453)`. Those are token-privilege errors, not parity errors, and
  the test passes where the privileges are held.

Two tests need an `experts.bin` fixture in the working directory that is a developer's local file
and is not in the repository: `expert_parity` and `pool_test` fail with "cannot read expert 0
(layer 0) from experts.bin" until one is supplied with `--file`.

The real-weight tools (`ple_q5_parity`, `expert_parity --file`, `pool_test --file`) want a **raw
GGUF or a raw `experts.bin`**, not a `.basalt` container. Handing them a container makes them
decode arbitrary bytes: they either report nonsense or fault. `ple_q5_parity` exits with
`0xc0000409` that way.

## Memory residency

Two different Windows limits matter, and they need two different privileges. Both are commonly
assigned to the account and left **disabled** in the token of an interactive process; the engine
enables them in its own token, but the right still has to be granted to the account, through
*Local Security Policy -> User Rights Assignment*:

- **Lock pages in memory** (`SeLockMemoryPrivilege`) - required for the large-page arena.
- **Increase a process working set** (`SeIncreaseWorkingSetPrivilege`) - required to raise the
  working-set minimum that `VirtualLock` counts against.

What the engine reports is the honest answer, not an assumption:

- The resident expert copy is filled by copying from the file mapping. Linux fills it with unbuffered
  reads straight into the arena; this platform offers no unbuffered descriptor for that fill
  (`platform::open_unbuffered` says so), so the bytes arrive through the mapping and the cache
  manager does the reading. It is the same bytes by the longer road: the copy pays a file-cache trip
  the unbuffered read avoids, and in exchange the mapped-view read-ahead described below does apply
  to this tier here.
- That copy's arena is ordinary aligned memory, and the driver is asked to register it once it is
  filled, rather than the memory being allocated page-locked before the first read. When the driver
  refuses, the same bytes fall back to a raised working-set minimum plus chunked `VirtualLock`, and
  the startup note says which of the two the run got.
- The resident expert copy is locked through a raised working-set minimum plus chunked
  `VirtualLock`. A quota that cannot be raised is reported in the note, and the lock is attempted
  anyway: a partially locked region is still better than a pageable one.
- The pinned arena asks for `MEM_LARGE_PAGES` first, like `hugetlb` on Linux. On this machine the
  large-page minimum is detected (2 MiB) and the privilege enables, but the system's large-page
  pool declines (`ERROR_NO_SYSTEM_RESOURCES`, 1450): drivers and the OS take that pool at boot, so
  whether it is available is a machine property. The arena then falls back to ordinary 4 KiB pages
  and `cudaHostRegister` still pins all of it: a 4 GiB arena registers in a
  single piece, so the per-slice fallback stays unused. The note names the size, the large-page
  minimum and the `VirtualAlloc` error, because a declined pool (1450), a missing privilege (1314)
  and a size that is not a multiple of the minimum (87) need different fixes and look alike
  otherwise.
- Never assume the backing from the code path: read the note the engine prints at startup. The
  difference between large pages and 4 KiB pages is more than a factor of two in TLB reach.

`ReadFile` with `FILE_FLAG_NO_BUFFERING` requires 512-byte multiples, which is why the direct-file
reader keeps its 4 KiB request contract rather than relaxing the alignment asserts.

The pinned arena has a second guard. If the whole arena cannot be registered, the slices are pinned
up to the GPU's shared-memory budget (`RAM/2 - 8 GiB`) rather than until the driver refuses one:
page-locked memory the GPU maps is charged to the device's shared segment, and filling that segment
makes later device allocations fail. The note says what was capped and to how much. The arena is
reserved and then committed in 1 GiB chunks, so a refusal names the size the commit reached and the
commit charge still free (RAM plus page file): a page-locked arena needs RAM and page file, and a
machine with a small page file runs out of commit before it runs out of RAM.

## Platform differences that are visible from outside

These are the places where the platform layer gives out and the engine says so rather than
pretending otherwise:

- **Peer-GPU expert tiers** are refused at start-up: the peer and multi-GPU flags stop the engine on
  Windows rather than being ignored. The port targets one GPU, as the Linux baseline does.
- **GPU recovery timeout (TDR)**: on a machine with a display attached, timeout detection and
  recovery resets the GPU when one submission runs past the limit (two seconds by default), failing
  long kernels. The `TdrDelay` (seconds) and `TdrLevel` (`0` disables recovery) values under
  `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` set that behaviour; a GPU driving no
  display is off the desktop windowing path and is not reset the same way, so kernel length that is
  safe on one machine is not on another.
- **Shared arena files** are refused with a note. They exist to populate a `tmpfs` mapping shared
  between processes, and there is no counterpart to that arrangement here.
- **Per-range release of mapped pages** (`madvise(MADV_DONTNEED)` / `posix_fadvise`) is a no-op.
  A Windows file mapping is released as a whole, and the pages of a read-only view are clean, so
  the OS can drop them on its own.
- **Read-ahead of a mapped file** is a range hint to the cache manager (`PrefetchVirtualMemory`), so
  experts read through the mapped expert file are prefetched there as they are with `madvise` on
  Linux. The descriptor form (`posix_fadvise`) has no Windows counterpart and asks for nothing:
  reads through a handle are cached reads, and the cache manager reads ahead around a sequential
  reader itself.
- **cgroup CPU and memory quotas** do not exist here. The engine reads the host's own numbers
  (`GlobalMemoryStatusEx`) and reports no container limit.
- A file mapping view must start at an allocation-granularity offset, so the expert tiers map the
  whole file and address the expert region inside the view, instead of mapping just that range.
- **A path is UTF-8 inside the engine and UTF-16 at the OS.** Every file the engine opens goes through
  `src/platform/file.cpp`, which converts the path and calls the wide entry point (`_wfopen`, `_wopen`,
  `CreateFileW`, `_wfullpath`, `_wchdir`, `GetModuleFileNameW`). The narrow calls (`fopen`, `_open`,
  `_access`) read the machine's ANSI code page, and `std::filesystem` converts a narrow path with it too,
  so a character outside that code page arrives as `?`. The wide opens are what the engine relies on; the
  manifest below is what makes the narrow spellings and `argv` agree with it.
- **The programs that name a path on the command line - the engine, the bench, the vision sidecar and the
  file tests - carry an application manifest** (`src/platform/basalt.exe.manifest`, merged into the image by
  `link.exe` with `/MANIFEST:EMBED` and `/MANIFESTINPUT:`): `activeCodePage` `UTF-8` and `longPathAware`.
  The code page needs **Windows 10 version 1903 (build 18362) or later**, and it makes the process's ANSI
  code page **UTF-8 (65001)**: `argv` and any narrow CRT call then read the
  same bytes the engine carries, and a pack path under a profile name with an accent survives being typed
  into a tool. On an older Windows the element is ignored and the machine's ANSI code page stays in force,
  so a character it cannot represent comes out of `argv` as `?` while the engine's own opens are unaffected,
  because those take the UTF-16 path. `longPathAware` needs **Windows 10 version 1607 (build 14393) or
  later** and lifts the 260-character limit for the wide opens; no `\\?\` prefix is used.
  `tests/platform/path_utf8_test.cpp` covers it: a directory name carrying an accented Latin letter,
  Japanese and Cyrillic opened through every file seam, a path of 323 characters, and `argv` as a child of
  the test sees it. Built **without** the manifest that test fails, and it fails unevenly, which is the
  failure this change is about: `argv` arrives with the accented letter intact (CP1252 has a byte for it)
  and the Japanese and the Cyrillic replaced by `?`; `std::filesystem` then makes one directory while
  `CreateFileW` is asked for another, so the tool reports "cannot open"; and the 323-character path is
  refused outright.
- **`basalt serve` does not replace the process.** There is no `exec` here, so the Python server is started as a
  child, this process waits for it, and it ends with the child's exit code: a supervisor watching `basalt serve`
  sees the server's status rather than the wrapper's. On Linux the process is replaced.
- `off_t` is 32 bits in MSVC. Every read that can land past 4 GiB uses an explicitly 64-bit offset.

## Compiler notes worth knowing

The MSVC behaviour that differs from the GCC build, and what the port does about it:

- `windows.h` defines `min`/`max` macros that break `std::min`/`std::max` in every file that
  includes it, so every inclusion in this tree defines `NOMINMAX` first, including the widely
  included headers before they pull Win32 types in.
- `/arch:AVX2` defines `__AVX__` and `__AVX2__` but **not** `__F16C__`; `/arch:AVX512` adds
  `F`/`BW`/`VL`/`DQ` but not `F16C`, `VNNI` or `VBMI` - although the intrinsics for all of those
  compile anyway. The sources that branch on those macros therefore silently take the scalar path
  unless the macros are defined alongside the flag, which the build does.
- MSVC gives the scalar F16C conversions `_cvtss_sh` and `_cvtsh_ss` no working code: its headers declare
  them nowhere, and with `__F16C__` defined the compiler claims the names as its own, so a definition
  compiled under that macro does not build as written. The pair is supplied by `src/kernels/cpu/ggml/
  f16c_scalar_msvc.c`, built from the vector intrinsics (the same instruction: `1.5` becomes `0x3E00` and
  back; `0.1` becomes `0x2E66` and back to `0.0999756`). That file is compiled *without* `__F16C__`, and
  the extracted ggml dots carry `#pragma function` plus prototypes for the names, so the calls reach the
  definitions instead of compiler-generated code. Getting this wrong fails quietly: every stored block
  scale reads as zero, the quants stay correct, and each CPU dot product lands ~1e13 off - the engine
  still runs, on garbage numbers.
- MSVC has no `_Float16`, and no `__builtin_cpu_supports`. The vision encoder's fp32-through-half
  quantization uses the F16C intrinsic pair, and the BF16 kernel's run-time probe reads the same
  CPUID leaves and XCR0 bits GCC's probe consults. The AVX-512 BF16 types themselves need no
  change: MSVC spells `__m256bh` and `__m512bh` as typedefs of the integer vectors.
- MSVC reports `__cplusplus` as `199711L` unless `/Zc:__cplusplus` is passed. The CUDA driver header
  defines `CUtensorMap` with `alignas(TENSOR_MAP_ALIGN)` only when the reported value looks like
  C++11 or later, so without that flag the type compiles with the natural 8-byte alignment instead of
  64 (or 128, as reported by GCC). `hc_fused_kernel` passes six TMA descriptors in a
  `__grid_constant__` argument struct; when the struct's alignment is wrong, every
  `cp.async.bulk.tensor`/`prefetch.tensormap` faults with `cudaErrorMisalignedAddress`
  (compute-sanitizer reports it as "Misaligned shared or local address"), which surfaces in the
  engine as a stuck verify window. The build passes `/Zc:__cplusplus` to cl.exe and nvcc's host
  pass for every Windows translation unit; Linux builds need nothing, because GCC reports the true
  standard version. Same class as CUTLASS issue 2905 and PyGPUkit issue 107.

`GGML_EXTENSION` (a GCC `__extension__`) is guarded: it breaks every CUDA translation unit
that includes the block-layout header on the Windows front end.

## Quality

**Quality has not been measured on the Windows build.** No perplexity run and no KL-divergence run
against the Q8_0 reference has been done here: `basalt-bench --quality` needs a KL-divergence base
that llama.cpp builds from a Q8_0 copy of the model, and no such base exists on this machine. What
*has* been measured for this build is in [precision.md](precision.md), under "Compiler parity": the
CPU expert kernel and the expert pool against a plain reference on both toolchains (`expert_parity`
and `pool_test`, 0 failures on each), and the greedy token streams of the two builds, which part
company a few tokens into a prompt. That page also names the kernels neither table covers. Read the
Windows tier as numerically unproven at the quality-gate level until a gate run lands on it.

## Benchmarks

Not yet measured by the maintainer on Windows; a headline number is still pending. A development
A/B run on this machine (RTX 5090, 575 W power limit; Windows 11, MSVC 19.44, CUDA 13.0; Q3 pack;
llama-swap-equivalent config: 524288 context, yarn, `--kv int8`, `--spec 7 --spec-min-p 0.5`,
`--ple-io ram`, `--kv-resident 32768`) measured, as a sanity check only:

- prefill: 64,437 prompt tokens read in 11 s (~5,860 tok/s)
- decode: 12,299 tokens (reasoning + answer) in 56 s (~220 tok/s steady state)

These are not a headline figure: headline numbers are the maintainer's Linux runs. Every number
here was measured on this machine at its 575 W power limit.

## Machine-specific observations

Machine-environment restrictions that look like code defects but are not:

- Creating an executable named `basalt-load.exe` under the workspace on this machine is denied:
  that exact filename is refused (`Access denied`) in that directory tree, while any other
  name is allowed. The program compiles and links fine under a different name. Check with a build
  of your own before treating that failure as a port defect.
- A process started from under one particular folder on this machine gets a token that cannot
  raise its own working-set quota (`ERROR_PRIVILEGE_NOT_HELD`, 1314), while the *same* executable
  run from another folder locks 256 MiB successfully. `platform_memory_test` fails in the first
  location and passes in the second. If that test fails, run the same binary from somewhere else
  before believing the port is wrong.
