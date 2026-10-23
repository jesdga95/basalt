# AGENTS.md

Basalt is a local inference engine for the Qwen3.8-Flash-Next mixture-of-experts model, built for NVIDIA Blackwell (sm_120) on Linux and Windows x64. It runs on an NVIDIA Blackwell GPU, optionally with a second GPU as a peer expert tier, and system RAM holding the rest. It is a fork of Strata (MIT): the C++/CUDA engine is in `src/` and `include/`, and the Python server with an OpenAI- and Anthropic-compatible API is in `serve/`.

## Working rules

`CONTRIBUTING.md` sets the project scope (CUDA + Blackwell only) and what every change must include (parity test, profile, quality check). It applies to agents too.

- **Commits:** small and logical, message style `area: what it does`. Never amend and never force-push; always make new commits. End each message with the co-author line your tool's instructions specify.
- **Platform code:** engine code never names an OS or a compiler. No `_WIN32`, `_MSC_VER`, `__linux__`, `<windows.h>`, `<sys/mman.h>`, `<unistd.h>` or raw OS calls (mmap, pread, pthread, fork/exec, dlopen, /proc, /sys) outside `src/platform/` and `include/basalt/platform/`. A missing seam is added there with a Linux and a Windows body; the Linux body is the code you would have written anyway. `tools/check_platform_boundary.sh` enforces it.
- **Log lines:** every engine stderr line goes through `include/basalt/log.hpp` with its real level (`info`, `warn`, `error`); never `fprintf(stderr, ...)`. Do not change the text of lines the server parses. `tools/check_stderr_logging.sh` enforces it.
- **Code comments:** brief and stateless. Say what the code does and why. No hardware names, measured numbers, dates or history in comments.
- **Don't push or publish anywhere** (GitHub, HF, any remote) without the maintainer's approval.
- **Don't write tools nobody asked for.**
- **Linux and Windows are both supported; Linux is where we test, and CI compiles and CPU-tests Windows.** Changes must not remove the platform seams (`src/platform/`, CMake `if(WIN32)`); see `CONTRIBUTING.md` and `docs/platforms.md`.
- **Headline benchmark numbers come from the maintainer's own runs.** Agent benches are development A/Bs only. Label every number with its GPU power limit.
- **Quality gate:** PPL/KLD against the Q8_0 reference, through the decode path (`basalt-bench --quality`). Byte identity with old output is not required, but say whenever numerics change. Bitwise-neutral refactors need only the greedy-identity check.
- **Keep `docs/precision/` current.** Any engine change that can alter precision (weight formats, activation or accumulation types, KV/state formats, reduction order, decode or MTP paths) updates the page that covers it in the same change, with fresh numbers where it quotes them.
- **Docs:** no file over 300 lines; split a longer topic into a folder of files; keep docs short, stateless and free of duplication.
- **Basalt's own kernels are the reference.** Parity tests compare a changed kernel against the current one and against a plain CPU reference; Strata's original kernels are not kept as oracles. Every new kernel gets a parity test (random inputs + real geometry) and a microbench line.
