# AGENTS.md

Basalt is a local inference engine for the Qwen3.8-Flash-Next mixture-of-experts model, built for Linux and NVIDIA Blackwell (sm_120), with Windows x64 community-supported. It runs on one RTX 5090, with a second card as a peer expert tier and system RAM holding the rest. It is a fork of Strata (MIT): the C++/CUDA engine is in `src/` and `include/`, and the Python server with an OpenAI- and Anthropic-compatible API is in `serve/`.

## Working rules

`CONTRIBUTING.md` sets the project scope (CUDA + Blackwell only) and what every change must include (parity test, profile, quality check). It applies to agents too.

- **Commits:** small and logical, message style `area: what it does`. Never amend and never force-push; always make new commits. End each message with the co-author line your tool's instructions specify.
- **Code comments:** brief and stateless. Say what the code does and why. No hardware names, measured numbers, dates or history in comments.
- **Don't push or publish anywhere** (GitHub, HF, any remote) without the maintainer's approval.
- **Never expose a server beyond 127.0.0.1 without `--api-key`.**
- **Don't write tools nobody asked for.**
- **Windows x64 is community-supported:** it builds, single GPU, no performance claims. Linux changes need no Windows testing but must not remove the platform seams (`src/platform/`, CMake `if(WIN32)`); the full tier is in `CONTRIBUTING.md`, and the benchmark rule below keeps headline numbers with the maintainer's runs.
- **Headline benchmark numbers come from the maintainer's own runs.** Agent benches are development A/Bs only. Label every number with its GPU power limit.
- **Quality gate:** PPL/KLD against the Q8_0 reference, through the decode path (`basalt-bench --quality`). Byte identity with old output is not required, but say whenever numerics change. Bitwise-neutral refactors need only the greedy-identity check.
- **Keep `docs/precision.md` current.** Any engine change that can alter precision (weight formats, activation or accumulation types, KV/state formats, reduction order, decode or MTP paths) updates that page in the same change, with fresh numbers where it quotes them.
- **Basalt's own kernels are the reference.** Parity tests compare a changed kernel against the current one and against a plain CPU reference; Strata's original kernels are not kept as oracles. Every new kernel gets a parity test (random inputs + real geometry) and a microbench line.
