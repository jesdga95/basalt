# AGENTS.md

Basalt is a local inference engine for mixture-of-experts models, built for NVIDIA Blackwell (sm_120) on Linux and Windows x64. A model runs on one Blackwell GPU, optionally with more GPUs as peer expert tiers, and system RAM holding the rest. The C++/CUDA engine is in `src/` and `include/`, each model's own code under `src/models/<model>`, and the Python server with an OpenAI- and Anthropic-compatible API is in `serve/`.

## Rules

[CONTRIBUTING.md](CONTRIBUTING.md) is the project's rulebook and applies to agents in full: scope, platform and model layering, one fast path, what every change must include, benchmark hygiene, commits and docs. The rules below are only for agents.

- **Design the end state first.** Before writing code, design the architecture properly and read similar code. If a model already implements the feature, reuse that logic or make it shared. Do not propose features or improvements before testing them and checking the architecture.
- **Flag odd numbers.** Sanity-check every result before reporting it (size against speed, run-to-run spread, earlier numbers) and report anything odd with its suspected cause.
- **Read before answering.** Check the code, docs and existing results before stating a fact about Basalt.
- **No amends, no force-pushes.** Always make new commits.
- **Don't push or publish anywhere** (GitHub, HF, any remote) without the user's approval.
- **Don't write tools nobody asked for.**
