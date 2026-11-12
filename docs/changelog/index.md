# Changelog

Releases, newest first. Versions are 0.x: a minor release adds a model or another large change; a patch release brings fixes and smaller features, and can rename or remove a flag (a removed or renamed flag breaks a config, as Basalt keeps no shims, so each entry's Upgrading section lists them). Each release is an annotated tag `vX.Y.Z` with a GitHub Release carrying its entry. From 0.3.2 each release also carries Linux and Windows x64 archives ([platforms](../platforms.md#release-archives)); or build from source ([building](../building.md)).

| Release | Date | Summary |
| --- | --- | --- |
| [0.3.3](0.3.3.md) | 2026-11-12 | Answer now, logprobs, documents and video in messages, session-affinity slots, faster Flash-Next long prompts on one card, a DeepSeek NVFP4 pack, a container image, `--no-answer-room` becomes `--answer-room 0`. |
| [0.3.2](0.3.2.md) | 2026-11-08 | Faster DeepSeek decode on two cards, release archives for Linux and Windows x64, an install that finds its files beside `bin/`. |
| [0.3.1](0.3.1.md) | 2026-11-08 | DeepSeek FP8/FP4 attention caches, drafting in every slot, a shared peer-card prompt path, faster decode kernels, new packs, one RAM reserve for every model (`--ram-reserve-mib`). |
| [0.3.0](0.3.0.md) | 2026-11-05 | Two models on one engine: DeepSeek-Flash joins Qwen3.8-Flash-Next, a router swaps between them, and the config format changes. |
| [0.2.1](0.2.1.md) | 2026-11-03 | Image sources follow where a request comes from, quality measured against a BF16 reference, an NVFP4 pack with Q8_0 dense projections. |
| [0.2.0](0.2.0.md) | 2026-11-01 | `.basalt` format version 2, about 13% faster cold decode, opt-in lossy routing flags, a thinking-loop safeguard. |
| [0.1.0](0.1.0.md) | 2026-10-29 | First tagged release: the engine as it stands, and the changes since 2026-10-15. |
