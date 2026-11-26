# Changelog

Releases, newest first. Versions are 0.x: a minor release adds a model or another large change; a patch release brings fixes and smaller features, and can rename or remove a flag (a removed or renamed flag breaks a config, as Basalt keeps no shims, so each entry's Upgrading section lists them). Each release is an annotated tag `vX.Y.Z` with a GitHub Release carrying its entry. From 0.3.2 each release also carries Linux and Windows x64 archives ([platforms](../platforms.md#release-archives)); or build from source ([building](../building.md)).

| Release | Date | Summary |
| --- | --- | --- |
| [0.5.7](0.5.7.md) | 2026-11-23 | Breaking: `gpu`, `gpu-mode` and `gpu-share` replace every model's own multi-GPU logic (`gpu: auto` is gone, a list of cards needs a mode); peer and split on the 27B and Inkling 2 Mini, one shared layer and a conformance check for every model. |
| [0.5.6](0.5.6.md) | 2026-11-23 | Inkling 2 Mini on a small card or two cards (experts on the CPU, layer split) with prompts 21 to 25% faster and a second pack (MX8); parallel slots decode together on every model; Hy4-flash stray tokens fixed; `--seed` in the CLI. |
| [0.5.5](0.5.5.md) | 2026-11-21 | Inkling 2 Mini: a looped mixture of experts on one card, with images, the DSpark drafter and up to 1M tokens of context where the card has room. |
| [0.5.0](0.5.0.md) | 2026-11-19 | Hy4-flash: a mixture of experts with a 1M-token context (elastic fp16 or FP8 caches), images, the DFly2 drafter and a shipped expert profile; per-layer expert copies, a bounded lend and partial reclaim in the shared expert tier. |
| [0.4.1](0.4.1.md) | 2026-11-16 | Qwen3.8-27B on two cards (prompt line, layer split), YaRN past 262,144 and K/V streaming; Flash-Next's RAM plan leaves room again on 64 GB boxes; `--parallel` prompts tokenized in the engine; DeepSeek tiers on up to four cards. |
| [0.4.0](0.4.0.md) | 2026-11-14 | Qwen3.8-27B, the first dense model: QUASAR's NVFP4 checkpoint with an FP8 head, the DFlash 2 drafter in windows of up to 16 with exact speculative sampling, batched slots; one checkpoint transfer for every model. |
| [0.3.3](0.3.3.md) | 2026-11-12 | Answer now, logprobs, documents and video in messages, session-affinity slots, faster Flash-Next long prompts on one card, a DeepSeek NVFP4 pack, a container image, `--no-answer-room` becomes `--answer-room 0`. |
| [0.3.2](0.3.2.md) | 2026-11-08 | Faster DeepSeek decode on two cards, release archives for Linux and Windows x64, an install that finds its files beside `bin/`. |
| [0.3.1](0.3.1.md) | 2026-11-08 | DeepSeek FP8/FP4 attention caches, drafting in every slot, a shared peer-card prompt path, faster decode kernels, new packs, one RAM reserve for every model (`--ram-reserve-mib`). |
| [0.3.0](0.3.0.md) | 2026-11-05 | Two models on one engine: DeepSeek-Flash joins Qwen3.8-Flash-Next, a router swaps between them, and the config format changes. |
| [0.2.1](0.2.1.md) | 2026-11-03 | Image sources follow where a request comes from, quality measured against a BF16 reference, an NVFP4 pack with Q8_0 dense projections. |
| [0.2.0](0.2.0.md) | 2026-11-01 | `.basalt` format version 2, about 13% faster cold decode, opt-in lossy routing flags, a thinking-loop safeguard. |
| [0.1.0](0.1.0.md) | 2026-10-29 | First tagged release: the engine as it stands, and the changes since 2026-10-15. |
