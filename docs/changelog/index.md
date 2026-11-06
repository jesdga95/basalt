# Changelog

Releases, newest first. Versions are 0.x: a minor release adds features or changes flags and defaults (a removed or renamed flag breaks a config, as Basalt keeps no shims); a patch release only fixes. Each release is an annotated tag `vX.Y.Z` with a GitHub Release carrying its entry; there are no binaries, build from source ([building](../building.md)).

| Release | Date | Summary |
| --- | --- | --- |
| [0.3.1](0.3.1.md) | unreleased | Experts on the SSD are said in the log, the console and the stats page; one RAM reserve and guard for every model (`--ram-reserve-mib`); default context 8,192. |
| [0.3.0](0.3.0.md) | 2026-11-05 | Two models on one engine: DeepSeek-Flash joins Qwen3.8-Flash-Next, a router swaps between them, and the config format changes. |
| [0.2.1](0.2.1.md) | 2026-11-03 | Image sources follow where a request comes from, quality measured against a BF16 reference, an NVFP4 pack with Q8_0 dense projections. |
| [0.2.0](0.2.0.md) | 2026-11-01 | `.basalt` format version 2, about 13% faster cold decode, opt-in lossy routing flags, a thinking-loop safeguard. |
| [0.1.0](0.1.0.md) | 2026-10-29 | First tagged release: the engine as it stands, and the changes since 2026-10-15. |
