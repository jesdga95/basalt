# `.basalt` file format changelog

The version is the `u32` after the magic in the file's header. The engine reads one version; the
[model file](model-file.md) page describes the current one.

## Version 2

The current version. The engine refuses version 1 files and names
[tools/migrations/basalt_v1_to_v2.py](../tools/migrations/README.md), which writes a version 2 file from a version 1 file without a
download or any re-quantizing.

- **Contains:** `meta`, `tokenizer/*`, `dense`, `dense.index`, `experts`, `experts.layout`, `gguf`, `mtp/*`, `hc8`,
  `vision/*`.
- **Added `hc8`, now required:** the int8 copies of the hyper-connection matrices the decode reads, the bits the
  engine used to compute at every start. A start reads them as they are and skips reading the BF16 matrices to
  quantize them (about half a second).
- **Removed `profile`:** the expert profile decides which experts start in VRAM. It is placement policy that improves
  independently of the weights, so it moved to the engine (`data/profiles`), where a better profile needs no new file.
- **Trimmed `meta`:** no `sampling` (the server reads the GGUF metadata's `general.sampling.*` keys), no `profile` or
  `hc8` records and no converter revision, provenance that changed whenever a section was replaced.

## Version 1

- **Contains:** `meta` (with `sampling`, `profile` and `hc8` records and the converter's git revision),
  `tokenizer/*`, `dense`, `dense.index`, `experts`, `experts.layout`, `gguf`, `mtp/*`, `profile`, optionally `hc8`,
  `vision/*`.
- The first single-file format: everything the engine, the server and the vision sidecar load. Without `hc8` the
  engine quantized the hyper-connection matrices at every start.
