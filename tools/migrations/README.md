# Migration

`basalt_v1_to_v2.py` turns a version 1 `.basalt` file into a version 2 file, which is what the engine reads. Nothing
is downloaded or re-quantized: every model section is copied byte for byte, the expert profile is dropped (the engine
ships its own, [data/profiles](../../data/profiles)), and the 8-bit hyper-connection table (`hc8`) is computed from the
file's own matrices, bit for bit what the engine used to compute at every start.

```sh
python3 tools/migrations/basalt_v1_to_v2.py Qwen3.8-FN-NVFP4.basalt Qwen3.8-FN-NVFP4.v2.basalt
```

It needs Python 3.10+ with numpy, and free disk space for a second copy of the file. The old file is only read; delete
it once the new one runs. What changed between the versions: [the file format changelog](../../docs/basalt-file-changelog.md).
