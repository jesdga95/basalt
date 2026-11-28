# `.basalt` model file

One file holds everything the engine, the server and the vision sidecar load: weights, tokenizer and chat template, MTP or DSpark
drafter, 8-bit hyper-connection copies, vision projector, and the GGUF metadata with default sampling. Flags and the expert profile are
not in it.

- **Layout:** a 64 KiB header (magic `BASALTMF`, version 2, up to 511 named sections with offset, size and SHA-256), then sections aligned
  to 2 MiB. Authoritative: [model_format.hpp](../include/basalt/core/model_format.hpp).
- **Versions:** the engine reads version 2 only. Version 1 files convert with
  [tools/migrations/basalt_v1_to_v2.py](../tools/migrations/README.md), without a download. What changed:
  [file format changes](basalt-file-changelog.md).

## Build a pack from a GGUF

```sh
python3 tools/convert/convert.py --gguf MODEL-00001-of-0000N.gguf --mmproj MMPROJ.gguf --out MODEL.basalt \
    [--mtp-source DIR [--mtp-dense FILE]] [--draft-vocab FILE] [--compat-bf16] [--name NAME]
```

`--help` describes every input. `--compat-bf16` converts GGUF projections stored as BF16 in another type. A pack built this way
without the official tuned drafter is slower than the official file.

Each pack's sections, inputs and converter command are on the page for its model.

## DeepSeek pack

Sections, the DSpark drafter and how to attach it: [DeepSeek packs](model-file/deepseek.md#deepseek-pack).

## DeepSeek NVFP4 pack

Built from NVIDIA's checkpoint: [DeepSeek packs](model-file/deepseek.md#deepseek-nvfp4-pack).

## Flash-Next packs

NVFP4, QAD and FP4-Local: [Flash-Next packs](model-file/qwen38-flash-next.md).

## Qwen3.8-27B pack

Sections, both checkpoints and the `NVFP4_PLANES` tensor type: [Qwen3.8-27B pack](model-file/qwen38-27b.md).

## Hy4-flash NVFP4 pack

The NVFP4 pack: [Hy4-flash packs](model-file/hy4-flash.md#hy4-flash-nvfp4-pack). The test pack from a Q4_K_M GGUF is on the same page.

## Inkling 2 Mini pack

[Inkling 2 Mini pack](model-file/inkling2.md).

## Gemma 4 pack

[Gemma 4 pack](model-file/gemma4.md).

## ERNIE-5.5-Flash pack

[ERNIE-5.5-Flash pack](model-file/ernie55.md).
