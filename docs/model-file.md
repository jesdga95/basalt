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

## DeepSeek pack

- `dense`: F32 and BF16 tensors. `gguf`: quantized non-expert tensors. `experts`: routed experts per layer.
- `engram`: the Engram table alone (IQ4_NL rows of 144 bytes), 2 MiB aligned.
- `vision/mmproj`: present only when built with `--mmproj`.
- `dspark/gguf`: DeepSeek's DSpark drafter. Attach it to an existing pack without copying weights:

```sh
python3 tools/convert/convert.py --dspark DRAFTER.gguf --into PACK.basalt --undo-dir DIR
```

`--roll-back --into PACK.basalt --undo-dir DIR` reverts it.

## Flash-Next NVFP4 pack

The official file is NVIDIA's checkpoint ([nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4),
revision `fc694b54`) with dense projections as Q6_K. The converter reads the safetensors directly: experts move bit for bit, dense
tensors are quantized once from BF16.

```sh
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --revision fc694b54 --local-dir nvidia

# 6-bit dense (the official file)
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia \
    --out Qwen3.8-FN-NVIDIA-NVFP4.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"

# 8-bit dense variant
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia --dense-q8-0 \
    --out Qwen3.8-FN-NVIDIA-NVFP4-Q8.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"
```

`--mtp-dense FILE` (the `mtp/dense` bytes of an official IQ3_S, IQ3_XXS or Q2_0 pack) swaps in a tuned drafter layer; the converter
records it in `meta.mtp.dense`. Output is about 122 GiB (123 GiB for the 8-bit variant). The base's vocabulary must be NVIDIA's.
