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
- `engram`: the Engram table alone (IQ4_NL rows of 144 bytes, or MXFP4 rows of 136 in the NVFP4 pack), 2 MiB aligned.
- `vision/mmproj`: present only when built with `--mmproj`; the engine and `basalt-vision --model FILE.basalt` read it in place.
- `dspark/gguf`: DeepSeek's DSpark drafter. Attach it to an existing pack without copying weights:

```sh
python3 tools/convert/convert.py --dspark DRAFTER.gguf --into PACK.basalt --undo-dir DIR
```

`--roll-back --into PACK.basalt --undo-dir DIR` reverts it.

## DeepSeek NVFP4 pack

From NVIDIA's checkpoint ([nvidia/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4)),
with the tokenizer, metadata, vision projector and DSpark drafter of a DeepSeek pack (`--sections-from`). The routed experts move
bit for bit; the FP8 projections (32 x 32 block scales) become Q8_0 of their exact values, BF16 where the base pack reads BF16;
the MXFP4 Engram rows are kept; each tensor is checked against the base pack's. `--stream-shards` lets the shards arrive while
it runs (downloaded in file order by another process) and deletes each once used, so the disk holds the output and a shard or
two (about 430 GB). The routed experts' shards and those holding the other tensors and the Engram tables are needed; the drafter
and vision shards are not.

```sh
python3 tools/convert/convert.py --sections-from DeepSeek-V4.1-Flash-1017-UD-IQ1_M.basalt --checkpoint nvidia-ds \
    --stream-shards --out DeepSeek-V4.1-Flash-1017-NVFP4.basalt --name "DeepSeek-V4.1-Flash-1017 NVFP4"
```

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

`--mtp-dense FILE` (the `mtp/dense` bytes of an official NVFP4, UD-IQ1_M, UD-IQ2_XS or UD-IQ3_XXS pack) swaps in a tuned drafter layer; the converter
records it in `meta.mtp.dense`. Output is about 122 GiB (123 GiB for the 8-bit variant). The base's vocabulary must be NVIDIA's.
