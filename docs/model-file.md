# The .basalt model file

A model is one file. It holds everything the engine, the server and the vision sidecar load: the weights, the
tokenizer and chat template, the MTP drafter, the 8-bit hyper-connection copies, the vision projector, and the GGUF
metadata with the default sampling. It carries nothing that is engine policy: no flags (those come from `--config` or
the command line) and no expert profile (the engine's own, [below](#expert-profiles)).

The layout diagram is in the [README](../README.md#models).

- **Layout:** a 64 KiB header (magic `BASALTMF`, version 2, a table of up to 511 named sections, each with its offset,
  size and SHA-256), then the sections, each aligned to 2 MiB so it can be memory-mapped or read with `O_DIRECT`.
- **Sections:** `meta` (JSON: the source GGUF metadata with its `general.sampling.*` defaults, every tensor's type,
  shape and offset, provenance and license), `tokenizer/*`, `dense`, `experts` and `experts.layout` (the routed
  experts, laid out per layer), `gguf` (every non-expert source tensor), `mtp/*` (the drafter), `hc8` (the int8 copies
  of the hyper-connection matrices the decode reads, computed by the converter so a start reads them as they are) and
  `vision/*`. The authoritative description is [include/basalt/core/model_format.hpp](../include/basalt/core/model_format.hpp).
- **Versions:** the engine reads version 2 and refuses any other. A version 1 file becomes a version 2 file with
  [tools/migrations/basalt_v1_to_v2.py](../tools/migrations/README.md), without a download. What changed:
  [the file format changelog](basalt-file-changelog.md).
- **Weights as published:** a pack built from a GGUF copies every tensor byte for byte, except with `--compat-bf16`,
  for GGUFs that store the small projections the engine reads as BF16 in another type (it converts them and records
  that in `meta`). The NVIDIA NVFP4 pack keeps NVIDIA's NVFP4 experts bit for bit and quantizes its dense tensors once
  from BF16 ([below](#nvfp4-pack)).

Build one from the source GGUF shards (give shard 1; the others are found beside it):

```sh
python3 tools/convert/convert.py --gguf MODEL-00001-of-0000N.gguf --mmproj MMPROJ.gguf --out MODEL.basalt \
    [--mtp-source DIR [--mtp-dense FILE]] [--draft-vocab FILE] [--compat-bf16] [--name NAME]
```

`python3 tools/convert/convert.py --help` describes every input.

## Packs

A pack you convert yourself from a GGUF, without the official files' tuned drafter, will be significantly slower than the official `.basalt` files. How precision is handled and how quality is measured: [precision](precision/index.md).

## NVFP4 pack

The official NVFP4 file is NVIDIA's NVFP4 checkpoint
([nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), revision `fc694b54`)
with the dense projections stored as 6-bit Q6_K instead of 8-bit Q8_0, which takes the tensor-core dense path in
decode. The per-tensor breakdown and its quality are on the Hugging Face card.

Basalt's converter reads NVIDIA's safetensors directly (`convert.py --checkpoint`). It moves each expert's NVFP4 codes
and scales into Basalt's blocks without changing a bit, quantizes each dense projection once from BF16, folds the "+1"
of the norm weights into F32, puts the GDN value heads in the order the engine reads them, and writes the metadata and
tokenizer from NVIDIA's `config.json`, tokenizer files and model card. `meta.checkpoint` records each step.

Build either variant yourself from NVIDIA's checkpoint. The base pack (version 2) supplies only the drafter and
vision projector, copied byte for byte; the `hc8` copies are computed from NVIDIA's weights. The official file takes
them from UD-Q4_K_XL. The base's vocabulary must be NVIDIA's:

```sh
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --revision fc694b54 --local-dir nvidia

# 6-bit dense (the official file)
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia \
    --out Qwen3.8-FN-NVIDIA-NVFP4.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"

# 8-bit dense variant
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia --dense-q8-0 \
    --out Qwen3.8-FN-NVIDIA-NVFP4-Q8.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"
```

`--mtp-dense FILE` replaces the dense part of the drafter copied from the base with a tuned layer. The file is the raw
bytes of the `mtp/dense` section of an official IQ3_S, IQ3_XXS or Q2_0 pack (for example
`ModelFile(path).read("mtp/dense")` from `tools/convert/model_file.py`); its size must match the base's section, and the
converter records its SHA-256 in `meta.mtp.dense` with the tuning `self_distilled`. Add it to either command:

```sh
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia \
    --mtp-dense tuned-mtp-dense.bin \
    --out Qwen3.8-FN-NVIDIA-NVFP4.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"
```

The output is about 122 GiB (the 8-bit variant 123 GiB). How NVFP4 is computed on each tier:
[precision](precision/index.md).

## Expert profiles

Which experts start in VRAM is the engine's expert profile for the model, `data/profiles/<key>.prof` in the source
tree, where `<key>` is `<general.architecture>-<block_count>x<expert_count>` from the model's metadata
(`qwen4exp-48x512`). Every pack of the model uses it; without a file for the model the experts are placed by first
use. The start line `profile PATH: N ranked pairs` names the one in use.

A profile is what a server learned: run one with `--expert-profile-save FILE` on a representative mix of prompts and
stop it (QUIT saves the file). `--expert-profile-save` with an existing file also starts from it, which tries a
profile before it replaces the one in `data/profiles`.
