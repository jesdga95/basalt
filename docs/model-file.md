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

## Qwen3.8-27B pack

The published pack comes from QUASAR's quantization-aware NVFP4 checkpoint
([QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4), revision `4a22293a`,
Apache-2.0, every projection NVFP4 in compressed-tensors' form) alone, no base pack: tokenizer, metadata, MTP layer and vision
tower come from the checkpoint's own files. `--drafter` adds the DFlash 2 block drafter
([z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), revision `50307d4c`).

```sh
hf download QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 --revision 4a22293a --local-dir quasar
hf download z-lab/Qwen3.8-27B-DFlash2 --revision 50307d4c --local-dir dflash2
python3 tools/convert/convert.py --checkpoint quasar --drafter dflash2 --out Qwen3.8-27B-QUASAR-NVFP4.basalt --name "Qwen3.8-27B QUASAR NVFP4"
```

- `gguf`: every tensor in llama.cpp's `qwen35` names. GDN, attention and MLP projections move bit for bit as `NVFP4_PLANES`
  (type 45, below), `1 / weight_global_scale` in FP32 (rounded once) as `<name>.scale`. The GDN a and b projections (NVFP4 in
  the checkpoint, 48 rows) become BF16 of their values, rounded once to nearest. The MTP layer's and the drafter's BF16
  projections Q8_0; token embedding BF16 as stored; norms F32 with the offset folded in (exact); GDN value heads in the GGUF
  order (rows of qkv and z, columns of the out projection, moved whole).
- The BF16 output head becomes `F8_E4M3` rows (type 44: one byte per value, Basalt's own type id), each row's FP32 scale (its
  amax / 448) in `output.scale` (F32 [rows]), each value the nearest E4M3 code. `--head nvfp4` rounds it to `NVFP4_PLANES`
  instead (tensor scale amax / (6 x 448), per 16 values the UE4M3 scale with the least squared error among half to four times
  amax / 6): 0.72 GB instead of 1.27 GB, nine times the squared error ([precision](precision/qwen38-27b.md#quality-against-bf16)).
- `vision/mmproj`: the BF16 tower written as llama.cpp's clip GGUF (`qwen3vl_merger`); `vision/text_vocab` beside it.
- The drafter's feature projection is stored as one block per target layer (`dflash.fc.N`), so the engine projects each
  layer's rows as the layer ends. 19.7 GiB with the drafter (21,149,505,930 bytes, SHA-256
  `5ec89e71b4bd461a5abf607825d3cd99e74d2476b8ba087fd690f09a067919ba` for the published file); about a minute on the default
  16 converter processes. The engine refuses a 27B pack whose MLP and head are NVFP4 in ggml blocks (type 40, before
  NVFP4_PLANES) and asks for a re-conversion.

NVIDIA's checkpoint ([nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), revision `482ca0f3`) converts
with the same command:

```sh
hf download nvidia/Qwen3.8-27B-NVFP4 --revision 482ca0f3 --local-dir nvidia
python3 tools/convert/convert.py --checkpoint nvidia --drafter dflash2 --out Qwen3.8-27B-NVFP4.basalt --name "Qwen3.8-27B NVFP4"
```

- Its MLP and output head are NVFP4 and move bit for bit as `NVFP4_PLANES` with each matrix's FP32 `weight_scale_2` as
  `<name>.scale`; its GDN and attention projections are FP8 and keep their E4M3 codes as `F8_E4M3` with each matrix's FP32
  `weight_scale` as `<name>.scale`. Everything else as above. About 22 GiB with the drafter.

### `NVFP4_PLANES` (tensor type 45)

An NVFP4 matrix of `rows` x `K` (rows a multiple of 128, K of 64) as two planes in one tensor, 36 bytes per 64 values like
ggml's NVFP4 blocks, the layout the dense products read with TMA (NInfer's and Infernix's `block_scale_k16_m128x4`):

- **Codes**, `rows * K / 2` bytes from the tensor's start: row-major, byte `b` of a row holds value `2b` in its low nibble and
  `2b + 1` in its high nibble (E2M1, as NVIDIA's checkpoints store them).
- **Scales**, `rows * K / 16` UE4M3 bytes right after the codes, one per 16 values: 512-byte tiles of 128 rows x 4 groups,
  row-block major, so the scale of row `r` and group `g` (`= k / 16`) sits at
  `((r / 128) * (K / 64) + g / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + g % 4`.
- The matrix's FP32 scale is a separate F32 [1] tensor `<name>.scale`; a value is `e2m1 * ue4m3 * scale`.

[nvfp4_planes.hpp](../include/basalt/kernels/nvfp4_planes.hpp) is the engine's copy of this layout.
