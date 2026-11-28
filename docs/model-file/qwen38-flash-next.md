# Flash-Next packs

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
records it in `meta.mtp.dense`. The base's vocabulary must be NVIDIA's.

## Flash-Next QAD pack

An alternative to the NVFP4 pack, from NVIDIA's quantization-aware distillation export
([nvidia/Qwen3.8-Flash-Next-NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4-QAD)): lower PPL, further from
BF16, no faster ([measured](../precision/qwen38-flash-next.md#qad-checkpoint)). The routed experts keep their NVFP4 codes and
scales; the MXFP8 projections and NVFP4 shared experts become Q6_K (Q8_0 where rows are not whole blocks), the indexer and
GDN a/b rows BF16, the FP8 n-gram shards Q8_0. The drafter and the vision tower come from a base pack.

```sh
hf download nvidia/Qwen3.8-Flash-Next-NVFP4-QAD --local-dir qad
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-NVFP4.basalt --checkpoint qad \
    --out Qwen3.8-FN-NVFP4-QAD.basalt --name "Qwen3.8-Flash-Next NVFP4-QAD"
```

## Flash-Next FP4-Local pack

`--checkpoint` also reads Model Optimizer's FP4-Local layout, as in
[nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local): the shards sit in `tensors/`,
the routed experts' scales are CSF stream pairs (`weight_scale.nvfp4_csf_fixed`, `weight_scale.nvfp4_csf_exceptions`), the
projections are MXFP8, the norm weights F32 and the n-gram table NVFP4.

- **Experts:** each CSF pair decodes to the UE4M3 scales bit for bit; codes and scales then move as in the NVFP4 pack.
- **Projections:** MXFP8 values (E4M3 codes, one E8M0 scale per 32) widen to f32 exactly, then become Q6_K (Q8_0 where rows
  are not whole blocks), or BF16 where the engine reads BF16, as in the QAD pack.
- **N-gram table:** E2M1 code × UE4M3 block scale × `weight_scale_2`, stored as Q8_0 (one rounding).
- **Drafter and vision:** copied from the base pack.

```sh
hf download nvidia/Qwen3.8-Flash-Next-FP4-Local --revision b9ce18a6 --local-dir fp4-local
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-NVFP4.basalt --checkpoint fp4-local \
    --out Qwen3.8-FN-FP4-Local.basalt --name "Qwen3.8-Flash-Next FP4-Local"
```
