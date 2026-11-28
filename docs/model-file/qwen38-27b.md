# Qwen3.8-27B pack

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
  amax / 6), a smaller head with more error ([precision](../precision/qwen38-27b.md#quality-against-bf16)).
- `vision/mmproj`: the BF16 tower written as llama.cpp's clip GGUF (`qwen3vl_merger`); `vision/text_vocab` beside it.
- The drafter's feature projection is stored as one block per target layer (`dflash.fc.N`), so the engine projects each
  layer's rows as the layer ends. The engine refuses a 27B pack
  whose MLP and head are NVFP4 in ggml blocks (type 40) and asks for a re-conversion.

NVIDIA's checkpoint ([nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), revision `482ca0f3`) converts
with the same command:

```sh
hf download nvidia/Qwen3.8-27B-NVFP4 --revision 482ca0f3 --local-dir nvidia
python3 tools/convert/convert.py --checkpoint nvidia --drafter dflash2 --out Qwen3.8-27B-NVFP4.basalt --name "Qwen3.8-27B NVFP4"
```

- Its MLP and output head are NVFP4 and move bit for bit as `NVFP4_PLANES` with each matrix's FP32 `weight_scale_2` as
  `<name>.scale`; its GDN and attention projections are FP8 and keep their E4M3 codes as `F8_E4M3` with each matrix's FP32
  `weight_scale` as `<name>.scale`. Everything else as above.

### `NVFP4_PLANES` (tensor type 45)

An NVFP4 matrix of `rows` x `K` (rows a multiple of 128, K of 64) as two planes in one tensor, 36 bytes per 64 values like
ggml's NVFP4 blocks, the layout the dense products read with TMA (NInfer's and Infernix's `block_scale_k16_m128x4`):

- **Codes**, `rows * K / 2` bytes from the tensor's start: row-major, byte `b` of a row holds value `2b` in its low nibble and
  `2b + 1` in its high nibble (E2M1, as NVIDIA's checkpoints store them).
- **Scales**, `rows * K / 16` UE4M3 bytes right after the codes, one per 16 values: 512-byte tiles of 128 rows x 4 groups,
  row-block major, so the scale of row `r` and group `g` (`= k / 16`) sits at
  `((r / 128) * (K / 64) + g / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + g % 4`.
- The matrix's FP32 scale is a separate F32 [1] tensor `<name>.scale`; a value is `e2m1 * ue4m3 * scale`.

[nvfp4_planes.hpp](../../include/basalt/kernels/nvfp4_planes.hpp) is the engine's copy of this layout.
