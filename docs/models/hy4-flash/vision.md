# Hy4-flash vision

The picture path: encoder, preprocessing, prompt layout, rotary positions, Engram. Companion to
[Hy4-flash](../hy4-flash.md#vision).

The ViT runs on `src/vision/encoder_cuda.cu` (and the CPU reference `encoder_cpu.cpp`) selected by the projector file's `hy_v4_flash`
type; Flash-Next's `qwen3vl_merger` path is untouched (its kernels are not edited, only new ones added). What differs from Qwen3-VL:

| | Qwen3-VL (Flash-Next) | Hy4-flash |
|---|---|---|
| Patch embedding | two temporal taps summed | one 16 x 16 conv, bias |
| Position table | 48 x 48, align-corners bilinear | 128 x 128 (row 0, the CLS slot, dropped), `interpolate(scale_factor=(size + 0.1) / 128, align_corners=False)`; an axis whose size equals the table's is copied |
| In-layer rotation | 2-D RoPE on q, k | none |
| After the layers | LayerNorm, merger (tanh GELU) | RMS norm (eps 1e-5), 2 x 2 merge conv 4,608 -> erf GELU -> 1 x 1 conv 3,072, a learned `image_newline` row after each line |
| Encoder FFN | tanh GELU | tanh GELU (`bias_activation_fusion`, as llama.cpp's hy_v4_flash) |

The projector is the side file's `mmproj-Hy4-flash-BF16.gguf` (llama.cpp's `hy_v4_flash` layout: F32 convs and position table, BF16
products, q, k and v separate). The encoder reads it as is: q, k, v become one product, and the merge conv is reordered into a 4,608 x 4,608
BF16 product over a group's four patch rows in merge order, at stage time. `convert.py --mmproj FILE` (arch `mmproj_only`) adds it
as `vision/mmproj`. `--vision` (serve: `dynamic`, the only mode) stages it once in pinned RAM (928 MiB) before the tier is sized;
a request's pictures then take a region on the main card (about 1.9 GB with the work buffers for 4,096 merged tokens) where a short
prompt's buffers go: what the tier left free, then the slots beside the base arena, then every slot the tier lends. `images_done`
frees it and the tier takes its slots back. A 600-token picture encodes in about 0.1 s, a 4,096-token one in 0.55 s.

Preprocessing (`vision.hpp`): `smart_resize` with factor 32, min 4,096 and max 4,194,304 pixels (4 and 4,096 merged tokens), a ratio over
200 refused; Pillow's bicubic to the canvas (bit-exact); (v / 255 - 0.5) / 0.5; patches in merge order. The filter:
`processor_config.json` says `resample: 1` (Lanczos) and llama.cpp's `hy_v4_flash` uses it, but transformers' `hy_v4_vl` processor
resizes with BICUBIC because the original's `PIL.Image.resize` call never passes the configured filter. Lanczos rows sit 3 to 7 %
(relL2) from bicubic ones on real pictures, and the model reads finer print with bicubic (the newspaper's price and issue number).

Prompt: `<image_start>` pad x (oh x (ow + 1)) `<image_end>`; the pads of the newline slots are in the run, so the block of
embedding rows is `oh * (ow + 1)` long and the engine's pad-run expansion (`encode_image` row count) already fits. The adapter names
`<hy_image_pad..>` as the image pad; the server's Hy4 dialect sets `image_pad` and `image_start` to the pack's tokens.

MRoPE: interleaved sections [12, 10, 10] over the 32 rotated pairs: pair i turns by h when i % 3 == 1 and i < 30, by w when
i % 3 == 2 and i < 30, by t otherwise. Text takes the running position on all three; a picture block starting at p puts row j (column
newline included, nx = ow + 1) at (p, p + j / nx, p + j % nx), and the text after it continues at p + max(oh, ow + 1)
(transformers' `get_rope_index` of `hy_v4_vl`, tested on five prompts). The session keeps each cell's triple (`RopeTrack`) and,
once a picture is in, a device table `cell_rope` that `mla_prepare`, `idx_rope_heads` and `idx_pool` read (null: the cell index, so
text-only output is unchanged). The indexer's pooled key turns by its group's first cell's triple, llama.cpp's convention (its pool
position is "the M-RoPE position of each new pool's first member"); vLLM's `rotate_indexer` takes three-axis group positions too.
llama.cpp's flash path itself rotates by 1-D positions (rope type NORM), so it is not a reference for picture positions. After a
picture the drafter turns its block by the text's rotary position, not the cell.

Engram: llama.cpp's hy-v4-flash treats an embedding (image) position as a token that cannot be hashed, so an image row's
own Engram contribution is masked (valid 0) and, to the rows after it, it is a padded history entry. The session passes -1 as the token
of an image row (`run_rows` accepts it only with embedding rows) and `engram_ids` reads -1 as dead on both counts. vLLM's flash
Engram has the same hook for a picture's own rows (`visual_token_ids`, masked) but leaves it empty and reads the pad ids as history.

Images enter `session.cpp` through three marked hooks (`[vision]`): `encode_image`/`images_done`/`prefill_embeddings` (chunks of up
to 256 rows through `run_rows` with the rows as the embeddings), `place_rope` in `run_rows`, and the `cell_rope` pointers in
`mla_args` and `indexer`. Video is out of the first cut. Checks: a picture's rows match the torch reference (transformers'
processor, the ViT on the checkpoint) at relL2 4 to 6e-3 for PNGs (JPEGs add about 1e-2: stb_image and libjpeg decode apart); fed
llama.cpp's own rows with 1-D positions, NVFP4 two cards keeps the BF16 GGUF's first 12 greedy tokens on three prompts.
