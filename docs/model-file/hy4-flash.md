# Hy4-flash packs

## Hy4-flash NVFP4 pack

The NVFP4 pack (`Hy4-flash-NVFP4.basalt`) comes from Tencent's
BF16 checkpoint ([tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash), 148 shards, Tencent Hy License 1.0), through three
inputs: the checkpoint converted to a BF16 GGUF by llama.cpp's `convert_hf_to_gguf.py --outtype bf16` (llama.cpp 0a054d0b0, which
has `hy_v4_flash`), the vision projector as llama.cpp's `mmproj-Hy4-flash-BF16.gguf`, and the DFly2 drafter, the checkpoint's
`model.dfly_layers.*` tensors (its last shard, `model-00148-of-00148.safetensors`, or a side file with those 87 tensors).

```sh
python3 tools/convert/convert.py --gguf Hy4-flash-BF16.gguf --nvfp4 --hf-dir tencent-Hy4-flash \
    --mmproj mmproj-Hy4-flash-BF16.gguf --dfly tencent-Hy4-flash/model-00148-of-00148.safetensors \
    --out Hy4-flash-NVFP4.basalt --name "Hy4-flash NVFP4"
python3 tools/convert/convert.py --verify Hy4-flash-NVFP4.basalt --gguf Hy4-flash-BF16.gguf
```

- Routed experts (layers 1 to 41) NVFP4 (type 40): blobs `[gate | up | down]` and an FP32 tail `{s_gate, s_up, s_down, 0}`; per matrix the tensor scale amax / (6 x 448), per 16 values the UE4M3 scale with the least squared error among half
  to four times amax / 6 (the 27B pack's rounding).
- The BF16 projections (attention, latent, shared expert, dense FFN, Engram key/value, indexer, embedding) Q8_0 in ggml's reference
  rounding; `--dense-q6k` makes the projections with whole 256-value rows Q6_K. The head stays BF16 (the engine rounds it to Q8_0 as it
  loads); norms, hc, router and biases F32 as stored.
- The Engram table: E4M3 codes (type 43), to nearest with ties to even, no scale; the exact fraction and RMS error go to
  `meta.quantized_from.engram`.
- Tokenizer and chat template are the GGUF's, refused unless they equal `--hf-dir`'s. `vision/mmproj` and `dfly` hold the side files
  whole; `--hf-dir` also gives `LICENSE` and `config.json`.
- Every rounding is deterministic and runs once,
  in pieces by worker processes. `--verify` repeats each rounding against the BF16 GGUF and compares the bytes; it takes as long as
  the build. How each tensor rounds: [precision](../precision/hy4-flash.md#weights).

## Hy4-flash test pack

From [AngelSlim/Hy4-flash-GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF)'s Q4_K_M (`general.architecture` `hy_v4_flash`)
plus the model's Hugging Face directory ([tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash): `LICENSE` and `config.json`
only). Nothing is requantized: every tensor keeps its GGUF bytes and name.

```sh
python3 tools/convert/convert.py --gguf Hy4-flash-Q4_K_M-00001-of-00002.gguf --hf-dir tencent-Hy4-flash \
    --out Hy4-flash-Q4_K_M-test.basalt --name "Hy4-flash Q4_K_M (test pack)"
python3 tools/convert/convert.py --verify Hy4-flash-Q4_K_M-test.basalt --gguf Hy4-flash-Q4_K_M-00001-of-00002.gguf
```

- `dense`, `dense.index`, `gguf`: as the DeepSeek pack (`gguf` holds every non-expert tensor, so float tensors are also in
  `dense`). Three-dimensional tensors (`attn_k_b`, `attn_v_b`) keep their GGUF shape in `meta.tensors`; their dense-index row is
  `(ne0, product of the rest)`. The projections (attention, latent, shared expert, layer 0's FFN, Engram key/value, indexer)
  and the token embedding are Q8_0, the head BF16, norms, hyper-connections, router and biases F32.
- `experts`, `experts.layout`: one row per MoE layer by layer id, 1 to 41 (layer 0 is dense and has no row). Gate and up are
  Q4_K, down Q6_K (19 layers) or Q4_K (22), so a layer's blob is one of two sizes.
- `engram`: the table, 96 x 660,006,416 `F8_E4M3_RAW` (llama.cpp's type 43: one E4M3 code per value, no scale), 2 MiB aligned.
- `mtp/dense`, `mtp/dense.index`, `mtp/experts`, `mtp/experts.layout`: the GGUF's next-n block (the last block), BF16 as stored,
  laid out like the main sections (`meta.mtp.dense_shapes` has the 3-D shapes). The official checkpoint has no such layer, so
  the engine may ignore these sections.
- `meta`: `arch` `hy4flash`, `geometry`, `engram`, `rope` (MRoPE sections from `config.json`), `hf_config` (the whole
  `config.json`: vision tower and DFly2 settings), `license`. `vision` and `dfly2` are null until those sections are attached.
- `--verify` reads the pack back: every section's SHA-256, each source tensor found where the index, layout or `meta.tensors`
  puts it, with its type, shape, size and SHA-256.
