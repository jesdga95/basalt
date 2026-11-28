# Inkling 2 Mini pack

The packs (`Inkling-2-Mini-NVFP4-DQ8.basalt` and `Inkling-2-Mini-NVFP4-MX8.basalt`) come from Thinking Machines Lab's official release alone
([thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0): both safetensors files
(`model` and the DSpark drafter `mtp`), `config.json`, `hf_quant_config.json`, the tokenizer files and the chat template. No other
input. [checkpoint_inkling2.py](../../tools/convert/checkpoint_inkling2.py) is what `convert.py --checkpoint` runs on a checkpoint of
model type `inkling2_loop`; `--drafter` and `--head` do not apply (the drafter is in the release).

```sh
hf download thinkingmachines/Inkling-2-Mini --local-dir Inkling-2-Mini
python3 tools/convert/convert.py --checkpoint Inkling-2-Mini --out Inkling-2-Mini-NVFP4-MX8.basalt --name "Inkling 2 Mini"
```

- `experts` and `experts.layout`: the routed experts of each of the 24 layers, 128 to a layer, one blob per expert in the order
  `[gate | up | down]` (the checkpoint's `w13` rows alternate gate and up; they are separated). `experts.layout` is a text file with
  one `layer gu_type d_type offset blob_bytes` line per layer after a header line.
  - Layers 0 to 19 (the release's NVFP4): NVFP4 (type 40) blobs. The E2M1 codes and the E4M3 scale of every 16
    values move bit for bit, the nibbles regrouped into 64-value blocks; the blob ends in an FP32 tail `{s_gate, s_up, s_down, 0}`
    with `s = amax / outer_scale_denominator` of the expert's matrices (gate and up share one `s`, from `w13`).
  - Layers 20 to 23 (the release's MXFP8), MX8 pack: MXFP8 (type 46) blobs, no tail: 128-value blocks of four
    UE8M0 scales then 128 E4M3 codes (132 bytes), the release's own bits.
  - The same layers in the DQ8 pack: Q8_0 blobs, no tail (34 bytes per 32 values). Each value is its E4M3 code times
    2^(UE8M0 - 127) in ggml's reference Q8_0 rounding; one Q8_0 block is one MXFP8 block. Rounding error:
    [precision](../precision/inkling2.md). `convert.py` writes the MX8 form; both load.
  - The engine reads a row of an expert 256 values at a time, so it allocates each layer's experts with `mmq::read_past_bytes`
    extra zeroed bytes after the last blob; the file holds none.
- `gguf`: every other text tensor in llama.cpp-style names. The attention projections (q, k, v, relative-bias, gate, sink,
  output) and the k and v short convolutions' weight projections Q8_0 (ggml's reference rounding of the BF16 values); the token embedding and the output head BF16 as stored, their
  padded rows past the real vocabulary (200,058 of 201,024) dropped, the engine rounding the head to Q8_0 as it loads; the loop
  controllers BF16 as stored; norms, the block-residual and message queries, the router and its bias, the short-convolution taps,
  the relative-bias tables and the clip scales F32 (exact). The drafter's tensors are in it as `dspark.*`, BF16 and F32 as stored.
- `vision/mmproj`: the vision tower as llama.cpp's clip GGUF (projector `inkling2v`), BF16 and F32 as stored.
- `tokenizer/*` and `meta`: the release's tokenizer files and chat template, its geometry (24 layers, the prelude, core and coda
  ranges, the loop's passes, the experts) and its license; the pretokenizer is the checkpoint's o200k split.
- Every rounding is deterministic and runs once, in pieces by
  worker processes. How each tensor rounds and what it costs in quality: [precision](../precision/inkling2.md).
