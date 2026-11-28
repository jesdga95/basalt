# Gemma 4 pack

From Google's BF16 checkpoint alone ([google/gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it),
[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) or
[google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it)), with its assistant drafter (the same name with
`-assistant`) when `--drafter` names it. [checkpoint_gemma4.py](../../tools/convert/checkpoint_gemma4.py) is what
`convert.py --checkpoint` runs on a checkpoint of model type `gemma4` or `gemma4_unified` (the 12B). The 31B needs
`--mlp-nvfp4` to fit a 32 GB card.

```sh
hf download google/gemma-4-26b-a4b-it --local-dir gemma4
hf download google/gemma-4-26B-A4B-it-assistant --local-dir gemma4-assistant
python3 tools/convert/convert.py --checkpoint gemma4 --drafter gemma4-assistant --out Gemma-4-26B-A4B-it.basalt --name "Gemma 4 26B A4B"
hf download google/gemma-4-31B-it --local-dir gemma4-31b
hf download google/gemma-4-31B-it-assistant --local-dir gemma4-31b-assistant
python3 tools/convert/convert.py --checkpoint gemma4-31b --drafter gemma4-31b-assistant --mlp-nvfp4 --out Gemma-4-31B-it.basalt --name "Gemma 4 31B"
```

- `experts`, `experts.layout`: each layer's routed experts, one NVFP4 (type 40) blob per expert `[gate | up | down]` (gate the
  first half of `gate_up_proj`), per matrix the tensor scale amax / (6 x 448), per 16 values the UE4M3 scale of least squared
  error among half to four times amax / 6, then the FP32 tail `{s_gate, s_up, s_down x per_expert_scale, 0}`. The layers that
  `--experts-mxfp8 LAYERS` names (e.g. `0,5,10-12`) hold MXFP8 (type 46) blobs instead, no tail: 128-value blocks of four UE8M0
  scales then 128 E4M3 codes, per 32 values the smallest power of two that brings the largest magnitude within 448, the codes
  to nearest; the down rows padded with zero columns to whole 128-value blocks; the per_expert_scale folded into the down
  weights before rounding. `meta` notes those layers (`checkpoint.experts_mxfp8`).
- `gguf`: the attention and MLP projections `F8_E4M3` (type 44) with one FP32 scale per matrix (amax / 448, `<name>.scale`), the
  MLP's inner width zero-padded to a multiple of 128; `token_embd.weight` BF16 as stored; `output.weight` the tied embedding as
  `F8_E4M3` rows with `output.scale` per row; `ffn_gate_inp.weight` the router F32 with its input scale and 1 / sqrt(hidden)
  folded in; norms and `layer_scalar` F32. The drafter's tensors as `assistant.*`, rounded the same way. With `--mlp-nvfp4`
  the MLP's `ffn_gate`, `ffn_up` and `ffn_down` are `NVFP4_PLANES` (type 45, unpadded: the widths must be multiples of 256),
  rounded as the experts are, the tensor scale as `<name>.scale`.
- `vision/mmproj`: the vision tower as llama.cpp's clip GGUF (projector `gemma4v`): the patch embedding as the F32 convolution,
  the column and row position tables, the norms and the standardisation F32, the products and the embedding projection
  (`mm.input_projection.weight`) BF16 as stored. The 12B's patch embedder and audio projection are not converted (the
  converter says so): its pack has no `vision/mmproj`.
- `tokenizer/*`: the checkpoint's SentencePiece-style BPE as the engine's tables: a token's bytes are its text with U+2581 as a
  space, spelled in GPT-2's byte alphabet; the byte-fallback tokens keep their `<0xNN>` text and type 6. The engine starts a
  piece from its characters (the fallback bytes for a character the vocabulary lacks) and merges by rank, as the checkpoint's
  tokenizer does.
- `meta`: `arch` `gemma4`, the geometry (`gemma4.*`: the layer pattern, both head widths and key/value head counts, the window,
  both rotary bases and the partial factor, `k_eq_v`, the embedding scale as BF16 rounds sqrt(hidden), the softcap, the experts,
  `gemma4.assistant.*`), the checkpoint's config and license.
