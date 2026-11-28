# ERNIE-5.5-Flash pack

The pack (`ERNIE-5.5-Flash-Preview-MXFP4.basalt`) is built directly from Baidu's
official Hugging Face release ([baidu/ERNIE-5.5-Flash-Preview](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview), Apache-2.0
preview weights): safetensors shards, `config.json`, `generation_config.json`, tokenizer files, and the bundled `model.mtp.*`
DSpark drafter tensors. [checkpoint_ernie55.py](../../tools/convert/checkpoint_ernie55.py) is what `convert.py --checkpoint` runs on
a checkpoint of model type `ernie5_5_moe`; [dspark_ernie55_attach.py](../../tools/convert/dspark_ernie55_attach.py) appends the
`dspark` section afterwards without reconverting the main weights.

```sh
hf download baidu/ERNIE-5.5-Flash-Preview --local-dir ERNIE-5.5-Flash-Preview
python3 tools/convert/convert.py --checkpoint ERNIE-5.5-Flash-Preview --out ERNIE-5.5-Flash-Preview-MXFP4.basalt --workers 16
python3 tools/convert/dspark_ernie55_attach.py --checkpoint ERNIE-5.5-Flash-Preview --into ERNIE-5.5-Flash-Preview-MXFP4.basalt --workers 16
```

- `experts` and `experts.layout`: the routed experts of the 42 MoE layers (layers 1 to 42), 512 per layer, one
  MXFP4 (type 39) blob per expert in `[w1 (gate) | w3 (up) | w2 (down)]` order, packed bit for bit from the
  release's paired E2M1 nibbles and UE8M0 block scales (17 bytes per 32 values). `experts.layout` carries one
  `layer 39 39 offset blob_bytes` line per MoE layer.
- `gguf`: every other language-model tensor under Basalt's `ernie55` names. 128×128 blockwise FP8 projections
  dequantized to F32 and rounded once to Q8_0 (ggml's reference rounding); `token_embd.weight`, `output.weight`, compressor
  projections and indexer `wk` / `weights_proj` / `flash_gate` kept as BF16; norms, mHC mixers (transposed to `[out, in]`) /
  biases / stacked `[pre, post, res]` scales, folded VHA PostMix (`I + V @ U^T`), attention sinks, compressor APE, split-feature
  routers (`[w0; w1]`), router bias & per-expert scale, and lm_head multimax parameters as F32. Optional FlashIndexer tensors
  (`flashindexer/flashindexer.safetensors`) are included automatically when present.
- `dspark`: the 3-stage SWA-mHC DSpark drafter (`model.mtp.*`): 64-byte-aligned dense, attention, mHC, Markov
  (`markov_w1` transposed BF16, `markov_w2` Q8_0) and confidence-head tensors, followed at a 4096-byte boundary by contiguous
  MXFP4 routed expert blobs in `[w1 | w3 | w2]` order, indexed in `meta["dspark"]`. Appending leaves every earlier section
  untouched and `--roll-back` restores the pack bit for bit.
- `tokenizer/*` and `meta`: the release's BPE vocabulary, merges, token types, metadata and license.
- How each tensor rounds and what it costs in quality: [precision](../precision/ernie55.md).
