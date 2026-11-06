# Adding a model

A model lives in `src/models/<arch>/` and `include/basalt/models/<arch>/`. Shared code never names it
(`tools/check_model_layering.sh` checks this).

## What a model writes

- An adapter (`include/basalt/model/adapter.hpp`), registered in `src/models/builtin.cpp`: validate the file, describe it, load it,
  name its tokens, pre-tokenizer and tool-call form.
- A `ModelSession` with `decode_window`, `prefill_chunk` and the state calls it supports.
- A dialect in `serve/models/`: prompt rendering, output parsing, stop texts.

The router then needs one config entry. Nothing in the server names a model.

## Fixtures, flags and loading

- Benchmark token ids go in `data/bench/<arch>`. `--quality` needs a BF16 reference.
- Flags: `parse_flag` and `finish_flags`. A flag a model lacks is the normal unknown-argument error. Token id flags default to the pack's tokens.
- Report loads through `core::LoadProgress`. Use `core/ram_load` for the RAM tier.
- Conversations: `save_cells` and `restore_cells` give the RAM and disk tiers what they need.
- A model with its own fast path derives from `serve::Runtime` and `serve::Request`; see `src/models/qwen4exp/`.
- Kernels: shared kernels stay in `src/kernels/`. Every new kernel gets a parity test and a microbench line.
