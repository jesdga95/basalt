# Adding a model

A model lives in `src/models/<arch>/` and `include/basalt/models/<arch>/`. Shared code never names it
(`tools/check_model_layering.sh` checks this).

## What a model writes

- An adapter (`include/basalt/model/adapter.hpp`): validate the file, describe it, load it, name its tokens, pre-tokenizer and
  tool-call form. It is `basalt::models::<arch>::make_adapter()`, in the library `model_<arch>`.
- A `ModelSession` with `decode_window`, `prefill_chunk` and the state calls it supports.
- A dialect in `serve/models/`: prompt rendering, output parsing, stop texts.

Registering it is one word: the id in `BASALT_BUILTIN_MODELS` (`cmake/models.cmake`), which adds it to `-DBASALT_MODELS`
([building](building.md#building-only-some-models)) and to the registration in `src/models/builtin.cpp`. Everything of the model in
`CMakeLists.txt` (its libraries, kernels, tests and tools) sits under `if(BASALT_MODEL_<ARCH>)`, so a build without it compiles none of it.

The router then needs one config entry. Nothing in the server names a model.

## Fixtures, flags and loading

- Benchmark token ids go in `data/bench/<arch>`. `--quality` needs a BF16 reference.
- Flags: `parse_flag` and `finish_flags`. A flag a model lacks is the normal unknown-argument error. Token id flags default to the pack's tokens.
- Report loads through `core::LoadProgress`. Use `core/ram_load` for the RAM tier.
- Conversations: `save_cells` and `restore_cells` give the RAM and disk tiers what they need.
- A model with its own fast path derives from `serve::Runtime` and `serve::Request`; see `src/models/qwen4exp/`.
- Kernels: shared kernels stay in `src/kernels/`. Every new kernel gets a parity test and a microbench line.
