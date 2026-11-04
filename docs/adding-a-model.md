# Adding a model

A model lives in `src/models/<arch>/` and `include/basalt/models/<arch>/`. Shared code never names it;
`tools/check_model_layering.sh` holds that, for includes, CMake targets, names, token ids and widths.

## The least a model writes

- An adapter (`include/basalt/model/adapter.hpp`), registered in `src/models/builtin.cpp`: validate the file, describe it,
  load it, name its tokens, its pre-tokenizer and its tool-call form.
- A `ModelSession`: `decode_window`, `prefill_chunk`, and the state calls it supports.
- A dialect in `serve/models/`: prompt rendering, output parsing, stop texts.

The shared server then runs it: the request keys, sampling through `sample_tokens`, grammar, the checkpoint chain, batch
slots taking each slot in turn, `VRAM`, `SLEEP`, `WAKE`, `INFO`, `PT` and `GENT`.

## Benchmarks

Fixtures hold token ids, so each model has its own in `data/bench/<arch>`; bench and probe pick that folder when it exists. `--quality` needs a BF16 reference; a model without one refuses its flags.

## Flags

The common flags are in `src/program/cli/arguments.cpp`. A model adds its own through `parse_flag`; any other flag is the
normal `unknown argument` error for that model. No flag's default differs by model. Token id flags default to the pack's
own tokens, which the adapter names.

## Loading

Report the load through `core::LoadProgress`: the total, a plan per target, then each phase. The server's bars and the
`/health` percent read those lines. Auto-picked settings go on one `auto flags:` stdout line.

The RAM tier goes through `core/ram_load`: give it a layout (`ram_arena_runs`, `ram_layer_runs` or your own runs) and it does
the unbuffered reads, the bar and the pin with its fall-backs. Lookup stays yours.

## Probe

`ModelSession::probe_env` names the CPU pool, the RAM blobs, the expert formats and, with a second card, a `ProbePeerPath`
timing your own peer launch. The probe asks the engine which tuning flags the model takes and runs only those tests.

## Conversations

`save_cells` and `restore_cells` hand positional state as `core::ConversationKv` entries, a row being `page_size` positions.
The RAM and disk tiers then work with no more code. A restore gets what a cut at any position left.

## A model with its own fast path

Derive from `serve::Runtime` and `serve::Request`, and instantiate `Request::decode_loop` with `kSpeculates` set. See
`src/models/qwen4exp/`. Its decode tail, persistent experts and mid chain take their widths from
`include/basalt/models/qwen4exp/geometry.hpp`.

## Kernels

Shared kernels stay in `src/kernels/`. A kernel whose split or sharing would change a model's numerics ships in the model's
folder. Fast math is set per file and follows the file. Every new kernel gets a parity test and a microbench line.
