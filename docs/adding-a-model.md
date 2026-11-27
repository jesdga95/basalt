# Adding a model

A model lives in `src/models/<arch>/` and `include/basalt/models/<arch>/`. Shared code never names it
(`tools/check_model_layering.sh` checks this).

## What a model writes

- An adapter (`include/basalt/model/adapter.hpp`): validate the file, describe it, make its `LoadedModel`, name its tokens,
  pre-tokenizer and tool-call form. It is `basalt::models::<arch>::make_adapter()`, in the library `model_<arch>`.
- A `ModelSession` with `decode_window`, `prefill_chunk` and the state calls it supports.
  A model whose conversations resume from checkpoints lists its running state in `running_state` (device regions, in a fixed
  order, and a host head) and takes the head back in `running_restored`; the shared engine copies it through page-locked blocks
  (`core/checkpoint_transfer.hpp`) and never reads the model's memory itself.
  A model whose caches hold state that only moves forward - a recurrent state, a conv history - has to restart that state when a read
  starts at `pos == 0`: the engine keeps one session per slot across conversations, and a model that keeps no checkpoints gets no other
  signal that a conversation begins (there is no reset hook on `ModelSession`, and the two calls that could bring a parked conversation
  back - `running_state` and `save_cells` - are the ones such a model declines; serve reads a declined restore as its cue to drop the
  resume and read the prompt from the start, `restore_image` in `serve/session_engine.cpp`, so `pos == 0` really is where a fresh
  conversation arrives). Skipping it does not corrupt one conversation, only every later one in that slot, so it is a
  perplexity number and not a crash (`glm5next`: `GlmCaches::reset` from both `prefill_chunk` and `decode_window`; `deepseek41` trims
  its ops at the same place).
  Not every forward-only buffer needs it, only the ones a read takes before it writes, so ask that of each in turn: a cache written at
  every position a window reaches cannot serve a stale row, and neither can a handoff keyed by absolute position that every window
  rewrites (`glm5next`'s MLA latents and indexer pool, and the drafter's kept tail in `mtp_prev_*`, which `mtp_keep_row` selects by
  position and `mtp_rows` overwrites on every drafting window).
- A dialect in `serve/models/`: prompt rendering, output parsing, stop texts.

Registering it is one word: the id in `BASALT_BUILTIN_MODELS` (`cmake/models.cmake`), which adds it to `-DBASALT_MODELS`
([building](building.md#building-only-some-models)) and to the registration in `src/models/builtin.cpp`. Everything of the model in
`CMakeLists.txt` (its libraries, kernels, tests and tools) sits under `if(BASALT_MODEL_<ARCH>)`, so a build without it compiles none of it.

The router then needs one config entry. Nothing in the server names a model.

## Fixtures, flags and loading

- Benchmark token ids go in `data/bench/<arch>`. `--quality` needs a BF16 reference.
- Flags: `parse_flag` and `finish_flags`. A flag a model lacks is the normal unknown-argument error. Token id flags default to the pack's tokens.
- The start (`ModelAdapter::start`) runs in three steps. The model's `plan` decides and allocates everything, in the order its
  sizing depends on (weights, then what is sized from the memory left: caches, tiers, a peer's prompt stages), and queues every
  range of the file it reads in one `LoadPlan` (`basalt/core/loader.hpp`), with copies of bytes the plan itself places (a second
  home, a stage's image) as memory sources; nothing is placed yet. The start announces the plan's bytes as the one total, places
  them in one unbuffered pass on a thread pool (the ranges bound for a card first, the main card then the others, each streamed
  through small pinned staging; then the ranges bound for RAM; the one bar names the stage and its own part: `loading model
  weights: cuda0 3.1/12.5`, ..., `ram`; each destination's share is summed from the plan and the result line splits the
  total by it: `loaded 76.8GiB 30.1s  cuda0=12.5GiB ram=64.3GiB`), then calls the model's `attach`: what needs the bytes landed (packed forms,
  checks, timing a peer's cards) and the sessions, when nothing was sized after them. A model makes its sessions in
  `plan` instead when a cache or tier sizes itself from the memory they leave. `core/ram_load` allocates and pins the RAM tier:
  `plan` allocates it, the pass gives it its pages as it reads into it, and it is pinned once its reads have landed (in
  `attach`, or as a step of the plan, `LoadPlan::after_reads`, when the pass copies from it), so a start holds no RAM before
  its bytes arrive; a tier sized before that pin leaves the VRAM the pin maps on its card free (`pin_mapping_bytes`);
  a peer's prompt stages take the shared `core::PromptStage` and its start lines (`basalt/core/prompt_stage.hpp`).
- Conversations: `save_cells` and `restore_cells` give the RAM and disk tiers what they need.
- Several GPUs: `ModelAdapter::gpu_modes()` states the most cards each `gpu-mode` takes (0: not run; the shared code refuses it before
  the load, listing the modes the model runs). A mode is a hook class the loaded model derives from, like batching:
  `model::PeerHooks` (what a peer holds, its room, what the model does with the shared deal) and `model::SplitHooks` (what each layer
  needs and what each card has; the shared driver cuts by the shares). The model receives `GpuCards` in its `LoadRequest` and writes
  each card's part of the start line; it never prices a mode or picks a cut. Check it with `--diag-gpu-check` and
  `tools/gpu_identity.py` against one card ([several GPUs](design/multi-gpu.md)).
- `--parallel` slots in one pass through the weights: derive the session from `model::BatchSession`, write its hooks and run
  the batch check, see [batched decoding](design/batching.md#adding-batching-to-a-model).
- A model with its own fast path derives from `serve::Runtime` and `serve::Request`; see `src/models/qwen4exp/`.
- A dense model of GDN and gated attention layers (the Qwen3.5 family) needs only its pack read into `dense::Geometry` and
  `dense::Weights` (`include/basalt/dense/model.hpp`): the shared dense stack runs the layers, the drafters (MTP layer or DFlash
  2 block drafter), images and video, and gives the session engine its sessions. See `src/models/qwen35/`.
- A Qwen chat model takes the shared pre-tokenizer, tool form and token names (`basalt/program/qwen_forms.hpp`) and the
  shared Qwen dialect (`serve/models/qwen.py`).
- Kernels: shared kernels stay in `src/kernels/`. Every new kernel gets a parity test and a microbench line.
