# Configuration reference

Every setting a user can pass to Basalt: the server options, the engine flags, the per-request sampling fields and
the model file's sampling defaults. For what a setting does to speed, memory and output precision see the
[running](../running.md) and [precision](../precision/index.md).

- [How settings are given](#how-settings-are-given)
- [Server](#server)
- [Model and GPUs](#model-and-gpus)
- [Context and KV cache](#context-and-kv-cache)
- [Expert cache and memory](#expert-cache-and-memory)
- [Prefill](#prefill)
- [Speculative decoding (MTP)](#speculative-decoding-mtp)
- [Parallel slots](#parallel-slots)
- [Conversation cache](#conversation-cache)
- [Vision](#vision)
- [Logging and diagnostics](#logging-and-diagnostics)
- [Control vectors](#control-vectors)
- [Sampling](#sampling)
- [Example configurations](examples.md)

## How settings are given

```sh
bin/basalt serve --config basalt_config.json [options]
bin/basalt serve MODEL.basalt [--python PY] [options]
```

`basalt serve` runs the Python server (`serve/`) over the engine binary. `MODEL.basalt` as the first argument stands for
`--model MODEL.basalt`; `--python` picks the interpreter (default: `.venv/bin/python` in the source tree, else
`python3`). Everything else goes to the server.

There are two kinds of setting:

- **Server options** belong to the Python server: model path, address, port, slots, vision, logging. On the command line
  they are `--name value`.
- **Engine flags** belong to the engine binary (`bin/basalt --help`): context, KV format, expert cache, prefill,
  speculation. The server passes them on. Put them on the command line like any other flag (`--kv int8`). The server passes
  them to the engine as typed, in order, and does not interpret their values, so a flag that takes several values
  works too (`--control-vector-layer-range 4 44`). It asks the engine binary which flags it accepts, and stops at
  startup on a flag name it does not know.

A config file is one JSON object with two optional blocks:

```json
{
  "server": { "model": "model.basalt", "port": 8080 },
  "engine": { "kv": "int8", "resident-experts": true, "control-vector-layer-range": [4, 44] }
}
```

- Keys are the flag names without the dashes (`_` works for `-`). Values are strings or numbers; a switch is `true`, and
  `false` leaves it off. A repeatable option (`allowed-host`, `cors-origin`) takes a list.
- `model` and `tokenizer` paths are relative to the config file.
- **Precedence:** command line > config file > built-in defaults. The engine receives the config file's flags, then
  the command line's, and a flag given twice takes its last value.
- An engine flag in `server`, or a server option in `engine`, stops the server at startup, as does an unknown name.
- `api-key` cannot be set in a config file; a file that sets it is refused.
- The built-in defaults are untuned (4096 context, fp16 KV, no speculation, no expert cache). A config file is how
  you tune them. The startup line `[server] engine: ...` lists what came from the file or the command line.

In the tables, "default" is what applies when the setting is absent everywhere.

## Server

Server options. Command line `--name`, config `server` block `"name"`.

| Setting | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- |
| `model` | path | none (required) | The `.basalt` model file. The weights, tokenizer, chat template, drafter, vision projector and default sampling all come from it. |
| `alias` | string | the name stored in the model file | The model name the API reports. Change it when a client expects a particular name. |
| `host` | address | `127.0.0.1` | The address to listen on. `127.0.0.1` is this machine only; `0.0.0.0` also serves the network. Never listen beyond `127.0.0.1` without `api-key`. |
| `port` | integer | `8080` | The TCP port. |
| `api-key` | string | none | Require this key on `/v1/*` (`Authorization: Bearer ...` or `x-api-key`). Command line only. Set it whenever `host` is not `127.0.0.1`. |
| `allowed-host` | string, repeatable | none | Also answer to this `Host` name; `"*"` turns the check off. Needed when clients reach the server by a name other than its address. |
| `cors-origin` | string, repeatable | none | Let web pages from this origin call `/v1/*`. |
| `fit-max-tokens` | switch | off | Clamp a request's `max_tokens` to the context that remains instead of rejecting it with 400. |
| `open` | switch | off | Open the local web page in the browser once the model is ready. |
| `lazy` | switch | off | Start the text-only API unloaded and load the model on the first request. Turns vision off. |
| `idle-unload` | seconds | off | Unload the model after this long without requests; the next request loads it again. Frees VRAM for other programs. |
| `min-free-vram-mib` | integer | off | Load an unloaded model only when this much VRAM is free, otherwise answer 503. Pairs with `lazy` and `idle-unload`. |
| `temperature`, `top-p`, `top-k`, `min-p`, `presence-penalty`, `frequency-penalty`, `repetition-penalty`, `penalty-last-n`, `seed` | number (`top-k`, `penalty-last-n`, `seed`: integer) | the model file's `sampling` value, else the engine's | The sampling default for a request that leaves that field out, over the model file's. Same meaning and allowed values as the [model file's sampling keys](#the-model-files-sampling-defaults). The server applies them to each request; the engine never sees them. |
| `tokenizer` | path | `pack/full/tokenizer` in the source tree (a byte tokenizer if absent) | Tokenizer directory. Mock engine only. |
| `mock-engine`, `script` | switch, string (repeatable) | off, a short greeting | Testing: run a scripted engine instead of the model, for clients and tests (no `model` needed); `script` is its answer, and given more than once, requests get them in turn. |
| `config` | path | none | The config file. Command line only. |

## Model and GPUs

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `gpu` | server | list of GPU numbers, e.g. `"0"` or `"0,1"` | every visible GPU | The cards to use, numbered as `nvidia-smi` numbers them. The first runs the model. A second card becomes the peer expert tier. Set it to keep other cards free or to pick which card is the main one. |
| `peer-device` | engine | integer | off; the server sets `1` when two or more cards are visible | A second GPU as an adaptive expert-cache tier: it holds experts and computes their prompt rows. Rarely set by hand; `gpu` does it. Not with `layer-split` or `expert-cache-device1..3`. |
| `peer-exclusive` | engine | switch | off | The peer card's experts live only there, not also in RAM. Uses less RAM, prefill is slower. Needs `resident-experts`. |
| `peer-reserve-mib` | engine | integer MiB | 600 | VRAM the peer tier leaves free on its card. |
| `peer-slots` | engine | integer | as many as fit | Expert slots on the peer card. Lower it to leave the card room for something else. |
| `peer-prefill-rows` | engine | integer | half of chunk x top-k | Prompt rows per layer the peer computes. `0` keeps the prompt path on the main card. |
| `layer-split` | engine | layer number, `K1,K2,...` or `auto` | off (one GPU) | Advanced: run the model in stages across GPUs, each stage a block of layers. The peer tier is the usual way to use a second card. |
| `split-device` | engine | GPU numbers | the next visible GPUs | With `layer-split`, the devices of the later stages. |
| `split-skip-if-fits` | engine | switch | off | Advanced. With `layer-split auto`, run on the first card alone when it holds every profiled expert, the session, the drafter and the reserve; a split then only adds hand-offs. |
| `batch-groups` | engine | integer | 1 | Advanced. With `parallel` and `layer-split`, put the slots in this many groups pipelined through the GPUs; must divide the slot count. |
| `trim-stage-weights` | engine | switch | off | Advanced. With an explicit `layer-split` list, each GPU loads only its own layers' dense weights. |
| `pcie-frac` | engine | number 0..1 | 0.55, scaled down on a slow link | The share of each layer's missed experts the GPU reads over PCIe from the pinned RAM copy while the CPU computes the rest. `0` keeps every miss on the CPU (used to pin placement in identity checks). |
| `pcie-mode` | engine | `auto`, `dma`, `kernel`, `direct` | `auto` (= `kernel`) | How that share reaches the GPU: `dma` copies through staging with the copy engine, `kernel` stages it with a copy kernel inside the graph, `direct` has the grouped kernel read the mapped arena. |

## Context and KV cache

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `max-context` | engine | integer tokens | 4096 | The longest context the engine allows. Raise it for long prompts; the KV cache grows with it. With `parallel`, this is shared by the slots. |
| `kv` | engine | `fp16`, `int8`, `q4_0`, `k8v4` | `fp16` | KV storage. `int8` halves the VRAM. `q4_0` is 4-bit after a Hadamard rotation, half of `int8` again, slightly lower precision. `k8v4` keeps K in int8 and V in rotated 4-bit (816 B per cell against `int8`'s 1056); not with `kv-resident`. Use `int8` for long contexts; go lower when VRAM is short. See [precision](../precision/index.md). |
| `kv-resident` | engine | integer cells, minimum 20480, or `0` | 0 (all KV in VRAM) | Keep only this many cells of each attention layer's KV in VRAM and the whole K/V in pinned RAM; the freed VRAM goes to expert slots. A context that fits in N cells is not streamed. For very long contexts on a small card. |
| `rope-scaling` | engine | `none`, `linear`, `yarn` | the model file's, else `none` | Extend the context past the trained length. Fixed for the run: K in the cache is stored after RoPE. |
| `rope-scale` | engine | number | the model file's, else 1 | The extension factor for `linear` and `yarn`. |
| `rope-freq-base`, `rope-freq-scale` | engine | number | 0 (the model's own) | Raw frequency base and angle shrink. Leave at 0 unless you know the values you want. |
| `yarn-orig-ctx`, `yarn-ext-factor`, `yarn-attn-factor`, `yarn-beta-fast`, `yarn-beta-slow` | engine | number | 0 (262144), auto, 1, 32, 1 | YaRN's own parameters, as in llama.cpp. |
| `mtp-window` | engine | integer cells | 32768 | The drafter attends to the last N cells; `0` is every cell. |

A server with `max-context` above 32768 has an elastic KV cache: each session backs its first 32768 cells at startup and
takes more from the expert cache only when a request needs it. It is off when the model is split over GPUs with
`layer-split` (a peer card is fine) and with `kv-resident` or `vram-elastic`. See [running](../running.md#concurrent-requests---parallel-n).

## Expert cache and memory

The model's experts live in VRAM (the expert cache), in RAM, or in the file. These settings decide where.

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `expert-cache` | engine | integer or `auto` | 0 (none) | Experts held in VRAM. `auto` uses every slot that fits after the weights, the sessions and `vram-reserve-mib`. Almost always `auto`. |
| `vram-reserve-mib` | engine | integer MiB, at least 256 | 700 (lowered automatically, down to 300, on a card that would otherwise get too few slots; a value you give is kept) | VRAM left free on the main card for graphs, scratch and the head when `expert-cache auto` sizes the cache. Below 256 is refused. Raise it if the machine pages GPU memory or other programs need VRAM. |
| `resident-experts` | engine | switch | off | Keep the experts the GPU does not hold in page-locked RAM, so none is read from the file again. Leaves a RAM floor free for the OS. Recommended whenever the RAM can hold them. |
| `resident-budget-gib` | engine | number GiB | off | Like `resident-experts` but with at most N GiB of experts in RAM, hottest first. Refused when it would leave less than the RAM floor free; the error names the largest safe value. |
| `resident-cpu-experts` | engine | switch | off | The lower-level form of `resident-experts` (with `mmap-experts`): experts in ordinary RAM. Use `resident-experts` instead. |
| `mmap-experts` | engine | switch | off | Opt out of the resident arena and back to a file mapping. Speed then depends on the OS page cache. |
| `shared-expert-arena` | engine | file path | off | Back the resident arena with one shared file so several processes use one copy. Put it on `/dev/shm`. |
| `pool-workers` | engine | integer | 0 (every physical core but the host loop's) | CPU expert pool threads, lowered automatically to fit a container CPU quota. |
| `pool-affinity` | engine | `all`, `auto`, `p-cores` | `all` | CPU affinity of the workers. `auto` and `p-cores` suit hybrid CPUs. |
| `expert-profile-save` | engine | file path | off | Save what the adaptive tier learned (experts in VRAM and routing counts) to this file on exit and periodically. `tools/convert/convert.py --profile` reads it. |
| `expert-profile-save-every` | engine | minutes | 10 (`0` = on exit only) | How often the profile is saved. |
| `adapt-every` | engine | integer | 4 (`0` = static residency) | How many decode windows pass between adaptive rounds. A round moves experts between the tiers (see below). `1` follows the conversation fastest. |
| `adapt-decay` | engine | number, 0 to 1 | 0.93 on a single card (with or without the peer); 0.7 when the experts are split over several cards | The factor the usage counts are multiplied by after each round. Lower forgets sooner; higher remembers longer. |
| `prefill-cpu-share` | engine | number, 0 to 1 | 0.6 (`0` = the GPU streams every expert) | The share of the streamed (not VRAM-resident) experts that the CPU computes while a prompt is read; the GPU streams and computes the rest. The value is fixed, not tuned from timing, so the output is identical from run to run at a given value. A weaker CPU may do better with a lower value. |
| `adapt-swaps` | engine | integer | 96 (`0` = static residency) | The most experts one round swaps into the main card's tier. |
| `peer-adapt-swaps` | engine | integer | same as `adapt-swaps` (`0` = the peer's experts stay fixed) | The most experts one round moves on the peer card. |
| `vram-elastic` | engine | switch | off | Hold the expert cache in segments so `POST /v1/vram` can lend VRAM to other programs between requests. |
| `vram-segment-mib` | engine | integer MiB | 512 | Segment size for `vram-elastic`. |
| `expert-cache-device1`, `-device2`, `-device3` | engine | integer or `auto` | off | Fill another GPU with experts directly, as a fixed tier. Not with `peer-device`. Advanced. |
| `expert-cache-remote-placement` | engine | `stripe`, `layer` | `stripe` | How those cards share the experts. |

### The adaptive tier

The engine counts how often each expert is routed to. After every `adapt-every` decode windows it runs a round: in each
layer it takes the most-used experts that are not in VRAM and swaps them in for the least-used resident ones, when the
gain is clear (a candidate needs a count of at least 2 and 1.5 above the expert it replaces), at most `adapt-swaps`
swaps per round, best gain first. Experts held by the peer card count against a margin of 10 before they move to the
main card, and the peer takes the next most-used experts the main card does not hold, up to `peer-adapt-swaps`. After
each round every count is multiplied by `adapt-decay`, so recent routing outweighs old. A count settles at about
1 / (1 - decay) times the routing per round, so the decay sets how long the tier remembers: 0.93 is roughly the last 14
rounds, 0.85 the last 7. Copies run on a side stream and land a few windows later; a window can wait for a landing while the tier is adapting. With `adapt-every 0` or `adapt-swaps 0`
the tiers stay as loaded (`expert-profile-save` then has nothing to learn). Adaptation moves experts between the CPU and
the GPUs, so output is not bit-identical from run to run; see [precision](../precision/index.md).

A tuning note: on the NVIDIA NVFP4 pack (6-bit dense) with an RTX 5090 and an RTX 5060 Ti as the peer, both at a 400 W
power limit, `--adapt-every 1 --adapt-decay 0.85` measured prose right after a topic switch 233.2 to 241.5 tok/s
(+3.6%), a conversation 264.5 to 275.6 (+4.2%), and structured output 491.4 to 487.0 (neutral), each an ABBA run of 4
processes per arm. A longer memory (`adapt-decay` 0.97 and up) looked 3 to 6% slower on prose in single runs (about
3% run-to-run noise).

The startup lines `RAM floor ...` and `RAM ... free, guard ...` give the figures behind the memory floors. See the
[running](../running.md#memory-floors).

## Prefill

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `prefill` | engine | `auto`, `auto:N` (N = 8192, 16384 or 32768), or a token count | 0 (the decode path) | Read prompts in batched chunks. `auto` takes the fewest equal chunks of at most 16384 tokens (32768 for packs whose expert blobs exceed 1.5 MiB, such as NVFP4) that the expert cache can lend buffers for; `auto:N` sets the ceiling; a number is a fixed chunk. Always set it to `auto` for serving. |
| `no-prefill-borrow` | engine | switch | off | Reserve the prompt buffers' VRAM for the whole session instead of lending expert slots. Costs expert cache; only for experiments. |
| `short-read` | engine | integer tokens | 64 | A text part of at most this many tokens goes through the decode windows instead of the batched path. `0` always uses the batched path. |

## Speculative decoding (MTP)

The drafter proposes tokens and the model verifies them. Unless `relaxed-drafting` is on, the output is exactly what
plain decoding would give.

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `spec` | engine | integer; `0` or `1` = off | 0 (no speculation, the drafter is not loaded) | The verify window: the last accepted token plus N-1 drafts. The model file does not set it; give it in the config or on the command line (7 is a good start) to turn speculation on. |
| `spec-min-p` | engine | number 0..1 | 0 (always draft to the cap; not set by the model file) | The drafter's cutoff: a draft enters the window only while it, and every draft before it, has at least this probability under the drafter. Higher values draft fewer, surer tokens. This is not the sampling `min_p` below. |
| `mtp-max-t` | engine | integer | 0 (= `spec`) | Cap on the drafter's own window; longer windows then come from `suffix-draft` matches. |
| `suffix-draft` | engine | integer tokens | 3 (`0` = drafter only) | Draft from an earlier repeat of the last N or more tokens of context (code edits, quoted text, tool-call JSON) when that is likelier to pay than the drafter. |
| `relaxed-drafting` | engine (`[P]`) | `true`, or a number in (0, 1] | off | Optional, and it changes output. Under sampling, a draft the model gives probability P or more is kept even when the model's own draw would have been another token. Faster, but no longer the model's exact sampling distribution; lower P is faster and further from exact. Bare `true` or flag alone is 0.5. Greedy requests, `logit_bias`, JSON `response_format`, control tokens and tool calls stay exact. Single-stream only. Numbers: [precision](../precision/features.md#relaxed-drafting). |

## Parallel slots

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `parallel` | server | integer 1..8 | 1 | Serve N requests at once in batch slots; further requests wait their turn. `max-context` is shared: each slot holds `max-context / N` tokens. The slots' KV cache comes out of the expert cache. Sets the engine's `batch`. |
| `slot-full-context` | server | switch | off | With `parallel`, give every slot the whole `max-context` instead of an equal share. Costs N times the KV cache. |
| `batch` / `slots` | engine | integer 2..8 | off | The engine flag `parallel` sets. Use `parallel`; setting both is an error. |

## Conversation cache

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `prompt-cache` | engine | integer | 6 | Conversation checkpoints kept between requests, about 118 MB of RAM each. `0` reads every prompt from the start. More checkpoints let more chats resume without re-reading. |
| `prompt-cache-every` | engine | integer tokens | 16384 (`0` = off) | Also checkpoint every N freshly read prompt tokens. |
| `prompt-cache-root` | engine | integer tokens | 2048 (`0` = never) | Checkpoint a prompt's first turn boundary (the end of the system prompt) when it is at least this long. |
| `conversation-cache-mib` | engine | integer MiB | 0 (off) | RAM budget for parked conversations, so separate chats can come back without re-reading. |
| `conversation-cache-slots` | engine | integer | 4 | At most this many parked conversations. |
| `conversation-cache-min-free-mib` | engine | integer MiB | 2560 | Physical RAM floor kept free when parking a conversation. |
| `turn-token`, `tail-role-token` | engine | token id | 248045, off | The token that opens a chat turn, and the role token of a trailing turn left out of a checkpoint. Only for a different chat template. |

## Vision

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `vision` | server | `off`, `on`, `peer`, `dynamic`, `cpu` | `off` | Image input; booleans are refused. `on`: encoder sidecar on the main GPU (about 2.2 GiB of expert cache). `peer`: the sidecar on the second `gpu` card. `dynamic`: inside the engine, borrowing expert-cache slots only while it reads an image. `cpu`: sidecar on the CPU, no VRAM, slow. Latencies: [running](../running.md#vision---vision). `on` and `peer` need the `basalt-vision` binary next to `bin/basalt`. Needs a model file with a vision projector. |

## Logging and diagnostics

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `debug` | server | switch | off | One log line per request and the raw model text after each reply. |
| `request-lines` | server | switch | off | Echo the engine's per-request summary on stdout, for a supervisor that sees only that output. |
| `profile` | engine | switch | off | Diagnostic: GPU stage times of the decode windows. |
| `wait-stats` | engine | switch | off | Diagnostic: the host's wait times in the decode loop. |
| `decode-timing` | engine | switch | off | Diagnostic: one line per request on where its decode time went. |
| `draft-log` | engine | file path | off | Diagnostic: one line per decode window, for studying draft policy. |
| `stop-eos` | engine | switch | off | Stop when the model emits an end-of-turn token; the server handles stopping itself. |
| `eos-ids` | engine | comma-separated token ids | 248044,248046 | The end-of-turn tokens `stop-eos` looks for. Setting it implies `stop-eos`. |
| `exp` | engine | `NAME=VALUE[,...]` | none | Diagnostics. `concur9_logit_hash=1` prints a hash of each accepted row's logits on stderr, for the identity checks in [precision](../precision/builds.md#reproduce). |

## Control vectors

A control vector (a GGUF in llama.cpp's format) is added to the residual stream; with the server, a request switches it
with `cvec` (0 or 1).

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `control-vector` | engine | file path | off | A control vector at scale 1. |
| `control-vector-scaled` | engine | `FILE:SCALE[,...]` | off | Control vectors with their scales. |
| `control-vector-layer-range` | engine | two layer numbers, `A B` | 1 to the last layer | The layers it follows, inclusive. |
| `cvec-mode` | engine | `add`, `project` | `add` | `add`: h += s v. `project`: h -= s (h.v) v with v a unit vector. |
| `cvec-dir` | engine | `per-layer`, `single:L` | `per-layer` | Each layer's own direction, or layer L's direction everywhere (`project`). Not with `layer-split`. |

## Sampling

A request carries its own sampling fields. A field the request leaves out takes the launch option of the same name
(`--temperature`, `--top-p`, ... in the [server options](#server); command line over the config file's `server`
block), then the model file's `sampling` default (below), then the engine's. Request fields always win. `/props`
reports the defaults in effect.

| Field | Type, values | What it does |
| --- | --- | --- |
| `temperature` | number >= 0 | `0` is greedy (argmax); higher is more random. Greedy requests stay exact under every other setting. |
| `top_p` | number in (0, 1] | Nucleus sampling: keep the smallest set of tokens whose probability adds up to this. `1` is off. |
| `top_k` | integer 1..64 | Keep only the K likeliest tokens. The sampler keeps at most 64; a larger value, or `0`, is 64. |
| `min_p` | number in [0, 1] | Sampling cutoff: drop tokens less likely than `min_p` times the likeliest token's probability. `0` is off. It acts on the model's tokens and is unrelated to the server's `spec-min-p`, which stops the drafter from proposing unlikely tokens. |
| `repetition_penalty` | number > 0 | Multiplicative penalty on tokens seen in the window; `1` is off. |
| `presence_penalty` | number | Flat penalty on any token seen in the window; `0` is off. |
| `frequency_penalty` | number | Penalty that grows with how often a token was seen; `0` is off. |
| `penalty_last_n` | integer > 0 | The window, in tokens, the three penalties look back over. Applies only when a penalty is set; 64 when omitted. |
| `seed` | integer > 0 | Fixes the random draw for a reproducible sampled reply. |
| `relaxed_drafting` | number in [0, 1], or `false` | This request's relaxed-drafting threshold; `false` or `0` forces exact sampling. Needs speculation; see [Speculative decoding](#speculative-decoding-mtp). |

Other request fields the API reads: `max_tokens` / `max_completion_tokens` (the Anthropic `max_tokens`; `0` or
`-1` means the rest of the context), `stop` / `stop_sequences`, `logit_bias`, `reasoning_effort`, `enable_thinking`
and `reasoning_budget_tokens` (a cap on thinking tokens; `0` is none).

### The model file's sampling defaults

The model file's `meta` section carries a `sampling` object. Its keys are the request fields above, with the same
meaning:

| Key | Allowed values |
| --- | --- |
| `temperature` | number >= 0 |
| `top_p` | 0 < value <= 1 |
| `top_k` | integer 1..64 |
| `min_p` | 0 <= value <= 1 |
| `presence_penalty` | number >= 0 |
| `frequency_penalty` | number >= 0 |
| `repetition_penalty` | number > 0 (`1` = off) |
| `penalty_last_n` | non-negative integer |
| `seed` | positive integer |

A bad value stops the server at startup; an unknown key is named in a warning and ignored. The defaults in effect
are printed at startup as `[server] sampling defaults: ...`, each marked `(command line)` or `(config)` when a launch
option set it and unmarked when it comes from the model file. A launch option overrides the model file's value for
that key; the others stay.
