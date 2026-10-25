# Configuration reference

Every setting a user can pass to Basalt: the server options, the engine flags, the per-request sampling fields and
the model file's sampling defaults. For what a setting does to speed, memory and output precision see the
[running](../running.md) and [precision](../precision/index.md).

- [How settings are given](#how-settings-are-given)
- [Server](#server)
- [Model and GPUs](#model-and-gpus)
- [Context and KV cache](#context-and-kv-cache)
- [Expert cache and memory](#expert-cache-and-memory)
- [Speculative decoding (MTP)](#speculative-decoding-mtp)
- [Parallel slots](#parallel-slots)
- [Conversation cache](#conversation-cache)
- [Vision](#vision)
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
- The built-in defaults pick the fastest settings that need no knowledge of the machine (speculation 8 with a drafter,
  expert cache auto, prefill auto, resident experts when the RAM holds them, adaptive rounds every window), and keep
  the settings that change output or memory use (4096 context, fp16 KV). A config file sets those. The startup line `[server] engine: ...` lists what came from the file or the command line.

In the tables, "default" is what applies when the setting is absent everywhere.

## Automatic settings

Four settings are picked by the engine at start when they are left out: `peer-exclusive`, `resident`
(`resident-budget-gib`), `pool-affinity` and `pool-workers`. A value in the config file or on the command line always
wins and is never changed.

| setting | rule |
| --- | --- |
| `peer-exclusive` | on only with a peer card and `--serve`, when the full expert set does not fit in free RAM but the set without the peer card's share does; otherwise off. |
| `resident` | the free RAM above the safety floor, capped at the experts the GPUs do not hold. |
| `pool-affinity` | `auto`. |
| `pool-workers` | the physical cores minus the host thread; on a hybrid CPU with `auto`, the P-cores minus one plus half the E-cores, with `p-cores` the P-cores minus one; fitted to a container CPU quota. |

The engine prints one line naming only the picked settings, in the order above, and the server relays it once per start:

```
[server] auto flags: peer-exclusive off, resident 51.2 GB, pool-affinity all, pool-workers 15
```

When every setting is given, no line is printed.

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
| `grammar` | switch (`--no-grammar` or `grammar: false` turns it off) | on | Hold every tool call and JSON response format (`json_object`, `json_schema`) to its schema inside the engine, so a malformed call or object cannot be produced (as in llama.cpp's Qwen3-Coder parser, a call's required parameters may come in any order, each once, with up to six required ones permuted and more kept in declared order, followed by the optional ones in any order, which may repeat); the model's JSON is returned as it is generated, with no check after the fact. A request with `strict: true` gets the grammar even when this is off; with it off, any other JSON `response_format` (or Responses `text.format`, or Anthropic `output_config.format`) is refused with HTTP 400. Needs an engine that has `--grammar`; the server turns the engine's grammar support on whenever it has the flag. Works in batch slots: a slot's first pick the grammar refuses is resampled under the mask (the solo rule) and its window ends there. |
| `debug` | switch | off | One log line per request and the raw model text after each reply. |
| `request-lines` | switch | off | Echo the engine's per-request summary on stdout, for a supervisor that sees only that output. |
| `temperature`, `top-p`, `top-k`, `min-p`, `presence-penalty`, `frequency-penalty`, `repetition-penalty`, `penalty-last-n`, `seed` | number (`top-k`, `penalty-last-n`, `seed`: integer) | the model file's `sampling` value, else the engine's | The sampling default for a request that leaves that field out, over the model file's. Same meaning and allowed values as the [model file's sampling keys](#the-model-files-sampling-defaults). The server applies them to each request; the engine never sees them. |
| `reasoning-budget-tokens` | integer | `0` (none) | The thinking budget for a request that names none (see [Sampling](#sampling)). At the budget the thinking is closed and the answer starts. It only helps when it sits below the request's `max_tokens`: at or above it the request runs to `max_tokens` in thinking and answers nothing. A request's own field wins, and `0` there turns it off. |
| `tokenizer` | path | `pack/full/tokenizer` in the source tree (a byte tokenizer if absent) | Tokenizer directory. Mock engine only. |
| `mock-engine`, `script` | switch, string (repeatable) | off, a short greeting | Testing: run a scripted engine instead of the model, for clients and tests (no `model` needed); `script` is its answer, and given more than once, requests get them in turn. |
| `config` | path | none | The config file. Command line only. |

## Model and GPUs

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `gpu` | server (also `basalt`, `basalt-bench`, `basalt-probe`) | GPU numbers, e.g. `"0"` or `"0,1"`, or `"auto"` | GPU 0 alone (a warning names the other visible cards) | The cards to use, numbered as `nvidia-smi` numbers them: exactly these, the first is the main card. `auto` takes every Blackwell card and makes the fastest the main one. The engine gives the other cards their roles (a second card becomes the peer expert tier). See [GPUs](gpus.md). |
| `peer-reserve-mib` | engine | integer MiB | 600 | VRAM the second card leaves free after its experts and the buffers of its prompt share, which the engine sizes itself. Raise it when the start prints `prefill peer: the peer's buffers do not fit`. See [GPUs](gpus.md#two-16-gb-cards). |

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

A server with `max-context` above 32768 has an elastic KV cache: each session backs its first 32768 cells at startup and
takes more from the expert cache only when a request needs it. It is off with `kv-resident` or `vram-elastic`. See [running](../running.md#concurrent-requests---parallel-n).

## Expert cache and memory

The model's experts live in VRAM (the expert cache), in RAM, or in the file. These settings decide where.

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `vram-reserve-mib` | engine | integer MiB, at least 256 | 700 (lowered automatically, down to 300, on a card that would otherwise get too few slots; a value you give is kept) | VRAM left free on the main card for graphs, scratch and the head when `expert-cache auto` sizes the cache. Below 256 is refused. Raise it if the machine pages GPU memory or other programs need VRAM. |
| `resident-experts` | engine | switch | on when the pack has an expert profile and the experts are not split over several cards | Keep the experts the GPU does not hold in page-locked RAM, so none is read from the file again. Leaves a RAM floor free for the OS; when the RAM cannot hold them it falls back to a plain file mapping. `mmap-experts` turns it off. |
| `resident-budget-gib` | engine | number GiB | auto when not set | Like `resident-experts` but with at most N GiB of experts in RAM, hottest first. Refused when it would leave less than the RAM floor free; the error names the largest safe value. |
| `mmap-experts` | engine | switch | off | Opt out of the default resident RAM copy and back to a file mapping. Speed then depends on the OS page cache. |
| `shared-expert-arena` | engine | file path | off | Back the resident arena with one shared file so several processes use one copy. Put it on `/dev/shm`. |
| `expert-profile-save` | engine | file path | off | Save what the adaptive tier learned (experts in VRAM and routing counts) to this file on exit and periodically. `tools/convert/convert.py --profile` reads it. |
| `expert-profile-save-every` | engine | minutes | 10 (`0` = on exit only) | How often the profile is saved. |
| `vram-elastic` | engine | switch | off | Hold the expert cache in segments so `POST /v1/vram` can lend VRAM to other programs between requests. Works with `resident-experts`: a shrink first copies the experts it gives back into RAM, so none is read from the file, and is refused (nothing changed) when the RAM above the floor cannot hold them; a grow frees those copies. |

### The adaptive tier

The engine counts how often each expert is routed to. Every few decode windows it runs a round: in each layer it swaps
the most-used experts that are not in VRAM in for the least-used resident ones, when the gain is clear. The peer card
takes the next most-used experts the main card does not hold. After each round every count is decayed, so recent routing
outweighs old. Copies run on a side stream and land a few windows later. Adaptation moves experts between the CPU and
the GPUs, so output is not bit-identical from run to run; see [precision](../precision/index.md). The round settings are
engine diagnostic flags; with adaptation off the tiers stay as loaded and `expert-profile-save` has nothing to learn.

The startup lines `RAM floor ...` and `RAM ... free, guard ...` give the figures behind the memory floors. See the
[running](../running.md#memory-floors).

## Speculative decoding (MTP)

The drafter proposes tokens and the model verifies them. Unless `relaxed-drafting` is on, the output is exactly what
plain decoding would give.

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
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

## Vision

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `vision` | server | `off`, `dynamic`, `cpu`, a GPU number | `off` | Image input; booleans are refused. A GPU number, as `gpu` numbers them: encoder sidecar always on that card (about 2.2 GiB of its expert cache), whatever the engine's cards are and no engine role needed. `dynamic`: inside the engine, borrowing expert-cache slots only while it reads an image. `cpu`: sidecar on the CPU, no VRAM, slow. Latencies: [running](../running.md#vision---vision). A GPU number needs the `basalt-vision` binary next to `bin/basalt`. Needs a model file with a vision projector. |

Diagnostic flags: `bin/basalt --help`

## Control vectors

A control vector (a GGUF in llama.cpp's format) is added to the residual stream; with the server, a request switches it
with `cvec` (0 or 1).

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `control-vector` | engine | file path | off | A control vector at scale 1. |
| `control-vector-scaled` | engine | `FILE:SCALE[,...]` | off | Control vectors with their scales. |
| `control-vector-layer-range` | engine | two layer numbers, `A B` | 1 to the last layer | The layers it follows, inclusive. |
| `cvec-mode` | engine | `add`, `project` | `add` | `add`: h += s v. `project`: h -= s (h.v) v with v a unit vector. |
| `cvec-dir` | engine | `per-layer`, `single:L` | `per-layer` | Each layer's own direction, or layer L's direction everywhere (`project`). |

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
| `min_p` | number in [0, 1] | Sampling cutoff: drop tokens less likely than `min_p` times the likeliest token's probability. `0` is off. It acts on the model's tokens. |
| `repetition_penalty` | number > 0 | Multiplicative penalty on tokens seen in the window; `1` is off. |
| `presence_penalty` | number | Flat penalty on any token seen in the window; `0` is off. |
| `frequency_penalty` | number | Penalty that grows with how often a token was seen; `0` is off. |
| `penalty_last_n` | integer > 0 | The window, in tokens, the three penalties look back over. Applies only when a penalty is set; 64 when omitted. |
| `seed` | integer > 0 | Fixes the random draw for a reproducible sampled reply. |
| `relaxed_drafting` | number in [0, 1], or `false` | This request's relaxed-drafting threshold; `false` or `0` forces exact sampling. Needs speculation; see [Speculative decoding](#speculative-decoding-mtp). |

Other request fields the API reads: `max_tokens` / `max_completion_tokens` (the Anthropic `max_tokens`; `0` or
`-1` means the rest of the context), `stop` / `stop_sequences`, `logit_bias`, `reasoning_effort`, `enable_thinking`
and `reasoning_budget_tokens` (a cap on thinking tokens; `0` is none). Anthropic's `thinking.budget_tokens` and
llama.cpp's `thinking_budget_tokens` are the same budget, and `reasoning_budget_tokens` wins when more than one is
sent. A request that names none of them takes the server's [`reasoning-budget-tokens`](#server).

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
