# Configuration reference

Every setting a user can pass to Basalt: the server options, the engine flags, the per-request sampling fields and
the model file's sampling defaults. For what a setting does to speed, memory and output precision see the
[running](../running.md) and [precision](../precision/index.md).

- [How settings are given](#how-settings-are-given)
- [Server](#server)
- [Model and GPUs](#model-and-gpus)
- [Context and KV cache](#context-and-kv-cache)
- [Expert cache and memory](#expert-cache-and-memory)
- [Prompt processing](#prompt-processing)
- [Speculative decoding (MTP)](#speculative-decoding-mtp)
- [Parallel slots](#parallel-slots)
- [Conversation cache](#conversation-cache)
- [Vision](#vision)
- [Control vectors](#control-vectors)
- [Sampling and thinking](sampling.md)
- [The adaptive tier](adaptive.md)
- [Example configurations](examples.md)
- [Diagnostics, measurement and `--exp` flags](diagnostics.md)

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
  startup on a flag name it does not know. It adds `--serve` itself, and `--vision` (with `--vision-mmproj` for
  `vision dynamic`) when `vision` is on.

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
  expert cache auto, prefill auto, resident experts when the RAM holds them, adaptive rounds every window on a server), and keep
  the settings that change output or memory use (4096 context, fp16 KV). A config file sets those. The startup line `[server] engine ...` lists what came from the file or the command line.

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
[server] auto     peer-excl=off resident=51.2GB pool-affinity=all pool=15
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
| `idle-sleep` | seconds | off | Put the model to sleep after this long without requests: its VRAM is freed, the process keeps its RAM copy, conversations and learned expert placement, and the next request wakes it in seconds. Linux only (the driver suspends the engine's GPU state); it stays awake when that state would not fit in the free RAM. [running](../running.md#sleep---idle-sleep) |
| `min-free-vram-mib` | integer | off | Load an unloaded model only when this much VRAM is free, otherwise answer 503. Pairs with `lazy` and `idle-unload`. |
| `grammar` | switch (`--no-grammar` or `grammar: false` turns it off) | on | Hold every tool call and JSON response format (`json_object`, `json_schema`) to its schema inside the engine, so a malformed call or object cannot be produced (as in llama.cpp's Qwen3-Coder parser, a call's required parameters may come in any order, each once, with up to six required ones permuted and more kept in declared order, followed by the optional ones in any order, which may repeat); the model's JSON is returned as it is generated, with no check after the fact. A request with `strict: true` gets the grammar even when this is off; with it off, any other JSON `response_format` (or Responses `text.format`, or Anthropic `output_config.format`) is refused with HTTP 400. Needs an engine that has `--grammar`; the server turns the engine's grammar support on whenever it has the flag. Works in batch slots: a slot's first pick the grammar refuses is resampled under the mask (the solo rule) and its window ends there. |
| `debug` | switch | off | One log line per request and the raw model text after each reply. |
| `request-lines` | switch | off | Echo the engine's per-request summary on stdout, for a supervisor that sees only that output. |
| `temperature`, `top-p`, `top-k`, `min-p`, `presence-penalty`, `frequency-penalty`, `repetition-penalty`, `penalty-last-n`, `seed` | number (`top-k`, `penalty-last-n`, `seed`: integer) | the model file's `sampling` value, else the engine's | The sampling default for a request that leaves that field out, over the model file's. Same meaning and allowed values as the [model file's sampling keys](sampling.md#the-model-files-sampling-defaults). The server applies them to each request; the engine never sees them. |
| `reasoning-budget-tokens` | integer | `0` (none) | The thinking budget for a request that names none (see [sampling](sampling.md#the-thinking-budget)). At the budget the thinking is closed and the answer starts. A request's own field wins, and `0` there disables budgeting. Positive budgets are also limited by `answer-room`. |
| `thinking-loop-tokens` | integer | `256` | End the thinking, and let the model answer, when it repeats one token this many times in a row inside the thinking; `0` turns it off. See [Thinking loops](sampling.md#thinking-loops). |
| `answer-room` | switch (`--no-answer-room` or `answer-room: false` turns it off) | on | Reserve space for closing the thinking and for the answer when the effective output cap is at least 1024 tokens. Applies to missing budgets and clamps positive budgets; an explicit request budget of `0` opts out. |
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
takes more from the expert cache only when a request needs it. With `parallel`, each slot whose share is above 4096
tokens does the same from 4096 cells, growing by 4096, and an idle conversation's KV moves to RAM once requests beside
it have run 15 s. It is off with `kv-resident` or `vram-elastic`. See [running](../running.md#concurrent-requests---parallel-n).

## Expert cache and memory

The model's experts live in VRAM (the expert cache), in RAM, or in the file. These settings decide where.

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `vram-reserve-mib` | engine | integer MiB, at least 256 (128 on Linux) | 700 (lowered automatically, down to 300, on a card that would otherwise get too few slots; a value you give is kept unless it is raised as described here) | VRAM left free on the main card for graphs, scratch and the head when `expert-cache auto` sizes the cache. A server's kernels and window graphs that load after the cache (96 MiB, plus 24 MiB per slot with `--parallel`) load into it; a reserve too small to hold them and keep the floor free is raised to that: on Linux with `--parallel 4`, any value below 320 MiB (128 + 96 + 4 x 24) becomes 320, and the `expert cache auto` start line gives both (`320 MiB reserved (150 asked, ...)`). Below the floor is refused. Raise it if the machine pages GPU memory or other programs need VRAM. |
| `resident-experts` | engine | switch | on when the pack has an expert profile | Keep the experts the GPU does not hold in page-locked RAM, so none is read from the file again. Leaves a RAM floor free for the OS; when the RAM cannot hold them it falls back to a plain file mapping. `mmap-experts` turns it off. |
| `resident-budget-gib` | engine | number GiB | auto when not set | Like `resident-experts` but with at most N GiB of experts in RAM, hottest first. Refused when it would leave less than the RAM floor free; the error names the largest safe value. |
| `mmap-experts` | engine | switch | off | Opt out of the default resident RAM copy and back to a file mapping. Speed then depends on the OS page cache. |
| `shared-expert-arena` | engine | file path | off | Back the resident arena with one shared file so several processes use one copy. Put it on `/dev/shm`. |
| `expert-profile-save` | engine | file path | off | Save what the adaptive tier learned (experts in VRAM and routing counts) to this file on exit and periodically. A start that finds the file places it instead of the model's profile, so the first request runs on what the last run learned: faster when the workload continues, slower when it changes ([adaptive tier](adaptive.md)). Needs the adaptive tier on. `tools/convert/convert.py --profile` reads it too. |
| `expert-profile-save-every` | engine | minutes | 10 (`0` = on exit only) | How often the profile is saved. |
| `resident-cpu-experts` | engine | switch | off (implied by `resident-experts`) | With `mmap-experts` and a pack with an expert profile: keep the experts the GPU does not hold in ordinary RAM, so none is read from the file again. `resident-experts` is this with the copy page-locked when possible and the file mapping as a fallback when it does not fit. |
| `vram-elastic` | engine | switch | off | Hold the expert cache in segments so `POST /v1/vram` can lend VRAM to other programs between requests. Works with `resident-experts`: a shrink first copies the experts it gives back into RAM, so none is read from the file, and is refused (nothing changed) when the RAM above the floor cannot hold them; a grow frees those copies. |
| `vram-segment-mib` | engine | integer MiB, at least 64 | 512 | The segment size of `vram-elastic`'s expert cache: the step VRAM is lent and taken back in. Below 64 turns `vram-elastic` off with a warning. |
| `expert-cache` | engine | `auto`, or an integer count of expert blobs | `auto` (on packs with an expert profile) | Keep this many expert blobs in VRAM and compute their rows on the GPU. `auto` takes as many as fit in the VRAM left after the weights, the session and `vram-reserve-mib`. |
| `adapt-every` | engine | integer windows | 1 on a server, 4 with a layer split (`0` = off) | An adaptive round every N decode windows. [Adaptive tier](adaptive.md). |
| `adapt-swaps` | engine | integer | 96 (`0` = off) | Most experts one adaptive round swaps into the main GPU; a server round after a large backlog swaps four times as many, and the RAM copy keeps four times N free places for them. [Adaptive tier](adaptive.md). |
| `peer-adapt-swaps` | engine | integer | the value of `adapt-swaps` | Most experts one round swaps into the peer card. Only with a peer card. |
| `adapt-decay` | engine | number | 0.85 on a server, 0.7 with a layer split | The factor the routing counts are multiplied by after each round; lower forgets sooner. [Adaptive tier](adaptive.md). |

### The adaptive tier

The engine moves the experts the conversation routes to most into VRAM, in rounds beside decode. What the four
settings above do, what they trade and the measured numbers: [adaptive tier](adaptive.md).

The startup lines `RAM floor ...` and `RAM ... free, guard ...` give the figures behind the memory floors. See the
[running](../running.md#memory-floors).

## Prompt processing

The prompt is read in batched chunks. These settings set the chunk size and how the streamed experts are shared.

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `prefill` | engine | `auto`, `auto:N` (N = 8192, 16384 or 32768), or a chunk size in tokens | `auto` | Read the prompt in chunks of this many tokens. `auto` takes the largest chunk the expert cache can lend, up to a ceiling; `auto:N` sets that ceiling. |
| `prefill-cpu-share` | engine | number from 0 to 1 | 0.6 | Share of the streamed prompt experts the CPU computes; `0` has the GPU stream them all. Fixed, so the output is the same from run to run. A weaker CPU may prefer a lower value. |
| `short-read` | engine | integer tokens | 64 (`0` = off) | A fresh part of a prompt of at most N tokens (a chat message, a short tool result) is read through the decode windows instead of the batched prompt path. |
| `peer-prefill-rows` | engine | integer rows per layer | half of chunk x top-k | Prompt rows per layer the peer card computes. `0` keeps the prompt path on the main card. |

## Speculative decoding (MTP)

The drafter proposes tokens and the model verifies them. Unless `relaxed-drafting` is on, the output is exactly what
plain decoding would give.

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `spec` | engine | integer | 8 with a drafter, else 0 | Verify windows of at most N tokens: the last accepted token plus up to N-1 drafts. `0` or `1` turns speculation off. |
| `suffix-draft` | engine | integer tokens | 3 (`0` = MTP only) | Prompt lookup: when the text repeats an earlier stretch of the context by at least N tokens, the window may be filled from what followed it there, where the draft policy expects it to pay. |
| `mtp-max-t` | engine | integer | 0 (= `spec`) | Caps the MTP's own windows; longer ones then come only from prompt lookup. |
| `mtp-window` | engine | integer cells | 32768 (`0` = all) | The drafter attends to the last N cells. With `parallel`, each slot's draft K/V keeps only those and the 16384 before them (a turn rolled back), not the slot's whole context, and the rest of that VRAM holds expert slots. |
| `relaxed-drafting` | engine (`[P]`) | `true`, or a number in (0, 1] | off | Optional, and it changes output. Under sampling, a draft the model gives probability P or more is kept even when the model's own draw would have been another token. Faster, but no longer the model's exact sampling distribution; lower P is faster and further from exact. Bare `true` or flag alone is 0.5. Greedy requests, `logit_bias`, JSON `response_format`, control tokens and tool calls stay exact. Single-stream only. Numbers: [precision](../precision/features.md#relaxed-drafting). |
| `route-tail-skip` | engine (`[R]`) | `true`, or an integer 1 to 9 | off | Optional, and it **changes the output: less accurate than the model.** In a decode window, an expert no GPU holds is neither copied nor computed when every entry naming it ranks R or lower by router weight within its token (0-based of 10); it adds zero and the other experts keep their weights. Lower R drops more. Bare `true` or flag alone is 9. Needs the device-planned MoE tail, else it stays off and says so at start. Listed in the start warning of lossy flags. Numbers: [precision](../precision/routing.md#route-tail-skip). |
| `route-prior` | engine (`[LAMBDA]`) | `true`, or a number in (0, 16] | off | Optional, and it **changes the output: less accurate than the model.** A decode window picks its ten experts as if each expert a GPU holds had LAMBDA added to its router logit; the weights stay the router's own. Higher routes harder toward held experts. Bare `true` or flag alone is 0.5. Needs the device-planned MoE tail, else it stays off and says so at start. Listed in the start warning of lossy flags. Numbers: [precision](../precision/routing.md#route-prior). |

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
| `prompt-cache-every` | engine | integer tokens | 16384 (`0` = off) | Also checkpoint every N freshly read prompt tokens (RAM only: the disk tier does not store them). |
| `prompt-cache-root` | engine | integer tokens | 2048 (`0` = never) | Checkpoint a prompt's first turn boundary (the end of the system prompt) when it is at least this long. |
| `turn-token` | engine | token id | 248045 (`<\|im_start\|>`) | The token that opens a chat turn; the last one in a prompt is where the conversation checkpoint goes. |
| `tail-role-token` | engine | token id | -1 (off) | When the turn right before the last one opens with this role token (a short trailing turn such as a reasoning-effort note), the checkpoint goes in front of that turn, so it holds the conversation and not the note. |
| `conversation-cache-mib` | engine | integer MiB | 0 (off) | RAM budget for parked conversations, so separate chats can come back without re-reading. |
| `conversation-cache-slots` | engine | integer | 4 | At most this many parked conversations. |
| `conversation-cache-min-free-mib` | engine | integer MiB | 2560 | Physical RAM floor kept free when parking a conversation. |
| `conversation-cache-disk-mib` | engine | integer MiB | 0 (off) | Park conversations on disk, at most this much (the least recently used go first), and read them back when a prompt starts with one, after a restart too. Above 0 turns it on; the folder is `conversations` in the model's cache folder. [running](../running.md#conversations-on-disk---conversation-cache-disk-mib) |
| `conversation-cache-dir` | engine | directory | model cache folder | Another folder for the disk tier. Does nothing while the budget is 0. |

## Vision

| Setting | Where | Type, values | Default | What it does and when to change it |
| --- | --- | --- | --- | --- |
| `vision` | server | `off`, `dynamic`, `cpu`, a GPU number | `off` | Image input; booleans are refused. A GPU number, as `gpu` numbers them: encoder sidecar always on that card (about 2.2 GiB of its expert cache), whatever the engine's cards are and no engine role needed. `dynamic`: inside the engine, borrowing expert-cache slots only while it reads an image. `cpu`: sidecar on the CPU, no VRAM, slow. Latencies: [running](../running.md#vision---vision). A GPU number needs the `basalt-vision` binary next to `bin/basalt`. Needs a model file with a vision projector. One image is capped at 64 megapixels before it is decoded. |

Diagnostic flags: [diagnostics](diagnostics.md)

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

The request fields, the defaults a request inherits, the thinking budget and the loop safeguard are in
[sampling](sampling.md).
