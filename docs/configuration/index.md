# Configuration reference

Every setting: server options, engine flags, and the sampling fields. Effects on speed and precision: [running](../running.md),
[precision](../precision/index.md).

- [How settings are given](#how-settings-are-given) · [Automatic settings](#automatic-settings) · [Server](#server) · [Model and GPUs](#model-and-gpus)
- [Context and KV](#context-and-kv-cache) · [Expert cache](#expert-cache-and-memory) · [Prompt](#prompt-processing) · [Drafting](#drafting)
- [Slots](#parallel-slots) · [Conversation cache](#conversation-cache) · [Vision](#vision) · [Control vectors](#control-vectors)
- Pages: [models and the router](models.md), [sampling](sampling.md), [adaptive tier](adaptive.md), [examples](examples.md), [diagnostics](diagnostics.md)

## How settings are given

```sh
bin/basalt serve --config basalt_config.json [options]
bin/basalt serve MODEL.basalt [--python PY] [options]
```

- **Server options** (the Python server): model, address, port, slots, vision, logging. `--name value` on the command line.
- **Engine flags** (`bin/basalt --help`): context, KV, expert cache, prefill, speculation. Given like any flag; the server passes them
  on in order and asks the engine which it accepts.

A config file is one JSON object. `models` lists the models; `engine` applies to every model that takes it; `server` is shared:

```json
{ "version": 2, "models": [{ "model": "model.basalt", "kv": "int8" }],
  "engine": { "resident-experts": true }, "server": { "port": 8080 } }
```

- Where a setting goes in a config file: `server` is the `server` block, `entry` a model entry or the top-level `engine`,
  `engine` (an engine flag) the same places, `model entry` a model entry only.
- Keys are flag names without dashes. Switches are `true`. A repeatable option (`allowed-host`, `cors-origin`, `image-url-host`,
  `image-dir`) takes a list. `model` paths are relative to the file.
- A row marked Flash-Next only or DeepSeek only stops the start in a model entry or on the command line of the other model; in the top-level `engine` it is left out for that model, with a start line ([checks at start](models.md#checks-at-start)).
- Precedence: command line, then config file, then defaults; `--model` and `--alias` do not go with `--config` (its entries name the models).
  A wrong block or an unknown name stops the start. `api-key` cannot be in a file.
- The defaults are the fastest settings that need no knowledge of the machine. The keys that change output or memory (8,192 context,
  fp16 KV) are kept for you to set.

## Automatic settings

When left out, the engine picks `peer-exclusive`, `resident` (`resident-budget-gib`), `pool-affinity` and `pool-workers`. A given value
always wins. It prints one `[server] auto` line, where `ram-experts` is the `resident` amount (the experts held in RAM, not the whole model):

| Setting | Rule |
| --- | --- |
| `peer-exclusive` | Flash-Next only. On only with a peer card, when the full expert set does not fit in free RAM but the set without the peer's share does |
| `resident` | Flash-Next only. Free RAM above the safety floor, capped at the experts the GPUs do not hold |
| `pool-affinity` | `auto` |
| `pool-workers` | physical cores minus the host thread; on hybrid CPUs, P-cores minus one plus half the E-cores |

## Server

Command line `--name`; the Where column says which config block takes it.

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `model` | model entry | path | required | The `.basalt` file: weights, tokenizer, chat template, drafter, vision projector and default sampling. |
| `alias` | model entry | string | name in the file | The model name the API reports. |
| `host` | server | address | `127.0.0.1` | Listen address. `0.0.0.0` serves the network; never without `api-key`. |
| `port` | server | integer | `8080` | TCP port. |
| `api-key` | command line | string | none | Require this key on `/v1/*` (`Authorization: Bearer` or `x-api-key`). Command line only. |
| `allowed-host` | server | string, repeatable | none | Also answer to this `Host` name; `"*"` turns the check off. |
| `cors-origin` | server | string, repeatable | none | Let web pages from this origin call `/v1/*`. |
| `fit-max-tokens` | entry | switch | off | Clamp `max_tokens` to the remaining context instead of a 400. |
| `open` | server | switch | off | Open the web page when ready. |
| `lazy` | server | switch | off | Start unloaded; load on the first request. Drops a sidecar encoder. |
| `idle-unload` | entry | seconds | off | Unload after this long idle. Frees VRAM for other programs. |
| `idle-sleep` | entry | seconds | off | Sleep after this long idle: VRAM freed, RAM kept, wake in seconds. Linux only. [running](../running.md#sleep---idle-sleep) |
| `min-free-vram-mib` | entry | integer | off | Load an unloaded model only when this much VRAM is free; else 503. |
| `grammar` | entry | switch (`--no-grammar`) | on | Hold tool calls and JSON formats (`json_object`, `json_schema`) to their schema inside the engine. Without it, other JSON formats are refused with 400. |
| `max-payload-mib` | server | MiB | `256` | Largest request body the server reads; a bigger one gets a 413. |
| `debug` | server | switch | off | One log line per request, and the raw model text. |
| `request-lines` | server | switch | off | Echo the per-request summary on stdout. |
| `temperature`, `top-p`, `top-k`, `min-p`, `presence-penalty`, `frequency-penalty`, `repetition-penalty`, `penalty-last-n`, `seed` | entry | number (`top-k`, `penalty-last-n`, `seed`: integer) | the model file's `sampling`, else the engine's | Default for a request that leaves the field out. See [sampling](sampling.md). |
| `reasoning-budget-tokens` | entry | integer | `0` | Thinking budget for requests that name none. A request's own value wins. |
| `thinking-loop-tokens` | entry | integer | `256` | End thinking when one token repeats this many times in a row; `0` turns it off. |
| `answer-room` | entry | switch (`--no-answer-room`) | on | Reserve room for closing the thinking and the answer (caps of 1,024 tokens or more). |
| `config` | command line | path | none | The config file. Command line only. |

## Model and GPUs

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `gpu` | entry, `basalt`, `basalt-bench`, `basalt-probe` | `"0"`, `"0,1"`, `"auto"` | GPU 0 alone | Cards, numbered as `nvidia-smi`. The first is the main card. `auto` takes every Blackwell card and makes the fastest main. [GPUs](gpus.md) |
| `peer-reserve-mib` | engine | MiB | 600 | VRAM the second card keeps free after its experts and prompt buffers. Raise it when the start says the peer's buffers do not fit. [GPUs](gpus.md#two-16-gb-cards) |

## Context and KV cache

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `max-context` | engine | tokens | 8192 | Longest context. Shared by `parallel` slots. The KV cache grows with it. |
| `kv` | engine | Flash-Next: `fp16`, `int8`, `q4_0`, `k8v4`; DeepSeek: `fp8`, `fp16` | Flash-Next `fp16`, DeepSeek `fp8` | KV storage. Flash-Next: `int8` halves VRAM; `q4_0` is 4-bit after rotation; `k8v4` keeps K int8 and V 4-bit (816 bytes per cell against `int8`'s 1,056); not with `kv-resident`. DeepSeek: `fp8` rounds as the model's reference (FP8 window rows, FP4 compressed rows and index keys, 920 bytes per position); `fp16` keeps the unrounded values (3,200 bytes per position). A model refuses the other's values. |
| `kv-resident` | engine | cells (min 20,480) or `0` | 0 | Flash-Next only. Keep N cells of KV in VRAM and the rest in pinned RAM. With `parallel` and `kv` `int8`, every slot streams its own K/V this way. |
| `rope-scaling` | engine | `none`, `linear`, `yarn` | model's, else `none` | Flash-Next only. Extend past the trained length. Fixed for the run. |
| `rope-scale` | engine | number | model's, else 1 | Flash-Next only. Factor for `linear` and `yarn`. |
| `rope-freq-base`, `rope-freq-scale` | engine | number | 0 (model's own) | Flash-Next only. Raw RoPE values. Leave at 0 unless you know them. |
| `yarn-orig-ctx`, `yarn-ext-factor`, `yarn-attn-factor`, `yarn-beta-fast`, `yarn-beta-slow` | engine | number | 0 (262144), auto, 1, 32, 1 | Flash-Next only. YaRN parameters, as in llama.cpp. |

## Expert cache and memory

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `vram-reserve-mib` | engine | MiB, at least 256 (128 on Linux) | 700 (lowered to 300 on a card with too few slots) | VRAM kept free on the main card for graphs, scratch and the head. Raised to the floor the kernels need (`--parallel 4` on Linux: at least 320). |
| `resident-experts` | engine | switch | on with a profile | Flash-Next only. Keep experts the GPU does not hold in page-locked RAM. Falls back to a file mapping when RAM is short. |
| `resident-budget-gib` | engine | GiB | auto | At most N GiB of experts in RAM, hottest first. |
| `mmap-experts` | engine | switch | off | Flash-Next only. Use a file mapping instead of the RAM copy. |
| `shared-expert-arena` | engine | file path | off | Flash-Next only. One shared file for the resident arena across processes (put it on `/dev/shm`). |
| `expert-profile` | engine | file | `data/profiles/<arch>-<layers>x<experts>.prof` | Which experts start in VRAM. |
| `expert-profile-save` | engine | file | off | Flash-Next only. Save the learned placement on exit and periodically; a start places from it. Needs the adaptive tier. |
| `expert-profile-save-every` | engine | minutes | 10 (`0` = on exit only) | Flash-Next only. Save period. |
| `resident-cpu-experts` | engine | switch | off (implied by `resident-experts`) | Flash-Next only. With `mmap-experts`: keep RAM-resident experts in ordinary RAM. |
| `vram-elastic` | engine | switch | off | Flash-Next only. Hold the expert cache in segments so `POST /v1/vram` can lend VRAM between requests. |
| `vram-segment-mib` | engine | MiB, at least 64 | 512 | Flash-Next only. Segment size for `vram-elastic`. |
| `expert-cache` | engine | `auto` or a count | `auto` | Expert blobs kept in VRAM. `auto` takes what fits after weights, session and reserve. |
| `ram-reserve-mib` | engine | MiB | 2048 | RAM left free for the rest of the system. The expert tiers take what is above it after the engine's own host buffers, and the serving RAM guard holds it. `0` keeps nothing and turns the guard off. [memory floors](../running.md#memory-floors) |
| `adapt-every` | engine | windows | 1 on a server, 4 with a layer split (`0` = off) | Flash-Next only. An adaptive round every N windows. [adaptive](adaptive.md) |
| `adapt-swaps` | engine | integer | 96 (`0` = off) | Flash-Next only. Most experts one round swaps into the main GPU. [adaptive](adaptive.md) |
| `peer-adapt-swaps` | engine | integer | `adapt-swaps` | Flash-Next only. Most experts one round swaps into the peer card. |
| `adapt-decay` | engine | number | 0.85 (0.7 with a layer split) | Flash-Next only. Factor applied to routing counts after each round. [adaptive](adaptive.md) |

## Prompt processing

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `prefill` | engine | `auto`, `auto:N` (N = 8192, 16384 or 32768), or tokens | `auto` | Flash-Next only. Prompt chunk size. |
| `prefill-cpu-share` | engine | 0 to 1 | 0.6 | Flash-Next only. Share of streamed prompt experts the CPU computes. Fixed, so output is the same run to run. |
| `short-read` | engine | tokens | 64 (`0` = off) | Flash-Next only. A fresh part of at most N tokens is read through decode windows. |
| `no-prompt-skip` | engine | switch | off | DeepSeek only. Run every row through every layer. Slower; the same math. [precision](../precision/deepseek41.md#prompt-skip) |
| `no-peer-prefill` | engine | switch | off | Flash-Next only. The peer card computes none of the prompt rows; the prompt stays on the main card. |

## Drafting

Unless `relaxed-drafting` is on, output is exactly plain decoding's.

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `spec` | engine | integer | 8 (MTP), 6 (DSpark), else 0 | Verify windows of at most N tokens. `0` or `1` turns it off. |
| `draft-experts` | engine | 32, 64, 96, 128, or 0 | 32 | DeepSeek only. Experts each drafter stage keeps (`0` = all 128). |
| `suffix-draft` | engine | tokens | 3 (`0` = MTP only) | Flash-Next only. Prompt lookup: fill a window from an earlier repeat of at least N tokens. |
| `mtp-max-t` | engine | integer | 0 (= `spec`) | Flash-Next only. Caps the MTP's own windows. |
| `mtp-window` | engine | cells | 32768 (`0` = all) | Flash-Next only. The drafter attends to the last N cells. |
| `relaxed-drafting` | engine (`[P]`) | `true` or 0 to 1 | off | Flash-Next only. **Changes output.** Under sampling, keeps a draft the model gives probability P or more. Greedy, `logit_bias`, JSON and tool calls stay exact. [precision](../precision/qwen38-flash-next.md#relaxed-drafting) |
| `route-tail-skip` | engine (`[R]`) | `true` or 1 to 9 | off | Flash-Next only. **Changes output.** Skips experts no GPU holds when they rank R or lower in every window entry. Needs the device MoE tail. [precision](../precision/qwen38-flash-next.md#lossy-routing-flags) |
| `route-prior` | engine (`[LAMBDA]`) | `true` or 0 to 16 | off | Flash-Next only. **Changes output.** Adds LAMBDA to held experts' router logits. Needs the device MoE tail. [precision](../precision/qwen38-flash-next.md#lossy-routing-flags) |

## Parallel slots

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `parallel` | entry | 1..8 | 1 | Serve N requests at once. Sets the engine's `batch`. With `kv-resident`, each slot keeps only those cells in VRAM. |
| `slot-full-context` | entry | switch | off | Give every slot the whole `max-context`, at N times the KV. |

## Conversation cache

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `prompt-cache` | engine | integer | 6 | Conversation checkpoints kept, about 118 MB of RAM each. `0` reads every prompt from the start. |
| `prompt-cache-every` | engine | tokens | 16384 (`0` = off) | Flash-Next only. Also checkpoint every N freshly read tokens (RAM only). |
| `prompt-cache-root` | engine | tokens | 2048 (`0` = never) | Checkpoint the end of the system prompt when it is at least this long. |
| `turn-token` | engine | token id | 248045 | The token that opens a chat turn; checkpoints go at the last one. |
| `tail-role-token` | engine | token id | -1 (off) | A short trailing turn that opens with this token keeps the checkpoint in front of it. |
| `conversation-cache-mib` | engine | MiB | 0 (off) | RAM budget for parked conversations. |
| `conversation-cache-slots` | engine | integer | 4 | Most parked conversations. |
| `conversation-cache-min-free-mib` | engine | MiB | 2560 | Physical RAM kept free when parking. |
| `conversation-cache-min-tokens` | engine | tokens | 0 | Park only conversations of at least this many tokens. |
| `conversation-cache-disk-mib` | engine | MiB | 0 (off) | Disk budget for parked conversations. [running](../running.md#conversations-on-disk---conversation-cache-disk-mib) |
| `conversation-cache-dir` | engine | directory | model cache folder | Another folder for the disk tier. |

## Vision

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `vision` | entry | `off`, `dynamic`, `cpu`, GPU number | `off` | Image input. A GPU number runs a sidecar on that card (about 2.2 GiB of its expert cache). `cpu` is slow. Needs a model with a vision projector. 32 MiB and 64 megapixels per image. [running](../running.md#vision---vision) |
| `image-url-host` | server | `HOST` or `HOST:PORT`, repeatable | none | Allow `http(s)` image URLs on this host from other machines. [image sources](image-sources.md) |
| `image-dir` | server | directory, repeatable | none | Allow image files under this directory from other machines. [image sources](image-sources.md) |
| `image-remote-only` | server | switch | off | Treat this machine's requests as remote too. [image sources](image-sources.md) |

Diagnostic flags: [diagnostics](diagnostics.md).

## Control vectors

A GGUF control vector added to the residual stream. A request switches it with `experimental_speed_projection`. A layer split
or a prompt split keeps it on every card: each stage applies the vector to the steered layers it reads.

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `control-vector` | engine | file | off | Flash-Next only. A vector at scale 1. |
| `control-vector-scaled` | engine | `FILE:SCALE[,...]` | off | Flash-Next only. Vectors with their scales. |
| `control-vector-layer-range` | engine | `A B` | 1 to last layer | Flash-Next only. Layers it applies to, inclusive. |
| `cvec-mode` | engine | `add`, `project` | `add` | Flash-Next only. `add`: h += s v. `project`: h -= s (h.v) v. |
| `cvec-dir` | engine | `per-layer`, `single:L` | `per-layer` | Flash-Next only. Each layer's own direction, or layer L's everywhere. |
