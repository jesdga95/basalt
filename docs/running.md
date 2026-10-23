# Running the server

The long-form notes behind the settings in [configuration](configuration/index.md): concurrent requests, the KV cache, memory floors and image input. Diagnostic flags: `bin/basalt --help`.

## Concurrent requests: `--parallel N`

`--parallel N` (1..8, also `"parallel"` in the config's `server` section) serves N requests at once in the
engine's batch slots; the default 1 is the single-stream path. Further requests wait in arrival order (a later one
may go first only to resume a slot that holds its conversation, a few times at most). A slot is freed when its
request finishes, is cancelled or its client leaves. `/metrics` reports `live.slots` (`total`, `active`,
`queue_depth`). As in llama.cpp, `--max-context` is shared: each slot holds `--max-context / N` tokens (`--parallel 4
--max-context 262144` gives four slots of 65536; ask for `--max-context 1048576` for four of 262144). The solo
path, used while one request runs alone, has the same length, so a request that fits a slot never outgrows it, and
one that does not is refused with a 400 naming that length. `/slots`, `/props` and `/v1/models` report the context
each slot has. `--slot-full-context` (also `"slot-full-context"` in the config's `server` section) gives every slot
the whole `--max-context` instead, at N times the KV cache; the startup line says so. The slots' KV cache, and one
solo session of the same length, come out of the expert cache.

Every chat, messages and responses request must name the served model in `model`: the alias (or the model's name) that `/v1/models` lists, or the model file's name. An unknown or missing name is a 404 in the API's own error shape, and replies always carry the served name.

With `--max-context` above 32768 the KV cache is elastic (not when the model is split over GPUs with `--layer-split`, a peer
card is fine, and not with `--kv-resident` or `--vram-elastic`): each session backs its first 32768
cells at startup (the K/V pools, the indexer's block keys, the RoPE tables and the draft layer's K/V alike), and the
rest of that VRAM holds the expert cache's last slots until a request needs it. A request backs its prompt plus 16384 cells before the prompt is read (and 16384 more whenever decode runs past that) by taking
those slots back; their experts become CPU misses read from their RAM copies, which the RAM copy keeps for exactly this.
The next request that needs less gives the cells back and the cache refills the slots from RAM. A short request so
decodes with the cache a 32k context would have, whatever `--max-context` is. The startup line `elastic KV: ...` says
how much of the cache is lent this way, and each move is logged (`elastic KV: N cells backed ...`).

While other slots are decoding, a new prompt is read beside them: the prompt path runs on its own thread, the slots'
decode windows keep running (their kernels take the higher stream priority), and the few steps that touch the draft
layer or the output run between two windows. A long prompt therefore does not stop the other requests. A prompt that
arrives while no other slot is decoding is read as in the single-stream path.

At startup the server prints one `[server] kv cache:` line per device with all the session state placed there, for
example `12.5 GiB on cuda2 (solo 262144 tokens + 2 slots x 262144 tokens + draft K/V 0.5 GiB)`. With elastic KV (the
default for a server whose `--max-context` is above 32768) it reads `0.6 / 4.4 GiB on cuda0 (elastic)`: the memory backed at startup, and what the cache can
grow to at `--max-context`. Because that memory is
taken from the expert cache, the engine warns (`warning: the N batch slots' sessions ... take X GiB of CUDA0
...`) when it leaves the cache short of the expert set, or at under half of what it would hold without the slots, and
names the remedies: a smaller `--max-context`, no `--slot-full-context`, fewer `--parallel` slots. If a slot's session
does not fit at all, the engine says which slot (`warning: --batch N: slot K's session does not fit`) and starts with
the slots that did fit; if fewer than two fit, `--batch is off` and the server runs one request at a time.

- `server` holds the server's options (`python -m serve.server --help`), named as the flags without the dashes:
  strings or numbers, `true`/`false` for a switch (`lazy`, ...), a list for the repeatable ones
  (`allowed-host`, `cors-origin`). `model` and `tokenizer` paths are relative to the file.
- `engine` holds engine flags (`bin/basalt --help`), the same way: a string or number, `true` for a switch, `false`
  to leave it off, a list for a flag with several values (`"control-vector-layer-range": [4, 44]`). Any flag the engine accepts works, on the command line as well (`--kv int8`, `--resident-experts`).
  A name the engine does not accept stops the server at startup, as does one in the wrong section.
- Precedence: command line > config file > built-in defaults.
- The API key is command line only (`--api-key`); a config file that sets it is refused.
- `"relaxed-drafting": 0.3` (or `true` for the default 0.5; `--relaxed-drafting [P]` on the command line) trades
  exactness for speed under sampling: a draft the model gives probability P or more is kept even when the model's own
  draw would have been another token. Lower P is faster and further from the model's true sampling distribution; the
  measured speed and cost per value are in [precision/features.md](precision/features.md#relaxed-drafting). Off by default:
  without it, speculative decoding returns exactly what plain sampling would. Greedy (temperature 0) requests,
  `logit_bias`, a request with a grammar (JSON `response_format`, tools), control tokens and everything inside a tool call stay exact. A request can set
  its own value with `"relaxed_drafting": P` (or `false` for exact). Single-stream decoding only; `--parallel` slots
  stay exact.
- The startup line `[server] engine: ...` lists the settings that came from the file or the command line, then the grammar state; with
  none it says `using default flags` (with or without `--config`), e.g. `using default flags, grammar off (cli)`.

## Memory floors

Basalt refuses settings that would leave the machine without memory, and never shrinks what you asked for:

- `--vram-reserve-mib` below 256 is refused at startup: with less VRAM free once everything is loaded, a later
  allocation can make the driver page GPU memory and requests stall.
- When some experts are read from the SSD (they fit in neither the GPUs nor the RAM copy), the RAM copy leaves a
  floor free for the OS file cache: the RAM margin (2 GiB or 5% of the RAM, the larger) plus the larger of 3.5 GiB plus two layers'
  experts (what one prompt chunk reads from the file) and 4 GiB. The startup line `RAM floor ...` gives the figures.
  When every expert has a place, only the margin is kept, as before.
- A server's RAM copy also leaves room for what serving takes after startup: the conversation checkpoints (each the
  model's whole running state, `--prompt-cache` of them per slot plus one), `--conversation-cache-mib`, one image's
  buffers and a request's own buffers. The startup line `RAM ... free, guard ...` gives the figures.
- With the adaptive tier, the RAM copy also holds free places for the experts that leave the GPU, so a topic change can
  swap quickly. The startup line `free places for the adaptive swaps` shows the count; if RAM is short, fewer are kept
  and a round swaps fewer.
- An explicit `--resident-budget-gib` that would leave less than that floor free is refused at startup; the error
  states the RAM free and the largest safe value ('max').
- While serving, a request that finds MemAvailable below the margin plus 3.5 GiB (the margin alone when no expert is
  read from the SSD) stops with an error (`RAM low (... free < ... guard): request stopped`, also returned to the
  client); the server keeps running and takes the next request.

## Vision: `--vision`

Image input is one server option, `--vision off | on | peer | dynamic | cpu` (config: `"vision": "peer"` in the
`server` section). It is `off` by default. The encoder is Basalt's own and shares its kernels in every mode, so all
five give bitwise identical embeddings.

| mode | where the encoder runs | VRAM cost | fresh-image latency (IQ3_S, 400 W, earlier build) |
| --- | --- | --- | --- |
| `off` | no image input | none | |
| `on` | sidecar `bin/basalt-vision` on the main GPU, always loaded | about 2.2 GiB of the main card's expert cache | TTFT 0.77 s at 1k image tokens, 1.4 s at 3k, 1.8 s at 4k |
| `peer` | the same sidecar on the second `--gpu` card | part of the peer card's cache, none on the main card | 0.86 s, 2.1 s, 2.9 s |
| `dynamic` | inside the engine, on the main GPU | none at rest (0.85 GiB of pinned RAM) | 0.77 s, 1.45 s, 1.8 s |
| `cpu` | sidecar on the CPU | none | about 6 s at 1k, 40 s at 3k, 70 s at 4k |

`dynamic` borrows expert-cache slots per image (the prompt path's lend), uploads the weights from the
pinned copy (about 30 ms), encodes on a side stream, and refills the same slots from RAM (about 70 ms), so the cache is
identical afterwards. Other `--parallel` slots pause for the encode. An image already seen is served from the
encoder cache in tens of milliseconds in every mode. `peer` on a single GPU warns and loads the encoder on that GPU;
with two cards, `on` says so at startup and names `peer` as the way to put it on the second card.

```json
{ "server": { "gpu": "0,1", "vision": "peer" } }
```

`on` and `peer` use the sidecar, which the main build also produces (`tools/vision` also builds on its own); the
server looks for it next to `bin/basalt`. `dynamic` needs no sidecar.

The image encoder is Basalt's own, written for this model's vision tower (no llama.cpp or ggml): bf16 tensor-core
products and flash attention with fp32 accumulation on the GPU, AVX-512/AVX2 on the CPU. It replaces llama.cpp's
mtmd encoder as a drop-in, is faster than it at every image size, and stays closer to an fp32 reference: mtmd's fp16
attention drifts on large images, ours does not.

## `PROBE rates` (engine command)

With `--serve`, the line `PROBE rates` on stdin, between requests, measures the machine's rates inside the loaded
engine in about half a second and prints one block on stdout: `PROBE begin rates v1`, lines `PROBE <section> key=value ...`,
`PROBE end ms=<wall>`. It changes no engine state (tiers, caches, KV, RNG, the CPU pool's active worker count) and saves
nothing. Sections:

| section | measures |
|---|---|
| `gpu`, `host`, `pack` | device, link generation and width, power limit, RAM, CPU pool, expert format |
| `link` | pinned host-to-device and device-to-host GB/s per GPU, best of six 64 MiB bursts |
| `gpu_mem` | device-to-device copy GB/s per GPU (counts the bytes read and written) |
| `host_mem` | GB/s the CPU pool reads from RAM-resident experts, by active worker count |
| `cpu_expert` | cold experts through the pool at 1, 2, 3 and 8 per call, at every measured worker count and the current one |
| `peer` | peer round trip by entry count (1 to 8) and the linear fit: base and per-entry microseconds |
| `copy` | one expert-sized copy into each GPU, and the peer round trip while such a copy runs |

A step that cannot run prints `n/a` with a reason (no peer, no RAM-resident experts, a non-native expert format).
During a batch the command answers `ERR PROBE: ...`.

`PROBE set workers=K share=F` (either key) sets the CPU pool's active worker count and the prompt path's CPU share for the
requests that follow, and answers `PROBE end workers=<count>`. It is for `basalt-probe`; the settings last until the engine stops.
