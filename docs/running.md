# Running the server

The long-form notes behind the settings in [configuration](configuration/index.md): concurrent requests, the KV cache, memory floors and image input. Diagnostic flags: `bin/basalt --help`.

## Start-up and loading

The server and the page are up at once; the model loads behind them. While it loads (at start, on the first `--lazy`
request, on `POST /load`, after an idle unload) the header reads `Loading weights` with the percent and each device's
GiB, and a `--idle-sleep` wake reads `Waking up`. `/metrics` has `live.state` `loading` with `live.load.kind` `weights`
or `wake`; `/v1/status` and `/health` carry `loading`. Requests wait, then are served; a failed load gives them a 503 in
the API's own error shape. The start-up load tries once and a failure ends the server; reloads try up to 3 times.

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
solo session of the same length, come out of the expert cache. Each slot keeps two conversation checkpoints, the end
of its system prompt and its last turn (each when at least `--prompt-cache-root` tokens lie before it), and a new
request reads only what follows the longest start it shares with any slot's checkpoint, decoding or idle: requests that
share a long system prompt read it once.

Every chat, messages and responses request must name the served model in `model`: the alias (or the model's name) that `/v1/models` lists, or the model file's name. An unknown or missing name is a 404 in the API's own error shape, and replies always carry the served name.

With `--max-context` above 32768 the KV cache is elastic (not with `--kv-resident` or `--vram-elastic`; a peer card is
fine): each session backs its first 32768
cells at startup (the K/V pools, the indexer's block keys, the RoPE tables and the draft layer's K/V alike), and the
rest of that VRAM holds the expert cache's last slots until a request needs it. A request backs its prompt plus 16384 cells before the prompt is read (and 16384 more whenever decode runs past that) by taking
those slots back; their experts become CPU misses read from their RAM copies, which the RAM copy keeps for exactly this.
The next request that needs less gives the cells back and the cache refills the slots from RAM. A short request so
decodes with the cache a 32k context would have, whatever `--max-context` is. With `--parallel` every slot is elastic
once its share of the context is above 4096 tokens: it backs 4096 cells at startup and grows by 4096, so idle and short
slots leave their VRAM to the expert cache. The startup line `elastic KV: ...` says
how much of the cache is lent this way, and each move is logged (`elastic KV: N cells backed ...`). The cache never goes
below the slots a prompt chunk needs: a request whose KV would take it there, even after the other sessions give back
the cells they do not hold, is refused with that error, and a reply that grows past it ends with it (a `--parallel`
slot ends as at its length limit); the engine keeps running.

While other slots are decoding, a new prompt of up to 16384 tokens is read at once: the slots wait for it (a few
seconds at most), and it is read as in the single-stream path: the same chunks and streamed ring, the CPU pool's
share of its experts. Its first token so comes about as soon as on an idle engine. A longer prompt is read beside them:
the prompt path runs on its own thread, the slots' decode windows keep running (their kernels take the higher stream
priority), and the few steps that touch the draft layer or the output run between two windows, so a long prompt does
not stop the other requests. A much shorter request that arrives meanwhile goes first at the end of the long prompt's
current chunk; the rest of the long prompt is read after it, beside the slots, in the same chunks. The cache slots its loan and its KV take hold, before they are taken, only experts the
decoding slots route least: the ones they routed lately trade places with less-routed experts of their layer, so the
slots keep their experts in VRAM while the prompt is read. A prompt that arrives while no other slot is decoding is read as in the single-stream
path.

On one card, a batch window's layer with enough CPU misses sends a share of them over PCIe: CUDA0 copies them from
their RAM copies into VRAM staging and computes them while the CPU pool computes the rest, both reading RAM at once.
The startup line `batch windows: a layer with N+ CPU misses sends a share of them over PCIe to CUDA0` gives the
threshold. A request decoding alone runs the single-stream path and is not affected.

At startup the server prints one `[server] kv` line per device with all the session state placed there, for
example `12.5GiB cuda2 solo=262144 slots=2x262144 draft=0.5GiB`. With elastic KV (the
default for a server whose `--max-context` is above 32768, or with `--parallel` for slots above 4096 tokens) it reads `0.6/4.4GiB cuda0 elastic`: the memory backed at startup, and what the cache can
grow to at `--max-context`. Because that memory is
taken from the expert cache, the engine warns (`warning: the N batch slots' sessions ... take X GiB of CUDA0
...`) when it leaves the cache short of the expert set, or at under half of what it would hold without the slots, and
names the remedies: a smaller `--max-context`, no `--slot-full-context`, fewer `--parallel` slots. If a slot's session
does not fit at all, the engine says which slot (`warning: --batch N: slot K's session does not fit`) and starts with
the slots that did fit; if fewer than two fit, `--batch is off` and the server runs one request at a time.

- `server` holds the server's options (`python -m serve.server --help`), named as the flags without the dashes:
  strings or numbers, `true`/`false` for a switch (`lazy`, ...), a list for the repeatable ones
  (`allowed-host`, `cors-origin`). `model` and `tokenizer` paths are relative to the file.
- `engine` holds engine flags (`bin/basalt --help`), the same way: a string or number, `true` for a switch, `false` to
  leave it off, a list for a flag with several values (`"control-vector-layer-range": [4, 44]`). Any flag the engine
  accepts works, on the command line as well (`--kv int8`, `--resident-experts`). A name the engine does not accept
  stops the server at startup, as does one in the wrong section.
- Precedence: command line > config file > built-in defaults.
- The API key is command line only (`--api-key`); a config file that sets it is refused.
- `"relaxed-drafting": 0.3` (or `true` for the default 0.5; `--relaxed-drafting [P]` on the command line) trades
  exactness for speed under sampling: a draft the model gives probability P or more is kept even when the model's own
  draw would have been another token. Lower P is faster and further from the model's true sampling distribution; the
  measured speed and cost per value are in [precision/features.md](precision/features.md#relaxed-drafting). Off by
  default: without it, speculative decoding returns exactly what plain sampling would. Greedy (temperature 0) requests,
  `logit_bias`, a request with a grammar (JSON `response_format`, tools), control tokens and everything inside a tool
  call stay exact. A request can set its own value with `"relaxed_drafting": P` (or `false` for exact). Single-stream
  decoding only; `--parallel` slots stay exact.
- The startup line `[server] engine ...` lists the settings that came from the file or the command line, then the
  grammar state; with none it says `defaults`, e.g. `grammar=off (from cli)`.

## The console log

Every server line starts with `[server]` and a label padded to a column; values are `key=value`. Rates are `t/s`,
sizes `GiB`/`MiB`, times `s`. Startup prints one block, starting with a legend:

```
[server] legend   pp=prompt read  tg=generation  c=cached  acc=drafts accepted  dl=tokens per window  hit=expert cache  pcie=routed experts over PCIe
[server] engine   ctx=262144 kv=int8 vram-reserve=500MiB pool=8 batch=4 (from config)
[server] sampling temp=1.0 top_p=0.95 top_k=20 min_p=0.0 pres=0.0
[server] vision   mode=dynamic gpu=0 (RTX 5090) load=per-image
[server] auto     peer-excl=off resident=64.0GB pool-affinity=auto
[server] stats    http://127.0.0.1:8080/
[server] loaded   68.4GiB 37.5s  cuda0=26.6GiB ram=41.2GiB cuda1=0.6GiB cuda1-copy=14.2GiB
[server] kv       2.6/16.3GiB cuda0 elastic slots=4
[server] slots    par=4 ctx/slot=262144 (slot-full-context), others queue
[server] ready    http://127.0.0.1:8080/v1  openai+anthropic  ctx=262144 images=on
```

`sampling` marks a value set on the command line `(cli)` or in the config file `(config)`; `auto` is the engine's
picked settings (see [configuration](configuration/index.md)). A warning or error that comes later reads
`[server] warning: text` or `[server] error: text`.

While a prompt is read, a slot prints `pp=<read>/<total>` with the read rate and the elapsed time about every 5 s:

```
[server] s2 read  pp=24576/66836 9150t/s 3s
```

While requests generate, one `live` line every 2 s covers every generating slot: `tg=<tokens>/<max> <rate>`.
Slots are counted from 1, as on the stats page. `total` is the sum of the slots' rates. Acceptance is not shown
live; the `done` line carries the engine's exact figures.

```
[server] live | s1 tg=454/2048 113.7t/s | s2 tg=436/2048 119.0t/s | total 233t/s
```

Each request ends with one slot-tagged `done` line:

```
[server] s1 done  pp=96/89c 0.1s  tg=1720 119.6t/s acc=65% dl=2.9  ttft=0.1s  end=stop  hit=89.7%  pcie=39.3%
[server] s2 done  pp=66836/0c 7.3s 9204t/s  tg=512 281.0t/s acc=58% dl=2.7  ttft=7.3s  end=length
```

- `pp=<prompt tokens>/<cached>c <read time>`, with the read rate once at least 512 tokens were newly read; the rate
  counts newly read tokens only.
- `tg=<tokens> <rate>`. `acc` is drafts accepted over drafts offered for the request, `dl` the average tokens per
  window; both are left out when the request did not draft.
- `ttft` is the time to the first token; `end` is the finish reason (`stop`, `length`, `cancelled`, `disconnect`,
  `error`).
- `hit` is the expert cache hit rate and `pcie` the share of routed experts read over PCIe; both appear when the
  engine reports them.

## The stats page

The server's own address (`[server] stats` above) serves one page that reads `/metrics`, `/slots` and `/api/logs`
once a second:

- GPUs: every card the engine uses, with VRAM and 60 s charts of load and power (one card adds temperature; three or
  more cards are a table with a chart of their mean load and total power). System: CPU, RAM, disk reads, driver and
  CUDA. Experts: where the last request's routed lookups were served (VRAM, peer card, CPU).
- Speed: decode and prefill rates with 60 s charts (one line per slot with `--parallel`), the last prompt and its prefix
  cache hit, and the fastest whole request since the server started (at least 256 output tokens for decode, 4096
  newly read prompt tokens for prefill).
- Slots (with `--parallel`): each busy slot's rate, output against its limit, context, time to first token, prefix
  hit and tokens per window; Accept is the slot's last finished request. Requests: the newest finished requests.
  Queue: the requests waiting for a slot or for the single stream.
- Logs: the console lines.

`/metrics` carries the history behind the charts in `history` (`tok_s`, `prefill_tok_s_mean`, `slot_tok_s` one series
per slot, `gpus` with `util`, `power` and `temp` per card) and the fastest requests in `best`.

## Conversations on disk: `--conversation-cache-disk-mib`

`--conversation-cache-disk-mib N` (an engine flag, also `"conversation-cache-disk-mib"` in the config's `engine`
section) parks conversations on disk under the RAM cache, in at most N MiB. It is off at 0, the default. It writes to
the folder `conversations` inside the model's cache folder (`~/.cache/basalt/<hash>/`, where the engine log is);
`--conversation-cache-dir DIR` only names another folder, and does nothing on its own. When the tier is on the server
prints `[disk] conversations=DIR disk-mib=N` at start. A conversation of at least 8192 tokens that leaves its session
(another one takes its slot), that the RAM cache evicts, or that the engine still holds when it ends (a stop, an idle
unload) is written to DIR: one file for its tokens and running state, and shared files for its K/V cache (one per 4096
cells) and for each checkpoint's state, each named by the tokens it covers and its bytes, so the same conversation
written again, or another that starts the same way, writes only the blocks and states that are new. A conversation whose
last checkpoint (the end of its last prompt) is at most 4096 tokens behind is stored as of that checkpoint, with no
running state of its own: the next turn goes on from there, and a reply sent back as it was is read again. A later
prompt that starts with a stored conversation (its tokens, or one of its checkpoints) reads it back cut at that point
and goes on from there, after a restart too: the wait is the drive reading the K/V cache instead of the prompt being
read again. A file is bound to the model file, the stored-conversation format and the settings that change the state
(the KV type, the rope, the draft layer, control vectors); every part of it carries a hash, and a file that does not
check out is deleted, never restored. A conversation is written again only once it has grown 4096 tokens past its stored
copy, and after a second without input (no slot decoding) the engine writes each one that has, so a crash or a kill
loses at most that much of a conversation. The budget caps the folder, least recently used conversations first (a shared
file goes with the last conversation that uses it), and a write always leaves 4 GiB of the volume free. Writes on the
drive that holds the model file can slow its next load: a cache directory on another drive avoids that.

## Sleep: `--idle-sleep`

`--idle-sleep SECONDS` (a server option, also `"idle-sleep"` in the config's `server` section) puts the model to sleep
after that long without requests. The engine first gives its expert caches' VRAM back, a peer card's tier included (each
expert has its RAM copy and the slot it held is remembered; an exclusive peer tier stays), and drops the RAM copy's
registration with the driver, then the driver moves the rest of its GPU state (weights, KV cache, buffers) to host
memory and releases the VRAM. The process, its RAM copy of the experts, the conversations it holds and the placement the
adaptive tier learned all stay. The next request wakes it: the driver puts the GPU state back at the same addresses, the
engine registers the RAM copy again and the caches refill from the parts already registered, so the request runs as it
would have before the sleep. The server checks first that the GPU state fits in the free RAM with 8 GiB to spare;
otherwise the model stays awake (`[server] warning: the model stays awake ...`). Linux only: the driver offers no such
suspend elsewhere. `--idle-unload` still ends the process, which frees the RAM too; with both, give `idle-unload` the
longer time.

## Memory floors

Basalt refuses settings that would leave the machine without memory, and never shrinks what you asked for:

- `--vram-reserve-mib` below 256 is refused at startup: with less VRAM free once everything is loaded, a later
  allocation can make the driver page GPU memory and requests stall.
- When some experts are read from the SSD (they fit in neither the GPUs nor the RAM copy), the RAM copy leaves a
  floor free for the OS file cache: the RAM margin (2 GiB or 5% of the RAM, the larger) plus the larger of 4 GiB and
  what one prompt chunk reads from the file (up to 3.5 GiB for the lent GPU slots, plus the SSD-held experts of two
  layers). The startup line `RAM floor ...` gives the figures. When every expert has a place, only the margin is kept;
  the copies kept for the lent slots and the adaptive swaps do not count toward that: they take the RAM left over.
- A server's RAM copy also leaves room for what serving takes after startup: the conversation checkpoints (each the
  model's whole running state, `--prompt-cache` of them per slot plus one), `--conversation-cache-mib`, one image's
  buffers and a request's own buffers. The startup line `RAM ... free, guard ...` gives the figures.
- With the adaptive tier, the RAM copy also holds free places for the experts that leave the GPU, so a topic change can
  swap quickly. The startup line `free places for the adaptive swaps` shows the count (four times `--adapt-swaps` on a server); if RAM is short, fewer are kept
  and a round swaps fewer.
- An explicit `--resident-budget-gib` that would leave less than that floor free is refused at startup; the error
  states the RAM free and the largest safe value ('max').
- While serving, the RAM guard watches MemAvailable against the margin plus a prompt chunk's file reads, up to 3.5 GiB
  (the margin alone when no expert is read from the SSD). Below it, the engine's parked conversations leave RAM (except
  the one the next request continues): with the disk tier on they go to disk (those it holds no copy of yet), otherwise
  they are dropped. The engine also frees the checkpoint blocks it keeps for reuse and returns its free heap to the OS,
  and a request that arrives waits up to 30 s for the RAM to come back before it starts; one that adds at most 256
  tokens to the live conversation (the server's wrap-up after a thinking budget) does not wait. Below the guard nothing
  new is parked, and the disk tier neither writes a conversation still in a session nor reads one back (its own floor
  sits 1 GiB above the guard's). A running request continues; it stops with an error only when the RAM is critical:
  under 1 GiB free, or below the guard while the kernel's memory pressure (`/proc/pressure/memory`, avg10) shows tasks
  stalled most of the time (some >= 60% or full >= 40%). Without a pressure reading (Windows, or a kernel without PSI)
  below the guard is critical. The error (`RAM low (... free < ... guard) ...: request stopped` or `RAM critical (...
  free < ... hard floor): request stopped`, also returned to the client) names `--resident-budget-gib` only when the
  engine's own growth took the RAM, otherwise how much other programs took. While the model sleeps the guard judges no
  reading (the driver holds the GPU state in RAM until the wake); the wake takes a fresh one. Windows does not count the
  pages read through the mapped expert file as available, so within 1 GiB of the guard the engine gives them back first
  and reads again.

## Vision: `--vision`

Image input is one server option, `--vision off | dynamic | cpu | <GPU number>` (config: `"vision": 1` in the
`server` section). It is `off` by default. The encoder is Basalt's own and shares its kernels in every mode, so all
of them give bitwise identical embeddings.

| mode | where the encoder runs | VRAM cost | fresh-image latency (IQ3_S, 400 W, earlier build) |
| --- | --- | --- | --- |
| `off` | no image input | none | |
| a GPU number | sidecar `bin/basalt-vision` on that card (the numbering `gpu` uses), always loaded | about 2.2 GiB of that card's expert cache | TTFT 0.77 s at 1k image tokens, 1.4 s at 3k, 1.8 s at 4k |
| `dynamic` | inside the engine, on the main GPU | none at rest (0.85 GiB of pinned RAM) | 0.77 s, 1.45 s, 1.8 s |
| `cpu` | sidecar on the CPU | none | about 6 s at 1k, 40 s at 3k, 70 s at 4k |

`dynamic` borrows the expert cache's last slots per image (any slot whose expert has a RAM copy), uploads the weights
from the pinned copy (about 30 ms), encodes on a side stream, and refills the same slots from RAM (about 70 ms), so the
cache is identical afterwards. Other `--parallel` slots pause for the encode. When a long conversation's elastic KV
holds that room, the sessions first give back the KV cells they do not hold (idle slots first) and take them back as
they grow. An image is refused only when the room cannot exist even then. An image already seen is served from the
encoder cache in tens of milliseconds in every mode. The sidecar's card is named, not derived from `gpu`: any card
works, whether or not it runs the engine, and the startup line names it.

```json
{ "server": { "gpu": "0,1", "vision": 1 } }
```

A GPU number uses the sidecar, which the main build also produces (`tools/vision` also builds on its own); the
server looks for it next to `bin/basalt` and only needs the named card to exist: the sm_120 rule is the engine's,
not the separate sidecar binary's. The card a sidecar is built for is chosen at its configure time
(`-DCMAKE_CUDA_ARCHITECTURES=<cc>`); pointed at another one it fails at startup with CUDA's own error, and the
server shows it. Sidecars on non-Blackwell cards are community-tested. `dynamic` needs no sidecar.

The image encoder is Basalt's own, written for this model's vision tower (no llama.cpp or ggml): bf16 tensor-core
products and flash attention with fp32 accumulation on the GPU, AVX-512/AVX2 on the CPU. It replaces llama.cpp's
mtmd encoder as a drop-in, is faster than it at every image size, and stays closer to an fp32 reference: mtmd's fp16
attention drifts on large images, ours does not.

One image is capped at 64 megapixels; a header past it is refused before it is decoded.

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

`PROBE adapt [every=N] [swaps=N] [decay=F] [profile=FILE]` puts the expert tiers back where they were at `READY`, or
with `profile=` places that saved profile's ranking into the same slots, copying only the slots whose expert differs.
It zeroes the routing counts, the round counter and the draft policy, drops the session's cached prompts, and sets the
adaptive tier's settings for the requests that follow, so the next request runs as the first after a load. It answers
`PROBE end adapt every= swaps= decay= moved= peer_moved= kept= ms=` (slots copied on each card, slots left as they
were, wall time). It needs the adaptive tier on and one main card (no layer split, at most one peer card, not
`--peer-exclusive`); a swap count above the start's `--adapt-swaps` finds fewer free RAM places for its rounds.
