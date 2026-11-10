# Running the server

Settings: [configuration](configuration/index.md). Diagnostic flags: `bin/basalt --help`. Several models: [models and the router](configuration/models.md).

## Start-up

The page is up at once and the model loads behind it. The header reads `Loading` with the percent over a progress bar while weights load; choosing a model in the picker loads it. `/health` and `/metrics`
show the load. A failed start-up load ends the server. Requests wait during a load and get a 503 if it fails.

## Concurrent requests: `--parallel N`

`--parallel N` (1 to 8) serves N requests at once in batch slots. Further requests wait in arrival order. `--max-context` is shared:
each slot holds `max-context / N` tokens (`--parallel 4 --max-context 262144` gives four slots of 65,536). A request longer than its slot
is refused with a 400 that names the length. `--slot-full-context` gives every slot the whole `--max-context`, at N times the KV cache.

Each slot keeps two checkpoints (end of the system prompt, last turn), so requests that share a long system prompt read it once.

### Session affinity

With `--parallel`, a request that names its session goes to the slot that session last used, so the next turn continues the K/V already
there. The session is the first of these headers that is set, then the request body's `session_id`: `x-llm-flow-id`,
`x-claude-code-session-id`, `x-session-id`, `x-session-affinity`. A request with none picks a slot by the longest shared prefix, then
the least recently used. When the session's slot is busy, the request takes that normal pick and the session's slot becomes the new
one. Claude Code sends one `x-claude-code-session-id` for a session and its parallel subagents, so their overlapping requests spread
over slots.

### Prompts beside decoding slots

A prompt of up to 16,384 tokens is read at once, and the slots wait a few seconds. A longer one is read beside them, on its own thread,
while the slots keep decoding. A short request that arrives meanwhile goes first at the end of the current chunk.

## The console log

Every line starts with `[server]`. Values are `key=value`. Start-up prints a legend, a line per model, `ready`, `loaded`, `kv`, `slots`.
While a prompt is read, a slot prints `pp=<read>/<total>` with the rate. While requests generate, a `live` line appears every 2 s.
Each request ends with one `done` line:

```
[server] s1 done  pp=96/89c 0.1s  tg=1720 119.6t/s acc=65% dl=2.9  ttft=0.1s  end=stop  hit=89.7%  pcie=39.3%
```

- `pp=<prompt tokens>/<cached>c`: the rate counts newly read tokens only.
- `acc`: drafts accepted over drafts offered. `dl`: tokens per window. Both are absent when the request did not draft.
- `end`: `stop`, `length`, `cancelled`, `disconnect` or `error`.
- `hit`: expert cache hit rate. `pcie`: share of routed experts read over PCIe.

The engine's own `basalt ...` lines go to `~/.cache/basalt/<id>/engine.log`, one folder per model file (`<id>` is 16 hex digits
taken from the file); `[server] engine log: PATH` names it when the engine fails to start. The folder also holds the conversation
cache. Nothing is unpacked from the model file: the server reads the tokenizer from its sections, and the engine and `basalt-vision`
read the vision projector inside it.

## The stats page

The server's address serves a page, updated every second from `/metrics`, `/slots` and `/api/logs`: GPU load, power, VRAM and temperature,
RAM and disk, decode and prefill rates with 60 s charts, slots, the newest requests, and the console lines.

Three quiet figures sit in it, each from `/metrics`:

- `N chats cached · X GiB` at the right of the Speed footer: the conversations parked in RAM (`conversation`: `parked`,
  `bytes`, `evictions`, and the last restore's `restored` tokens and `restored_ms`, in the tooltip). The server reads them
  from the engine's `conversation cache:` lines and clears them when the engine unloads, sleeps or restarts. A model that
  prints no line when it parks (DeepSeek) shows them after its first restore, without evictions.
- `N requests served · M tokens written` under the Slots table (`totals`: requests, prompt, reused and written tokens, prompt
  and decode time since the server started; the rest in the tooltip).
- The header word reads `Thinking`, `Answering` or `Tool call` (`live.phase`) while exactly one request runs.

The log panel shows the engine's long cache lines short (`cache: restored 187k tokens in 912 ms`); the log and `/api/logs`
keep the raw line.

## Conversations on disk: `--conversation-cache-disk-mib`

Off at 0. With a budget, conversations of at least 8,192 tokens are written to `conversations` in the model's cache folder when they leave
RAM, a session, or at exit. Only the system prompt's and the last three turns' states are stored. Files are bound to the model and settings
they were made with, and checked by hash; a bad file is deleted. The folder is capped by the budget (least recently used first), and a
write always leaves 4 GiB of the volume free. After a restart, a prompt that starts with a stored conversation reads it back.

## Sleep: `--idle-sleep`

After the given seconds without requests, the engine frees its VRAM (the driver moves the GPU state to host memory). The process and its
RAM copy of the experts stay. The next request wakes it in seconds. Linux only. The server stays awake when the GPU state would not fit in
free RAM with 8 GiB to spare. `--idle-unload` ends the process instead.

## Memory floors

Basalt refuses settings that would leave the machine without memory, and never shrinks what you asked for:

- `--vram-reserve-mib` below 256 is refused on Windows and WSL2 (128 on Linux): less free VRAM makes the driver page GPU memory.
- `--ram-reserve-mib` (default 2048) is the RAM left free for the rest of the system. The engine allocates its host buffers
  first, or holds their RAM when it page-locks them later; the expert tiers then take the free RAM less the reserve and less
  the driver's tracking of the pages they pin (`RAM floor ...` in the start log). When some experts stay in the file, the
  tiers also leave the OS file cache they are read through a floor: one prompt chunk's file reads and two layers' share of
  those experts, at least 4 GiB. Programs left idle beside a small reserve may be swapped out.
- While serving, the RAM guard holds the reserve: below it, it parks conversations to disk or drops them. A request waits up to 30 s
  for RAM to return. A running request stops with an error only when RAM is critical (under 1 GiB free, or below the reserve under
  memory pressure). `--ram-reserve-mib 0` turns the guard off.
- Experts on the SSD give one warning, the same for every model: `N experts on SSD (X GiB) · free X GiB RAM`. The console
  repeats it and the stats page turns its Experts block amber.
- The disk tier keeps its floor a GiB above the reserve.
- `--resident-budget-gib` that would leave less than the floor is refused with the largest safe value.

## Vision: `--vision`

`--vision off | dynamic | cpu | <GPU number>` (config: `"vision"` in `server`). The encoder is Basalt's own, and all modes give bitwise
identical embeddings. All of them read the projector in place from the model file's `vision/mmproj` section. The text can still
differ between modes: the sidecar's VRAM cost shrinks the expert cache, and which experts run on the GPU or the CPU shifts the
last bits of the logits, so a greedy answer can fork after some tokens (`dynamic` does the same with a different cache size).

| Mode | Encoder runs | VRAM cost | Fresh-image TTFT (IQ3_S, 400 W): 1k / 3k / 4k image tokens |
| --- | --- | --- | --- |
| `off` | none | none | |
| GPU number | sidecar `bin/basalt-vision` on that card | about 2.2 GiB of its expert cache | 0.77 s / 1.4 s / 1.8 s |
| `dynamic` | inside the engine, on the main GPU | none at rest (0.85 GiB pinned RAM) | 0.77 s / 1.45 s / 1.8 s |
| `cpu` | sidecar on the CPU | none | about 6 s / 40 s / 70 s |

`dynamic` borrows the cache's last slots per image and refills them afterwards (about 70 ms), so the cache is identical after. Which
images a request may name: [image sources](configuration/image-sources.md). An image may not sit in a system prompt.
