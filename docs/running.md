# Running the server

Settings: [configuration](configuration/index.md). Diagnostic flags: `bin/basalt --help`. Several models: [models and the router](configuration/models.md).

## Start-up

The page is up at once and the model loads behind it. The header reads `Loading` with the percent over a progress bar while weights load; choosing a model in the picker loads it, and the Unload button beside the model name unloads the loaded one until the next request; each reads `Loading…` or `Unloading…` until the server shows the change, and a refused one shows its error in the header. `/health` and `/metrics`
show the load: `load.percent` and `load.gib`, the GiB placed so far in one figure over every byte the model loads (a card, a second card, RAM), where it was one figure per target. A failed start-up load ends the server. Requests wait during a load and get a 503 if it fails.

## Concurrent requests: `--parallel N`

`--parallel N` (1 to 8) serves N requests at once in batch slots. Further requests wait in arrival order. `--max-context` is shared:
each slot holds `max-context / N` tokens (`--parallel 4 --max-context 262144` gives four slots of 65,536). A request longer than its slot
is refused with a 400 that names the length; the start log's `slots` line shows the context each slot gets. `--slot-full-context` gives every slot the whole `--max-context`, at N times the KV cache.

Each slot keeps two checkpoints (end of the system prompt, last turn), so requests that share a long system prompt read it once.

The engine tokenizes every prompt and matches it against what each slot holds. A reply the client sends back unchanged continues the
slot's own tokens even when the model wrote it in other pieces than the tokenizer cuts the same bytes into (not with images).

### Session affinity

With `--parallel`, a request that names its session goes to the slot that session's conversation sits in, so the next turn continues
the K/V already there. The session is the first of these headers that is set, then the request body's `session_id`: `x-llm-flow-id`,
`x-claude-code-session-id`, `x-session-id`, `x-session-affinity`, `x-grok-conv-id`, `x-grok-session-id`. A request with none picks a
slot by the longest shared prefix, then the least recently used. A session may hold several slots: a client may send one session id for a
session and its parallel subagents (Claude Code's `x-claude-code-session-id`, OpenCode's `X-Session-Id` and `x-session-affinity`, which
name the parent session), and each conversation keeps the slot its K/V is in. A request that continues none of the session's own slots
takes an unowned one first; with every slot owned, the session's own least recently used one comes before another session's.

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
- `acc`: drafts accepted over drafts offered. `dl`: tokens per window, from the engine's own window count over the whole request (its solo and batch runs and every pass). With `--parallel` a window serves every busy slot in a fixed number of rows, so `dl` falls toward 1.0 as the slots fill. Both are absent when the request did not draft.
- `end`: `stop`, `length`, `cancelled`, `disconnect` or `error`.
- `hit`: expert cache hit rate. `pcie`: share of routed experts read over PCIe.

### Request timing fields

The API's `timings` object and the matching `done`-line values are request-level wall-clock totals, not single-window CUDA timings. `prompt_ms` covers prompt reading; `decode_ms` covers decoding after prompt setup, including host time and gaps between decode windows or engine passes that belong to the request. A request that moves between solo and batch execution accumulates the applicable parts, so compare these fields with other request-level measurements. Use the engine log or `basalt-bench` window metrics when you need a per-window or kernel-oriented rate.

The engine's own `basalt ...` lines go to `~/.cache/basalt/<id>/engine.log`, one folder per model file (`<id>` is 16 hex digits
taken from the file); `[server] engine log: PATH` names it at start. A log of 10 MiB or more moves to `engine.log.1` at the
next start. An engine that dies of a fault or an unhandled exception writes a `basalt crash: error: ...` line (the fault, the stage
it was in) and the server's warning quotes it with the exit status. The folder also holds the conversation
cache. Nothing is unpacked from the model file: the server reads the tokenizer from its sections, and the engine and `basalt-vision`
read the vision projector inside it.

## The stats page

The server's address serves a page. Its figures arrive as one combined snapshot over `/api/state/stream`, once per
second and on observed request, queue or model state changes (checked every 100 ms); a shared publisher serves all
clients without per-client snapshot queues, and reconnecting clients receive its latest snapshot. This is a combined
view, not a transaction across all engine and router state; `/metrics`, `/slots` and `/api/logs` remain JSON endpoints
for external clients. Each SSE endpoint permits up to eight concurrent clients. The Logs tab receives live console
lines from `/api/logs/stream`, reconnects from its last cursor, and reports server restarts or lines lost from the
bounded buffer. Both log endpoints require the API key when one is configured; the page sends it in the Authorization
header. The top of the page always shows:


- Decode and prefill: the live rate with a 60 s chart; when idle, the last request's decode and the last prompt read of at least 512
  new tokens (a cached follow-up has no meaningful read rate). The decode number's tooltip has the best decode and the conversation
  cache; the prefill number's has the last prompt, its prefix cache hit and the best prefill (the fastest read listed when the
  server has none yet).
- Experts: where the last request's expert lookups were served (VRAM, a peer card, CPU and RAM, the SSD), as a legend, and a map
  of 640 experts sampled evenly from every layer (`experts_map`): its colour is where the expert lives now, its brightness how often
  decode looked it up lately, an ember edge a lookup off the GPUs in the last second. A dense model shows its weights by bytes.
  Experts on the SSD turn the heading amber.
- With no model loaded, a slow field of basalt columns on its own (8 frames a second, paused while the tab is hidden, one still
  frame under reduced motion).

Under it, four tabs (keys `1` to `4`, `L` for the logs, or `#tab=req|slots|hw|logs` in the address):

- Requests: the queue, the recent requests, tokens per request, the last request, the best since the server started, the
  conversation cache (`conversation`: `parked`, `bytes`, `evictions`, `restored`, `restored_ms`) and the totals (`totals`).
- Slots: a card per slot with its current request, or for an idle slot its last one, its context and its drafting; past four
  slots, compact rows. With no model loaded the cards stay, empty.
- Hardware: each card's load, power, temperature, VRAM and PCIe traffic, and the CPU, RAM, disk reads, experts in VRAM, driver and
  uptime.
- Logs: the console lines, all or warnings only. The engine's long cache lines are shown short (`cache: restored 187k tokens in
  912 ms`); the log and `/api/logs` keep the raw line. The footer always shows the newest line and the request and token totals.

The engine's own figures, which the server asks for with `STATS` once the engine is ready:

- `/metrics` `experts_map`: `total` experts, and for each sampled one `tier` (0 VRAM, 1 peer card, 2 RAM, 3 file),
  `routed` (decode lookups a second, 0 once nothing runs) and `recent` (its lookups, fading by 0.85 a second).
- `/slots` per slot: `started` (epoch seconds), `path` (the endpoint) and `prompt_ms` (the read time) of the running request;
  for its current or last request `accept_pos` (drafts accepted over offered by draft position) and `accept_hist` (tokens per
  decode window, a sample a second over 60 s); for a model with routed experts `hit`: its windows' lookups split `vram`, `peer`
  (a bus copy or another card) and `cpu`, with `lookups_s`, `peer_s` and `ram_s` a second. A batch window several slots share
  counts its lookups for each of them; a model whose slots run through the stages of `gpu-mode split` counts none per slot.
- `/metrics` `engine`: `commit` (the build's), and `expert_slots`, the experts every card holds now (fewer while asleep).
- `/metrics` `hardware`: each card's `sm_clock` and `mem_clock` (MHz), `throttle` (what holds the clocks down, empty for nothing),
  `expert_slots` and `h2d_gbps` (the host-to-device link as the start measured it; Flash-Next measures it, the other models
  leave it out); `cpu_ccd` (the load of each group of cores that share an L3 cache), `ram_parts` (`experts` the engine holds in
  RAM, `page_cache`, `other`; Linux only) and `disk_hist` (the read rate's last 60 s).
- **Prometheus:** `GET /metrics` returns Prometheus text exposition format (`text/plain; version=0.0.4`) when `Accept` asks for `text/plain` or `application/openmetrics-text`, or with `?format=prometheus` (JSON otherwise, so the stats page is unchanged). Every metric is named `basalt_*` (requests, tokens, latency histograms, draft acceptance, engine state, GPU and RAM); a dashboard made for another engine's names can use Prometheus relabeling.

The header word reads `Thinking`, `Answering` or `Tool call` (`live.phase`) while exactly one request runs. The conversation cache
figures come from the engine's `conversation cache:` lines and clear when the engine unloads, sleeps or restarts; a model that prints
no line when it parks (DeepSeek) shows them after its first restore, without evictions.

## Conversations on disk: `--conversation-cache-disk-mib`

Off at 0. With a budget, conversations of at least 8,192 tokens are written to `conversations` in the model's cache folder when they leave
RAM, a session, or at exit. Only the system prompt's and the last three turns' states are stored. Files are bound to the model and settings
they were made with, and checked by hash; a bad file is deleted. The folder is capped by the budget (least recently used first), and a
write always leaves 4 GiB of the volume free. After a restart, a prompt that starts with a stored conversation reads it back.

## Sleep: `--idle-sleep`

After the given seconds without requests, the engine frees its VRAM (the driver moves the GPU state to host memory). The process and its
RAM copy of the experts stay (a dense model has no cache to give back first: the driver moves all its GPU state, both cards' on two cards). The next request wakes it in seconds. Linux only. The server stays awake when the GPU state would not fit in
free RAM with 8 GiB to spare. `--idle-unload` ends the process instead.

With several models, [`--model-cache-mib`](configuration/models.md#parking-models-in-ram) keeps a model the router switches away from asleep in RAM in the same way, so its next turn wakes it instead of loading it.

## Lending VRAM: `POST /v1/vram`

With `--vram-elastic`, `POST /v1/vram` with `{"reserve_mib": N}` resizes the expert cache to keep N MiB free (`null`: the start's reserve).
A resize needs a gap between requests. Idle, it applies at once and answers `200` with the new figures. While a request runs it is
queued and answered `202 {"status": "pending"}`: it applies when the running requests have finished, and requests that arrive meanwhile
wait for it, so steady traffic cannot hold it off. A later POST replaces a queued one (`"replaced": true`). A malformed body is a
`400`. While the model sleeps, the reserve is kept and applied at the wake (`200 {"status": "asleep"}`), as when it is not loaded.
`GET /v1/status` shows `vram.pending` (the queued reserve, or `null`), `vram.applied` and `vram.error` (the engine refused
the queued one).

## Memory floors

Basalt refuses settings that would leave the machine without memory, and never shrinks what you asked for:

- `--vram-reserve-mib` below 256 is refused on Windows and WSL2 (128 on Linux): less free VRAM makes the driver page GPU memory.
- `--ram-reserve-mib` (default 2048) is the RAM left free for the rest of the system. The engine allocates its host buffers
  first, or holds their RAM when it page-locks them later; the expert tiers then take the free RAM less the reserve and less
  the driver's tracking of the pages they pin (`RAM floor ...` in the start log). Flash-Next's copy and top-up also leave 2 GiB beside the reserve
  for the host memory the engine takes after they read the free RAM (CUDA graphs, the first request), so the free RAM settles
  above the guard, not on it. When some experts stay in the file, the
  tiers also leave the OS file cache they are read through a floor: one prompt chunk's file reads and two layers' share of
  those experts, at least 4 GiB. Programs left idle beside a small reserve may be swapped out.
- While serving, the RAM guard holds the reserve: below it, it parks conversations to disk or drops them. A request waits up to 30 s
  for RAM to return. A running request stops with an error only when RAM is critical (under 1 GiB free, or below the reserve under
  memory pressure). `--ram-reserve-mib 0` turns the guard off.
- Experts on the SSD give one warning, the same for every model: `N experts on SSD (X GiB) · free X GiB RAM`. The console
  repeats it and the stats page adds an SSD tier and turns its Experts heading amber.
- The disk tier keeps its floor a GiB above the reserve.
- `--resident-budget-gib` that would leave less than the floor is refused with the largest safe value.
- The automatic RAM copy leaves the adaptive swaps' free places room beside the experts, so they need no setting of
  their own. An explicit `--resident-budget-gib` within the places' own size of the RAM room gives back the coldest
  experts for them: an evicted expert with nowhere to go is not evicted at all, and the tier moves nothing.

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

`dynamic` borrows the cache's last slots per image and refills them afterwards (about 70 ms), so the cache is identical after.
Hy4-flash reads images with `dynamic` only: its main-card expert slots go back for the request's pictures and fill again as after a
prompt. Which
images a request may name: [image sources](configuration/image-sources.md). An image may not sit in a system prompt.

## Video

With image input on, Qwen3.8-Flash-Next and Qwen3.8-27B read video (DeepSeek, Hy4-flash and Inkling 2 Mini do not). A user message names a video as it names an image, from
the same [sources](configuration/image-sources.md):

| API | Part |
| --- | --- |
| OpenAI chat | `{"type": "video_url", "video_url": {"url": "..."}}`; llama.cpp's `{"type": "input_video", "input_video": {"url": "..."}}` |
| Responses | `{"type": "input_video", "video_url": "..."}` |
| Anthropic | `{"type": "video", "source": {"type": "base64", "media_type": "video/mp4", "data": "..."}}`, or a `url` source |

Frames are sampled as the model's own processor samples them: 2 a second, 4 to 768, evenly spaced. A part may set `fps` (up to 60)
and `max_frames` (4 to 768). All frames are scaled to one size so the whole video fits `video-max-tokens` (default 8192 image
tokens), and each pair of frames is read as one picture with its time in the prompt. Audio and rotation metadata are not read: a
phone's portrait clip arrives sideways.

Decoding uses FFmpeg 6, 7, 8 or 9 from the machine ([platforms](platforms.md#ffmpeg-for-video)). At start the server prints
`video  on, FFmpeg 8 (...)`, or `video  off` with the reason, and then a video is refused with that reason. A video may be 1 GiB.

Qwen3.8-Flash-Next's frames decode on the encoder's GPU with NVDEC (H.264, HEVC, AV1, VP9 and the older codecs FFmpeg's NVDEC takes); a stream NVDEC
refuses decodes on the CPU, with the same pixels. Qwen3.8-27B's frames always decode on the CPU. With `dynamic` the expert cache lends NVDEC the memory for the decode when no
reply is running (tens of ms to lend and refill); beside a running reply the video decodes on the CPU, and the reply keeps going
until the frames are encoded. Another question about the same video, in the same chat or a new one, starts from the video
already read.

NVFP4 pack, `dynamic`, 400 W + 180 W, cold, time to the first token: a 30 s 1280x720 clip (8,202 prompt tokens) 2.7 s; a minute of
3840x2160 HEVC 5.2 s (decode 2.6 s, encode 0.3 s, prompt read 1.9 s); of 1920x1080 H.264 4.3 s. A new question on the same 30 s
clip: 0.4 s.
