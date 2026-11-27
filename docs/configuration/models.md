# Models and the router

Several models share one server. A request's `model` field picks one, and one is loaded at a time. A swap ends one engine process
and starts the next, or [parks](#parking-models-in-ram) it.

```json
{
  "version": 2,
  "engine": { "conversation-cache-disk-mib": 32768, "idle-sleep": 300, "vram-reserve-mib": 500 },
  "models": [
    { "alias": "Flash-Next", "model": "flash-next.basalt", "gpu": "all", "gpu-mode": "peer", "kv": "int8", "max-context": 262144, "parallel": 4,
      "vision": "dynamic" },
    { "alias": "DeepSeek", "model": "deepseek.basalt", "gpu": "all", "gpu-mode": "peer", "max-context": 1048576, "vision": "dynamic" }
  ],
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

- `models`: each entry has a `model` (relative to the file) and an optional `alias`, the name the API reports.
- `engine`: applies to every model that takes it. An entry's own value wins.
- `server`: shared by the whole server: `host`, `port`, `allowed-host`, `cors-origin`, `image-url-host`, `image-dir`,
  `image-remote-only`, `max-payload-mib`, `model-cache-mib`, `lazy`, `open`, `debug`, `request-lines`.
- An entry takes engine flags and these server options: `gpu`, `vision`, `parallel`, `kv-pool`, `idle-sleep`,
  `idle-unload`, `min-free-vram-mib`, `fit-max-tokens`, `grammar`, `video-max-tokens`, the reasoning settings and the sampling
  defaults.
- `version` is `2`. A file in the old shape (no `models`) is refused.
- The command line wins for every model.

## Checks at start

Each entry is checked before anything loads, by the engine's own rules:

| | Model entry or command line | Top-level `engine` |
|---|---|---|
| Flag the model does not have | stops the start | left out for that model; a start line says so |
| Value the model refuses (`max-context` past its maximum) | stops the start | stops the start |
| Unknown name | stops the start | stops the start |

`kv` takes each model's own values (`fp8` or `fp16` for DeepSeek); another value stops the start. Two aliases that are the same, or two entries sharing a cache
directory, stop the start too.

## Requests and turns

A request's `model` is the alias or the file's name on every API. Any other name is a 404, and so is no name at all while several
models are configured. With one model configured, a request may leave `model` out or send it as null and that model answers.
`/v1/models` shows `loaded`, `loading`, `unloaded` or `failed`.

Clients wait through a swap; they do not get a 503 for it. A streamed response sends keep-alive comments every 10 s while it waits.

- A request for the loaded model runs at once while nothing waits. Once a request for another model waits, later arrivals queue behind it.
- The head of the queue picks the next model; its swap starts when the running requests finish. The model takes every request queued
  for it at that moment, in one batch.
- A failed load answers its batch with a 503 naming the model, and the queue goes on. A load tries twice. With one model, a failed start
  load ends the server.

## What a swap keeps

| | Across a swap |
|---|---|
| Conversations on disk | kept in each model's folder |
| Conversations in RAM and slots | written to disk at exit when `conversation-cache-disk-mib` is on; otherwise lost |
| Sleeping engine | wakes on the next request for it |
| `POST /v1/vram` | the machine's; applies to each model that can resize its tier |
| Metrics | per model (`GET /metrics?model=NAME`) |

`POST /v1/load` with `{"model": NAME}` swaps ahead of a request. `/health` and `/metrics` show the swap's progress. The console prints
`[server] swap: Flash-Next -> DeepSeek, 2 waiting`.

## Parking models in RAM

`model-cache-mib` (server section or command line, off by default) keeps a model the router switches away from in RAM. Its engine is
put to sleep as `idle-sleep` does it: the VRAM is freed, the process, its RAM copy of the experts and its conversations stay, and the
driver holds the rest of its GPU state in RAM. Its next turn wakes it, a RAM-to-VRAM copy in place of a load from the SSD, with the
same state it had. A swap pays for the model it parks (the driver copies its GPU state to RAM) and for the one it wakes (the copy
back), so parking gains most for a model that loads slowly, such as one with many experts to read, and little for one a fast drive
loads in seconds. The budget counts what each parked model holds in RAM (its process and the GPU state the driver moved), not the
size of its file.

- A swap parks the model it leaves when its state fits: within the budget, with 8 GiB of RAM still free once the state is held (as
  for `idle-sleep`), and above the floor, which is the highest `ram-reserve-mib` of any model plus 1 GiB. Otherwise that model is
  ended as without the setting, and the console says why (`not parked: ...`).
- Parked models over the budget end least recently used first. A model that alone holds more than the budget is never parked.
- While models are parked, the server checks the available RAM every 2 s; under the floor it ends the least recently used one
  (`unparked ...`). The engines' own RAM guard then never has to act for the parked ones.
- Ending a parked model drops the conversations it held only in RAM; those on disk (`conversation-cache-disk-mib`) stay. A parked
  model that is stopped with the server writes its disk conversations as an engine does at exit.
- Parked RAM is not free for the loaded model: engines size their RAM tiers from what is free when they load.
- It needs two or more models. Windows (and any machine whose driver cannot suspend a process) refuses the setting at the start
  (`idle-sleep` only warns there). The start prints `parking`; `/v1/models` shows `parked_mib` for each model when it is on.
