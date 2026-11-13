# Models and the router

Several models share one server. A request's `model` field picks one, and one is loaded at a time. A swap ends one engine process
and starts the next.

```json
{
  "version": 2,
  "engine": { "conversation-cache-disk-mib": 32768, "idle-sleep": 300, "vram-reserve-mib": 500 },
  "models": [
    { "alias": "Flash-Next", "model": "flash-next.basalt", "gpu": "auto", "kv": "int8", "max-context": 262144, "parallel": 4,
      "slot-full-context": true, "vision": "dynamic" },
    { "alias": "DeepSeek", "model": "deepseek.basalt", "gpu": "auto", "max-context": 1048576, "vision": "dynamic" }
  ],
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

- `models`: each entry has a `model` (relative to the file) and an optional `alias`, the name the API reports.
- `engine`: applies to every model that takes it. An entry's own value wins.
- `server`: shared by the whole server: `host`, `port`, `allowed-host`, `cors-origin`, `image-url-host`, `image-dir`,
  `image-remote-only`, `lazy`, `open`, `debug`, `request-lines`.
- An entry takes engine flags and these server options: `gpu`, `vision`, `parallel`, `slot-full-context`, `idle-sleep`,
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

A request's `model` is the alias or the file's name on every API. Any other name is a 404. `/v1/models` shows `loaded`, `loading`,
`unloaded` or `failed`.

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
