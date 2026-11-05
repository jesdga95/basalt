# Models and the router

A server runs one model or several. Several share one address: a request's `model` field picks one, and one is loaded at a
time. A swap ends one engine process and starts the next.

## The config file

```json
{
  "version": 2,
  "engine": { "conversation-cache-disk-mib": 32768, "idle-sleep": 300, "pool-workers": 8, "vram-reserve-mib": 500 },
  "models": [
    { "alias": "Qwen", "model": "qwen.basalt", "gpu": "auto", "kv": "int8", "max-context": 262144, "parallel": 4,
      "slot-full-context": true, "vision": "dynamic" },
    { "alias": "DeepSeek", "model": "deepseek.basalt", "gpu": "auto", "max-context": 1048576, "vision": "dynamic" }
  ],
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

- `models`: at least one entry, each with `model` (relative to the config file). Its `alias` is the name the API reports; without
  one, the name stored in the model file. A third model is one more entry.
- `engine`: what applies to every model that takes it. An entry's own setting wins over it.
- `server`: what the whole server shares: `host`, `port`, `allowed-host`, `cors-origin`, `image-url-host`, `image-dir`,
  `image-remote-only`, `lazy`, `open`, `debug`, `request-lines`. API key and image-source rules do not change per model.
- An entry or `engine` takes engine flags and these server options: `gpu`, `vision`, `parallel`, `slot-full-context`,
  `idle-sleep`, `idle-unload`, `min-free-vram-mib`, `fit-max-tokens`, `grammar`, `reasoning-budget-tokens`,
  `thinking-loop-tokens`, `answer-room` and the sampling defaults. `alias` and `model` belong to an entry.
- `version`: this format is `2`, which a file without the key also means. A file from before it (no `models`) is refused:
  `Basalt requires v2 config, please upgrade`. [`tools/migrations/config_v1_to_v2.py`](../../tools/migrations/README.md) prints
  the new shape.
- The command line wins, for every model. `--model` and `--alias` do not go with `--config`; `basalt serve --model X` alone is a
  one-model server.

## Checks at start

Every entry is checked before anything loads: the file, its tokenizer and its flags, by asking the engine with the model's own
rules. No flag's default differs by model.

| The setting | In a model entry or on the command line | In the top-level `engine` |
| --- | --- | --- |
| A flag the model does not have | stops the start, naming it | left out for that model; a start line says so |
| A value the model refuses (`max-context` past its maximum) | stops the start | stops the start |
| A name no model has | stops the start | stops the start |

`kv` belongs in the entries of models that have a choice of KV format: [DeepSeek](../deepseek41.md) has none, so `kv` in its entry
stops the start. `gpu: auto` is the engine's choice for each model.

Two names that answer for two entries, and two entries sharing a `conversation-cache-dir`, stop the start too.

## Requests

A request's `model` is the alias, or the model file's name, on every API: chat completions, messages, responses. Any other name
is the normal 404. `/v1/models` lists every configured model, with `status.value` `loaded`, `loading`, `unloaded` or `failed`.

Clients wait through a swap; no request gets a 503 for one. A stream gets its headers and keep-alive comments every 10 s while it
waits, and an error after that arrives as the stream's own error event. A client that hangs up gives its place up.

## Turns

- A request for the loaded model runs at once while nothing waits. Once a request for another model waits, later arrivals queue
  behind it, whatever their model.
- The head of the queue picks the next model. Its swap starts when every running request has finished.
- A model takes every request queued for it at that moment, in one batch; the others keep their order.

So no request waits behind more than the requests that arrived before it plus one batch per model ahead of it, and a swap serves
all that waited for that model.

## A failed load

The requests of that batch get the load's error (a 503 naming the model) and the next in the queue goes on: the server stays up.
`/v1/models` shows `failed` with the reason until a load succeeds. The next request for that model tries again. A load tries
twice, for VRAM another process is still releasing. With one model configured, a failed load at start ends the server.

## What a swap keeps

| | Across a swap |
| --- | --- |
| Conversations on disk | Kept, in each model's own folder; they restore when the model returns. |
| Conversations in RAM (`conversation-cache-mib`) and in slots | Written to the disk tier at the engine's exit when `conversation-cache-disk-mib` is on, else lost and read again. |
| A sleeping engine | Gets its GPU state back to exit, without refilling its caches. Idle sleep and idle unload apply to whichever model is loaded. |
| `POST /v1/vram` | The machine's: it applies to the loaded model and, once loaded, to each model loaded later that can resize its tier; for one that cannot, a warning says so. |
| Finished requests, best speeds | Per model; `GET /metrics?model=NAME` shows that model. |
| Vision | Per model. A numbered sidecar runs only while its model is loaded. |
| Hardware readings | One sampler over the cards of every model. |

At start the first entry loads, unless `lazy`. `POST /v1/load` with `{"model": NAME}` swaps ahead of a request, taking its turn in
the queue.

## Progress

`/health` has `model` (the model loading, else the loaded one), `percent` and `swap` (`from`, `to`, `phase`, `elapsed_s`, `waiting`). `/metrics`
and `/v1/status` have `swap` and `models`. The stats page has a picker: it chooses whose numbers the page shows, and Load swaps to
it. The console prints the load bars of each swap, and:

```
[server] swap: Qwen -> DeepSeek, 2 waiting
[server] swap: DeepSeek ready in 41.0 s
```
