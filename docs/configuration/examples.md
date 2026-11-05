# Example configurations

Settings are described in [configuration](index.md). The model path is relative to the config file. Start the server with `bin/basalt serve --config basalt_config.json`.

## One GPU (32 GB) and RAM

The experts the card cannot hold stay in page-locked RAM.

```json
{
  "version": 2,
  "models": [{ "model": "model.basalt", "gpu": "0", "max-context": 262144, "kv": "int8" }],
  "engine": { "resident-experts": true, "vram-reserve-mib": 700 },
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

## A GPU and a second, smaller card as the peer expert tier

Listing two cards makes the second one the peer tier. `"vision": 1` runs the image encoder on that card too.

```json
{
  "version": 2,
  "models": [{ "model": "model.basalt", "gpu": "0,1", "vision": 1, "max-context": 262144, "kv": "int8" }],
  "engine": { "resident-experts": true, "vram-reserve-mib": 700 },
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

## A low-VRAM card

A smaller KV format and a shorter context leave more VRAM for experts; the rest stay in RAM.

```json
{
  "version": 2,
  "models": [{ "model": "model.basalt", "gpu": "0", "max-context": 65536, "kv": "q4_0" }],
  "engine": { "resident-experts": true, "vram-reserve-mib": 1024 },
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

Several models on one server: [models and the router](models.md).

Optional on any of them: `"relaxed-drafting": 0.5` in a model entry trades exactness under sampling for speed (see
[Speculative decoding](index.md#speculative-decoding-mtp)).
