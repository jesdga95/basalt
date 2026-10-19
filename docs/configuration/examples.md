# Example configurations

Settings are described in [configuration](index.md). The model path is relative to the config file. Start the server with `bin/basalt serve --config basalt_config.json`.

## One GPU (32 GB) and RAM

The experts the card cannot hold stay in page-locked RAM.

```json
{
  "server": {
    "model": "model.basalt",
    "gpu": "0",
    "host": "127.0.0.1",
    "port": 8080
  },
  "engine": {
    "expert-cache": "auto",
    "prefill": "auto",
    "spec": 7,
    "spec-min-p": 0.5,
    "max-context": 262144,
    "kv": "int8",
    "resident-experts": true,
    "vram-reserve-mib": 700
  }
}
```

## A GPU and a second, smaller card as the peer expert tier

Listing two cards makes the second one the peer tier. `vision: peer` puts the image encoder there too, and
`peer-exclusive` keeps the peer's experts out of RAM.

```json
{
  "server": {
    "model": "model.basalt",
    "gpu": "0,1",
    "host": "127.0.0.1",
    "port": 8080,
    "vision": "peer"
  },
  "engine": {
    "expert-cache": "auto",
    "prefill": "auto",
    "spec": 7,
    "spec-min-p": 0.5,
    "max-context": 262144,
    "kv": "int8",
    "resident-experts": true,
    "peer-exclusive": true,
    "vram-reserve-mib": 700
  }
}
```

## A low-VRAM card

A smaller KV format and a shorter context leave more VRAM for experts; the rest stay in RAM.

```json
{
  "server": {
    "model": "model.basalt",
    "gpu": "0",
    "host": "127.0.0.1",
    "port": 8080
  },
  "engine": {
    "expert-cache": "auto",
    "prefill": "auto",
    "spec": 5,
    "spec-min-p": 0.5,
    "max-context": 65536,
    "kv": "q4_0",
    "resident-experts": true,
    "vram-reserve-mib": 1024
  }
}
```

Optional on any of them: `"relaxed-drafting": 0.5` in `engine` trades exactness under sampling for speed (see
[Speculative decoding](index.md#speculative-decoding-mtp)).
