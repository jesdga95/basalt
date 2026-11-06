# basalt-probe

`bin/basalt-probe` starts the engine (as `basalt-bench` does), measures the machine's rates and the cold speed on three fixtures, and
prints suggested values. It is manual: it changes and saves nothing.

```
basalt-probe --model PACK.basalt -- ENGINE FLAGS...
```

| Option | Meaning |
|---|---|
| `--model FILE` | the model file |
| `--engine BIN`, `--fixtures DIR`, `--logs DIR` | as for `basalt-bench` |
| `--gpu A,B` or `auto` | the cards, as for the server; default GPU 0 |
| `-- ARGS` | engine flags, as for `basalt-bench`; the report prints the flags it used. `--expert-profile-save FILE` is compared, never written |

## What it does

1. Starts the engine and reads the rates with `PROBE rates` (GPU copy and link rates, host read rate, CPU expert time per worker count).
2. Runs `conv`, `struct` and `prose` once each, greedy, 400 tokens. These are **cold** speeds: the first request after load counts.
3. Measures its suggestions (below).

The token budget is part of "cold": the tiers keep adapting, so a longer request reads faster.

## Suggested values

Printed only when a value beats the built-in default by more than the measured noise:

- **`--pool-workers N`**: the fewest workers within 2% of the fastest measured CPU expert time.
- **`--adapt-every`, `--adapt-swaps`, `--adapt-decay`**: tried from the start placement (`PROBE adapt`) with a prose-then-struct switch.
  `64 / 256 / 0.7` and `1 / 160` are tested against the engine's own, which runs twice to set the noise band.
- **`--prefill-cpu-share F`**: 8192-token chunks at share 0, 1 and the balance. Printed only when at least 8% faster.
- **`--expert-profile-save FILE`**: suggested when a saved profile is faster beyond the band.

## Engine command: `PROBE rates`

With `--serve`, `PROBE rates` on stdin between requests answers in about half a second with `PROBE begin rates v1`, lines
`PROBE <section> key=value` (sections `gpu`, `host`, `pack`, `link`, `gpu_mem`, `host_mem`, `cpu_expert`, `peer`, `copy`) and
`PROBE end ms=`. It changes no engine state. `PROBE set workers=K share=F` sets the pool's worker count and the prompt CPU share;
`PROBE adapt [every=N] [swaps=N] [decay=F] [profile=FILE]` resets the tiers and sets the adaptive settings for the next request.
