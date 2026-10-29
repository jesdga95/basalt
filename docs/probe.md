# basalt-probe

`bin/basalt-probe` starts the engine as a child process (`--serve`, like `basalt-bench`), asks it for the machine's
rates, measures cold decode speed on the three standard fixtures and prints a short report with suggested values. It is
manual: it prints and saves nothing. It is a separate program, not a subcommand of `basalt`.

```
basalt-probe --model PACK.basalt -- ENGINE FLAGS...
```

| option | meaning |
|---|---|
| `--model FILE` | the model file |
| `--engine BIN`, `--fixtures DIR`, `--logs DIR` | as for `basalt-bench` |
| `--gpu A,B,..` or `--gpu auto` | the cards, as for the server ([GPUs](configuration/gpus.md)); default GPU 0 alone |
| `-- ARGS` | engine flags, as for `basalt-bench` (`--kv`, `--spec`, `--expert-cache`, ...). Nothing is added except a `--max-context` that fits the fixtures when none is given: the report prints the flags that produced its numbers |

## What it does

1. Starts the engine and waits for READY.
2. Sends `PROBE rates` ([running](running.md)) and prints the rates as the first report line: each GPU's memory
   copy rate, the host-to-device link rate of the first GPU, and the host read rate at the engine's worker count.
3. Reads the tier placement: the primary GPU's expert slots from the engine's INFO line, the peer's from its log.
4. Runs `conv`, `struct` and `prose` once each, greedy, 400 tokens each, in that order. These are the **cold**
   speeds: the first request after load counts, nothing is discarded, and nothing is repeated.
5. Measures the values it suggests (below).

The token budget is part of what "cold" means: the engine keeps adapting its expert tiers as it decodes, so a longer
request reads faster. The speeds are at the engine's context. The fixtures are short prompts, so they are measured at
the fixture depth with the context allocated, not with the KV filled to it.

## Suggested values

The probe prints `suggested:` lines for values its own measurements say beat the engine's built-in defaults on this
machine, each as the flag and value with one short measured figure. Detail goes to stderr. Nothing is applied or saved;
a value equal to the default is not printed.

- **`--pool-workers N`**: the CPU expert time per expert (8 experts per call) is measured at 1, 2, 4, ... and every
  pool worker count (`PROBE rates`). The pick is the point of diminishing returns: the fewest workers within 2% of the
  fastest measured count. It is printed only when it is below the pool's count, as `(N us/expert, best M us)`: the
  pick's CPU time and the fastest count's. Decode with the pick ties the default; the host read rate alone saturates
  earlier and is not used.
- **`--prefill-cpu-share F`**: the engine prefills 8192-token chunks (`PROBE set share=F`, then a one-token request) at
  share 0 (all GPU) and 1 (all CPU); the balance is `t0 / (t0 + t1)`, rounded to 0.05. A chunk at the balance and at the
  default (ABBA) must then be at least 8% faster, because a live chunk carries about 5% of noise; otherwise nothing is
  printed. The line shows the measured gain, `(prefill +9%)`. The share changes which experts the CPU rounds, so it is fixed per machine, never timing-driven at run time.

### What it covers

The probe suggests a value only where a quick measurement on this machine can pick it and a measured run shows the
pick matters. The other settings with machine-dependent defaults, and why it leaves them:

| setting | covered | why | seconds added |
|---|---|---|---|
| `pool-workers` | yes | the CPU expert time per worker count, from `PROBE rates` | 0 (part of the rates) |
| `prefill-cpu-share` | yes | timed 8192-token chunks at share 0, 1, the balance and the default | the chunks, a few seconds |
| `adapt-every`, `adapt-swaps`, `adapt-decay` | no | the defaults were fastest cold on both card sets measured, and the swap counts tried on a small cache split both ways within run noise ([adaptive tier](configuration/adaptive.md)); nothing the probe measures moves the best value, and a cold comparison needs a fresh engine per value | 0 |
| `peer-adapt-swaps` | no | fewer peer swaps lost on every workload measured; it follows `adapt-swaps` | 0 |
| `vram-reserve-mib`, `peer-reserve-mib` | no | lower reserves add about 1% more slots, inside run noise; they guard against paging and the start says when one is short | 0 |
| `spec` | no | fixed at start (no per-request value), and spec 7 against 8 differs by 1-2%, inside one run's noise | 0 |
| `prefill`, `expert-cache` | no | sized at start from what the cards can hold and lend | 0 |
| `peer-exclusive`, `resident-budget-gib`, `pool-affinity` | no | picked at start from the RAM and the CPU layout (`[server] auto`) | 0 |
| `suffix-draft`, `mtp-max-t`, `mtp-window`, `short-read` | no | properties of the model and the workload, not the machine | 0 |
| `kv`, `max-context` | no | memory and precision choices, not speed | 0 |

The engine's `PROBE set workers=K share=F` sets the pool's active worker count and the share for the requests that
follow, between requests; the probe uses it and the engine is stopped afterwards.

## Report

```
basalt-probe
  5090 1541 GB/s | 5060 Ti 388 GB/s | PCIe 28.9 GB/s | RAM 58 GB/s
  experts 5090 9553 / 5060 Ti 5711 / RAM rest | KV int8 | ctx 262k
  decode cold  prose 194 conv 255 struct 223 tok/s (measured)
  suggested: --pool-workers 8 (52 us/expert, best 51 us)
  suggested: --prefill-cpu-share 0.45 (prefill +9%)
  config: --gpu 0,1 --kv int8 ...
  Measured on this machine. Hand-tuned settings can still beat these.
```

(The numbers above only show the layout.) Timing and engine progress go to stderr.
