# Diagnostics and measurement flags

Flags for measuring, debugging and testing the engine. They are not settings: each is off by default, and the
[configuration reference](index.md) lists the settings.

The binary column says who takes the flag:

- **CLI**: `bin/basalt` without `--serve`, the one-shot mode: `bin/basalt --model FILE.basalt --tokens IDS [flags]`
  reads the prompt, generates and exits.
- **serve**: `bin/basalt --serve`, the server's engine; give these in the config's `engine` block or on the server's
  command line like any engine flag.
- **engine**: both modes.
- **bench**: `bin/basalt-bench`. Every other `--flag` given to it is an engine flag, passed on.

## One-shot generation

| Flag | Binary | Value, default | What it does |
| --- | --- | --- | --- |
| `--tokens` | CLI | comma-separated token ids | The prompt, pre-tokenized. This or `--tokens-file` is required. |
| `--tokens-file` | CLI | path | The prompt from a file, ids separated by commas or whitespace. |
| `--max-new` | CLI | integer, 16 | Tokens to generate. |
| `--greedy` | CLI | switch, on | Argmax decoding. |
| `--seed` | CLI | integer | Sample instead, with this Philox seed, under `--top-k` (20), `--top-p` (0.95) and `--temperature` (1.0). |
| `--stop-eos` | CLI | switch, off | Stop at an end-of-turn token. `--serve` always stops there. |
| `--eos-ids` | engine | id list, `248044,248046` | The end-of-turn tokens; implies `--stop-eos`. The server expects the default. |
| `--gpu-plan` | engine | switch | Print the `--gpu` choice as one JSON line and exit; needs no model. The server calls it. |
| `--dump-hc8` | engine | path | Load, write the 8-bit hyper-connection copies (the converter's `hc8` section) to the file, exit. |

## Dumps

All CLI, all off.

| Flag | Value | What it writes |
| --- | --- | --- |
| `--dump-residual` | path | The residual at the end of the run (hc x n_embd, f32), for bisecting the head. |
| `--dump-mixed` | path | The post-attention residual the head reads, at the end of the run (n_embd, f32). |
| `--dump-routing` | path | The routed expert ids and weights per layer per position. Not with `--no-pool`. |

## Timing and profiles

| Flag | Binary | What it does |
| --- | --- | --- |
| `--stats` | CLI | Print the per-token wall time, its host phases and the expert pool's breakdown. |
| `--no-pool` | CLI | Skip the CPU expert pool: the GPU-only floor, without the experts the CPU computes. |
| `--no-capture` | CLI | Run the layers directly instead of replaying graphs. Needs `--no-pool`. |
| `--stage-timing` | CLI | Per-stage shares by kernel count, not a time profile (an uncaptured interval includes host gaps). Needs `--no-capture`. |
| `--graph-only` | CLI | Replay the per-layer `pre` graphs only and time them. Leaves out the `post` graphs and the head, so it is not the GPU floor. |
| `--gpu-only-full` | CLI | Replay every layer's graphs and the head with no pool: the per-token GPU floor. Not with a layer split. |
| `--gpu-stages` | CLI | Capture each layer as three graphs (mixer, FFN and router, post), time them from outside the capture, print and exit. Not with a layer split. |
| `--profile` | serve | GPU stage times of the decode windows, per request. Turns programmatic dependent launch off, so not for A/B timing. |
| `--wait-stats` | serve | The host's wait times in the decode loop, one report per 200 windows. |
| `--decode-timing` | serve | One line per request: where its decode time went. |
| `--draft-log` | serve | Path. One line per verify window (width, accepted, time), for studying the draft policy. |

## Speculation tests

| Flag | Binary | Value | What it does |
| --- | --- | --- | --- |
| `--spec-oracle` | CLI | path | With `--spec N` on a pack without a drafter: draft from a token file (the expected continuation), the exactness test of the verify path. Prompt lookup windows still take precedence. |
| `--spec-corrupt` | CLI | integer, 0 | Replace every Nth draft with a wrong token, to test rejection. |

## The n-gram table reads

The per-layer n-gram (PLE) table stays on disk and its rows are read as needed.

| Flag | Binary | Value, default | What it does |
| --- | --- | --- | --- |
| `--ple-io` | engine | `direct`, `ram`; `direct` | `direct`: unbuffered reads, the table never enters RAM or the file cache. `ram`: map the file with the whole table locked in RAM at start. |
| `--ple-row-cache` | engine | rows, 1048576 | A bounded cache of fetched rows, 90 B each; `0` turns it off. |
| `--ple-inflight` | engine | integer, 256 | Outstanding reads. |
| `--ple-delay-us` | engine | microseconds, 0 | Fault injection: each row read completes no earlier than this. |

## Quality gate

`basalt-bench --quality` runs it; [how quality is measured](../precision/quality.md) has the method.

| Flag or key | Binary | Value, default | What it does |
| --- | --- | --- | --- |
| `--quality` | bench | chunks, 20 | Run the PPL/KLD/top-1 gate through the decode path. Alone it runs only the gate; with `-p` or `-t` it runs after them. |
| `--kld-base` | bench | path | The llama.cpp KL-divergence base file. |
| `--quality-t` | bench | tokens, 3 | Tokens per verify window in the gate. |
| `--ppl-base` | serve | path | The base file that `ppl=` requests score against. |
| `--ppl-out` | serve | path | Where `ppl=` requests write their per-row lines. |
| `ppl=K` | request key | chunk index | Score chunk K of `--ppl-base`, the rest of the chunk teacher-forced through verify windows. |
| `ppl_t=T` | request key | tokens, 3 | The window width of a `ppl=` request. |

The request keys go on a `GEN` line of the engine's stdin protocol, before the token ids.

## basalt-bench

`basalt-bench --help` gives the full text and how the fixtures are made; [building](../building.md) has the community
preset (`--community`, `--submitter`, `--community-file`).

| Flag | Value, default | What it does |
| --- | --- | --- |
| `-t` | any of `conv,struct,prose` | Decode fixtures, always run in that order (draft state carries across requests). |
| `-p` | depths in tokens (`32k` style accepted) | Prefill depths. Without `-t` or `-p` the bench runs `-p 16384 -t conv,struct,prose`. |
| `-r` | integer, 1 | Repetitions of the whole set; above 1 the table shows mean and standard deviation, the first repetition cold. |
| `--seed` | integer >= 1 | Seed every decode request, so the sampled fixtures give the same text each run. |
| `--model`, `--gpu`, `--spec` | as for the engine | Passed to the engine; `--spec 0` is a plain-decode baseline. |
| `--set` | `"FLAG VALUE ..."` | Replace or add engine arguments; repeatable. Arguments after `--` do the same. |
| `--engine` | path | The engine binary; default the one built with the bench. |
| `--fixtures` | directory, `data/bench` | Where the pre-tokenized prompts are. |
| `-o` | `md`, `json`; `md` | A markdown table, or the full detail as JSON on stdout. |
| `--power-label` | watts | The power limit to print instead of the one read from the GPU. |
| `--logs` | directory | Keep the engine log, the draft log and the quality rows there. |
| `--no-windows` | switch | Do not ask the engine for the per-window log; no window statistics. |
| `--print-command` | switch | Print the engine command and exit. |
| `--hardware` | switch | Measure and print the CPU, RAM read speed and each GPU's PCIe link and copy speed, then exit; no model. |

`basalt-device`, a helper in the build directory, takes `--list-devices` (every GPU the runtime sees and whether this
build runs on it), `--link-rates` (one `pcie_h2d BUS_ID GB_PER_S` line per GPU, which `--hardware` reads) and
`--selftest`; with no flag it prints the main GPU and the memory plan.

## Experiment keys (`--exp`)

`--exp NAME=VALUE[,NAME=VALUE...]` (engine) sets experiment switches. They exist to test a change, carry no stability
promise and go away when the experiment ends. A key the engine does not read is ignored; `--exp` may be given more
than once. In a config file: `"engine": { "exp": "host_ev=1" }`.

| Key | Default | What it does |
| --- | --- | --- |
| `concur9_logit_hash` | 0 | `1`: print a hash of each accepted row's head logits on stderr (single-stream and batch slots). |
| `concur5_digest` | 0 | `1`: batch slots print a `BDIG` digest of their state on stdout after each window, and windows do not overlap. |
| `spec_log` | 0 | `1` with `--draft-log FILE`: write `FILE.tk`, each verified row's target distribution beside the drafter's top 4. |
| `host_ev` | 0 | `1`: record events at the decode loop's graph launches; each request then prints the GPU time of a window's verify, commit and draft and the idle gap before the next window (median and mean). |
