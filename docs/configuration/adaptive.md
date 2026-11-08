# Adaptive expert tier

Flash-Next only. The GPUs hold part of the experts. The adaptive tier counts what the conversation routes to and, in rounds beside decode, moves the most
used experts into VRAM. The defaults are the fastest measured here.

## How a round works

Routing counts rank the experts not in VRAM. A round swaps the most-routed ones in for the least-routed resident ones. A server round
takes three windows: copy out, copy in on a side stream, then use. Counts decay after each round. On the main card every layer ranks
together; on a peer card the main card takes its share first. Moving experts changes rounding, so output with adaptation on is not
bit-identical run to run.

## Settings

| Setting | Default | Trade-off |
| --- | --- | --- |
| `adapt-every` | 1 on a server, 4 with a layer split | Fewer, later rounds cost less link, but a topic change or a fresh engine waits N windows. |
| `adapt-swaps` | 96 | Bigger rounds catch up faster and move more bytes (about 2.6 MiB per NVFP4 expert each way). The RAM copy keeps four times N free places. |
| `peer-adapt-swaps` | `adapt-swaps` | Same, on the peer's link. `0` freezes the peer tier. |
| `adapt-decay` | 0.85 (0.7 with a layer split) | Lower follows topic changes sooner. Higher keeps a long-run placement. |
| `expert-profile-save` | off | Saves the placement; a continuing workload starts on it, a different one starts wrong. |

`hit=` in the `done` line counts only the main card's VRAM lookups, so a higher `hit` is not always faster. Compare `tg`.

## Measured

RTX 5090 at 400 W (an RTX 5060 Ti at 180 W for two-card arms), NVFP4 pack, one fresh server per arm, one session: a cold 1,200-token prose request, then prose, code, prose and
chat requests. Decode tok/s, one run per arm:

| Arm (two cards: 400 W + 180 W) | Cold | Whole session |
|---|---:|---:|
| defaults | 285.5 | 297.8 |
| `adapt-every 64`, `adapt-swaps 256`, `adapt-decay 0.7` | 232.4 | 212.1 |
| `adapt-every 1`, `adapt-swaps 160` | 287.1 | 295.5 |

| Arm (RTX 5090 alone, 400 W) | Cold | Whole session |
|---|---:|---:|
| defaults | 241.8 | 204.8 |
| `adapt-every 64`, `adapt-swaps 256`, `adapt-decay 0.7` | 160.0 | 143.7 |
| `adapt-every 1`, `adapt-swaps 160` | 230.1 | 206.2 |

A long cadence is what hurts: the first round of a fresh engine and every topic change wait for it. Single-knob tests on two cards:
`adapt-swaps 256` alone and `adapt-decay 0.7` alone stay within noise of the defaults.

[basalt-probe](../probe.md) tries these values on your machine.
