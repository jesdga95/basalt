# The adaptive expert tier

The GPUs hold only part of the model's experts. The adaptive tier follows what the conversation routes to and moves
the experts it uses most into VRAM, in rounds that run beside decode. These settings shape the rounds. The built-in
values are the fastest measured here; this page says what each one trades and what was measured.

## How a round works

The engine counts how often each expert is routed to. A round ranks the experts that are not in VRAM by their count
and swaps the most-routed ones in for the least-routed resident ones, when the gain is clear. On a server a round has
three steps, one per decode window: the leaving experts are copied to RAM, the incoming ones are copied into their
slots on a side stream, and the next window uses them (each layer waits only for its own copies). A new round starts
only once the last one has landed, so "every window" means one round about every three windows. After each round
every count is multiplied by the decay, so recent routing outweighs old.

On one card without a peer tier the round ranks every layer together, so a layer that routes to more experts takes
slots from the layers that route to fewer. With a peer card the main card takes its share first and the peer card then
takes the next most-routed experts the main card does not hold. A layer split ranks each stage's own layers.

Moving an expert changes which tier computes its rows (the CPU rounds differently from the GPU), so output with
adaptation on is not bit-identical from run to run; see [precision](../precision/index.md).

## Settings

| Setting | Default | What it does | What it trades |
| --- | --- | --- | --- |
| `adapt-every` | `1` on a server, with or without the peer tier; `4` with a layer split and outside `--serve` | A round every N decode windows. `0` keeps the experts as loaded. | Fewer, later rounds cost the link less, but a request runs on the old placement until the next round: a fresh engine waits N windows for its first round, and a topic change waits as long. |
| `adapt-swaps` | `96` (`0` = off) | Most experts one round swaps into the main card. A server round after one that ranked at least twice that many candidates swaps four times as many (a cold start or a topic change). | Bigger rounds catch up faster but move more bytes per round (about 2.6 MiB per NVFP4 expert, each way when the leaving expert has no RAM copy), and the copies share host memory with the CPU experts. The RAM copy keeps four times N free places (384 places, about 1 GiB, at the default), taken from the RAM that holds experts. |
| `peer-adapt-swaps` | the value of `adapt-swaps` | Most experts one round swaps into the peer card. | As above, on the peer's link; `0` freezes the peer tier, which costs code-like requests most. |
| `adapt-decay` | `0.85` on a server, with or without the peer tier; `0.7` with a layer split and outside `--serve` | The factor every count is multiplied by after each round. | Lower forgets sooner and follows a topic change faster; higher keeps a long-run placement but adapts late. The decay is per round, so with a larger `adapt-every` the same value remembers far longer in windows. |
| `expert-profile-save` | off | Saves the learned placement (the experts in VRAM, then the counts) on exit and every `expert-profile-save-every` minutes (default 10) between requests. A start that finds the file places it instead of the model's own profile. Needs the adaptive tier on. | A continuing workload starts on what it learned; a different one starts on the wrong experts and must swap them out first. |

The `done` line's `hit=` is the share of the main card's lookups served from its VRAM. Experts the peer card computes
and experts read over PCIe are not among those lookups, so a higher `hit` is not always a faster request: compare
`tg` rates.

## Measured here

An RTX 5090 at 400 W, alone or with an RTX 5060 Ti at 180 W as the peer tier, the NVFP4 pack, 8 CPU pool workers,
the same server settings in every arm. Each arm is one fresh server and one session of 31 greedy requests: a cold
prose request of 1200 tokens (the number that counts), 8 prose requests, 12 code requests, 4 prose requests and 6
short chat requests of 400 tokens each. Rates are the engine's decode rates (tok/s); "switch" is the first three code
requests after the prose. One run per arm: repeats of a cold prose request vary by about 1.2% on two cards and 0.3% on one
(4% with the smaller cache below).

Two cards (400 W + 180 W):

| Arm | cold | cold hit | prose | switch | code | back to prose | chat | whole session | hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| defaults | 285.5 | 95.9% | 288.3 | 312.1 | 330.3 | 246.6 | 289.9 | 297.8 | 94.0% |
| `adapt-every 64`, `adapt-swaps 256`, `adapt-decay 0.7` | 232.4 | 92.2% | 268.7 | 134.5 | 193.0 | 149.6 | 222.5 | 212.1 | 83.4% |
| the same, started on its saved profile | 278.1 | 96.6% | 274.8 | 134.5 | 197.8 | 144.6 | 210.7 | 214.9 | 83.5% |
| `adapt-every 1`, `adapt-swaps 160` | 287.1 | 95.8% | 283.4 | 310.2 | 326.8 | 256.3 | 282.8 | 295.5 | 93.9% |
| defaults, started on their saved profile | 289.2 | 96.2% | 286.8 | 311.7 | 333.5 | 251.1 | 286.5 | 298.4 | 94.0% |

The RTX 5090 alone (400 W):

| Arm | cold | cold hit | prose | switch | code | back to prose | chat | whole session | hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| defaults | 241.8 | 93.3% | 207.9 | 208.7 | 209.5 | 184.8 | 185.7 | 204.8 | 88.2% |
| `adapt-every 64`, `adapt-swaps 256`, `adapt-decay 0.7` | 160.0 | 83.0% | 181.1 | 96.8 | 136.4 | 103.8 | 134.1 | 143.7 | 74.5% |
| the same, started on its saved profile | 197.2 | 88.3% | 178.8 | 90.3 | 133.4 | 103.6 | 145.2 | 146.6 | 75.0% |
| `adapt-every 1`, `adapt-swaps 160` | 230.1 | 92.5% | 208.9 | 216.5 | 212.9 | 189.7 | 187.3 | 206.2 | 88.3% |
| defaults, started on their saved profile | 227.9 | 92.6% | 207.1 | 208.1 | 207.9 | 184.1 | 179.5 | 201.6 | 88.1% |

One knob at a time on two cards: `adapt-every 64` alone costs the most (cold 195.1, switch 120.2, session 183.7);
`adapt-swaps 256` alone (cold 290.7, session 290.7) and `adapt-decay 0.7` alone (cold 287.3, session 296.0) stay
within noise of the defaults. A long cadence is what hurts: the first round of a fresh engine comes after N windows,
and every topic change runs on the old placement for as long. Larger and fewer rounds do not pay for that; the
default's round about every three windows keeps the main card's link at about 7-8 GB/s of expert copies during
decode, and it is still the fastest arm. A saved profile helps the long-cadence arm's first request (it starts on
learned experts) but not the rest of its session, and it slowed the defaults' first prose request on one card,
because that profile had learned the session's last workload (chat), not prose.

With a smaller cache (the RTX 5090 alone with `expert-cache 4300`, half its usual slots, as on a smaller card), `adapt-swaps` 48, 96 and
160 gave cold 130.3, 127.1 and 132.4 (two default runs), 132.3 tok/s: the swap count moves nothing beyond run
noise there either.

## Other machines

Not measured: two alike cards (in the expert tier or a layer split), a main card with 16 GB or less (far fewer slots,
so more misses and bigger backlogs), slower or narrower links (every round takes longer to land), and
a RAM size where the free places push experts out of the RAM copy. What the measurements above suggest for those, as
untested guidance:

- Keep `adapt-every` small on a server. A long cadence starves the cold start and every topic change, and a saved
  profile only fixes the start of a workload that matches the one it learned.
- A slower link favours fewer swaps per round over fewer rounds: lower `adapt-swaps` before raising `adapt-every`.
- With little RAM, `adapt-swaps` sets the free places (four times N experts): lowering it gives that RAM back to the
  experts.
- Try a change on your own mix with fresh servers and compare the `done` lines' `tg` rates, the first request first.
