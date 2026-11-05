# DeepSeek-V4.1 packs

A `deepseek41` pack runs through the same command, `basalt serve PACK`; the server reads the model's architecture from
the pack and picks the chat format, tool-call syntax and thinking switch to match.

- **First start:** the engine deals the experts out in routing order and loads them once per start. One card: as many
  as fit go into a pinned RAM tier, the most routed first, the VRAM slots hold copies of its head, and the rest are read
  from the pack by the CPU, the most routed of them read ahead into the page cache after the load. Two cards
  (`--gpu 0,1`): the second card holds experts nowhere else has; when RAM cannot take all the others, the main card
  holds the most routed of them too, nowhere else, as many as leave room for a prompt at the full context, and its other
  slots copy the RAM tier. When everything then has a place, nothing is read from the pack after READY but the Engram
  table and the token embedding. The startup line `basalt deepseek41: expert homes of 384 per layer: ...` gives the split.
  While the model runs, both cards' slots follow the routing: an expert that keeps missing takes the slot used longest ago,
  on either card, and a home that loses its slot goes to RAM, into the place the new expert leaves.
  Which experts are resident can change the rounding of the output ([precision](precision/deepseek41.md)).
- **Context:** `--max-context N` (or `"max-context"` in the config's `engine` section) sets it, as for every model, up
  to the model's own maximum (1,048,576 tokens, from the pack); without it the context is 4,096 tokens. A value above the
  model's maximum, or whose caches need more than half the free VRAM, is refused at start.
  The caches grow with the position: a short chat maps a few MiB and a 1M-token context about 3.2 GiB. They
  grow into the memory the tier leaves free, then into the tier's spare slots, and a prompt that does not fit is
  refused; a prompt that fills the context leaves the tier smaller for the rest of that run. The rotary extension (YaRN) is part of the model: the `--rope-*` and
  `--yarn-*` flags are refused for it.
- **Long prompts:** each prompt takes the tier's loan back, maps the caches it needs and sizes its workspaces to the
  memory left: 16,384 tokens per chunk, halved down to 256. A read of 4,096 tokens or more (on two cards, more than 2,048) gives the main card's copies
  back for its duration and streams the experts from the main card's homes, RAM, the second card (through the host) and
  the pack; the copies reopen after it, from the routing order on two cards, empty on one, where the first tokens after
  it decode slower. On two cards a prompt of up to 2,048 tokens reads beside the copies and keeps them, in chunks of 1,024 and workspaces that stay for the next one, and up to 12 tokens read one decode step at a time. A chunk under 8,334 tokens stages only the experts it routes; in it the second card computes its own experts of up to 16 rows and the CPU the RAM experts of up to 8 rows, beside the bus copies of the rest. A longer chunk stages every expert ahead of its routing. After a long
  context, a tail of up to 256 tokens that finds no room for workspaces reads one decode step at a time; a longer one
  borrows the copies too. If even the smallest chunk does not fit, or the caches cannot grow, the request fails with an
  error naming the free memory and the next one runs. When a new conversation starts, caches a long one left mapped
  beyond 32,768 positions are released to the tier.
- **Prompt skip:** in a long read, rows further than 2,668 positions from the read's end (the end of the system prompt, the
  last prompt token) stop at layer 20's compressor, which is all a later token reads of them. From the last prompt token on
  the math is a full read's; a real run differs from a full read by the rounding of another batch shape, as another chunk size
  does ([precision](precision/deepseek41.md#prompt-skip)). State is saved or resumed only at a read's end, which every
  checkpoint and conversation image is. `--no-prompt-skip` runs every row through every layer.
- **KV format:** the caches are fp16, the only format this model has; `--kv` is the normal `unknown argument` error.
- **VRAM limits:** `--expert-cache` caps the VRAM slots and `--vram-reserve-mib` or `POST /v1/vram` sets what stays free, through
  prompts, images and the tier reopening after them. A prompt that wants more memory takes the copies first and halves its chunk
  to fit above the reserve; the caches map only above it. If the smallest chunk or the caches do not fit there, the request
  fails with an error naming the memory and the next one runs: the reserve is not spent. `POST /v1/vram` resizes the copies at
  once; homes stay.
- **Requests:** one at a time, or `--parallel N` slots with `--slot-full-context` as for any model. The engine keeps the tokens it holds (the prompt and what it
  generated) and a checkpoint before the last prompt token. A request whose prompt starts with all the held tokens, or
  with the checkpoint's (a next turn after a thinking reply: the template drops the thinking from the history), reads
  only the new tail; any other request reads its whole prompt again. `pp=N/Mc` in the done line shows how many were kept.
  A cancel stops a long read at the end of its chunk.
- **Second GPU:** `--gpu 0,1` makes the second card an expert tier: it computes its experts with the main card's kernels,
  so an expert gives the same bits on either card, while the CPU pool computes the misses; the rows come back through
  mapped host memory. Without it the first card runs alone and a warning names the other. One second card is used.
- **Counters:** the `done` line and the stats page show the same `hit=` and `pcie=` as any model: `hit` is the main card's
  share of what it and the CPU served, `pcie` the share a bus copy or the second card served.
- **Images:** `--vision dynamic` encodes inside the engine from the pack's `vision/mmproj` section, which exists when the
  pack was built with `--mmproj` ([model file](model-file.md#deepseek-v41-pack)). The sidecar modes do not apply. It holds no
  memory at rest: for the pictures of a request the encoder takes the memory of the extra copies, of all copies on one
  card when that is short, and gives it back before the prompt reads.

## Drafting

Off by default. A pack built with `--dspark` carries DeepSeek's DSpark drafter in its `dspark/gguf` section
([model file](model-file.md#deepseek-v41-pack)), and `--spec N` (`"spec"` in the config's `engine` section) runs it. A window
holds the last accepted token and up to N-1 drafts, as on any model; a block proposes 5 tokens, so at most 6. Without `--spec`,
or with `--spec 0` or `1`, the section stays unread. `--spec` on a pack without the section decodes plain with a warning, and so
do `--parallel` slots, which draft nothing. One engine line says what started: `basalt deepseek41: drafter on GPU 0: 7.47 GiB,
windows of at most 6 tokens`.

The drafts stay in a window while the drafter's own chance that all of them are accepted holds against a threshold adapted per
request. The model verifies every row and a rejected row leaves no trace, so the output is plain decode's within the
rounding of where experts sit ([precision](precision/deepseek41.md#verify-windows)).

It runs on the main card and borrows the model's token embedding and output matrix. Its memory is taken before the tiers size
themselves and stays through `SLEEP` and `POST /v1/vram`. What moves first is the second card's or the main card's homes; when
RAM cannot take them, the last experts of the routing order stay in the file.

### What it costs

Measured with a 400 W RTX 5090 and a 180 W RTX 5060 Ti, 121 GiB of RAM, `--pool-workers 8`, the Q8_0 drafter on the main card.
It takes 7.47 GiB there, and the start line goes from `main card 34, second card 46, RAM 304, file 0` to `main card 10, second
card 46, RAM 303, file 25`. A block takes 2.24 ms (8.63 ms with `--diag-drafter-gpu 1`).

A long conversation, 125-141K tokens of context, thinking and tools on, temperature 1, 600 tokens per turn, `--spec 8`, the turns
after the first, one cold server run each:

| | tok/s | Tokens generated | Tokens per window | Prompt reads |
| --- | --- | --- | --- | --- |
| Plain | 64.06 | 4,239 | 1 | 30.2 s |
| Drafter | 51.62 | 9,015 | 1.97 | 39.9 s |

It loses here. With the drafter in memory a window took 17.9, 29.0, 38.3, 49.1, 63.0 and 81.9 ms at 1 to 6 rows, and the draft 2.5
ms more, against 15.6 ms for a plain token, so a window of 3 rows has to keep 2.6 tokens to break even. The first draft of a window
was accepted 65% of the time.

Greedy, every window full, 800 tokens, short context:

| `--spec` | Prose tok/s | Tokens per window | Code tok/s | Tokens per window |
| --- | --- | --- | --- | --- |
| Plain | 72.9 | 1 | 75.1 | 1 |
| 3 | 55.1 | 1.95 | 87.8 | 2.76 |
| 5 | 46.3 | 2.18 | 88.4 | 4.28 |
| 8 (6 rows) | 40.3 | 2.17 | 64.1 | 3.83 |

**When it can pay off:** on text it predicts well, as the code above, and with memory to spare. Most of what a window costs beyond
its rows is the memory. The same 378 windows of 5 rows replayed, 800 tokens: 28.6 ms each with no drafter in memory, 44.5 with it
on the main card, 34.6 with it there and RAM raised by `--diag-tier-ram` so that no expert stays in the file. No machine with more
VRAM has been measured.
