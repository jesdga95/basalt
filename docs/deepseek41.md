# DeepSeek-V4.1 packs

A `deepseek41` pack runs through the same command, `basalt serve PACK`; the server reads the model's architecture from
the pack and picks the chat format, tool-call syntax and thinking switch to match.

- **First start:** the engine sizes a VRAM expert tier from the free memory, then loads as many experts as fit into a
  pinned RAM tier; the rest are read from the pack by the CPU. The load runs once per start and the startup lines
  `basalt deepseek41: VRAM expert tier ...` and `basalt deepseek41: RAM expert tier ...` give the sizes. Which experts are resident
  can change the last bits of the output ([precision](precision/deepseek41.md)).
- **Context:** `--max-context N` (or `"max-context"` in the config's `engine` section) sets it, as for every model, up
  to the model's own maximum (1,048,576 tokens, from the pack). Without it the engine takes that maximum, lowered
  until the compressed caches of a full context take at most a quarter of the free VRAM, and prints the result
  (`basalt deepseek41: context N tokens`); the server's `context N tokens` line, its Monitor and its request checks use that
  number. A value above the model's maximum, or whose caches need more than half the free VRAM, is refused at start.
  The caches grow with the position: a short chat maps a few MiB and a 1M-token context about 3.2 GiB. They
  grow into the memory the tier leaves free, then into the tier's spare slots, and a prompt that does not fit is
  refused; a prompt that fills the context leaves the tier smaller for the rest of that run. The rotary extension (YaRN) is part of the model: the `--rope-*` and
  `--yarn-*` flags are refused for it.
- **Long prompts:** each prompt takes the tier's loan back, maps the caches it needs and sizes its workspaces to the
  memory left: 8192 tokens per chunk, halved down to 256. A read of 4,096 tokens or more gives the whole VRAM tier
  back for its duration and reads every expert from RAM and the pack; the tier reopens empty, so the first tokens after
  it decode slower. After a long context, a tail of up to 256 tokens that finds no room for workspaces reads one decode
  step at a time; a longer one borrows the tier too. If even the smallest chunk does not fit, or the caches cannot grow,
  the request fails with an error naming the free memory and the next one runs. When a new conversation starts, caches a
  long one left mapped beyond 32,768 positions are released to the tier.
- **Requests:** one at a time, with no `--parallel` slots. The engine keeps the tokens it holds (the prompt and what it
  generated) and a checkpoint before the last prompt token. A request whose prompt starts with all the held tokens, or
  with the checkpoint's (a next turn after a thinking reply: the template drops the thinking from the history), reads
  only the new tail; any other request reads its whole prompt again. `pp=N/Mc` in the done line shows how many were kept.
- **Second GPU:** not used by this model outside `--spec`.
- **Speculation:** off by default. A pack built with `--dspark` carries the DSpark drafter; `--spec N` (2 to 6) loads it
  onto the second GPU (about 8.5 GB) and drafts N - 1 tokens per window, verified in one pass. It applies to greedy
  requests, keeps the output identical to plain decoding up to the numerics of a multi-row window
  ([precision](precision/deepseek41.md)), and pays only where the second GPU's draft overlaps the verify; on one
  card it is a loss, so leave it off there. `--spec` on a pack without the section is refused at start.
- **Images:** `--vision dynamic` encodes inside the engine from the pack's `vision/mmproj` section, which exists when the
  pack was built with `--mmproj` ([model file](model-file.md#deepseek-v41-pack)). The sidecar modes do not apply.
