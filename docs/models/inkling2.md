# Inkling 2 Mini

Thinking Machines Lab's Inkling 2 Mini (Apache-2.0). Runs with the same command as any model: `basalt serve PACK`. What it
supports next to the others: [models](index.md). Rounding and quality: [precision](../precision/inkling2.md).

## Pack

The default download is `Inkling-2-Mini-NVFP4-MX8.basalt` (19.6 GB, needs 0.5.6 or later, SHA-256
`28ec70723feaa037942a02f55b9d5e898276f3e62a3b1fa083745f932a041b99`), built from the official release with its vision encoder
and DSpark drafter:

- Routed experts, layers 0 to 19: the release's NVFP4 codes and scales, bit for bit (14.7 GiB for all 24 layers).
- Routed experts, layers 20 to 23: the release's MXFP8 codes and UE8M0 scales, bit for bit (`GGML_TYPE_MXFP8 = 46`).
- Attention, gate, short-convolution and output projections: Q8_0 of the BF16 values.
- Output head: BF16 in the file; the engine rounds it to Q8_0 as it loads. Token embedding BF16. Norms, router, biases and the
  loop's small tensors F32; loop controllers BF16.
- Vision encoder and projector (BF16, 0.6 GB) and the DSpark drafter (BF16, 0.6 GB), whole.

`Inkling-2-Mini-NVFP4-DQ8.basalt` (19.7 GB), the 0.5.5 pack, stores layers 20 to 23 as Q8_0 of the release's MXFP8 values,
everything else as above. The MX8 pack reads a prompt about 5% faster on the card, with the release's native bits on
block-scaled MXFP8 tensor cores, at level quality ([precision](../precision/inkling2.md#against-the-reference)). Both load.

Build it from the release directory: `tools/convert/convert.py --checkpoint DIR` reads both safetensors files, the quant config,
the tokenizer and the template, and writes the MX8 form.

## Architecture

The `inkling2` architecture (code in `src/models/inkling2`): 24 physical layers, 2,304 wide, no rotary positions anywhere in
the text model.

- **Loop:** a prelude (layers 0 to 5), a core (6 to 17) run three times, and a coda (18 to 23): 48 layer visits a token, each
  with its own cache and short-convolution state. Passes 2 and 3 start again from the prelude's output and mix in messages from
  the pass before through learned controllers.
- **Attention residuals:** a layer's input is a softmax mix over the finished blocks of 6 layers and the current one, not a
  plain residual stream.
- **Attention:** 24 query heads, 8 key/value heads of 128. Layers 5, 11, 17 and 23 attend every position (8 visits a token);
  the other 20 attend a window of 1,024. Keys and values pass a causal short convolution; a learned relative bias over the last
  1,024 positions, a sink per head and an output gate.
- **MoE:** 128 routed experts per layer, top 8 by sigmoid, 1,152 wide, with a softcapped SwiGLU; no shared expert.
- **Vision:** a 23-layer ViT reads a picture at its own size, padded to multiples of 40 pixels; one token per 40 x 40 cell.

## Memory

- **One card.** The main card holds the weights: 1.10 GiB dense, 14.71 GiB experts, the drafter 0.48 GiB. The token
  embedding (0.86 GiB) stays in page-locked RAM and the card gathers its rows. Two cards: [peer or split](#two-cards).
- **Context:** `--max-context` from 1,024 to 1,048,576, default 8,192.
- **Per position:** the 8 full-attention visits keep every position, 32 KiB with `--kv fp16`, 18 KiB with `--kv int8`, 16 KiB
  with `--kv fp8`, plus the drafter's full layer (fp16 in every mode); the window layers' rings do not grow.
- **Experts in RAM when the context needs the room:** when the asked context does not fit beside every expert, the least
  routed experts (by the shipped profile `data/profiles/inkling2-24x128.prof`) are planned into page-locked RAM, just enough,
  and decode computes them on the CPU. After routing, a visit with RAM experts publishes its choices and the input rows
  those experts need to mapped memory, then rings the CPU coordinator. The CPU pool (`--pool-workers`) runs those pairs
  while the card runs the rest. The card takes the CPU's rows only when the completion ticket matches the visit. Visits
  with no RAM pairs skip the hand-off, and the watchdog can release a stalled wait. The
  full visits' caches and the drafter's full layer then back their first 32,768 positions and grow as prompts need them, and
  the experts planned into RAM are copied back to the card while the caches leave room. A conversation that starts over, or
  resumes needing at most half the positions the caches back (a compacted one, or another on a shared prefix), trims them
  to what its prompt needs and copies experts back into the memory freed; a conversation's own turns never trim. A short chat at
  `--max-context 1048576 --kv fp8` keeps every expert on the card. On a smaller card the spill is static: the caches are whole
  from the start and the RAM experts are copied back while the card has room. Either way the copies leave the card's spare
  free: 1 GiB at the start and 512 MiB after a cache grow, or `--vram-reserve-mib N` for both (at least 128, 256 where the
  driver pages GPU memory); with the flag they also leave 256 MiB for what loads after them (kernels at their first launch,
  the windows' graphs). The flag is a load-time promise, not just a runtime floor: the load sizes the sessions and the
  spill so the card holds the asked reserve after loading — the fit check refuses a `--max-context` that cannot leave it
  room, and a load that still leaves less than the asked reserve is refused with an error naming the knobs (lower
  `--max-context` or `--spec`, or use a bigger card) — so a reserve that cannot be met is never silently served at a
  smaller one. The spare is where the copies stop, not a floor: a cache grow with no copy left to send back, or
  another program, can take it. Placement counts the card's free memory itself: it reads it when the copies are first
  placed, at a VRAM resize and at a wake, and counts every map and release of the copies and the caches after that (what
  loads later comes out of the spare and the 256 MiB). A grow or a copy that cannot map what was counted (another
  program took the memory, or graphs and modules needed more than the spare) reads the card again, sends copies back for it
  and goes on; a grow that does not fit even then refuses its request, as when the context does not fit. The log says
  `placement changed by a memory shortage at` the position (or the start, wake or resize) where it happened. The copies
  lie back to back in one address range; a demote packs the survivors together so whole pages go back. A window whose
  graph finds no device memory runs direct, and windows of its shape keep running without a graph (no copy moves); a
  batched window's graph has no such fallback. A VRAM resize the copies cannot meet is refused before any copy moves. A
  copy table update, a copy's upload, or a map or cache backing that fails for any reason but a clean lack of memory (a
  CUDA failure) stops the engine: every later request is refused. With one session, a refused decode window ends its
  request with an error and the engine serves the next. Nothing changes when everything fits.
- **Which experts get a copy:** the profile's order until the session has looked experts up, then the lookups of the rows its
  windows keep (rejected drafts do not count), decayed over 320 kept rows, discounted by the visits of the expert's layer: a
  visit waits for its RAM experts only while the CPU's share outlasts the card's, so a looped layer's counts are divided by
  its three visits to the power 0.75. The least looked up give way first when the caches
  grow. With one session and an elastic spill, at a request's start and every 256 decode positions after it, up to 64 copies
  move to clearly hotter RAM experts of the same size.
- **Prompt loan:** with an elastic spill, one session and one card, the prompt path's device memory that decode leaves idle (the
  chunk buffers' arena and the experts' prompt staging) holds copies of the next highest-ranked RAM experts, chosen after the
  turn's first decode window. Their uploads run beside the next windows, a bounded part after each, and the copies are used
  from 32 positions after that first window (that window waits for an upload still running), except those a copy of the
  card's own reached first. A prompt read through the staging takes the memory
  back first; a read through the decode path leaves the loan in place. Loaned copies use the card's arithmetic
  ([precision](../precision/inkling2.md)).
- **Same tokens on every run:** with one session, placement follows the token stream and the start state unless the log
  shows `placement changed by a memory shortage`: it only changes at positions the requests fix (a request's start, the
  rebalances, the loan, a cache grow) and by counted memory, and a decode window never crosses such a position, so the
  same greedy requests give the same tokens run to run whatever the windows' widths. The log's `copy set` names the copied
  experts after each change. A different free VRAM at start, a VRAM command, a sleep, a conversation restored from the
  parked store or several sessions decoding together can give a different placement.
- **Sleep and lending:** `--idle-sleep` and `POST /v1/vram` ([running](../running.md#sleep---idle-sleep)) use the same copies.
  At a sleep the card's copies of the RAM experts go back to RAM (the CPU computes them) and the driver moves the rest
  of the GPU state to host memory; the wake takes the copies back while the card has room. `reserve_mib` resizes them the
  same way: copies go back when the card has less than that free, return when it has more. Only the copies are lendable:
  the experts a pack keeps on the card have no RAM copy, so a reserve above what the copies free is refused (`400`), and an
  engine with every expert on the card gives nothing back before the driver moves it. No `--vram-elastic` (Flash-Next's
  segmented cache) is needed or accepted. At `--max-context 1048576 --kv fp8` on an RTX 5090 with images on the card holds
  28.4 GB awake and 20 MiB asleep; the wake takes about 2 s and every token after it equals the one before.

## Two cards

`gpu` names two cards and `gpu-mode` what they do ([GPUs](../configuration/gpus.md)): `"gpu": "0,1", "gpu-mode": "peer"` or
`"split"`. The model runs both on two cards and refuses `tensor` and a third card. With `gpu` unset and two cards visible the start warns once that GPU 1 is unused.
At default settings one card is the fastest setup up to 1M tokens of context: the experts that do not fit are computed on the CPU
and the elastic copies bring them back while the caches leave room, so most runs need neither mode. Use a mode when a card is too
small or held back (a desktop, another process).

- **Peer:** the main card runs every layer, and the hottest experts that spilled from it live in the second card's VRAM (no RAM
  copy). In decode the peer computes those experts' rows beside the CPU pool and the card merges them as before; in a prompt the
  rows with an expert on the peer go to the peer and its partial sums come back. The elastic copies also bring a peer's experts to
  the main card while it has room, and a layer all of whose experts are copied runs on the main card alone. A peer that would hold
  nothing (the main card holds every expert) is refused with `use gpu 0`.
  - Speed (RTX 5090 at 400 W, RTX 5060 Ti at 180 W, `--max-context 262144 --kv fp8`, ABBA x1, one card with the CPU experts against
    the peer, with `--expert-cache` forcing the spill):

    | | 1,572 spilled (`--expert-cache 1500`) | 672 spilled (`--expert-cache 2400`) |
    | --- | --- | --- |
    | Prompt read, 16K | 6,234 against 7,445 tok/s | 9,855 against 10,393 |
    | Prompt read, 64K | 5,670 against 6,560 | 8,514 against 8,488 |
    | Conversation | 116.5 against 290.5 | 244 against 345.5 |
    | Struct | 112.5 against 404 | 246.5 against 597.5 |
    | Prose | 128.5 against 254.5 | 242 against 297.5 |

    At 1M tokens, where the elastic copies bring all 1,938 spilled experts back, one card reads 8,992 tok/s and decodes 365, 650 and
    313 (conversation, struct, prose) and the peer 8,895 and 357, 645 and 311: the peer buys nothing there.
  - Quality: with the experts on the peer the BF16 KLD is level with the CPU's (`--expert-cache 1500`: 0.00440 and 97.25% top-1
    against 0.00444 and 97.15% on one card, 100 chunks). A peer's prompt differs from a resident card's at summation-order rounding
    (about 0.01 KLD on `--diag-gpu-check`, the same as reordering any expert sum); its decode rows are exact, and with every spilled
    expert copied back it is bitwise one card.
- **Split:** the second card runs the layers from the cut on, with their experts, caches and rings; the first runs the rest, the
  embedding gather, the head and the drafter. Every expert stays on a card: the cut divides each layer's experts, dense rows and
  caches at the asked context by the shares (free VRAM by default), and a part that does not fit its card is refused with its
  numbers. Nothing is priced against the main card alone.
  - **Crossings:** the rows a card reads that the other wrote go through page-locked RAM (no peer access needed): 2 crossings a
    window at a loop edge, 6 with the cut inside the loop. A window, both cards' launches and its crossings, is one graph; prompt
    chunks cross the same way.
  - **Speed:** about 25% slower than one card where one card holds or copies back its experts (768K tokens: prompt 6,985 against
    9,083 tok/s, conversation 276 against 356); it does not fit at 1M. The split's value is a card too small to hold the model.
  - **Exact:** the crossings change no bit: the same tokens and logit rows as one card (a diagnostic with both halves on one card,
    drafting on or off). A second card of another model can round differently, so greedy tokens may part: layers 12 to 23 on an
    RTX 5060 Ti score 0.00304 KLD and 97.70 % top-1 over 100 chunks, level with one card ([precision](../precision/inkling2.md)).
- `--diag-gpu-cut L` (diagnostic) forces the cut at layer L.
- **Pictures and sleep:** a picture borrows its room from the first card's experts; a sleep moves both cards' state.

Decode and a 16K prompt (`basalt-bench`, seeded, the RTX 5090 held to 16 GB by another process, 400 W + 180 W):

| | RTX 5060 Ti alone, ~1,100 experts in RAM | RTX 5090 at 16 GB + RTX 5060 Ti, split | split at 12 (`--diag-gpu-cut 12`) |
| --- | --- | --- | --- |
| Conversation | 59.8 tok/s | 246 | 176 |
| Struct | 109 | 430 | 323 |
| Prose | 49.1 | 209 | 142 |
| Prompt read, 16K | 2,770 tok/s | 9,140 | 6,180 |

From the two cuts, two RTX 5060 Ti (each layer on a 5060 Ti, 2 crossings) would decode conversation at about 120 tok/s,
struct 245 and prose 95, and read a 16K prompt at about 4,100 tok/s: an estimate, not a run. An RTX 5090 next to an
RTX 5060 Ti runs the 5090 alone (355, 642 and 306 tok/s); a split forced at 12 there decodes at the middle column's speed.

## Long prompts

Prompts of more than 16 tokens read in chunks of up to 4,096 rows on the tensor cores: the NVFP4 experts on the FP4 block-scaled
tensor cores (each activation row as two FP4 terms, [precision](../precision/inkling2.md#where-each-stage-rounds)), the other
products through MMQ. With experts in RAM the cut is 64 tokens: a shorter read, such as a turn's new tokens, runs as decode
windows, because a chunk's fixed cost there is more than eight windows. The drafter sees every chunk. With experts in RAM a
chunk stages each spilled layer's experts on the card, each from its card copy while the spill still holds one and from RAM
where the caches' growth sent it back; a chunk of at most 1,024 rows (a tool result, a short turn) stages only the experts its
rows route to; a longer chunk copies a layer's experts on a second stream once the spilled visit before it has read the
staging, under the kernels in between. A read of more than 4,096 tokens with experts in RAM runs chunks of up to 8,192 rows,
half the stagings, when the card's copies can make room for the wider buffers (about 1 GiB, and the drafter's 288 MiB tap):
the lowest-ranked copies go back to RAM for the read and the same copies return after it, so shorter reads and decode keep
the placement they had. At `--max-context 1048576 --kv fp8` on an RTX 5090, needles at 131K and 400K are found; decode runs
at 264 tok/s on a short chat, 173 at 131K and 108 at 400K (plain decoding), and a 400K prompt reads at about 4,110 tok/s.
Needles at 10, 50 and 90 % of 31,943 and 127,958-token prompts are found through the server
(`--max-context 262144`). On an RTX 5090 at 400 W with every expert on the card, prompts of 2K, 16K, 32K and 128K tokens read at
about 14,100, 13,300, 12,200 and 8,900 tok/s.

## Drafting

On when the pack has the DSpark drafter. One block pass drafts up to seven tokens: three dense layers over the anchor and seven
mask rows, reading keys and values made from the target's visits 5, 23 and 41 and its final state, then a recurrent Markov
head over the target's head (screened through 2-bit copies of its rows). The draft policy picks each window by expected
tokens per millisecond (`include/basalt/spec/draft_policy.hpp`). `--spec N` caps the window (default 8: the last accepted token
plus the block); `--spec 1` turns it off. The target checks every draft: output is plain decoding's.
A window also takes a draft copied from an earlier repeat of at least 3 tokens of the context when the policy expects
more from it than from the block: a 2,405-token copy prompt decodes 13 % faster. The model takes no `--suffix-draft` flag, so the
threshold stays 3.

## Parallel slots

`--parallel N` starts N sessions, each with its own caches, rings and drafter context; the forward's scratch, the prompt path
and the drafter's tap are one for all of them. The busy slots decode together, one pass through the weights of up to 8 rows
([batched decoding](../design/batching.md)), the rows going to the drafts likeliest kept. A request alone in its slot gives the
same greedy tokens as with `--parallel 1`, and requests decoding together each give the tokens they give alone.

## Images

`--vision dynamic` ([vision](../running.md#vision---vision)) keeps no card memory, so the context that keeps every expert on
the card is the same as without it. A request's pictures borrow the encoder's region (up to 1.37 GiB) from the experts: the
card copies of RAM experts go back first, then whole layers' experts are copied to RAM and come back byte for byte at the
same addresses after the pictures (one layer, about 0.3 s a request on an RTX 5090 with every expert on the card). The
engine's own encoder reads a picture at its size, padded to multiples of 40 pixels, one token per 40 x 40 cell: a 640 x 488
picture takes 208 tokens. A picture of more than 10,609 cells
(about 4,120 x 4,120) is scaled down to fit (bicubic). Transparent pixels are laid over the reference processor's grey
checkerboard (8 x 8 squares of 204 and 153), as the model saw in training; the model tends to call such a background black.

## Chat, thinking and tools

- **Template:** the release's Jinja template, with `<|message_*|>` roles and content blocks.
- **Thinking:** the template writes an effort level into a system message: `reasoning_effort` `none` (0), `minimal` (0.1),
  `low` (0.2), `medium` (0.7), `high` (0.9, the template's default), `xhigh` and `max` (0.99), or a number from 1 to 100 as
  hundredths (at most 0.99). There is no hard switch: at `none` the model still writes a short thinking block (a few hundred
  tokens), so leave room in `max_tokens`. As a practical floor, use 1,024 tokens for short answers and 2,048 when evaluating
  answer quality; a 512-token cap can end with `length` before the answer is complete. If that happens, raise `max_tokens` or
  set an explicit `reasoning_budget_tokens` / `answer-room` value; a reserved answer room helps but cannot guarantee a complete answer.
- **Structured output:** under a JSON schema or `json_object` the model thinks first, as its effort asks, and the grammar
  holds only the answer block; the answer cannot end before its value is whole. At `none` it answers at once.
- **Tools:** the model writes `<|content_invoke_tool_json|>{"name": ..., "args": {...}}`; several calls per turn are supported.
  The server also reads the forms the model drifts into: a call whose header has no name, or has no header at all (the name
  is in its JSON), a call written after a block it left open or cut off before its end, and a call written as XML
  (`<invoke name="..."><parameter name="...">...</parameter></invoke>`, bare or inside `<tool_call><function_calls>`, with
  or without `string="true"`) when it names one of the request's tools. Tags quoted in code, or naming no tool of the
  request, stay text. A call cut off by `max_tokens` is not a call (the finish is `length`), but the arguments it streamed
  are closed into valid JSON.

## Limits

- Two cards: peer (the spilled experts on the second card) or split (never with experts in RAM), no third card. Experts in RAM run on the CPU,
  so a spilled card decodes slower than a full one: an RTX 5060 Ti alone at 180 W with a Ryzen 9 9950X and about 1,100 experts in RAM decodes conversation at 59 tok/s,
  struct at 118 and prose at 53, about 2.6 times as fast as reading them over the bus; prompts read as before (2,430 tok/s
  at 16K).
- The server's `decode expert cache hit rate` counts a lookup as a hit when its expert was on a GPU when it ran, the second
  card's included (a copy that moves later leaves its past lookups where they were); the engine's `experts: ... on the card, ... on GPU 1, ... on the CPU` line at the end of a session splits them.
- No video.

## Speed

`basalt-bench` (seeded, `-r 2`), RTX 5090 alone at 400 W, the pack above, drafting on:

| | Inkling 2 Mini | llama.cpp, IQ4_NL GGUF | Qwen3.8-27B (Basalt) |
| --- | ---: | ---: | ---: |
| Decode: conversation | 347 tok/s | 94.5 | 261 |
| Decode: struct | 628 tok/s | 92.5 | 768 |
| Decode: prose | 296 tok/s | 94.5 | 192 |
| Prompt read | 15,820 tok/s (16K) | 3,093 (`llama-bench` pp512) | 10,757 (16K) |

Drafts accepted: conversation 68.0 %, struct 97.5 %, prose 50.7 %. llama.cpp (29b52d398, `-fa 1`, full offload, no drafting)
ran the same token ids through `llama-server` with the same lengths and temperatures; `llama-bench` tg128 reads 96.3 tok/s.
The struct fixture runs to its 8,192-token cap on both engines.
