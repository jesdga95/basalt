# Inkling 2 Mini

Thinking Machines Lab's Inkling 2 Mini (Apache-2.0). Runs with the same command as any model: `basalt serve PACK`. What it
supports next to the others: [models](index.md). Rounding and quality: [precision](../precision/inkling2.md).

## Pack

One file, `Inkling-2-Mini-NVFP4-DQ8.basalt` (19.7 GB), built from the official release with its vision encoder and DSpark
drafter:

- Routed experts, layers 0 to 19: the release's NVFP4 codes and scales, bit for bit (14.7 GiB for all 24 layers).
- Routed experts, layers 20 to 23: the release's MXFP8 values as Q8_0.
- Attention, gate, short-convolution and output projections: Q8_0 of the BF16 values.
- Output head: BF16 in the file; the engine rounds it to Q8_0 as it loads. Token embedding BF16. Norms, router, biases and the
  loop's small tensors F32; loop controllers BF16.
- Vision encoder and projector (BF16, 0.6 GB) and the DSpark drafter (BF16, 0.6 GB), whole.

Build it from the release directory: `tools/convert/convert.py --checkpoint DIR` reads both safetensors files, the quant config,
the tokenizer and the template.

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
  embedding (0.86 GiB) stays in page-locked RAM and the card gathers its rows. No second card.
- **Context:** `--max-context` from 1,024 to 1,048,576, default 8,192.
- **Per position:** the 8 full-attention visits keep every position, 32 KiB with `--kv fp16`, 16 KiB with `--kv fp8`, plus the
  drafter's full layer; the window layers' rings do not grow.
- **Experts in RAM when the context needs the room:** when the asked context does not fit beside every expert, the least
  routed experts (by the shipped profile `data/profiles/inkling2-24x128.prof`) are planned into page-locked RAM, just enough,
  and decode computes them on the CPU: after a layer's router its rows go to RAM, the CPU pool (`--pool-workers`) runs the
  pairs whose expert is in RAM while the card runs the rest, and the card takes their rows before it combines them. The
  full visits' caches then back their first 32,768 positions and grow as prompts
  need them, and the experts planned into RAM are copied back to the card while the caches leave room, the least routed
  giving way first when they grow. A short chat at `--max-context 1048576 --kv fp8` keeps every expert on the card. On a
  smaller card the spill is static: what does not fit stays in RAM. Nothing changes when everything fits.
- **Sleep and lending:** `--idle-sleep` and `POST /v1/vram` ([running](../running.md#sleep---idle-sleep)) use the same copies.
  At a sleep the card's copies of the RAM experts go back to RAM (the CPU computes them) and the driver moves the rest
  of the GPU state to host memory; the wake takes the copies back while the card has room. `reserve_mib` resizes them the
  same way: copies go back when the card has less than that free, return when it has more. Only the copies are lendable:
  the experts a pack keeps on the card have no RAM copy, so a reserve above what the copies free is refused (`400`), and an
  engine with every expert on the card gives nothing back before the driver moves it. No `--vram-elastic` (Flash-Next's
  segmented cache) is needed or accepted. At `--max-context 1048576 --kv fp8` on an RTX 5090 with images on the card holds
  28.4 GB awake and 20 MiB asleep; the wake takes about 2 s and every token after it equals the one before.

## Long prompts

Prompts of more than 16 tokens read in chunks of up to 4,096 rows on the tensor cores: the NVFP4 experts on the FP4 block-scaled
tensor cores (each activation row as two FP4 terms, [precision](../precision/inkling2.md#where-each-stage-rounds)), the other
products through MMQ. The drafter sees every chunk. With experts in RAM a chunk stages each spilled layer's experts on the card,
from their card copies when it has them all. At `--max-context 1048576 --kv fp8` on an RTX 5090, needles at 131K and 400K are
found; decode runs at 264 tok/s on a short chat, 173 at 131K and 108 at 400K (plain decoding), and a 400K prompt reads at about
4,110 tok/s. Needles at 10, 50 and 90 % of 31,943 and 127,958-token prompts are found through the server
(`--max-context 262144`). On an RTX 5090 at 400 W with every expert on the card, prompts of 2K, 16K, 32K and 128K tokens read at
about 14,100, 13,300, 12,200 and 8,900 tok/s.

## Drafting

On when the pack has the DSpark drafter. One block pass drafts up to seven tokens: three dense layers over the anchor and seven
mask rows, reading keys and values made from the target's visits 5, 23 and 41 and its final state, then a recurrent Markov
head over the target's head (screened through 2-bit copies of its rows). The draft policy picks each window by expected
tokens per millisecond (`include/basalt/spec/draft_policy.hpp`). `--spec N` caps the window (default 8: the last accepted token
plus the block); `--spec 1` turns it off. The target checks every draft: output is plain decoding's.
A window also takes a draft copied from an earlier repeat of the context (`--suffix-draft`, default 3) when the policy expects
more from it than from the block: a 2,405-token copy prompt decodes 13 % faster.

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
  tokens), so leave room in `max_tokens`.
- **Tools:** the model writes `<|content_invoke_tool_json|>{"name": ..., "args": {...}}`; several calls per turn are supported.
  The server also reads the forms the model drifts into: a call whose header has no name, or has no header at all (the name
  is in its JSON), a call written after a block it left open or cut off before its end, and a call written as XML
  (`<invoke name="..."><parameter name="...">...</parameter></invoke>`, bare or inside `<tool_call><function_calls>`, with
  or without `string="true"`) when it names one of the request's tools. Tags quoted in code, or naming no tool of the
  request, stay text.

## Limits

- One card: no second card and no layer split. Experts in RAM run on the CPU, so a spilled card decodes slower than a full
  one: an RTX 5060 Ti alone at 180 W with a Ryzen 9 9950X and about 1,100 experts in RAM decodes conversation at 59 tok/s,
  struct at 118 and prose at 53, about 2.6 times as fast as reading them over the bus; prompts read as before (2,430 tok/s
  at 16K).
- The server's `decode expert cache hit rate` line reads 100 % on a spilled load; the engine's `experts: ... on the card`
  line at the end of a session is the real share.
- No video.

## Speed

`basalt-bench` (seeded, `-r 2`), RTX 5090 alone at 400 W, the pack above, drafting on:

| | Inkling 2 Mini | llama.cpp, IQ4_NL GGUF | Qwen3.8-27B (Basalt) |
| --- | ---: | ---: | ---: |
| Decode: conversation | 347 tok/s | 94.5 | 261 |
| Decode: struct | 628 tok/s | 92.5 | 768 |
| Decode: prose | 296 tok/s | 94.5 | 192 |
| Prompt read | 12,310 tok/s (16K) | 3,093 (`llama-bench` pp512) | 10,757 (16K) |

Drafts accepted: conversation 68.0 %, struct 97.5 %, prose 50.7 %. llama.cpp (29b52d398, `-fa 1`, full offload, no drafting)
ran the same token ids through `llama-server` with the same lengths and temperatures; `llama-bench` tg128 reads 96.3 tok/s.
The struct fixture runs to its 8,192-token cap on both engines.
