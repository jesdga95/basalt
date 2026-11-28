# Gemma 4

Google's Gemma 4 (Apache-2.0) instruction models: the 26B A4B (routed experts), the 31B and the 12B (dense). One adapter
runs every size; each runs with the same command as any model: `basalt serve PACK`. What it supports next to the others:
[models](index.md).

## Pack

Built from Google's BF16 checkpoint and, for drafting, its assistant drafter by
`tools/convert/convert.py --checkpoint DIR [--drafter DIR] [--experts-mxfp8 LAYERS] [--mlp-nvfp4]`
([model file](../model-file/gemma4.md)):

| Model | Checkpoint, drafter | Layers (window, full) | Width, MLP | Pack |
| --- | --- | --- | --- | --- |
| 26B A4B | `google/gemma-4-26B-A4B-it`, `-assistant` | 30 (25, 5) | 2,816, 2,112 + 128 experts of 704, top 8 | 17.2 GiB with vision (16.0 without) |
| 31B | `google/gemma-4-31B-it`, `-assistant` | 60 (50, 10) | 5,376, 21,504 | 24.3 GiB with `--mlp-nvfp4`, vision included |
| 12B | `google/gemma-4-12B-it`, `-assistant` | 48 (40, 8) | 3,840, 15,360 | 13.4 GiB, text only |

- Routed experts: NVFP4 rounded once from BF16, the router's per-expert scale folded into each expert's down scale. The layers
  `--experts-mxfp8` names hold MXFP8 instead (E4M3, a power-of-two scale per 32 values, the per-expert scale folded into the
  down weights): about twice the bytes of an NVFP4 layer, closer to the BF16 model.
- Attention and MLP projections: FP8 E4M3 with one FP32 scale per matrix; the MLP's inner width padded with zeros to a
  multiple of 128. With `--mlp-nvfp4` the MLP's three projections are NVFP4 planes (rounded once as the experts are): the
  31B in FP8 (29 GiB of projections) does not fit a 32 GB card beside its caches.
- Output head (the tied embedding): FP8 E4M3 rows with an FP32 scale per row. Token embedding BF16, kept in RAM.
- Router F32 with its input scale folded in; norms and layer scalars F32.
- The assistant drafter's projections and head FP8 the same way.
- Vision tower and its projection into the text model's width: BF16 and F32 as stored (`vision/mmproj`, 1.1 GiB). The 12B
  (Gemma 4 Unified) has no tower but a patch embedder straight into the text model, and an audio projection: neither is
  converted, so its pack reads text only.

## Architecture

The `gemma4` architecture (code in `src/models/gemma4`):

- **Layers:** sliding-window layers (a window of 1,024, every rotary pair turning) interleaved with full-attention layers
  (partial "proportional" rotary: the first quarter of the pairs turns, the rest pass through). Queries and keys are RMS-normed
  per head with weights, values without; attention is unscaled.
- **Caches:** a sliding-window layer keeps a ring of its window and a prompt chunk's rows (a chunk writes its keys
  straight to their cells; a checkpoint holds while the session has not read a chunk past it), never the whole context. A
  full layer keeps one cell per position. A full layer whose keys are its values (`attention_k_eq_v`) stores each position
  once, the normed projection: V is that cell, K is rotated from it times the key norm's weights as the attention reads
  it. `--kv fp8` stores both as unit-scale E4M3.
- **Feed-forward:** a dense GELU MLP on every layer; on a layer with experts it runs beside the routed experts (softmax router,
  top-k renormalised), each part post-normed, then the sum normed into the residual and scaled by the layer's scalar.
- **Head:** the final norm, the tied embedding, the logit softcap.
- **Drafter:** Google's assistant model: the anchor's embedding beside the model's last normed state, projected to its width,
  through its layers, whose queries read the caches of the model's last sliding and last full layer at the anchor's position;
  its head picks the draft and its state goes to the next step. `--spec` caps the window (default: the drafter's own draft
  count plus the anchor).
- **Vision:** a 27-layer ViT on 16 x 16 patches (column and row position tables, RMS norms around attention and a gated
  MLP, queries and keys RMS-normed per head with 2D rotary positions); each cell of 3 x 3 patches is pooled to one row,
  standardised and projected to the text model's width. A picture's rows stand in for its `<|image|>` tokens and are read as
  one prompt chunk: the window layers let them attend to each other both ways, the full layers keep them causal.

## Prompt and decode

- **Prompt:** a read longer than a decode window runs in chunks of up to 2,048 rows: the projections and the MLP as W8A8
  products on the FP8 tensor cores (activations rounded per row; an NVFP4 MLP W4A4 on the FP4 tensor cores), the router
  over the chunk in fp32, the experts on the FP4 tensor cores (two FP4 terms per activation, about 8 bits), an MXFP8
  layer's on the MXFP8 block-scaled tensor cores (activations rounded to E4M3 per 32 values), with the GELU between the
  two products, the attention flash style on the fp16 tensor cores. A full layer's keys are rotated from its values once
  a chunk into an fp16 buffer the attention reads in place.
- **Decode:** the projections and the head read their FP8 (or the MLP's NVFP4) weights once per window of up to 16 rows;
  the experts run grouped by expert; the attention splits each row's keys in chunks by its own position, each chunk on the
  fp16 tensor cores (the query as two fp16 terms, so the scores keep its fp32 precision), then merges them, so a row's
  result does not depend on the rows beside it (batch windows and drafting give plain decoding's tokens). Precision:
  [Gemma 4](../precision/gemma4.md).

## Memory

The projections and the head sit on the main card; the embedding stays in RAM. The 26B A4B's routed experts are placed by
the shared expert tier: every one the card holds beside the caches is a home there (all of them on a 32 GB card), the rest,
the least routed first by the shipped profile (`data/profiles/gemma4-30x128.prof`), are homes in page-locked RAM.

- **Decode:** the card computes its experts from a table of their slots; a layer whose rows route to a RAM expert hands
  those rows to the host through mapped memory, the CPU pool computes them while the card computes its own, and the card
  takes their rows back before the layer ends. RAM experts a session keeps routing to come back to the card as copies
  in the room beside the homes. With every expert on the card the windows hold no hand-off at all.
- **Prompt:** a chunk reads every expert on the card: each layer's RAM experts are copied to a staging on the card (two
  layers' buffers, one filling while the other is read), whose room the copies beside the homes give up for the prompt.

The sliding layers' rings are a fixed size; the full layers' cells grow with `--max-context`, up to the models' 262,144.
`--batch N` slots (up to 16) each take their own caches. What fits beside the weights (one slot, the drafter loaded, vision
on for the 26B A4B and the 31B, as the engine plans it at start):

| Model | Card | Weights on the card | fp16 caches | `--kv fp8` |
| --- | --- | --- | --- | --- |
| 26B A4B | 32 GB | 14.6 GiB, every expert | 262,144 (11.9 GiB left free) | 262,144 (12.9 GiB left free) |
| 26B A4B | 16 GB | 2.6 GiB and the experts that fit | 262,144 with 71 of each layer's 128 experts on the card (104 at 8,192) | 262,144 with 83 of 128 (108 at 8,192) |
| 31B | 32 GB | 20.6 GiB | about 123,600 | about 238,200 |
| 12B | 32 GB | 11.5 GiB | 262,144 (15.2 GiB left free) | 262,144 (16.4 GiB left free) |
| 12B | 16 GB | 11.5 GiB | about 73,700 | about 192,500 |

The 31B's weights do not fit a 16 GB card.

With `--vision dynamic` the image encoder keeps no card memory beside the caches: its kernels' few MiB come out of the
card's spare, and vision on or off give the same context.

## Chat

The dialect (`serve/models/gemma4.py`) renders the checkpoint's template: thinking on by default (`<|think|>` in the system
turn, the reply's `<|channel>thought ... <channel|>`), off with reasoning effort `none`. Tool calls are
`<|tool_call>call:NAME{key:value,...}<tool_call|>`, string values between `<|"|>` marks; the parser turns them into JSON
arguments.

## Images

`--vision dynamic` ([vision](../running.md#vision---vision)). The picture is resized (aspect kept, bicubic) to the largest
size of whole 48 x 48 cells within 280 cells, up or down, as the checkpoint's processor does; one token per cell, so a
picture takes up to 280 tokens. Transparent pixels are laid over white. The encoder's weights are staged in page-locked RAM
(1.1 GiB) and keep no card memory at rest: the tower's 27 layers stream through two slots of a region (0.35 GiB) that a
request's pictures borrow from the prompt path's buffers (idle while pictures are read: they are read before the chunk that
holds them), and the buffers come back when the pictures are read. On an RTX 5090 a request with one fresh picture gives
its first token in 70 to 120 ms (the region's copy, the encoder and the prompt).

## Speed

An RTX 5090 at 400 W, `--kv fp8`, drafting on, one slot, vision on for the 26B A4B and the 31B, the second of two runs of
three chat requests (thinking off: an explanation at temperature 0.6, a JSON list at 0 and an essay at 0.6) and a 16K-token
prompt:

| Model | Context | Explanation | JSON | Essay | Prompt read, 16K |
| --- | --- | ---: | ---: | ---: | ---: |
| 26B A4B | 262,144 | 438 tok/s | 705 | 380 | 21,882 tok/s |
| 31B | 229,376 | 165 tok/s | 263 | 125 | 5,526 tok/s |
| 12B | 262,144 | 259 tok/s | 485 | 223 | 10,922 tok/s |

An RTX 5060 Ti alone at 180 W with a Ryzen 9 9950X (`basalt-bench`, fp16 caches, a context sized to its fixtures): the
26B A4B with 24 of each layer's 128 experts in RAM decodes the three fixtures at 144, 218 and 128 tok/s and reads a 16K
prompt at 6,061 tok/s; the 12B at 76, 143 and 62 tok/s and 3,376 tok/s.

## Quality

`basalt-bench --model PACK --quality REF.safetensors` runs the [quality suite](../precision/quality-suite.md) against a
model's BF16 reference; the results: [precision](../precision/gemma4.md#against-the-reference).
