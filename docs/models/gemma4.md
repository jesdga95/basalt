# Gemma 4

Google's Gemma 4 (Apache-2.0), first the 26B A4B instruction model. Runs with the same command as any model:
`basalt serve PACK`. What it supports next to the others: [models](index.md).

## Pack

Built from Google's BF16 checkpoint (`google/gemma-4-26b-a4b-it`) and, for drafting, its assistant drafter
(`google/gemma-4-26B-A4B-it-assistant`) by `tools/convert/convert.py --checkpoint DIR [--drafter DIR]`
([model file](../model-file.md#gemma-4-pack)):

- Routed experts: NVFP4 rounded once from BF16, the router's per-expert scale folded into each expert's down scale.
- Attention and MLP projections: FP8 E4M3 with one FP32 scale per matrix; the MLP's inner width padded with zeros to a
  multiple of 128.
- Output head (the tied embedding): FP8 E4M3 rows with an FP32 scale per row. Token embedding BF16, kept in RAM.
- Router F32 with its input scale folded in; norms and layer scalars F32.
- The assistant drafter's projections and head FP8 the same way.
- Vision tower and its projection into the text model's width: BF16 and F32 as stored (`vision/mmproj`, 1.1 GiB).

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
  products on the FP8 tensor cores (activations rounded per row), the router over the chunk in fp32, the experts on the FP4
  tensor cores (two FP4 terms per activation, about 8 bits) with the GELU between the two products, the attention flash
  style on the fp16 tensor cores. A full layer's keys are rotated from its values once a chunk into an fp16 buffer the
  attention reads in place.
- **Decode:** the projections and the head read their FP8 weights once per window of up to 16 rows; the experts run grouped
  by expert; the attention splits each row's keys in chunks by its own position, each chunk on the fp16 tensor cores (the
  query as two fp16 terms, so the scores keep its fp32 precision), then merges them, so a row's result does not depend on
  the rows beside it (batch windows and drafting give plain decoding's tokens). Precision: [Gemma 4](../precision/gemma4.md).

## Memory

Every weight sits on one card; the embedding stays in RAM. The sliding layers' rings are a fixed size; the full layers' cells
grow with `--max-context`. `--batch N` slots each take their own caches.
With `--vision dynamic` the image encoder's region (1.28 GiB) stays free beside the caches, so the context is that much
smaller.

## Chat

The dialect (`serve/models/gemma4.py`) renders the checkpoint's template: thinking on by default (`<|think|>` in the system
turn, the reply's `<|channel>thought ... <channel|>`), off with reasoning effort `none`. Tool calls are
`<|tool_call>call:NAME{key:value,...}<tool_call|>`, string values between `<|"|>` marks; the parser turns them into JSON
arguments.

## Images

`--vision dynamic` ([vision](../running.md#vision---vision)). The picture is resized (aspect kept, bicubic) to the largest
size of whole 48 x 48 cells within 280 cells, up or down, as the checkpoint's processor does; one token per cell, so a
picture takes up to 280 tokens. Transparent pixels are laid over white. The encoder's weights are staged in page-locked RAM
(1.1 GiB) and keep no card memory at rest: a request's pictures take a region of the card (1.28 GiB) from the room the model
keeps free, and it goes back when they are read. On an RTX 5090 a request with one fresh picture gives its first token in 70
to 120 ms (the region's copy, the encoder and the prompt).
