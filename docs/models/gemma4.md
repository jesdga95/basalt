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

## Architecture

The `gemma4` architecture (code in `src/models/gemma4`):

- **Layers:** sliding-window layers (a window of 1,024, every rotary pair turning) interleaved with full-attention layers
  (partial "proportional" rotary: the first quarter of the pairs turns, the rest pass through). Queries and keys are RMS-normed
  per head with weights, values without; attention is unscaled.
- **Caches:** a sliding-window layer keeps a ring of its window and a window's rows (the positions a verify window can add
  before it is settled), never the whole context. A full layer keeps one cell per position. A full layer whose keys are its
  values (`attention_k_eq_v`) stores each position once, the normed projection: V is that cell, K is rotated from it times the
  key norm's weights as the attention reads it. `--kv fp8` stores both as unit-scale E4M3.
- **Feed-forward:** a dense GELU MLP on every layer; on a layer with experts it runs beside the routed experts (softmax router,
  top-k renormalised), each part post-normed, then the sum normed into the residual and scaled by the layer's scalar.
- **Head:** the final norm, the tied embedding, the logit softcap.
- **Drafter:** Google's assistant model: the anchor's embedding beside the model's last normed state, projected to its width,
  through its layers, whose queries read the caches of the model's last sliding and last full layer at the anchor's position;
  its head picks the draft and its state goes to the next step. `--spec` caps the window (default: the drafter's own draft
  count plus the anchor).

## Memory

Every weight sits on one card; the embedding stays in RAM. The sliding layers' rings are a fixed size; the full layers' cells
grow with `--max-context`. `--batch N` slots each take their own caches.

## Chat

The dialect (`serve/models/gemma4.py`) renders the checkpoint's template: thinking on by default (`<|think|>` in the system
turn, the reply's `<|channel>thought ... <channel|>`), off with reasoning effort `none`. Tool calls are
`<|tool_call>call:NAME{key:value,...}<tool_call|>`, string values between `<|"|>` marks; the parser turns them into JSON
arguments.
