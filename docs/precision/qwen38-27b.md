# Qwen3.8-27B precision

What the dense stack computes in, where it rounds, and what that costs against the BF16 model. The method:
[how quality is measured](quality.md).

## Weights

| Tensors | Stored as | From NVIDIA's checkpoint |
| --- | --- | --- |
| MLP gate, up, down; output head | NVFP4 (E2M1, UE4M3 per 16, FP32 per matrix) | bit for bit |
| GDN qkv, z, out; attention q, k, v, o | F8_E4M3 (E4M3 codes, FP32 per matrix) | bit for bit |
| GDN a, b; conv; norms | BF16, F32 | as stored (norms with the offset folded in, exact) |
| Token embedding | BF16, in page-locked RAM | as stored |
| MTP layer, block drafter | Q8_0 from BF16 | drafting only: cannot change output |

## Where each stage rounds

- **Products:** NVFP4 matrices (MLP, head): activations rounded to q8_1 (32 values, fp16 scale) and multiplied on int8 tensor
  cores (llama.cpp's MMQ) in decode windows and prompt chunks below 1,024 rows; from 1,024 rows the MLP runs W4A4 (below).
  An NVFP4 matrix's FP32 scale is applied to its fp32 output. F8_E4M3 projections: in prompt chunks the activations are
  rounded to E4M3 once per row (scale amax / 448) and multiplied on FP8 tensor cores (W8A8, as NVIDIA's checkpoint
  specifies; its static input_scale is not used); in decode windows the codes are converted to fp16 exactly and multiplied by
  fp16 activations (each row scaled by a power of two) with fp32 accumulation. Both apply the matrix's FP32 scale at the end.
- **Decode windows** run with one 8-row tile whatever their rows, so a token's arithmetic is the same in a drafted window and in
  plain decoding: greedy output with drafting is bitwise plain greedy, alone or beside other slots. Tokens read through the
  prompt path round differently from decoded ones, so this holds where every request reads the same prompt the same way. A
  thinking budget stops the engine on its last token and the answer continues the engine's own state, so it holds there too.
  When the server cuts off a thinking loop, the tokens the engine wrote past the cut are read again through the prompt path,
  and that reply can differ in the last bits between runs.
- **GDN:** fp32 state and conv history. Decode windows replay accepted rows into the state; prompt chunks run the chunked delta
  rule with q/k in BF16 and the state in fp32 (shared with Flash-Next). The prompt's a/b projections (BF16 weights) take the
  normed rows as two tf32 terms on tensor cores (about fp32's precision).
- **Attention:** K and V stored fp16, or with `--kv int8` as int8 codes and an fp16 scale per 64 values (amax / 127, codes rounded
  to nearest against the stored scale). Decode reads them with fp32 dot products and softmax (int8: code x scale in fp32).
  Prompt chunks: QK^T with fp16 q and k on tensor cores, fp32 accumulation (int8: q rounded to int8 per 64 values, amax / 127,
  and each 64-value group's integer sum times both scales in fp32); softmax in fp32; P V with fp16 p and v on tensor cores,
  each tile of 32 cells summed in fp16 and added to the fp32 output (int8: code x scale rounded to fp16).
- **Residual and norms:** fp32.

## Quality against BF16

Reference: [quality](quality.md#the-bf16-reference). 100 chunks, 25,500 rows, one RTX 5090 at 400 W, `--kv fp16`:

| Pack | PPL | KLD mean | KLD median | KLD p99 | Top-1 |
|---|---:|---:|---:|---:|---:|
| NVFP4 + F8_E4M3 projections (NVIDIA's checkpoint, bit for bit) | 6.7850 | 0.0214 | 0.0118 | 0.179 | 92.87% |
| The same, prompt MLP W4A4 on every chunk (`--diag-fp4-rows 1`) | 6.7815 | 0.0239 | 0.0128 | 0.204 | 92.55% |
| Earlier pack: projections Q8_0 of the FP8 values | 6.7889 | 0.0213 | 0.0118 | 0.178 | 92.93% |

The F8 projections hold NVIDIA's values exactly but read prompt rows in E4M3 (one scale per row) where the Q8_0 pack read
them in q8_1 (one scale per 32): perplexity 6.7889 -> 6.7850, KLD 0.0213 -> 0.0214, top-1 92.93 -> 92.87%.

The base scores 6.7411 on the same rows. Each chunk's first half is read as one 257-row prompt (the MMQ path) and the scored
rows are then decoded against it. The second row forces the FP4 prompt path (W4A4, normally from 1,024 rows) onto that read,
so every scored row sees a cache and recurrent state written by it: KLD +0.0025, top-1 -0.4 points. Decode never uses it.

## Agreement with Hugging Face

Four short prompts (5, 15, 21 and 60 tokens; the last a chat-templated question), every position's logits through the decode
path against Hugging Face transformers' own Qwen3.5 modules run in fp32 on the CPU, with the same NVFP4 and FP8 weights
dequantized (`basalt --diag-serial-prompt --diag-logits`):

| Prompt | Positions | Top-1 agreement | Mean KL | Max KL |
| --- | --- | --- | --- | --- |
| 0 | 5 | 5 of 5 | 0.0002 | 0.0003 |
| 1 | 15 | 15 of 15 | 0.0002 | 0.0006 |
| 2 | 21 | 21 of 21 | 0.0001 | 0.0008 |
| 3 (chat) | 60 | 60 of 60 | 0.0029 | 0.042 |

The prompt path's last position agrees too (same top-1, KL at most 0.002). For scale, the same reference run on the BF16
model against the NVFP4 weights: 91 of 101 top-1, mean KL 0.01 on plain text and 0.18 on the chat prompt: the checkpoint's
quantization, not the engine.

A 4,508-token prompt read by the prompt path against the same prompt read one token at a time: the same next token, KL 0.018
(0.001 before the MLP's prompt products moved to W4A4 from 1,024-row chunks).
