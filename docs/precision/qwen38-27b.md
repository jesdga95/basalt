# Qwen3.8-27B precision

What the dense stack computes in, where it rounds, and what that costs against the BF16 model. The method:
[how quality is measured](quality.md).

## Weights

The published pack, from QUASAR's quantization-aware checkpoint ([model file](../model-file.md#qwen38-27b-pack)):

| Tensors | Stored as | From QUASAR's checkpoint |
| --- | --- | --- |
| MLP gate, up, down; GDN qkv, z, out; attention q, k, v, o | NVFP4_PLANES (E2M1, UE4M3 per 16, FP32 per matrix) | bit for bit (the FP32 scale is 1 / weight_global_scale, rounded once) |
| Output head | F8_E4M3 rows (E4M3 codes, FP32 per row: amax / 448) | its BF16 values rounded once to nearest |
| GDN a, b | BF16 | its NVFP4 values rounded once to nearest (within 2^-9 relative) |
| GDN conv; norms | BF16, F32 | as stored (norms with the offset folded in, exact) |
| Token embedding | BF16, in page-locked RAM | as stored |
| MTP layer, block drafter | Q8_0 from BF16; the block drafter's projections rounded to NVFP4 at load (its conv stays Q8_0) | drafting only: cannot change output |

A pack converted from NVIDIA's checkpoint holds its MLP and output head as NVFP4_PLANES and its GDN and attention projections
as F8_E4M3 (E4M3 codes, FP32 per matrix), both bit for bit; the rest as above.

## Where each stage rounds

- **Products:** NVFP4 matrices: in decode windows each E2M1 code times its UE4M3 group scale is converted to bf16 exactly and
  multiplied by the window's bf16 rows on the bf16 tensor cores with fp32 accumulation; prompt chunks run them W4A4 on every
  chunk: each row rounded to FP4 (E2M1 with a UE4M3 scale per 16 values against a per-row FP32 scale, amax / (6 x 448)) and
  multiplied on the FP4 tensor cores, fp32 accumulation, the GDN and attention products writing BF16 rows for the next step.
  An NVFP4 matrix's FP32 scale is applied to its fp32 output. The FP8-row head runs in decode windows only (a prompt chunk
  gives no logits; its last token runs as a decode window): codes converted to bf16 exactly, multiplied by the window's bf16
  rows with fp32 accumulation, each output row times its row's FP32 scale. F8_E4M3 projections (NVIDIA's pack): in prompt
  chunks the activations are rounded to E4M3 once per row (scale amax / 448) and multiplied on FP8 tensor cores (W8A8, as
  NVIDIA's checkpoint specifies; its static input_scale is not used); in decode windows as the head, with the matrix's FP32
  scale at the end.
- **Decode windows:** a row's products are summed in K slices fixed by the matrix's shape and added in order, and the
  attention splits a row's cells by position alone, so a token's arithmetic is the same in a drafted window and in plain
  decoding: greedy output with drafting is bitwise plain greedy, alone or beside other slots. Tokens read through the
  prompt path round differently from decoded ones, so this holds where every request reads the same prompt the same way. A
  thinking budget stops the engine on its last token and the answer continues the engine's own state, so it holds there too.
  When the server cuts off a thinking loop, the tokens the engine wrote past the cut are read again through the prompt path,
  and that reply can differ in the last bits between runs.
- **GDN:** fp32 state and conv history. Decode windows replay accepted rows into the state; prompt chunks run the chunked delta
  rule with q/k in BF16 and the state in fp32 (shared with Flash-Next). The prompt's a/b projections (BF16 weights) take the
  normed rows as two bf16 terms on tensor cores (about fp32's precision), the gates applied in the same kernel.
- **Attention:** K and V stored fp16, or with `--kv int8` as int8 codes and an fp16 scale per 64 values (amax / 127, codes rounded
  to nearest against the stored scale); with `--kv-resident` the positions in RAM are read where they live, the same bytes in
  the same order (bitwise the output with every position on the card). With `--kv nvfp4` each 16 values take an E4M3 scale
  (amax / 6) and E2M1 codes rounded to nearest against it; K, and every query read against it, is first turned by the
  normalized 256-point Hadamard transform (orthonormal, so the dot products are unchanged before rounding), V is stored as it
  is. Decode reads them with fp32 dot products and softmax (int8: code x scale in fp32; nvfp4: code x scale exact in fp32,
  then fp16), in chunks of 64 cells folded in a fixed order. A prompt chunk on an nvfp4 cache first rounds the layer's
  positions to int8 by the int8 cache's rule into one model-wide copy, and reads that as below.
  Prompt chunks, fp16 cache: QK^T with fp16 q and k on tensor cores, fp32 accumulation; P V with fp16 p and v, each tile of
  32 cells summed in fp16 and added to the fp32 output. int8 cache: q rounded to int8 per 64 values (amax / 127), each
  64-value group's integer sum times both scales in fp32; P V in 8 bits (Infernix's prompt-pv8): per 64-value V group
  each probability times its cell's fp16 scale, rounded to u8 against the row's largest product in the 64-cell tile
  (255 = that maximum), times the stored V codes on the int8 tensor cores with exact int32 sums, scaled back into the
  fp32 output; long prompts split the cells and merge the splits in fp32. Softmax in fp32. Measured against serial
  decode (the prompt path vs the same prompt one token at a time, int8 KV, natural text, NVIDIA's pack): KL 0.059 at 16,023 tokens and
  0.016 at 64,023, the same next token (one row each, BF16 activations in both paths). The gate's int8-KV rows below read
  this path.
- **Residual and norms:** prompt chunks keep the residual stream in BF16 (each add in fp32, rounded once) and pass BF16
  activations between kernels (qkv, z, the conv output, q/k/v, the normed rows for GDN's a/b); norms and softmax in fp32.
  Decode windows keep the residual in BF16 by the same rule (every add in fp32, rounded once, in the window gemm's epilogue),
  and the products read BF16 rows that the producing kernels write.

## Quality against BF16

Reference: [quality](quality.md#the-bf16-reference). 100 chunks, 25,500 rows, one RTX 5090 at 400 W, the current build.

| Pack | PPL | KLD mean | KLD median | KLD p99 | Top-1 |
|---|---:|---:|---:|---:|---:|
| QUASAR, every projection NVFP4, the head FP8 rows (the published pack), `--kv fp16` | 6.8388 | 0.0262 | 0.0119 | 0.252 | 92.82% |
| The same, `--kv int8` | 6.8260 | 0.0259 | 0.0118 | 0.252 | 92.89% |
| The same, `--kv nvfp4` | 6.8600 | 0.0283 | 0.0134 | 0.267 | 92.69% |
| QUASAR with the head rounded to NVFP4 (`--head nvfp4`), `--kv fp16` | 6.8622 | 0.0312 | 0.0168 | 0.265 | 91.83% |
| The same, `--kv int8` | 6.8498 | 0.0308 | 0.0168 | 0.267 | 91.87% |
| NVIDIA's checkpoint: NVFP4 MLP and head, F8_E4M3 GDN and attention, `--kv fp16` | 6.7843 | 0.0239 | 0.0128 | 0.209 | 92.52% |
| The same, `--kv int8` | 6.7814 | 0.0240 | 0.0129 | 0.203 | 92.56% |
| Reference (a converter experiment, FP32 prompt rows): QUASAR with the head Q8_0, near its BF16 values, `--kv fp16` | 6.8335 | 0.0256 | 0.0111 | 0.257 | 93.13% |

The head carries most of QUASAR's gap: rounding its BF16 head to NVFP4 costs KLD 0.0058 over a Q8_0 head, and FP8 rows (an
FP32 scale per row) win back most of it at 0.55 GB more on the card. Against NVIDIA's pack, QUASAR's has the higher top-1 and
the lower median, NVIDIA's the lower mean and p99. NVIDIA's GDN and attention projections keep 8 bits where QUASAR's take 4;
QUASAR's weights take 2.4 GiB less on the card and decode faster ([guide](../models/qwen38-27b.md#speed)).

The base scores 6.7411 on the same rows. Each chunk's first half is read as one 257-row prompt and the scored rows are then
decoded against it, so every scored row sees a cache and recurrent state the prompt path wrote, W4A4 on the NVFP4 matrices at
any chunk length (as the checkpoints and Infernix do). Decode never uses it.

## Two cards

`gpu-mode peer` and `gpu-mode split` give the tokens of one card, greedy, at 65,536 context on conversation, struct, prose and a
16,672-token prompt (`tools/gpu_identity.py` against `--gpu 0`): the second card runs the main card's kernels on the same rows.

## Context past 262,144

`--rope-scaling yarn --rope-scale F` rotates every query and key by YaRN, as transformers' Qwen3.5 rotary and llama.cpp's
`rope_multi` do for the interleaved M-RoPE: each of the 32 rotated pairs keeps its trained frequency (pairs 0 to 14 at the
defaults, beta 32 and 1 over the trained 262,144), takes 1/F of it (pairs 22 to 31) or a blend between, whichever position
(t, h, w) it rotates by, and cos and sin are scaled by 0.1 ln F + 1. The frequencies are worked out once in double precision;
prompt chunks and decode windows rotate by the same table. Every position turns the scaled way, short ones too, which costs a
little there (the gate on 20 chunks, `--kv int8`; llama.cpp's YaRN on unsloth's Q4_K_M, same text, for scale):

| | PPL | KLD mean | Top-1 | llama.cpp PPL |
| --- | ---: | ---: | ---: | ---: |
| Off | 7.0321 | 0.0257 | 93.14% | 6.9154 |
| `--rope-scaling yarn --rope-scale 2` | 7.0708 | 0.0274 | 92.73% | 6.9339 |

Past the trained length, a passphrase hidden at 10%, 50% and 90% of a 380,541-token prompt (synthetic filler, `--max-context
400000 --kv int8`, YaRN x2) was found at every depth, and at 25% and 75% of 500,573 tokens (`--max-context 524288
--kv-resident 200000`).

## Sampling with drafts

A sampled request (temperature above 0, `top_k` 1 to 64) that gives no `seed` drafts by speculative sampling. The block
drafter draws each draft from its own distribution q, the softmax of its selector's scores over its 16 candidates at the
request's temperature, on a Philox stream of its own. The window keeps draft d with probability min(1, p(d) / q(d)), p the
target row's distribution after the sampler chain, and at the first draft it does not keep it draws the token from
max(p - q, 0), renormalized; a window whose drafts are all kept draws its last token from p. Every token then has p's
distribution exactly. `sampler_drawn_parity` checks it on each sampler implementation the engine runs: 180,000 draws a case
give a total-variation distance of 0.001 to 0.003 from p (the sampling noise) and keep the draft as often as the sum of
min(p, q).

Which tokens come out depends on the drafts, and so on the windows the draft policy picks from measured round times. A request
with a `seed` keeps the plain rule instead (the drafter's best candidates, kept where the row's own draw is the draft), so its
reply is the same at any window. Greedy requests, `top_k` 0 or above 64, lookup drafts and the MTP drafter also keep the
plain rule, which is exact too.

## Agreement with Hugging Face

Four short prompts (5, 15, 21 and 60 tokens; the last a chat-templated question), every position's logits through the decode
path against Hugging Face transformers' own Qwen3.5 modules run in fp32 on the CPU, with NVIDIA's pack's NVFP4 and FP8 weights
dequantized (`basalt --diag-serial-prompt --diag-logits`):

| Prompt | Positions | Top-1 agreement | Mean KL | Max KL |
| --- | --- | --- | --- | --- |
| 0 | 5 | 5 of 5 | 0.0001 | 0.0001 |
| 1 | 15 | 15 of 15 | < 0.0001 | 0.0001 |
| 2 | 21 | 21 of 21 | < 0.0001 | 0.0002 |
| 3 (chat) | 60 | 59 of 60 | 0.0010 | 0.013 |

The prompt path's last position has the same top-1 on all four, KL at most 0.001 on the plain text and 0.11 on the chat
prompt, where the prompt MLP reads its rows in FP4. For scale, the same reference run on the BF16
model against the NVFP4 weights: 91 of 101 top-1, mean KL 0.01 on plain text and 0.18 on the chat prompt: the checkpoint's
quantization, not the engine.
