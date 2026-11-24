# Nemotron 3.5 Super precision

Where Nemotron 3.5 Super rounds and how its output compares. Method: [quality](quality.md). Guide:
[Nemotron 3.5 Super](../models/nemotron.md).

## Summary

- **The weights are the checkpoint's bits.** Every NVFP4 and FP8 matrix keeps its codes and scales; nothing is
  re-rounded by the converter except `A = -exp(A_log)`, computed once in float64.
- **Against a float64 reference of the same weights:** decode KL 3.8e-4 and top-1 54/54 on a 54-token chat; a prompt then
  decode KL 6.4e-4, top-1 28/28 ([below](#against-the-reference)).
- **The MTP head's drafts** match the reference's: the first at 53 of 53 positions, the second at 51 of 53.
- **No `basalt-bench --quality` base yet:** that needs a BF16 run of the full model.

## Weights

| Tensors | Stored as | From the checkpoint |
| --- | --- | --- |
| Routed experts (up, down) | NVFP4 (E2M1, a UE4M3 scale per 16, an FP32 scale per matrix) | bit for bit, regrouped into 64-value blocks |
| Shared experts | NVFP4 planes | bit for bit |
| Mamba in / out, attention q / k / v / o, output head | FP8 E4M3, an FP32 scale per matrix | bit for bit |
| Latent projections (down to 1,024, back up) | BF16 | as stored |
| MTP head: attention q / k / v / o, `eh_proj` | BF16 | as stored |
| Token embedding | BF16 | as stored |
| Norms, router, its bias, convolutions, `dt` bias, `D` | F32 | as stored |
| `A` | F32 | `-exp(A_log)` in float64, rounded once |

## Where each stage rounds

| Stage | Decode (windows of up to 16 rows) | Prompt (chunks) |
| --- | --- | --- |
| FP8 and BF16 products | BF16 activations, FP32 accumulate | Mamba in / out: FP8 activations (W8A8); the rest as decode |
| Shared expert | BF16 activations | FP4 activations (W4A4) |
| Routed experts, on the card | q8_1 activations (dp4a) | q8_1 activations (MMQ, int8 tensor cores) |
| Routed experts, on the CPU | Q8_0 activations | staged on the card, as above |
| Mamba-2 scan, norms, router, attention | FP32 (attention cache FP16) | the same kernels |

The routed experts' activations stay 8-bit on every path. Four-bit activations for them were measured and dropped: on the
FP4 tensor cores the squared ReLU's wide range (the down product's input) took the prompt's KL from 6.8e-4 to 3.4e-3.

On a small synthetic pack the prompt path's added error comes from the Mamba projections' FP8 activations (KL 1.7e-3)
and the shared expert's FP4 activations (3.7e-4); on the real model the whole prompt path adds about 2e-4.

## Against the reference

`tests/models/nemotronh/reference.py` runs the pack's own weights (dequantized) in float64 over a 54-token chat;
`nemotronh_model_parity` compares the engine's logits with it. RTX PRO 4000, 37 experts a layer on the card, the rest in RAM,
the placement frozen:

| Path | KL mean | KL max | Top-1 |
| --- | --- | --- | --- |
| Decode, one row at a time | 3.8e-4 | 5.6e-3 | 54/54 |
| A prompt of 26, then decode | 6.4e-4 | 2.4e-3 | 28/28 |
| Windows of 1 to 8 rows keeping some | bitwise one-row decode | | |
| State saved at 27, restored | bitwise | | |

The reference also runs the MTP head after each token: its two drafts and their probabilities. The engine's, one row at
a time from the start: the first draft the same at 53 of 53 (its probability off by 6.7e-3 on average), the second at 51
of 53. After the prompt of 26: 26 of 27 and 25 of 26. The misses are near ties.

With the tier moving experts between the card (q8_1) and the CPU (Q8_0), windows against one-row decode give KL 4.2e-4 and
the restored state 1.5e-4: the two activation roundings differ, so moved experts are not bitwise. The MTP
drafts there: 53 of 53 and 52 of 53, after the prompt 27 of 27 and 26 of 27.
