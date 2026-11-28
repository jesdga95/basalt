# DeepSeek-V4.1-Flash precision

Where DeepSeek rounds and how its output compares. Guide: [DeepSeek](../models/deepseek41.md).
Method: the [quality suite](quality-suite.md) (preferred). The KLD, PPL and top-1 figures on this page were measured with [wikitext KLD](quality.md#wikitext-kld).

Test setup: RTX 5090 at 400 W with an RTX 5060 Ti at 180 W, 121 GiB of RAM. Geometry: 5120 hidden, 40 layers,
384 routed experts (top-6) plus one shared, a 128-token window, a two-level top-512 indexer, and an Engram table on two layers.

## Summary

- **Placement decides rounding.** A GPU expert and a CPU expert differ by about 1% of a layer's routed output, so a different RAM
  tier size can change the text. A fixed prompt on a fixed memory budget repeats exactly.
- **Drafting does not change output.** With placement fixed, every row of a verify window equals plain decode's, bit for bit.
- **Against BF16:** the published pack scores KLD 0.2779 and top-1 83.88% ([table](#against-the-bf16-reference)).

## Where each stage rounds

| Stage | Format | Arithmetic |
|---|---|---|
| Routed experts (UD-IQ1_M) | IQ1_S, IQ2_XXS or IQ2_XS gate/up; IQ1_S, IQ2_XXS, IQ2_S or IQ2_XS down | q8_1 activations (GPU), Q8_K (CPU); fp32 scales |
| Routed experts (UD-Q2_K_XL) | IQ2_XXS, IQ2_XS, IQ2_S or IQ3_XXS gate/up; IQ2_XXS, IQ3_XXS, IQ3_S or MXFP4 down | q8_1 activations (GPU), Q8_K (CPU; MXFP4: Q8_0); fp32 scales |
| Attention and shared-expert projections | Q6_K, Q5_K or Q8_0 | q8_1 (decode); fp16 block-scaled int8 mma (prompt) |
| Engram table | IQ4_NL, Q5_K, Q6_K, MXFP4 or MXFP8, 256 values per row | dequantized to fp32 on the host |
| Compressor and indexer projections | BF16 | fp32 activations and accumulation |
| Norms, router, hyper-connections | F32 | fp32 |
| Window rows (the latest 128 latents) | E4M3, a power-of-two scale per 32 values (`--kv fp16`: fp16) | decoded exactly; fp32 scores, softmax and accumulation |
| Compressed rows | E2M1, an E4M3 scale per 16 values (`--kv fp16`: fp16) | decoded exactly; fp32 scores, softmax and accumulation |
| Indexer keys and queries | E2M1, a power-of-two scale per 32 values (`--kv fp16`: fp16 keys, fp32 queries) | fp32 scores |
| Output head | Q6_K | fp32 logits |

The NVFP4 pack differs in four rows: routed experts NVFP4 (E2M1 codes, an E4M3 scale per 16 values, an FP32 scale per matrix;
q8_1 activations on the GPU, Q8_0 on the CPU), attention and shared-expert projections Q8_0 of NVIDIA's FP8 values, the Engram
table MXFP4 (E2M1 codes, a power-of-two scale per 32 values), the output head Q8_0. It is not scored against BF16.

All activations between stages are fp32. A decode kernel's output does not depend on how many tokens share its launch (bitwise).

With the default `--kv fp8`, the attention caches round as the model's reference does (`act_quant`, `fp4_act_quant`; the code is Project Maya's, by way
of Strata) and keep the codes and scales: 920 bytes per position instead of 3,200 in fp16. A code times its scale is the
rounded value exactly, so the caches hold what fp16 copies of the rounded values would, except values below fp16's range.
Logits equal a build that keeps those rounded values in fp16 caches, bit for bit (greedy, pinned placement, two cards and one,
200 tokens of prose and of structured output, a 16,384-token prompt). `--kv fp16` keeps the values unrounded.

## Experts and placement

- **Tiers round differently.** A GPU expert and the same expert on the CPU differ by about 1% of a layer's routed output (q8_1 and Q8_K
  round activations differently). So which experts are resident can change a token.
- **Repeatable.** Placement moves only at token boundaries and follows routing. A fixed prompt on a fixed memory budget repeats exactly.
  A different RAM tier size can change the text.
- **Clamped SwiGLU** (routed and shared): gate capped, up clamped to ±limit, `silu(gate) * up` in fp32.

## Prompt path

Prompt chunks run MMQ products (q8_1, int8 mma, fp32 accumulation). Against decode they change numerics:

| Part | Change |
|---|---|
| Routed experts in prompt chunks | a different int8 reduction order |
| Shared expert | about 1e-3 per layer |
| Attention projections | about 2e-3 relative RMS per layer |
| Compressed rows | the same packed rows as decode (bitwise on random inputs); selected ids identical |
| Sparse attention | fp16 tensor cores over the decoded rows (exact in fp16), 2.1e-4 relative RMS against fp32; bitwise independent of token count |
| Router logits | 3.5e-4 relative RMS against llama.cpp's batched path |

With a second card, each layer's experts split between the cards, both on MMQ. An expert gives the same bits on either card for
the same launch, but MMQ's stream-k split depends on how many experts share a launch, so moving experts between launches moves the
rounding on either card. The last prompt row's KL, two cards, 400 W + 180 W, RAM tier pinned:

| Prompt | Split across the cards vs the main card's launches | Main card's launches halved vs whole |
|---|---:|---:|
| 600 tokens | 2.9e-2 | 7.4e-3 |
| 2,500 tokens | 4.6e-3 | 1.8e-2 |
| 15,400 tokens | 9.4e-3 | 1.4e-2 |

The halved launches' smaller buffers also moved four experts a layer into the main card's homes. Greedy text parts after 3 to 24
tokens either way. A fixed prompt on a fixed placement repeats exactly.

Against BF16 (100 chunks, the same placement: homes main card 34, second card 45, RAM 305), the split moves the score within
noise:

| Prompt experts | PPL | KLD mean | Top-1 |
|---|---:|---:|---:|
| Main card only | 3.0337 | 0.27693 | 83.93% |
| Split across the cards | 3.0378 | 0.27709 | 83.88% |

Paired by chunk, the KLD difference is +0.00016 (95% CI -0.00113 to +0.00145) and top-1 -0.05 points (-0.24 to +0.14).

### Prompt skip

Past layer 20, a row reads only its own 127-position window and layer 20's compressed rows. A row more than 2,668 positions from the
read's end (20 × 127 + 128) therefore stops at layer 20's compressor. `--no-prompt-skip` runs every row.

- The math is the same: an 8K read matches a full read bit for bit in the layer outputs, the last prompt token and 60 generated tokens.
- In a real read the rounding moves. The last prompt row against a full read: KL 2.4e-6, 1.7e-4 and 7.7e-5 at 8K, 32K and 64K. Another
  chunk size on a full read moves the same row by KL 3.2e-5, 8.4e-4 and 1.2e-3.
- State is saved or resumed only at a read's end.

## Verify windows

`--spec` runs up to 8 rows at once (the default is 6). With placement fixed (`--diag-tier-frozen --diag-tier-ram 304`), every row's logits equal plain decode's, bit for bit, in windows of 2, 3, 5 and 8 rows, for the drafter too. Sampled text at
temperature 1 matches plain decode. With tiers moving, rows round as under any placement: forced through the plain text, 15 to 25 prose
picks and 1 to 6 code picks differ over 800 tokens.

Where a promotion lands decides which experts the CPU computes later, so it moves the rounding too. Sending one to the second card
only when what it replaces there has gone unused 1,280 lookups longer than the main card's oldest (rather than any longer) scores
within noise against BF16 (100 chunks, homes main card 35, second card 45, RAM 304): KLD 0.27971 with the any-longer rule and 0.27944
with the margin, paired by chunk -0.00027 (95% CI -0.00199 to +0.00144), top-1 83.89% and 83.77% (-0.12 points, -0.32 to +0.07).

## Against llama.cpp

Basalt's kernels are its own ports of the same graph. Per-layer tensors on the same weights, relative RMS: router logits 1e-7 (decode),
hyper-connection chain 1e-7 (decode) to 5e-5 (prompt), `attn_out` 4e-3 to 1.1e-2 (llama.cpp's own prompt noise is the same size). Top-512
selection overlap 0.998 over a 6,250-token prompt. Greedy decoding reproduces llama.cpp's first 16 tokens on two of three fixtures.

## Against the BF16 reference

Reference: [quality](quality.md#the-bf16-reference). 25,500 rows each; the published packs on two cards (400 W + 180 W), UD-Q2_K_XL
on one:

| Pack | PPL | KLD mean | KLD median | KLD p99 | Top-1 |
|---|---:|---:|---:|---:|---:|
| UD-IQ1_M (the published pack) | 3.0456 | 0.2779 | 0.0566 | 2.86 | 83.88% |
| UD-IQ2_XXS (published, fp16 caches) | 3.1074 | 0.2408 | 0.0408 | 2.53 | 85.15% |
| GSQ-RCO IQ2_XXS (tested, not shipped) | 3.6353 | 0.4311 | 0.0894 | 4.39 | 79.97% |
| UD-Q2_K_XL (converted, not published) | 2.7356 | 0.1501 | 0.0201 | 1.76 | 88.42% |

The UD-Q2_K_XL row is one RTX PRO 6000 Blackwell Max-Q at 300 W with 121 GiB of RAM (homes main card 153, RAM 159, file 72).

The cache roundings cost nothing measurable: the same run with `--kv fp16` scores PPL 3.0387, KLD 0.2777, top-1 83.97%.

The Engram projection as one MMQ GEMM and the indexer scores on the tensor cores change the prompt's rounding, not its quality
(the same rows with and without: PPL 3.0520 and 3.0492, KLD 0.2800 and 0.2791, top-1 83.81% and 83.87%).

Prompt skip cannot be scored this way (512-token chunks). On a 64K prompt, the 200 tokens after it match `--no-prompt-skip` closely
(KL mean 6.2e-4 between the two runs).

## Lossy routing flags

What the flags do: [quality](quality.md#lossy-routing-flags). Here k is 6, so `--route-tail-skip` takes R 1 to 5 (bare 5), and
`--route-prior` adds LAMBDA to `sqrt(softplus(logit)) + bias`, which runs about 0.5 to 3 (bare 0.1).

Two cards (400 W + 180 W), the default placement (RAM tier 308 of 384 experts per layer). Speed: `basalt-bench -t conv,struct,prose
--seed 1`, one cold run each (off twice, before and after the rest); ms a window is the step time, which drafting luck does not move.
CPU and bus picks: the run's `TIER final` line. Quality: the gate at 100 chunks against the BF16 base.

| Flags | CPU / bus picks | conv tok/s, ms/w | struct tok/s, ms/w | prose tok/s, ms/w | PPL | KLD | Same top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 5.0% / 2.6% | 93.7, 20.80 | 245.5, 23.32 | 97.7, 17.10 | 3.0482 | 0.2803 | 83.84% |
| `--route-prior 0.1` | 2.8% / 1.1% | 112.2, 17.67 | 274.8, 21.24 | 111.9, 15.33 | 3.0446 | 0.2824 | 83.87% |
| `--route-tail-skip 5` | 4.0% / 1.9% | 110.6, 19.04 | 260.9, 22.67 | 108.5, 15.64 | 3.0457 | 0.2799 | 83.86% |
| `--route-tail-skip 3` | 2.5% / 0.8% | 132.5, 16.09 | 283.6, 20.80 | 133.5, 14.09 | 3.0421 | 0.2865 | 83.70% |
| both (0.1, 5) | 2.7% / 1.1% | 128.3, 18.33 | 274.1, 21.22 | 112.8, 14.77 | 3.0446 | 0.2808 | 83.99% |

On two cards only 7.6% of picks miss both GPUs, so each flag has less to win and less to cost than on one card: `--route-prior 0.1`
cuts conv's window by 15% for +0.002 KLD, `--route-tail-skip 5` (31% of the misses dropped) by 8% within the gate's noise, and
`--route-tail-skip 3` (77% dropped) by 23% for +0.006. A KLD step under about 0.002 is placement noise: routing follows the tiers,
which follow the run's history.

`--route-prior` on one card: UD-IQ1_M on a 32 GB card with a 123 GiB RAM tier
at 475 W, 100 chunks against the BF16 base (2.7314 PPL there), `--seed 1`, two runs per row:

| `--route-prior` | Picks in VRAM | PPL | KLD | Same top-1 | tg conv | tg struct | tg prose |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 69.8-70.9% | 3.0507 | 0.2800 | 83.77% | 66 | 108 | 70 |
| 0.1 | 81.0-81.2% | 3.0595 | 0.2847 | 83.68% | 82 | 216 | 83 |
| 0.25 | 86.3% | 3.1515 | 0.3082 | 83.24% | 93 | 284 | 96 |
| 0.5 | 91.2% | 3.4108 | 0.3787 | 81.03% | 125 | 307 | 121 |

0.1 adds 0.0047 KLD (1.7% of the pack's own error) for a quarter more speed on conv and double on struct; 0.25 costs six times
the error for little more. Prefill does not move: the batched prompt path takes no flag. The gain grows with the window's width,
since a window of five rows asks the CPU for five rows of every expert it computes. The 20-chunk quick check overstates the low
boosts by about half (its text is easier: base PPL 2.2717 there against 2.7314 over all 100).

Reproduce (the base is [basalt-kld-bases](https://huggingface.co/datasets/jesdga/basalt-kld-bases)):

```sh
basalt-bench --model DeepSeek-V4.1-Flash-1017-UD-IQ1_M.basalt --gpu 0,1 --kld 100 --kld-base dsv41-bf16-base.kld -- --route-prior 0.1
basalt-bench --model DeepSeek-V4.1-Flash-1017-UD-IQ1_M.basalt --gpu 0,1 -t conv,struct,prose --seed 1 -- --route-prior 0.1
```
