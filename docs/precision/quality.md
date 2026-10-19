# How quality is measured

The precision tables are in [index.md](index.md).

## How quality is measured

`basalt-bench --quality` (`tools/ppl_gate.py`) scores wikitext-2 (`wiki.test.raw`) in 512-token chunks. The first half
of each chunk is prefilled and the second half (rows 256 to 510) is scored through the decode path, against a llama.cpp
Q8_0 base file written by `llama-perplexity --kl-divergence-base`, read in llama.cpp's own `.kld` format.

The harness (`include/basalt/program/ppl_gate.hpp` (the KLD function)) computes KL(base ‖ test) over the full vocabulary from the
fp32 log-softmax of the head logits; the base is decoded as scale*u16+min, and terms with base log-prob ≤ -16 are
skipped (a tiny underestimate). The prompt half is checked equal to the base tokens, the window tokens are the base
tokens, and KLD is flat across positions, so the two are aligned.

### Against llama.cpp

llama.cpp (commit 436f6f89e, CUDA) scores the same second half of each chunk, with the whole chunk in one batched
forward pass. 100 chunks, base PPL 3.286070, 400 W:

| Pack | Engine | PPL | KLD mean | KLD median | KLD p99 | Same top-1 |
|---|---|---|---|---|---|---|
| GSQ-RCO IQ3_S | llama.cpp | 3.3728 | 0.1526 | 0.0340 | 1.750 | 87.20% |
| GSQ-RCO IQ3_S | Basalt | 3.3723 | 0.15264 | – | – | 87.16% |
| GSQ-RCO IQ3_XXS | llama.cpp | 3.4675 | 0.2428 | 0.0596 | 2.607 | 84.05% |
| GSQ-RCO IQ3_XXS | Basalt | 3.4678 | 0.2433 | – | – | – |

At 20 chunks (Basalt: the README table; llama.cpp: its running value at chunk 20):

| Pack | Basalt KLD / top-1 | llama.cpp KLD / top-1 |
|---|---|---|
| GSQ-RCO IQ3_S | 0.177 / 89.4% | 0.1753 / 89.22% |
| GSQ-RCO IQ3_XXS | 0.277 / 86.8% | 0.2780 / 86.65% |
| UD-Q4_K_XL | 0.068 / 93.0% | 0.0579 / 93.96% |

NVFP4 and UD-Q4_K_XL in Basalt only (same harness and base, T 3, int8 KV, `--resident-experts`, expert tiers pinned (`--adapt-swaps 0 --peer-adapt-swaps 0`), 400 W; a
development measurement):

| Pack | Chunks | PPL | KLD mean | KLD median | KLD p99 | Same top-1 |
|---|---|---|---|---|---|---|
| UD-Q4_K_XL | 100 | 3.3201 | 0.05656 | 0.01213 | 0.6586 | 92.27% |
| NVIDIA-checkpoint NVFP4 (dense Q8_0) | 100 | 3.2873 | 0.06167 | 0.01345 | 0.6923 | 91.80% |
| NVIDIA-checkpoint NVFP4 (dense Q8_0), combined build | 100 | 3.2870 | 0.06115 | 0.01365 | 0.6992 | 91.87% |
| NVIDIA-checkpoint NVFP4, dense Q6_K (official) | 100 | 3.3081 | 0.06799 | 0.0150 | 0.775 | 91.46% |
| UD-Q4_K_XL | 20 | 2.3223 | 0.0685 | 0.00514 | 0.9928 | 93.02% |

The NVIDIA-checkpoint (dense Q8_0) row was run on the build before the FP4 prompt path, T 3, int8 KV, resident experts, 400 W, base PPL 3.28607.
Against UD-Q4_K_XL it has 0.0051 higher KLD and 0.47 points lower top-1, and a lower PPL (3.2873 against 3.3201), which
is not a quality signal.

The combined-build row (FP4 prompt experts, chunked delta rule, FP16-MMA prompt attention, AVX2 CPU experts, 6,144-token
stream cutoff) used the same flags as the row above, 400 W. The 256-token prompt half is below the 1,024-token FP4 minimum,
so the gate exercises MMQ prompt experts, and the change from that row (KLD -0.0005, top-1 +0.07 points, PPL -0.0003) is
inside run noise. Long prompts (2,048-token halves, 12 chunks, PPL only) on the same build: 2.9579 against 2.9547 for
MMQ (standard error 0.037 each), paired per-row NLL difference +0.0011 (+-0.0014).

IQ3_S and IQ3_XXS agree between the two engines to within 0.002 KLD and 0.2 points of top-1 at both chunk counts. The
first 20 chunks read higher than the 100-chunk set (higher KLD and higher top-1 for the IQ3 packs), so the README table,
which uses 20 chunks, reads higher than the 100-chunk numbers; the 100-chunk numbers are the ones to compare. The one gap
is UD-Q4_K_XL at 20 chunks (0.068 vs 0.0579), explained in the next section.

## Why Q8_0 against Q8_0 is not zero

The Basalt Q8_0 pack scores KLD 0.0456 and top-1 95.6% (4 chunks, fp16 KV) against the stock llama.cpp Q8_0 base. This
is mostly the base's own noise. llama.cpp quantizes the activations of its Q8_0 matmuls. In the hyper-connection read
(down projection 10240 → 320, then up) the sum has cancellation, so Q8_0 quantization of the input (0.2 to 0.4% per
element) becomes about 1% in the mixed stream at layer 0. Basalt's hc read uses fp32 activations and matches the exact
math to 1e-5.

At layer 0, computed in numpy from llama.cpp's own hc_norm output and the GGUF Q8_0 hc weights, against an fp64 result
with no activation quantization: Basalt's mixed stream has relative error 0.00000 and llama.cpp's 1.06%. llama.cpp
against an emulation of Q8_0 activation quantization differs by 0.009%.

With the hc read cast to F32 in llama.cpp (a local patch), 4 chunks, 400 W:

| Comparison | KLD mean | Same top-1 |
|---|---|---|
| Basalt Q8_0 vs stock base | 0.0456 | 95.6% |
| Basalt Q8_0 vs F32-hc base | 0.0233 | 97.16% |
| llama.cpp Q8_0 with F32 hc vs stock base | 0.0436 | 96.0% |
| llama.cpp with ssm_alpha/beta activation quantization also removed vs F32-hc base | 0.0241 | 96.86% |

Basalt's 0.0233 is the level of two independent implementations of this model (0.0241 between two llama.cpp variants).
The stock base's hc noise lifts every pack's KLD by about 0.01 to 0.02 (Basalt, 4 chunks):

| Pack | Stock base | F32-hc base |
|---|---|---|
| UD-Q4_K_XL (hc Q8_0) | 0.0825 | 0.0748 |
| GSQ-RCO IQ3_S (hc BF16) | 0.2100 | 0.1950 |

The noise cancels only when the tested llama.cpp pack also has Q8_0 hc and quantizes the same way, which is why
llama.cpp against its own base scores better on UD-Q4_K_XL (0.0579 vs 0.068 at 20 chunks). Reported KLD numbers
therefore include about 0.01 to 0.02 of the reference's own noise. They are kept against the stock base because anyone
can regenerate it.

This is not a CPU-tier effect: the 4-chunk Q8_0 result holds whatever share of experts sits on the CPU pool (0.0456 to
0.0488 at the default placement, 0.0466 with 79% on the CPU). Other causes tested and excluded: KV mode, window width T,
`--gr-fp32-activations`, dense_tc against native_mmvq (not bit-identical: dense_tc takes exact int32 dots per 32-value
block, while mmvq rounds per 8-value quarter block, and the summation orders differ; switching it off made no
measurable difference to this gap), and refitting the hc8 scales.
