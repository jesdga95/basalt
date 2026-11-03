# How quality is measured

The precision tables are in [index.md](index.md).

`basalt-bench --quality` (`tools/ppl_gate.py`) scores wikitext-2 (`wiki.test.raw`) in 512-token chunks. The first half
of each chunk is prefilled and the second half (rows 256 to 510) is scored through the decode path, against a llama.cpp
base file written by `llama-perplexity --kl-divergence-base`, read in llama.cpp's own `.kld` format. The base is the
original model in BF16 ([below](#the-bf16-reference)).

The harness (`include/basalt/program/ppl_gate.hpp` (the KLD function)) computes KL(base ‖ test) over the full vocabulary from the
fp32 log-softmax of the head logits; the base is decoded as scale*u16+min, and terms with base log-prob ≤ -16 are
skipped (a tiny underestimate). The prompt half is checked equal to the base tokens, the window tokens are the base
tokens, and KLD is flat across positions, so the two are aligned.

## The BF16 reference

The base is unsloth's BF16 GGUF of Qwen3.8-Flash-Next through `llama-perplexity` (commit 436f6f89e), 100 chunks of 512,
`-b 8192 -ub 8192`, dense layers on the GPU and the experts read from disk: PPL 3.2749 (3.2733 on the scored rows),
the same tokens the gate reads. Every official pack's KLD, top-1 and PPL against it, with this gate (100 chunks, T 3,
int8 KV, `--resident-experts`, tiers pinned, RTX 5090 at 400 W + RTX 5060 Ti at 180 W), is on the
[Hugging Face card](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt#quality). llama.cpp's own Q8_0 GGUF through
`llama-perplexity --kl-divergence` (whole chunk in one batched pass) scores KLD 0.0325 and top-1 93.98% against it; the
Basalt Q8_0 pack scores 0.0200 and 95.34%.

Earlier numbers were measured against a llama.cpp Q8_0 base. That base carries llama.cpp's own Q8_0 rounding (next
section), so every pack read 0.003 to 0.010 KLD higher against it, and the Basalt Q8_0 pack is closer to BF16 (0.0200)
than to it (0.0304). All numbers in these pages are against the BF16 base.

## Why llama.cpp's Q8_0 reads further from BF16

llama.cpp quantizes the activations of its Q8_0 matmuls. In the hyper-connection read (down projection 10240 → 320,
then up) the sum has cancellation, so Q8_0 quantization of the input (0.2 to 0.4% per element) becomes about 1% in the
mixed stream at layer 0. Basalt's hc read uses fp32 activations and matches the exact math to 1e-5.

At layer 0, computed in numpy from llama.cpp's own hc_norm output and the GGUF Q8_0 hc weights, against an fp64 result
with no activation quantization: Basalt's mixed stream has relative error 0.00000 and llama.cpp's 1.06%. llama.cpp
against an emulation of Q8_0 activation quantization differs by 0.009%. This is the main reason llama.cpp running the
Q8_0 GGUF scores KLD 0.0325 against BF16 where the Basalt Q8_0 pack scores 0.0200 (table above).
