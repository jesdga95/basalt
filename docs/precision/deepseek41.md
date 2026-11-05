# DeepSeek-V4.1-Flash precision

Where DeepSeek rounds and how its output compares. Method: [quality](quality.md). Guide: [DeepSeek](../models/deepseek41.md).

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
| Routed experts (UD-IQ1_M) | IQ1_S/IQ1_M gate/up, IQ1_S/IQ1_M/IQ2_S down | q8_1 activations (GPU), Q8_K (CPU); fp32 scales |
| Attention and shared-expert projections | Q6_K, Q5_K or Q8_0 | q8_1 (decode); fp16 block-scaled int8 mma (prompt) |
| Engram table | IQ4_NL, 256 values per row | dequantized to fp32 on the host |
| Compressor and indexer projections | BF16 | fp32 activations and accumulation |
| Norms, router, hyper-connections | F32 | fp32 |
| Latents, compressed rows, indexer keys | fp16 | fp32 scores, softmax and accumulation |
| Output head | Q6_K | fp32 logits |

All activations between stages are fp32. A decode kernel's output does not depend on how many tokens share its launch (bitwise).

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
| Compressed rows | 1e-5 (fp16 storage included); selected ids identical on random inputs |
| Sparse attention | fp16 tensor cores, 2e-4 relative RMS against fp32; bitwise independent of token count |
| Router logits | 3.5e-4 relative RMS against llama.cpp's batched path |

On the second card, a chunk under 8,334 tokens runs its experts of up to 16 rows on the decode kernels. Against MMQ alone, the last
prompt row's KL is 3.3e-5 (2,500 tokens) and 8.9e-4 (4,300 tokens), top-1 the same.

### Prompt skip

Past layer 20, a row reads only its own 127-position window and layer 20's compressed rows. A row more than 2,668 positions from the
read's end (20 × 127 + 128) therefore stops at layer 20's compressor. `--no-prompt-skip` runs every row.

- The math is the same: an 8K read matches a full read bit for bit in the layer outputs, the last prompt token and 60 generated tokens.
- In a real read the rounding moves. The last prompt row against a full read: KL 2.4e-6, 1.7e-4 and 7.7e-5 at 8K, 32K and 64K. Another
  chunk size on a full read moves the same row by KL 3.2e-5, 8.4e-4 and 1.2e-3.
- State is saved or resumed only at a read's end.

## Verify windows

`--spec` runs up to 8 rows at once (the default is 3). With placement fixed (`--diag-tier-frozen --diag-tier-ram 304`), every row's logits equal plain decode's, bit for bit, in windows of 2, 3, 5 and 8 rows, for the drafter too. Sampled text at
temperature 1 matches plain decode. With tiers moving, rows round as under any placement: forced through the plain text, 15 to 25 prose
picks and 1 to 6 code picks differ over 800 tokens.

## Against llama.cpp

Basalt's kernels are its own ports of the same graph. Per-layer tensors on the same weights, relative RMS: router logits 1e-7 (decode),
hyper-connection chain 1e-7 (decode) to 5e-5 (prompt), `attn_out` 4e-3 to 1.1e-2 (llama.cpp's own prompt noise is the same size). Top-512
selection overlap 0.998 over a 6,250-token prompt. Greedy decoding reproduces llama.cpp's first 16 tokens on two of three fixtures.

## Against the BF16 reference

Reference: [quality](quality.md#the-bf16-reference). Two cards, 400 W + 180 W, 25,500 rows each:

| Pack | PPL | KLD mean | KLD median | KLD p99 | Top-1 |
|---|---:|---:|---:|---:|---:|
| UD-IQ1_M (the published pack) | 3.0334 | 0.2779 | 0.0550 | 2.86 | 83.88% |
| GSQ-RCO IQ2_XXS (tested, not shipped) | 3.6353 | 0.4311 | 0.0894 | 4.39 | 79.97% |

Prompt skip cannot be scored this way (512-token chunks). On a 64K prompt, the 200 tokens after it match `--no-prompt-skip` closely
(KL mean 6.2e-4 between the two runs).
