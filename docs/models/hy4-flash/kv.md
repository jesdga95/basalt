# Hy4-flash KV cache

What the attention caches hold under `--kv`, how the FP8 rows round, how the caches grow, and what 1M costs. Companion to
[Hy4-flash](../hy4-flash.md#1m-context).

## Formats

| Cache | Default (16-bit) | `--kv fp8` | vLLM, `--kv-cache-dtype fp8` |
|---|---|---|---|
| Sparse layers' latent rows (12 layers, every position) | 576 fp16 values, 1,152 B | `fp8_ds_mla` row, 656 B | `fp8_ds_mla` row, 656 B |
| Window rings (30 layers, 512 + 2,048 rows) | 16-bit | 16-bit | `fp8_ds_mla` pages (the same backend as the sparse layers) |
| Indexer pooled keys (12 layers, one per 4 positions) | FP16, 256 B | FP16 | E4M3 after a 128-point Walsh-Hadamard rotation, a power-of-two scale, whatever the KV type |
| Drafter block rows (scratch) | 16-bit | FP8 rows, decoded | written into the target's FP8 cache |

A position costs 14.6 KB with 16-bit rows and 8.6 KB with FP8 rows (12 rows and three pooled-key bytes in four): 3.6 / 2.1 GiB at
256K, 14.3 / 8.4 GiB at 1M.

The 16-bit rows are fp16 (`--kv fp16`, the default), as Basalt's other models keep theirs and as the BF16 reference's llama.cpp
run kept its cache (its default `-ctk f16`): the latent is at most sqrt(512) x 0.652 = 14.8 and the rope values at most 89 over all 42
layers' weights, far inside fp16's range, with 3 more mantissa bits than BF16.

## The FP8 row

vLLM's `fp8_ds_mla` layout (FlashMLA's sparse FP8 cache, `concat_and_cache_ds_mla` in vLLM's `csrc/cache_kernels.cu`), written by
`mla_prepare`'s FP8 overload:

- bytes 0..511: E4M3 codes of the 512 latent values, rounded to BF16 first (vLLM's cache input is the BF16 normed latent);
- bytes 512..527: four fp32 scales, one per 128 values: `2^ceil(log2(max(amax / 448, 1e-4)))`, a power of two;
- bytes 528..655: the 64 rotated rope values, BF16.

A code is `E4M3(v / scale)`, round to nearest even, saturating. A code times its power-of-two scale is exact in BF16 and in FP16
(the scales are at least 2^-13, E4M3 values at least 2^-9), so every reader decodes the row exactly: the decode attention in fp32,
the prompt attention's tensor cores in fp16, the drafter's gather into its 16-bit scratch. The drafter writes its own block rows as
FP8 rows first and decodes them, as vLLM computes (it writes the block's rows into the target's FP8 cache); acceptance only changes
speed.

## Elastic caches

As Flash-Next's (`core/elastic_kv.hpp`): with `--max-context` above 32,768 a session backs its first 32,768 positions at start, and
the start refuses only when those do not fit. A prompt or a generation that needs more maps them out of the expert tier (its spare
slots first, then the slots it lends); a prompt that needs fewer positions than are backed gives the rest back before it runs, and
the tier takes them at its end. A prompt whose positions would leave the tier below what a prompt chunk needs (the expert ring and
a GiB of buffers: about 1,300 slots) is refused with `ERR` and the engine serves the next request. The start line and
`/v1/status` (`context.elastic_kv_ceiling`) state the positions that fit.

## Selection at long range

The indexer scores every pooled key (one per 4 positions) for every row and keeps the best 511 groups. A prompt scores 64 rows at a
time on the tensor cores (q in two fp16 halves): 564 us for 64 rows against 131,072 pools, 122 TFLOP/s, at the compute floor. The
selection is a radix search over 11, 11 and 10 bits of the scores' order keys: 255 us for the same batch, 439 us for one row of
262,144 pools (a 1M context). Summed over a 1M prompt the indexer takes about 2.5 minutes (*est*, from these figures).

## Measured

NVFP4 pack, RTX 5090 (main) + RTX 5060 Ti (expert tier), both at their default power limits, `--max-context` sized to the run.

| | 16-bit rows | `--kv fp8` |
|---|---|---|
| Prompt read, 112,960 tokens (tiers pinned) | 3,291 tok/s | 3,208 tok/s |
| Prompt read, 225,880 tokens (tiers pinned) | | 3,041 tok/s |
| Needle at 10 / 50 / 90 % of 253,410 tokens | pass / pass / pass | pass / pass / pass |
| Decode after a 116,202-token prompt (`--spec 6`) | 68.4 tok/s, hits 83 % | 69.9 tok/s, hits 84 % |
| Decode after a 253,410-token prompt (`--spec 6`) | 54 to 62 tok/s, hits 64 to 80 % | |
| Needle at 50 % of 1,015,207 tokens | | pass (read in 486 s, 2,090 tok/s) |
| Decode after the 1,015,207-token prompt (`--spec 6`) | | 43.1 tok/s, hits 60 % |

The needle is a passphrase sentence in wikitext, asked for with thinking off; prompts of 253K tokens read in 94 to 98 s. After a
long prompt the tier takes back the slots its room still holds (114 to 122 of 129 per layer at 253K, 75 at 1M with FP8 rows) and
decode refills them.
