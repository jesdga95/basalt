# Two-stage verify head

The verify window's head reads every vocabulary row (248,320) for every column. Two heads run it in two stages: every
row is screened first and only the rows that can reach the top of a column are scored exactly. A Q8_0 head screens on
the top bits of its own codes with a proven bound, so every consumer reads the exact head's values; a Q6_K head screens
through a 2-bit copy and keeps approximate logits in the tail. The decode-path tables are in [index.md](index.md).

## Q8_0 head: bit planes, exact

A Q8_0 head (the NVIDIA NVFP4 pack's) is held in VRAM as bit planes (q80p.cuh): each 2,720-byte row keeps its 80 fp16
block scales, then for every code u = q + 128 its bits 2..5, its bits 6..7 and its bits 0..1, each in a plane of its
own. Every kernel that reads the head (the window's exact pass, the token path, the draft head's rows by map and its
2-bit copy) reads the planes back into the Q8_0 codes and runs native_mmvq's arithmetic: the same bits as the Q8_0 rows
(q80_vhead_parity, q80_proxy_parity).

In the verify window the head runs in two stages (`q80p_vhead`, q80_vhead.cu):

1. The screen reads each row's scales and bits 2..7 (2,080 of the 2,720 bytes). With h = u >> 2 and l = u & 3 (0..3), a
   block's sum q·y is 4 Σ h·y + Σ l·y - 128 Σ y, so taking l as 1.5 misses it by at most 1.5 Σ |y|. A row's logit is
   therefore within w = 1.5 Σ_b d_w d_x Σ|y| of c = Σ_b d_w d_x (4 Σ h·y - 126.5 Σ y). The exact kernel's logit and the
   screen's c and w are fp32 sums of at most 320 terms whose magnitudes add up to at most 85 w (the logit) and 252 w
   (c), so their rounding stays under 1% of w; the screen widens w by 1/16 and 1e-6.
2. Each block of rows adds its largest lower bound c - w per column to a histogram. The lower edge of the bin holding
   the K-th largest block maximum is at or below the K-th largest lower bound (the maxima belong to distinct rows), so
   at or below the column's K-th largest exact logit: the column's threshold.
3. Every row whose upper bound c + w reaches the threshold (or is not a number) is scored exactly; every other row keeps
   c + w, strictly below the threshold.

So every row whose exact logit reaches the K-th largest is scored, and every other entry lies strictly below the K-th
largest exact logit: the column's K largest entries, in the sampler's order (value, then the lower id), are the exact
head's ids in the same order with the same fp32 bits. A column whose activation scales are not finite is scored whole,
and a row whose bounds are not finite is scored. K is the request's (a device word set before each window): 1 for a
greedy pick, top_k for a sampled list of 64 or fewer, 0 (every row exact) otherwise. A token's entries do not depend on
the window's width.

| Consumer of the logits | Head in the window | Why it reads the exact head's values |
|---|---|---|
| Greedy pick (argmax, the lower id on a tie); draft acceptance compares picks | two-stage, K = 1 | the exact top row and value are in the column, every other entry below them |
| Sampled list of 1..64 entries (top_k; the server default is 20), with top_p, min_p and temperature | two-stage, K = top_k | the chain reads the logits only through its top_k list: top_p's normalization, min_p's reference, the temperature softmax and the draw all run over the list, which holds the same ids, order and values. The draft probability (relaxed drafting) and the kept-id list are read off the same list |
| Sampled list longer than 64 (top_k 0 or above 64), top_p or min_p alone | exact pass (K = 0) | every row exact |
| Penalties other than neutral (repeat, frequency, presence) | exact pass | a penalty moves rows across the list's edge; neutral values leave every logit's bits unchanged |
| `logit_bias` | exact pass | a bias can lift any row into the list |
| Grammar | two-stage; a refused row is drawn again after the exact pass over the window | the mask can leave only rows outside the list |
| Scored requests (`ppl=`, the quality gate), the logit hash, the top-k log | the rows are read through `Verifier::copy_logits`, which runs the exact pass first | every row exact |
| Batch slots; the head split by vocabulary with a peer card faster than CUDA0 | exact pass | the slots sample with their own parameters; both cards run the exact kernel |

Rows scored exactly per column on real verify-head inputs (256 inputs dumped from the bench fixtures on the IQ3_S pack,
against the NVFP4 pack's head, all rows): K = 1: 2 on average (at most 21); K = 20: 301 (p99 1,254, at most 1,460);
K = 64: 1,214 (p99 5,234, at most 6,814). The bound is an L1 bound over 2,560 terms, about 50 times the screen's typical
error. Fewer screened bits widen it past use (five bits: 9.6% of the rows at K = 20), and a 2-bit copy has no useful
bound.

Cold microbench (q80_vhead_parity, 248,320 rows, L2 flushed, 5090 at 400 W), µs at 1..8 columns: the Q8_0 head before
(native_mmvq up to three columns, the one-wave kernel from four) 434 / 438 / 447 / 458 / 463 / 468 / 479 / 479; the
exact pass on the planes 447 / 447 / 448 / 455 / 460 / 470 / 481 / 496; the two-stage head at K = 20 338 / 344 / 345 /
348 / 357 / 362 / 370 / 378. The screen streams 2,080 of the 2,720 bytes of every row at about 1.5 TB/s; scoring the
listed rows takes 6-15 µs.

In the engine (NVFP4 pack, the 5090 at 400 W with the 5060 Ti at 180 W as the peer tier, cold prose, a node trace of
one reply with the same text in both builds, 541 windows): the head takes 350 µs per window (the screen 332, the exact
rows 18 on average) against 439 µs for the Q8_0 head.

With the tiers pinned, both cards and the 5090 alone write the same greedy tokens as the Q8_0 head (the three bench
prompts, 1,200 tokens), seeded sampled replies the same text (temperature 0.6 and top_k 20, prose and conv, two seeds,
1,200 tokens), and one request per consumer in the table above the same tokens (18 requests of 400 tokens: greedy,
top_k 1, 5, 20 at temperature 1.0, 64, 100 and 0, min_p alone, three penalty settings and neutral ones, two logit
biases, a json_object grammar sampled and greedy, relaxed drafting); the quality gate's 5,100 rows (20 chunks) are byte
identical (PPL 2.30097, KLD 0.0750037).

### Split with a slower peer card

With a peer card whose memory is slower than CUDA0's, the window's two-stage head is split by vocabulary: CUDA0
screens rows [0, r0), the peer the rest on its own copy of them, r0 sized so that both finish together (201,984 and
46,336 rows with a 5090 and a 5060 Ti). The cards meet in pinned host memory: CUDA0 posts the window's q8_1 columns
and K and publishes each column's threshold as its screen finds it; the peer screens against the higher of its own
threshold and CUDA0's, scores the rows that reach it with the same kernel, and lists every scored entry at or above it
(row, fp32 bits); CUDA0 writes the list into its columns, whose peer rows are -inf otherwise. Either threshold is at or
below the K-th largest logit of its card's rows, so at or below the column's K-th largest: every row of the column's K
largest is scored on its card and kept or listed with the exact kernel's bits (the same code on both cards). The K
largest entries of a column, their order and their values are the exact head's. With K = 0, and in every batch
window, CUDA0's exact pass takes every row. A card that waits past its limit for the other marks the window, which is discarded with an error.

With the tiers pinned the split writes the same greedy tokens (the three bench prompts, 1,200 tokens, both cards) and
the same seeded prose text as the head on CUDA0 alone (q80_vhead_parity's split section: the same top-K lists and
draws at K = 1, 20, 64 and 0 on either card). In the engine (5090 at 400 W, 5060 Ti at 180 W, bench prose seed 101,
the same text in both builds) a window costs 50-80 µs less at every width (6.65 vs 6.71 ms at three tokens).

## Q6_K head: approximate tail

A Q6_K output head (the IQ3_S pack's) runs in two stages in the verify window (`q6k_vhead`, q6k_head.cu). Every one of the
248,320 rows is scored from a 2-bit copy (the draft head's fit, head_proxy.cuh) on int8 tensor cores against the same
q8_1 codes the exact kernel reads. Each block of rows adds its per-column maximum to a histogram; the lower edge of the bin
holding the 64th largest block maximum is at or below the column's 64th largest approximate logit, and every row within 4
of it is rescored exactly with the plain Q6_K kernel's arithmetic (the same bits). The other rows keep their approximate
logits, so log-probabilities stay finite. The block partition does not depend on window width, and every row above its
column's bound is rescored however many qualify, so a token's logits have the same bits at every T.

What it changes: the argmax and the sampler's top-64 list come from exact rows unless the copy's error exceeds the margin;
the softmax tail outside the rescored rows carries the copy's error (rms about 0.7 logit). A request whose sampled list
is longer than 64 (top_k 0 or above 64) reads every row, so its windows run the plain Q6_K head again over the same
columns (`Verifier::exact_head_rows`): every logit exact.

Offline, on 3,614 real verify-head inputs dumped from the engine (conv, struct, prose fixtures), all 248,320 rows, exact =
Q6_K x q8_1 in float:

| Fixture | Inputs | Top-64 recall, margin 4 | Margin 6 | Margin 8 | Rows rescored per column, margin 4 (mean / p99) | Greedy pick unchanged | KLD exact -> two-stage (mean) |
|---|---|---|---|---|---|---|---|
| conv | 1,232 | 99.971% | 100% | 100% | 1,333 / 6,286 | 100% | 4.7e-4 |
| prose | 1,124 | 99.979% | 100% | 100% | 1,098 / 2,988 | 100% | 5.1e-4 |
| struct | 1,258 | 99.968% | 100% | 100% | 1,981 / 12,790 | 100% | 2.5e-5 |

Margins 6 and 8 rescore 5,569-18,649 and 27,962-78,806 rows per column. Neither a least-squares scale for the copy nor a
per-column correction of the tail's bias lowers the KLD (the tail's mass is unbiased; the cost is its noise).

Quality gate (IQ3_S, 100 chunks, against the Q8_0 base, tiers pinned (expert cache 9000 slots, peer 7562), 400 W;
the gate leaves the PCIe share dynamic, so two runs of one build differ):

| Head | Run | PPL | KLD mean | KLD median | KLD p99 | Same top-1 |
|---|---|---|---|---|---|---|
| exact Q6_K | 1 | 3.3778 | 0.15355 | 0.03400 | 1.7316 | 87.36% |
| exact Q6_K | 2 | 3.3728 | 0.15288 | 0.03397 | 1.7082 | 87.18% |
| two-stage | 1 | 3.3852 | 0.15507 | 0.03484 | 1.7681 | 87.24% |
| two-stage | 2 | 3.3827 | 0.15472 | 0.03497 | 1.7381 | 87.40% |

The two-stage head raises KLD by about 0.0017 (1.1%) and PPL by 0.2%, more than the 0.0007 between two runs of one build:
the gate scores the whole softmax, including the approximate tail. Same top-1 is unchanged. Greedy decoding with the tiers
pinned (`tools/greedy_identity.py`, the three bench prompts, 900 tokens) is token-identical to the exact head.
