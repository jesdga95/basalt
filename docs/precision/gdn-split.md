# Precision of the GDN mixer split

The decode-path tables are in [index.md](index.md); the quality method is in [quality.md](quality.md).

## What the split changes

With a peer card (`--peer-device`) whose memory is faster than the main card's, the start plans each GDN layer's mixer
(qkv and z projections, recurrence, gated norm, out projection) split by key head: the peer takes a share that follows
both cards' memory bandwidth, in whole key-head units (a quarter at least, the main card keeps one at least), and one
start line says so. With a slower peer, the usual card order, the split is not taken: the whole mixer stays on the main
card and the output is bitwise what it was before the split existed (greedy identical, 1,200 tokens over 3 requests,
NVFP4 pack, 400 W main card and 180 W peer). Every change below applies only when the split is on.

- **Split-K out projection.** Each card sums the columns of its own value heads into an fp32 partial; the main card adds
  the peer's partial to its own. One fp32 add replaces the single K-ordered sum, so the result can differ in the last bit.
- **Peer projections on dp4a.** The peer runs qkv, z and the out partial as dp4a over the q8_1 rows with an exact int32
  sum per 32-value block, summed over blocks in another order than the main card's tensor-core kernels. The recurrence,
  conv and gated norm run the same code on the peer's key heads: `gdn_head_split_parity` checks each rank's heads
  bitwise against the whole layer.
- **q8_1 scale-step tie.** The out projection's activation blocks are quantized from the norm output. When the two sum
  orders place a block's largest value at an fp16 rounding tie, the block scale lands one fp16 step away while every int8
  code matches. The parity test counts one step like a one-step value flip and fails on a larger step: 60 of 60 cases pass
  in both card orders, with that tie as the only scale difference.

Quality gate, swapped card order (400 W peer card, 180 W main card), NVFP4 pack, 20 chunks, T 3, int8 KV, `--spec 0`,
base build / split: PPL 2.3016 / 2.3071, KLD 0.07656 / 0.07640, top-1 92.71% / 92.73%. The paired per-row KLD difference is
-0.00016 +- 0.0012 and 143 of 5,100 rows change their argmax: reduction-order noise, no shift.

Long-context check, same card order, greedy through the serve path (spec 7, int8 KV, a fresh engine per build), the first
100,000 and 65,000 tokens of a 200K document as a bare prompt, 128 tokens. The text splits early: at token 8 (100K) and
token 17 (65K) the builds pick different next words, then continue as different fluent text. The top two
tokens at that step are close in both builds: a logit bias of -0.1 on the leader flips the split-off build and -0.3 flips
the split-on build at 100K (the leaders differ by under 0.1 and under 0.3), and -0.1 flips both at 65K. So the flip is a
near tie moved by the changed sums, and greedy text on the two builds is not comparable past such a tie.
