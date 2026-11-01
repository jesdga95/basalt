# Jev target scores

Jev reads the native verifier's target logits using the loaded pack's formats. For screened Q6_K and bit-plane
Q8_0 output heads, upstream's exact-head helpers supply full rows before allowed-token probabilities are
calculated. Q8_0 batch windows already request exact rows, including beside a split two-stage head. Other exact
heads retain their normal row. Weights are never requantized; KV storage and chat sampling stay
as configured. The candidate distribution is conditional on its allowed tokens, not the full vocabulary.

The solo score window runs one target token, committing recurrent/indexer/PLE state. Prefix checkpoints include
those states as well as the existing positional KV history. The final token has no draft continuation and is
never offered as a live chat tail; subsequent work restores a pre-window checkpoint or recomputes its prefix.
Scoring rows use greedy sampling internally to avoid allocating a wide sampler; the drawn token is discarded.
MTP still fills prompt state under the existing prefill path, using the selected `spec` setting.

Branch waves use the native segmented verifier without committing their windows. All segments read one common
FP32 GDN recurrence/conv history. Each has private QSA cache/indexer storage and PLE history; suffix tokens are
causal within their own segment. The final row of each segment supplies its logits. Exact-head replay covers
the whole wave before host reads, including screened heads. Shared recurrence is neither copied nor quantized.

The segmented QSA K/V post kernel also accepts FP16 pools, using the existing FP32-to-FP16 conversion and page
addressing from `kv_append_step`. Norm, RoPE, indexer and attention arithmetic retain their native order. INT8
appends keep their existing path; solo FP16 recording is unchanged. The QSA segment fixture compares FP16 post
against separate native kernels plus a host FP16 append oracle, with shuffled slots, block boundaries, mixed
lengths and cache bytes.
The GDN segment fixture also checks shared read-only recurrence against solo kernels and the CPU oracle,
including byte-for-byte unchanged source state. It never commits segments that alias that source state.

No stored tensor format or chat configuration is changed. The solo scorer adds no GPU KV allocation; branch
buffers are demand-sized and bounded. See [configuration and API](../configuration/decisions.md) for scheduling
and cache ownership. Kernel parity, quality and latency measurements belong to the actual running pack/GPU;
host protocol timing is separate from model execution.
