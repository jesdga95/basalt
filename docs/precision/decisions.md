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
Decision prompts skip MTP preparation. A runtime watermark retains the longest draft-valid prefix across
target-only requests; a subsequent chat drops checkpoints beyond it before restoring draft state. Restoring a
parked chat or a chat slot supplies its own draft prefix. Chat speculation and weight formats stay configured.

Short branch waves use the native segmented verifier without committing their windows. All segments read one
common FP32 GDN recurrence/conv history. Continuing sequences instead copy that state privately and use the
existing segmented commit kernels between windows. Each has private QSA cache/indexer storage and PLE history;
tokens remain causal within their sequence, without padding. Each final real token supplies its logits, read
before its scratch row can be reused. Exact-head replay covers terminal windows before host reads, including
screened heads. The shared prefix stays unchanged; private recurrence remains FP32.
Intermediate windows skip vocabulary projection and sampling under a distinct graph-cache key. Their target
layers and state commits retain native arithmetic; terminal windows and ordinary chat retain their heads.

Decision windows share the chat batch scheduler's adaptive expert-tier maintenance. Exchanges advance only
between windows, with their copy worker joined before serial-verifier or prefill-loan transitions. The existing
adaptation switches and cadence apply. As in chat, moving an expert between CPU and GPU can change rounding;
pinning the tiers disables that source of variation. Model weights and KV formats stay unchanged.

The segmented QSA K/V post kernel also accepts FP16 pools, using the existing FP32-to-FP16 conversion and page
addressing from `kv_append_step`. Norm, RoPE, indexer and attention arithmetic retain their native order. INT8
appends keep their existing path; solo FP16 recording is unchanged. The QSA segment fixture compares FP16 post
against separate native kernels plus a host FP16 append oracle, with shuffled slots, block boundaries, mixed
lengths and cache bytes.
The GDN segment fixture checks shared read-only recurrence against solo kernels and the CPU oracle,
including byte-for-byte unchanged source state. It also covers private segmented commits against solo commits
and the CPU oracle. Segments that alias the common prefix never commit. Sequence scheduling fills bounded
windows, preserves independent offsets and removes completed sequences from subsequent windows.

No stored tensor format or chat configuration is changed. The solo scorer adds no GPU KV allocation; branch
buffers are demand-sized and bounded. See [configuration and API](../configuration/decisions.md) for scheduling
and cache ownership. Kernel parity, quality and latency measurements belong to the actual running pack/GPU;
host protocol timing is separate from model execution.
