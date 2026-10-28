// include/basalt/kernels/verify_kernels.hpp - kernels for the speculative VERIFY window, where T
// tokens (the last accepted token and T-1 drafts) go through a layer at once.
//
// The GDN recurrence is the one piece of state that cannot simply be overwritten by the next window, so its
// kernels come in two halves:
//
//   * the verify half reads the conv history and the recurrent state, runs all T tokens in order (bitwise the
//     single-token kernels' arithmetic per token) and writes ONLY the per-token outputs - the state is left
//     untouched;
//   * the commit half, launched once acceptance is known, replays the first `n_keep` tokens from the inputs the
//     verify half stored and writes the state.  Rejected tokens therefore never touch the state and there is no
//     snapshot of the 113 MB recurrent state to keep.
//
// Everything that varies per window (token ids, `n_keep`) is read from DEVICE memory so the kernels can be captured.
#pragma once

#include <cstdint>

namespace basalt::kernels {

inline constexpr int kVerifyMaxT = 8;

/// For token t of T: conv over [history(3) | qkv_0 .. qkv_t] -> SiLU -> L2 norm of the q/k heads -> h[t].
/// `history` is NOT written.  Bitwise `fused_gdn_conv_l2` per token.
void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin = 0);
/// history <- the last 3 entries of [history | qkv_0 .. qkv_{n-1}], n = *n_keep (0 leaves it as it was).
void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream);
/// DEC_COMMIT: n_layers consecutive GDN layers' commits in one launch (layer l's state at state + l *
/// state_stride, its conv history right after the S x h_v x S state; h / qkv at l * h_stride, gate / beta at
/// l * gb_stride): gdn_conv_commit and gdn_step_norm_multi's state-only commit, bitwise.
struct GdnCommitLayers {
    float* state = nullptr;
    const float *h = nullptr, *qkv = nullptr, *gate = nullptr, *beta = nullptr;
    size_t state_stride = 0, h_stride = 0, gb_stride = 0;
    int C = 0, h_k = 0, h_v = 0, n_layers = 0;
    const int32_t* n_keep = nullptr;
};
/// From `blocks` persistent blocks (grid-stride); each loads its next item while the current one runs.
void gdn_commit_layers(const GdnCommitLayers& a, void* stream, int blocks = 32);
/// Counters of the commit's dynamic items, two sets used by epoch parity (claim, done, started, pre; one line each),
/// then the newest epoch, the last epoch a finisher finished and the lazy layers' claims.  Zeroed once (then
/// gdn_commit_lazy_none); a set and the lazy claims are reset by the begin kernel of the epoch that uses them.
struct CommitCtl {
    uint32_t w[32 * 11];
};
/// The commit epoch's start: reads the epoch (> 0) from `epoch` (a device alias of mapped memory), resets that epoch's
/// counter set, publishes `started` and writes the epoch to `begun` (a device alias of mapped memory the host polls).
void gdn_commit_begin(CommitCtl* ctl, const int32_t* epoch, uint32_t* begun, void* stream);
/// gdn_commit_layers with its items claimed from the epoch's counter, so another grid can take items too.  Block 0
/// publishes `pre` on entry: the work stream-ordered before this launch (the indexer and PLE commits) is complete.
void gdn_commit_layers_dyn(const GdnCommitLayers& a, CommitCtl* ctl, void* stream, int blocks = 32);
/// The first node of a verify window.  With the newest epoch not yet finished (its begin kernel has run: the host
/// checks `begun` before launching the window) the blocks claim the epoch's remaining items, then wait until all are
/// done and `pre` is published, and the epoch is marked finished; otherwise every block returns.
void gdn_commit_finish(const GdnCommitLayers& a, CommitCtl* ctl, void* stream, int blocks);
/// The newest epoch's commit of the layers in `a` (as gdn_commit_layers), its items claimed in order from the epoch's
/// lazy counter: every item of the first `must_layers` not claimed yet, then further ones while *stop != stop_value
/// (device word; null: none).  A launch after the epoch's earlier ones finds their items done.
/// `pf` (or null): what the launch also reads into L2 for the next layer while the layer it runs beside waits for a
/// tier: once *stop holds stop_value with bit 31 (the combine waits for the CPU pool) or bit 30 (for the peers' rows)
/// and while it still does, if the layer has entries off CUDA0 at all (ids against ptab, that layer's CUDA0 blob table).
constexpr int kLazyPfRanges = 8;
struct GdnLazyPf {
    const void* p[kLazyPfRanges] = {};   ///< byte ranges (16-byte multiples) in the order they are used; empty ones last
    uint64_t n[kLazyPfRanges] = {};
    const int32_t* ids = nullptr;
    int n_ids = 0, n_expert = 0;
    const unsigned long long* ptab = nullptr;
};
void gdn_commit_lazy(const GdnCommitLayers& a, CommitCtl* ctl, int must_layers, const uint32_t* stop,
                     uint32_t stop_value, void* stream, int blocks, const GdnLazyPf* pf = nullptr);
/// The lazy counter of a new CommitCtl: nothing owed until the first begin kernel.
bool gdn_commit_lazy_none(CommitCtl* ctl);
/// A window of several sequences ("segments"): each segment is n consecutive rows with its own GDN state.  The table
/// has one entry per window row (a segment's rows all carry the segment's fields), uploaded to the device per window.
/// A kernel launched over the entries lets only the segment's first row (j == 0) do the segment's work.
struct SegRow {
    float* gdn;        // the segment's GDN state of layer 0 (layer l's state at gdn + l * state floats)
    int32_t row0;      // the segment's first window row
    int32_t j;         // this row's index in its segment
    int32_t n;         // the segment's rows
    int32_t n_keep;    // rows of the segment to commit
    int32_t slot;      // the sequence's slot
    int32_t pos;       // this row's position
};
static_assert(sizeof(SegRow) == 32, "SegRow is copied as eight words");
/// gdn_conv_l2_multi over a segmented window: every row's conv from its own segment's history (hist_off floats past
/// `gdn`: the layer's conv history), all n_rows rows in one launch; bitwise the per-segment calls.
void gdn_conv_l2_seg(const SegRow* rows, size_t hist_off, const float* qkv, const float* conv_w, float* h, int channels,
                     int qk_heads, float eps, int n_rows, void* stream);
/// gdn_step_norm_multi's verify form (state read, not written) over every segment of a window, one launch.
/// state_off: floats from `gdn` to the layer's state.
void gdn_step_norm_seg(const SegRow* rows, size_t state_off, const float* h, int conv_channels, const float* gate,
                       const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                       int n_rows, void* stream);
/// gdn_commit_layers over every segment of a window: segment s keeps rows[s.row0].n_keep rows of its own state.  The
/// layers' rows (h, qkv, gate, beta) are laid out as in GdnCommitLayers with the window's rows in each; state of
/// layer l at gdn + state_off + l * state_stride.
struct GdnCommitSeg {
    const SegRow* rows = nullptr;
    size_t state_off = 0, state_stride = 0, h_stride = 0, gb_stride = 0;
    const float *h = nullptr, *qkv = nullptr, *gate = nullptr, *beta = nullptr;
    int C = 0, h_k = 0, h_v = 0, n_layers = 0, n_rows = 0;
};
void gdn_commit_layers_seg(const GdnCommitSeg& a, void* stream, int blocks = 32);
/// The PLE history of each row's slot as data: `slots[slot]` is that slot's history (`n` floats).  Gather: work row t =
/// the slot's history; scatter: the slot's history = row (row0 + n_keep - 1) of `work` for each segment's first row
/// (n_keep <= 0 leaves it).
void ple_hist_gather(float* work, float* const* slots, const SegRow* rows, int64_t n, int n_rows, void* stream);
void ple_hist_scatter(float* const* slots, const float* work, const SegRow* rows, int64_t n, int n_rows, void* stream);
/// alpha/beta for T columns of x (T, n_embd): gate (T, h_v), beta (T, h_v).  Bitwise `fused_gdn_ab` per column.
void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream);
/// The recurrence + output norm for T tokens (h = (T, conv_channels) as q|k|v, gate/beta (T, h_v), z/y
/// (T, value_dim)).  With `n_keep == nullptr` the state is read and NOT written (verify); otherwise the first
/// *n_keep tokens are run and the state is written (commit; `y` may be scratch).  Bitwise `fused_gdn_step_norm`.
void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin = 0);
/// Spin until *flag >= value (a mapped host flag).  The value is fixed at capture, so several rings can be
/// outstanding at once (the split verify window keeps two).
void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream);
/// Sets *flag = value (a mapped host flag) behind everything queued on the stream, written out to the system first: the
/// other end of a wait_flag_ge on another device.
void signal_flag(uint32_t* flag, uint32_t value, void* stream);
/// Up to kWordCopyMax disjoint device word ranges copied by one launch (the pipelined pair's state snapshot).
constexpr int kWordCopyMax = 32;
struct WordCopy {
    const int32_t* src = nullptr;
    int32_t* dst = nullptr;
    int n = 0;
};
struct WordCopyList {
    WordCopy e[kWordCopyMax];
    int n = 0;
};
void word_copy_list(const WordCopyList& l, void* stream);
/// the GPU's %globaltimer (ns) into buf[i] (a one-thread kernel: the verify window's stage profiler).
void gpu_stamp(unsigned long long* buf, int i, void* stream);

// ---- a layer whose routed experts are all in VRAM needs nothing from the host
/// One group's plan, built on the device when every routed expert of its n*k entries is resident: the host pool's
/// layout (counts | start | dst | tok | pad | ptr | ptr2 | start2, `capx` entries) and order (distinct experts in
/// routing order, their entries ascending), no PCIe groups.  *skip = ring when it did, else 0.
/// Without `skip` (the zero-doorbell graph) an entry whose expert is not in VRAM leaves the plan empty and sets
/// *plan_err (mapped host memory) to 1, so the host sees it instead of a window on a stale plan.
void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream, uint32_t* plan_err = nullptr);
/// wait_flag_ge that also returns when *skip == value (device memory).
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream);
/// copy_i32_from_mapped unless *skip == value.
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream);
/// copy_from_mapped, or zeros when *skip == value (n a multiple of 4, 16-byte aligned).
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream);

/// Rows of the S2/S4/S8 embedding for T token ids read from DEVICE memory; out (T, n).  Bitwise `embedding_gather`.
void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream);
/// R[t][c][:] = x[t][:] for the hc streams.
void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream);
/// dst[:n] = src[(*index) * stride + :n]  (index read from device memory; a negative index copies nothing).
void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream);

/// Plan v0.3 P6: copy *n (device memory) blobs of `blob_bytes` from mapped host memory (src[k], device aliases)
/// into dst + k * blob_bytes with coalesced 16-byte loads - the PCIe share of a layer's missed experts, staged
/// into VRAM before the grouped expert kernel reads them.  Launched for a capacity of `cap` blobs.
void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream);
/// ptr[k] = base + k * blob_bytes for k < *n (the staged copies `fetch_blobs` made).
void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream);

// ---- the MTP draft layer (src/core/mtp.cpp)
/// R[t][c][:] = h[t][c][:] + e[t][:]  (the embedding branch added to every stream).
void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream);
/// Every routed expert is resident (slot = expert id): slot[i] = ids[i], dst[i] = i, *count = n.
void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream);
/// dst row i = src row ids[i] (row_bytes each, multiple of 16), for n rows.
void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream);
/// ids[t] = table[ids[t]] for n entries (a subset index back to a token id).
void map_ids(int32_t* ids, const int32_t* table, int n, void* stream);
/// probs[t] = softmax(logits[t])[ids[t]] for n_rows rows of n_vocab (the probability of each row's argmax).
void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream);
/// Row 0 of R, inj and mixed <- row *row_dev of each (rows R_stride, inj_stride, mixed_stride floats apart); nothing
/// when that row is 0.  The draft round takes its draft row from the catch-up with it (MTP_REUSE_ROW).
void mtp_take_row(float* R, int64_t R_stride, float* inj, int64_t inj_stride, float* mixed, int64_t mixed_stride,
                  const int32_t* row_dev, void* stream);
/// For draft steps that are conditional (IF) graph nodes (MTP_COND): handle h[0] <- the cumulative rule of
/// draft_cum.hpp for chain step j (hist[0..j) the earlier steps' probabilities, probs[*row] step j's) against the
/// threshold whose float bits are at *th_bits, and h[1..n) <- 0 (n <= 8).
/// False (nothing launched) when the build has no conditional graph nodes.
bool mtp_draft_cond(const unsigned long long* h, int n, const float* probs, const int32_t* row,
                    const int32_t* th_bits, void* stream, const float* hist, int j);
/// The draft round's staging in one launch: each of up to 6 int segments (dst, host-mapped src, count) by a block of its
/// own, and `n_f` floats (a multiple of 4, 16-byte aligned) from `f_src` to `f_dst` by the other blocks; the same copies
/// as copy_i32_from_mapped and copy_from_mapped.
struct MtpStageSeg { int32_t* dst; const int32_t* src; int n; };
void mtp_stage(const MtpStageSeg* segs, int n_segs, float* f_dst, const float* f_src, int64_t n_f, void* stream);
/// The draft chain's step end in one launch: tok_dst[0] = ids[row], out[j] = ids[row] and out_p[j] = probs[row] for
/// row = *row_dev (the residual is not carried), then, with n > 0, mtp_draft_cond's handles for the same row.  False
/// (nothing launched) when n > 0 and the build has no conditional graph nodes.
bool mtp_step_end(const int32_t* ids, const int32_t* row_dev, int32_t* tok_dst, int32_t* out, int j, const float* probs,
                  float* out_p, const unsigned long long* h, int n, const int32_t* th_bits,
                  const float* hist, void* stream);
/// Dense-attention step records for `n` cells: [cell, cell+1, (cell+1)/4, cell+1] from cells[i] (device memory).
void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream);
/// A sliding attention window: for `n` step records (kStepCount ints each, n_kv at [1]) the selection becomes the
/// last `window` cells: ids[q * ids_stride + j] = max(0, n_kv - window) + j and the record's width = the count.
void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream);

}  // namespace basalt::kernels
