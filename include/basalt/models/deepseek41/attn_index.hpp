// include/basalt/models/deepseek41/attn_index.hpp - DeepSeek-V4.1 attention, index side: the window ring, the KV
// compressor and the two-level sparse indexer, for decode calls of T = 1..kMaxT tokens.
//
// Activations are fp32, the caches hold packed FP8/FP4 rows or fp16 rows (CacheFormat), projection weights are bf16
// or fp32 as published.
// All pointers are device pointers. The result for a token does not depend on T or on its place in the call. `stream`
// may be null. Exact layouts are in the comments below; the cache rows are what the attend kernel reads.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::models::deepseek41 {

constexpr int kMaxT = 8;                ///< tokens per call
constexpr int kLatent = 512;            ///< KV latent width (K = V, one head); the last kRopeDim dims are rotated
constexpr int kRopeDim = 64;
constexpr int kWindow = 128;            ///< a query attends positions [max(0, p-127), p]
constexpr int kWindowSlots = 136;       ///< ring capacity (window + kMaxT, rounded up); position p sits in slot p % kWindowSlots
constexpr int kIdxHeads = 32;
constexpr int kIdxDim = 128;            ///< indexer key/query width; the last kRopeDim dims are rotated
constexpr int kTopK = 512;              ///< selected compressed rows per query
constexpr int kMaxCandBlocks = 2048;    ///< capacity of the level-one candidate list

/// The caches' rows round the way the model's reference does and keep its codes and group scales, codes first, each
/// row padded to 16 bytes. A reader's code times its scale is the rounded value, exactly.
///   window rows       kLatent E4M3 codes, then kLatent / 32 power-of-two scales (UE8M0: the fp32 exponent byte)
///   compressed rows   kLatent / 2 bytes of E2M1 codes (value 2i in the low nibble), then kLatent / 16 E4M3 scales
///   index keys        kIdxDim / 2 bytes of E2M1 codes, then kIdxDim / 32 UE8M0 scales
/// The indexer's queries take the index keys' rounding and stay fp32.
constexpr int kWinRowBytes = (kLatent + kLatent / 32 + 15) / 16 * 16;       ///< 528
constexpr int kCompRowBytes = (kLatent / 2 + kLatent / 16 + 15) / 16 * 16;  ///< 288
constexpr int kIdxRowBytes = (kIdxDim / 2 + kIdxDim / 32 + 15) / 16 * 16;   ///< 80

/// The caches' format (--kv): Fp8 the packed rows above (the default), Fp16 the unrounded values as fp16 rows of
/// kLatent or kIdxDim values; the queries are then not rounded either.
enum class CacheFormat { Fp8, Fp16 };
constexpr int win_row_bytes(CacheFormat f) { return f == CacheFormat::Fp8 ? kWinRowBytes : kLatent * 2; }
constexpr int comp_row_bytes(CacheFormat f) { return f == CacheFormat::Fp8 ? kCompRowBytes : kLatent * 2; }
constexpr int idx_row_bytes(CacheFormat f) { return f == CacheFormat::Fp8 ? kIdxRowBytes : kIdxDim * 2; }

/// Rotary parameters of one layer kind (interleaved pairs over the last kRopeDim dims, YaRN when ext_factor != 0).
/// Layers with compress ratio 0 use {10000, 1, 0, 0, 0, 0}; the others use the compressed base with the YaRN factor
/// (1/factor), ext_factor 1, beta 32/1 and the original context. The rotation has no magnitude scaling.
struct RopeParams {
    float freq_base = 10000.f;
    float freq_scale = 1.f;
    float ext_factor = 0.f;
    float beta_fast = 0.f;
    float beta_slow = 0.f;
    int n_ctx_orig = 0;
};

constexpr int kRopePairs = kRopeDim / 2;

/// Host: theta multipliers per pair, theta = position * out[i]. Upload as a [kRopePairs] fp32 device array.
void rope_freq_table(const RopeParams& p, float* out);

// ---- window ring ----------------------------------------------------------------------------------------------

/// kv = rope(rms_norm(kv_raw) * norm_w) per token, stored as a window row at slot pos % kWindowSlots of
/// ring[kWindowSlots][win_row_bytes(fmt)]; kv_out[T][512] (may be null) receives it in fp32 before the rounding.
/// kv_raw is wkv . x, [T][512]. pos[T] int32.
void kv_window(const float* kv_raw, const float* norm_w, const int32_t* pos, int n_tokens, float rms_eps,
               const float* rope_freq, float* kv_out, uint8_t* ring, CacheFormat fmt, void* stream);

// ---- compressor (kv-source layers) ----------------------------------------------------------------------------

/// Bytes of device workspace compress() needs.
size_t compress_workspace_bytes();

struct CompressArgs {
    int ratio = 2;                      ///< 1 (HCA layers) or 2 (CSA layers)
    float rms_eps = 1e-20f;
    int n_embd = 5120;
    const uint16_t* w_kv = nullptr;     ///< bf16 [512][n_embd]  (attn_compressor_kv)
    const uint16_t* w_gate = nullptr;   ///< bf16 [512][n_embd]  (attn_compressor_gate); null for ratio 1
    const float* norm_w = nullptr;      ///< [512]  (attn_compressor_norm)
    const uint16_t* w_idx_k = nullptr;  ///< bf16 [128][512] (indexer.attn_k); null when the layer has no indexer key
    const float* idx_k_norm_w = nullptr;///< [128]
    const float* rope_freq = nullptr;   ///< the layer's table
    float* state_kv = nullptr;          ///< [ratio][512] fp32, partial-group rows by position % ratio; zero at start
    float* state_score = nullptr;       ///< [ratio][512] fp32
    CacheFormat format = CacheFormat::Fp8;
    uint8_t* comp_kv = nullptr;         ///< [cap][comp_row_bytes]: compressed latents, row r = group r, rotated at r * ratio
    uint8_t* comp_ik = nullptr;         ///< [cap][idx_row_bytes]: index keys (normed, last 64 dims rotated); may be null
};

/// x[T][n_embd] is the attention-normed input, pos[T] the positions (consecutive within a sequence). For each token
/// whose position closes a group ((pos+1) % ratio == 0) row pos/ratio of comp_kv (and comp_ik) is written.
/// Rows are visible to a query at position p when row < (p+1)/ratio.
void compress(const CompressArgs& a, const float* x, const int32_t* pos, int n_tokens, void* workspace, void* stream);

/// A speculative window: as compress(), but the carried state is only read. The rows of the call stay in `workspace`
/// (compress_workspace_bytes, one per layer) until compress_window_commit stores the accepted ones into the state, so
/// a rejected tail leaves no trace. Group rows and index keys are written by position and are simply overwritten when
/// those positions run again.
void compress_window(const CompressArgs& a, const float* x, const int32_t* pos, int n_tokens, void* workspace, void* stream);
/// Folds the first `n_accepted` tokens of the last compress_window call (same `pos`, `workspace`) into the state.
void compress_window_commit(const CompressArgs& a, const int32_t* pos, int n_accepted, void* workspace, void* stream);


// ---- indexer --------------------------------------------------------------------------------------------------

enum class IndexMode {
    Full,       ///< score every visible row, keep the top kTopK (the first-level layers before the candidate source)
    Level1,     ///< as Full, and also pick the candidate blocks (newest block pinned) into cand_ids
    Level2,     ///< score only the rows inside cand_ids' blocks, keep the top kTopK
};

/// Bytes of device workspace indexer_select() needs for up to max_comp compressed rows.
size_t indexer_workspace_bytes(int max_comp);

struct IndexerArgs {
    IndexMode mode = IndexMode::Full;
    int ratio = 2;
    int n_embd = 5120;
    int n_comp = 0;                     ///< row capacity of the buffers: >= max over the call's tokens of (pos+1)/ratio;
                                        ///< only the rows the last token can see are read or scored
    int block_size = 8;                 ///< candidate block size (rows)
    int cand_blocks = kMaxCandBlocks;   ///< candidate blocks kept by Level1 (at most kMaxCandBlocks)
    const uint16_t* w_q = nullptr;      ///< bf16 [4096][1280]  (indexer.attn_q_b)
    const uint16_t* w_proj = nullptr;   ///< bf16 [32][n_embd]  (indexer.proj)
    const float* rope_freq = nullptr;   ///< the layer's table
    CacheFormat format = CacheFormat::Fp8;
    const uint8_t* comp_ik = nullptr;   ///< [>=n_comp][idx_row_bytes]: the kv-source layer's index keys
    int32_t* cand_ids = nullptr;        ///< [T][kMaxCandBlocks] int32: written by Level1, read by Level2
    int32_t* cand_n = nullptr;          ///< [T] int32
    int32_t* out_ids = nullptr;         ///< [T][kTopK] int32: row ids ascending, -1 padded
    int32_t* out_n = nullptr;           ///< [T] int32: number of valid ids
};

/// qr[T][1280] is the q_a-normed query latent, x[T][n_embd] the attention-normed input.
/// score(q, j) = sum_h relu(iq[h] . ik[j]) * iw[h] / sqrt(32*128), iw = w_proj . x, iq = rope(w_q . qr), rounded as
/// an index key in the Fp8 format.
/// Query at position p sees rows j < (p+1)/ratio. The top kTopK by score are kept (ties to the lower row id);
/// fewer than kTopK are kept when fewer rows are visible. Level1/Level2 as in IndexMode.
/// Reuse layers do nothing: they read the ids of the latest index-source layer.
void indexer_select(const IndexerArgs& a, const float* qr, const float* x, const int32_t* pos, int n_tokens,
                    void* workspace, void* stream);
/// The two parts of indexer_select on the same workspace: the head weights iw from x, then the scores and the
/// selection from qr and iw.
void indexer_weights(const IndexerArgs& a, const float* x, int n_tokens, void* workspace, void* stream);
void indexer_rank(const IndexerArgs& a, const float* qr, const int32_t* pos, int n_tokens, void* workspace, void* stream);

/// The K highest finite scores of one token (ties to the lower row), row ids ascending, `out` padded with -1 to out_ld.
/// `n` rows are the buffer's capacity and pos[0] says how many are visible, as for the indexer. With `workspace`
/// (top_rows_workspace_bytes) the rows are read over many blocks; without it over one, as the prompt path does.
void top_rows(const float* scores, int n, int K, int32_t* out, int out_ld, int32_t* out_n, const int32_t* pos, int ratio,
              int block_size, void* workspace, void* stream);
size_t top_rows_workspace_bytes(int max_rows);

/// y[t][o] = dot(w[o][0..in), x[t][0..in)) with bf16 weights, in a multiple of 8, T <= kMaxT.
void gemv_bf16(const uint16_t* w, const float* x, int n_tokens, int in, int out, float* y, void* stream);

}  // namespace basalt::models::deepseek41
