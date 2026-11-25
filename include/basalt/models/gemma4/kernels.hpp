// include/basalt/models/gemma4/kernels.hpp - the Gemma 4 kernels: the embedding rows, the attention's per-head norms,
// rotary embedding and cache writes, attention over a sliding window's ring or a full layer's cells (a full layer whose
// keys are its values stores each position once: K = rope(V * k_norm), V the unweighted-normed projection), and the glue
// between the products (the post-sublayer norms and residuals, GELU, the logit softcap).
//
// All activations are fp32 rows ([rows][width], row-major) and every pointer is device memory unless a comment says
// otherwise. A row's result never depends on how many rows share its call or where it sits among them. `stream` may be
// null (the call then synchronises).
#pragma once

#include "basalt/kernels/window_segs.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::models::gemma4 {

constexpr int kRows = 16;          ///< rows of a decode window (one conversation's or a batch's): the products' columns
constexpr int kMaxHeads = 32;      ///< query heads a layer may have
constexpr int kMaxGroup = 16;      ///< query heads per key/value head

/// One attention layer's shape as the kernels see it.
struct LayerShape {
    int n_head = 0, n_kv = 0, hd = 0;   ///< query heads, key/value heads, head width (256 or 512)
    int rot = 0;                        ///< rotated pairs: dims i and i + hd/2 for i < rot (the rest pass through)
    int window = 0;                     ///< a sliding-window layer's window (a position sees the last `window`); 0: full
    bool one = false;                   ///< keys are the values (k_eq_v): one cache holds V, K is derived from it
};

/// The cache element type: fp16, or unit-scale E4M3 (keys are normed, so they fit).
enum class KvType : int { F16 = 0, F8 = 1 };
constexpr int kv_elem_bytes(KvType t) { return t == KvType::F8 ? 1 : 2; }

/// One conversation's caches of a layer: [cap][n_kv][hd] each, position p in cell p % cap (a window layer's ring; a full
/// layer's cap is its context, so the cell is the position). A layer with `one` keeps only `k` (V; K derived).
struct SegKv {
    void* k = nullptr;
    void* v = nullptr;
    int64_t cap = 0;
};
/// The caches of the rows of one call: segment i's rows (at.row0[i]..) take c[i]'s; at.n = 0: every row takes c[0].
struct KvSegs {
    basalt::kernels::WindowSegs at;
    SegKv c[basalt::kernels::kWindowSegsMax];
};

/// The attention's preparation: per row and head, the projections normed (q and k by their weights, v unweighted), q and
/// k rotated at the row's position, and k and v written to the row's cache cell.
struct Prep {
    const float* q = nullptr;            ///< projection outputs: row r's heads at q + r * ldq (likewise k, v)
    const float* k = nullptr;
    const float* v = nullptr;            ///< unused with sh.one (V is the key projection, normed without a weight)
    int64_t ldq = 0, ldk = 0, ldv = 0;
    const int32_t* pos = nullptr;        ///< each row's position [rows]
    const float* q_norm = nullptr;       ///< [hd]
    const float* k_norm = nullptr;       ///< [hd]
    const float* inv_freq = nullptr;     ///< [hd / 2] rotary frequencies (pairs past sh.rot unused)
    float eps = 1e-6f;
    LayerShape sh;
    KvType kv_type = KvType::F16;
    float* q_out = nullptr;              ///< out: [rows][n_head][hd]
    KvSegs kv;
    bool q_only = false;                 ///< the queries alone (a drafter reading the model's caches)
};
void attn_prep(const Prep& p, int rows, void* stream);

/// Attention of each row over its conversation's cells: a window layer's positions (pos - window, pos], a full layer's
/// [0, pos]; scores q . k unscaled, softmax in fp32. The keys are split in chunks by the row's own key count (so a row's
/// bits do not depend on the other rows), each chunk a block, the chunks merged by a second kernel. `max_keys` bounds
/// every row's key count (the launch's grid; a captured window passes a bound that holds for every replay).
struct Attn {
    const float* q = nullptr;            ///< [rows][n_head][hd], prepared
    const int32_t* pos = nullptr;
    const float* k_norm = nullptr;       ///< sh.one: K = rope(V * k_norm)
    const float* inv_freq = nullptr;
    LayerShape sh;
    KvType kv_type = KvType::F16;
    KvSegs kv;
    int64_t max_keys = 0;
    float* scratch = nullptr;            ///< attn_scratch_floats
    float* out = nullptr;                ///< [rows][n_head][hd]
};
size_t attn_scratch_floats(int rows, int n_head, int hd);
void attn_decode(const Attn& a, int rows, void* stream);

// ---- glue -------------------------------------------------------------------------------------------------------------

/// out[r] = table[tokens[r]] * scale: BF16 rows of `width` (the table may be mapped host memory).
void embed_rows(const uint16_t* table, const int32_t* tokens, int rows, int width, float scale, float* out, void* stream);
/// x[r] += rms(y[r]) * w (the post-sublayer norm and the residual).
void add_norm(float* x, const float* y, const float* w, float eps, int rows, int width, void* stream);
/// out[r][i] = gelu_tanh(g[r][i]) * u[r][i] for i < n, 0 for n <= i < ld (the padding the next product reads).
void gelu_mul(const float* g, const float* u, int rows, int n, int64_t ld_gu, int64_t ld, float* out, void* stream);
/// The feed-forward's end: mlp = rms(mlp_out) * n1 when the layer has experts (then + rms(sum_s w[r][s] e[r * k + s]) * n2,
/// in slot order), or mlp_out alone; x = (x + rms(mlp) * n_post) * scalar[0].
struct FfnEnd {
    float* x = nullptr;
    const float* mlp = nullptr;
    const float* experts = nullptr;      ///< [rows * k][width]; null: no experts
    const float* weights = nullptr;      ///< [rows][k]
    int k = 0;
    const float* n1 = nullptr, *n2 = nullptr, *n_post = nullptr;
    const float* scalar = nullptr;       ///< device [1]
    float eps = 1e-6f;
};
void ffn_end(const FfnEnd& f, int rows, int width, void* stream);
/// x = cap * tanh(x / cap), n values.
void softcap(float* x, int64_t n, float cap, void* stream);

}  // namespace basalt::models::gemma4
