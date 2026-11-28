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

/// A layer kind's rotary angles as two tables of (cos, sin) float pairs, [256][hd / 2] for the position's low 8 bits and
/// [positions / 256][hd / 2] for the rest: the angle of position p and pair i is (p & ~255) f_i + (p & 255) f_i, so its
/// cos and sin are one product of the two entries (rope_tables).
struct RopeTab {
    const float* lo = nullptr;
    const float* hi = nullptr;
};
/// The tables of `positions` positions for head width hd, base `base`, the first `rot` pairs rotated (f_i =
/// base^(-2i / hd), computed in double; past `rot`, cos 1 and sin 0): host arrays to upload, lo then hi.
void rope_tables(int hd, int rot, double base, int64_t positions, float* lo, float* hi);
inline size_t rope_lo_floats(int hd) { return (size_t) 256 * hd; }
inline size_t rope_hi_floats(int hd, int64_t positions) { return (size_t) ((positions + 255) / 256) * hd; }

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
    RopeTab rope;
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
    RopeTab rope;
    LayerShape sh;
    KvType kv_type = KvType::F16;
    KvSegs kv;
    int64_t max_keys = 0;
    float* scratch = nullptr;            ///< attn_scratch_floats
    float* out = nullptr;                ///< [rows][n_head][hd]
    void* act = nullptr;                 ///< set: the output as the next product's bf16 rows (window_gemm.hpp) in place of out
};
size_t attn_scratch_floats(int rows, int n_head, int hd);
void attn_decode(const Attn& a, int rows, void* stream);

/// A layer whose keys are its values (sh.one): its keys 0..n-1 as fp16 rotated keys, K = rope(V * k_norm), into k16
/// ([cell][n_kv][hd], cell = key % kv.cap), and with an E4M3 cache its values as fp16 into v16 (null with an fp16
/// cache). The prompt attention then reads them as an fp16 cache with stored keys, its tiles straight from memory.
void derive_keys(const SegKv& kv, const float* k_norm, RopeTab rope, const LayerShape& sh, KvType kv_type, int64_t n,
                 void* k16, void* v16, void* stream);

/// The prompt attention: `rows` consecutive positions pos0.. of one conversation, each over the keys attn_decode gives it,
/// read from the cache (the rows' own keys written by attn_prep first: a window layer's ring holds the window and the
/// rows), flash style on the fp16 tensor cores (q, k, v and the probabilities rounded to fp16; fp32 products and
/// softmax). A full layer whose keys are its values derives each key tile from its values as it stages it. The key
/// tiles sit on a fixed grid of positions, so a row's result does not depend on the call's other rows. With `span`, the
/// call's rows are one picture's, read together: a window layer's row also sees the later rows of the call (the picture
/// attends to itself both ways); a full layer's row stays causal.
struct AttnPrompt {
    const float* q = nullptr;            ///< [rows][n_head][hd], prepared
    int64_t pos0 = 0;
    bool span = false;
    const float* k_norm = nullptr;       ///< sh.one: K = rope(V * k_norm)
    RopeTab rope;
    LayerShape sh;
    KvType kv_type = KvType::F16;
    SegKv kv;
    float* out = nullptr;                ///< [rows][n_head][hd]
};
void attn_prompt(const AttnPrompt& a, int rows, void* stream);

// ---- glue -------------------------------------------------------------------------------------------------------------

/// out[r] = table[tokens[r]] * scale: BF16 rows of `width` (the table may be mapped host memory).
void embed_rows(const uint16_t* table, const int32_t* tokens, int rows, int width, float scale, float* out, void* stream);
/// x[r] += rms(y[r]) * w (the post-sublayer norm and the residual).
void add_norm(float* x, const float* y, const float* w, float eps, int rows, int width, void* stream);
/// out[r][i] = gelu_tanh(g[r][i]) * u[r][i] for i < n, 0 for n <= i < ld (the padding the next product reads).
void gelu_mul(const float* g, const float* u, int rows, int n, int64_t ld_gu, int64_t ld, float* out, void* stream);
/// gelu_mul into the next product's bf16 rows (window_gemm.hpp) [rows][ld].
void gelu_pack(const float* g, const float* u, int rows, int n, int64_t ld_gu, int64_t ld, void* act, void* stream);
/// The experts' inputs: xr[r] = rms(x[r]) (unweighted: the router's), xq the q8_1 blocks (36 bytes, 32 values each) of
/// rms(x[r]) * w (width a multiple of 32).
void moe_inputs(const float* x, const float* w, float eps, int rows, int width, float* xr, void* xq, void* stream);
/// The feed-forward's end: mlp = rms(mlp_out) * n1 when the layer has experts (then + rms(sum_s w[r][s] e[r * k + s]) * n2,
/// in slot order), or mlp_out alone; x = (x + rms(mlp) * n_post) * scalar[0]. With `next_w`, then the next step's input
/// rms(x) * next_w: as bf16 rows of window_gemm.hpp into `act`, or as fp32 rows into `xn`.
struct FfnEnd {
    float* x = nullptr;
    const float* mlp = nullptr;
    const float* experts = nullptr;      ///< [rows * k][width]; null: no experts
    const float* weights = nullptr;      ///< [rows][k]
    int k = 0;
    const float* n1 = nullptr, *n2 = nullptr, *n_post = nullptr;
    const float* scalar = nullptr;       ///< device [1]
    float eps = 1e-6f;
    const float* next_w = nullptr;
    void* act = nullptr;
    float* xn = nullptr;
};
void ffn_end(const FfnEnd& f, int rows, int width, void* stream);

// ---- the experts off the card (experts.hpp): the decode window's hand-off to the host through mapped memory

/// The hand-off's words (mapped host memory): `ring` is 1 | layer << 8 | rows << 16 once a layer's rows and choices are
/// out, `ticket` its sequence number; the host sets `done` to the ticket once the rows of the pairs off the card are in;
/// `abort` (non-zero) releases every wait.
struct MoeBell {
    alignas(64) uint32_t ring;
    uint32_t ticket;
    alignas(64) uint32_t done;
    uint32_t abort;
};
constexpr int kMoePairs = 128;   ///< a window's (row, slot) pairs at most
/// When a pair's expert is off the card (tab[e] == 0): advances `seq` (device), publishes the choices to ids_m and the
/// experts' inputs xr[r] * w of the rows with such a pair to x_m (width floats each, mapped), then the ticket and the ring.
/// rows <= kRows, rows * k <= kMoePairs.
void moe_ring(const int32_t* ids, int rows, int k, const unsigned long long* tab, const float* xr, const float* w, int width,
              int layer, float* x_m, int32_t* ids_m, MoeBell* bell, uint32_t* seq, void* stream);
/// Pair p off the card takes row p of `host_rows` (mapped) into row p of `out` once the bell's done matches `seq`; the
/// card's pairs stay.
void moe_merge(const int32_t* ids, int rows, int k, const unsigned long long* tab, const float* host_rows, float* out, int width,
               MoeBell* bell, const uint32_t* seq, void* stream);
/// x = cap * tanh(x / cap), n values.
void softcap(float* x, int64_t n, float cap, void* stream);

}  // namespace basalt::models::gemma4
