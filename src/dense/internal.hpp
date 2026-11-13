// src/dense/internal.hpp - the dense stack's model, session state and forward passes (see basalt/dense/model.hpp).
#pragma once

#include "basalt/dense/model.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace basalt::vision {
class Mmproj;
class CudaEncoder;
}

namespace basalt::dense {

constexpr int kWindow = 8;        // rows one session's decode window holds
constexpr int kBatch = 16;        // rows a batch window (several sessions' windows) holds
constexpr int kChunk = 2048;      // tokens a prompt chunk reads at once
constexpr int kAttnSplits = 3;    // key splits the prompt attention may use: the most its plan picks for a full chunk
constexpr int kTypeF8 = 44;       // F8_E4M3 projections (Linear::type)
constexpr int kTypeNvfp4P = 45;   // NVFP4 in planes (nvfp4_planes.hpp)

/// Device buffers carved from one allocation.
class Arena {
public:
    ~Arena();
    bool reserve(size_t bytes);
    template <class T> T* take(size_t n) { return static_cast<T*>(take_bytes(n * sizeof(T))); }
    void* take_bytes(size_t n);
    void rewind(size_t mark) { used_ = mark; }
    size_t mark() const { return used_; }
    size_t size() const { return size_; }
private:
    uint8_t* base_ = nullptr;
    size_t size_ = 0, used_ = 0;
};

/// A batch window's device table: per window row its GDN segment, where its GDN rows go in its session, per segment its
/// first cell.
struct BatchTable {
    kernels::SegRow seg[kBatch];
    kernels::GdnRowDst dst[kBatch];
    int32_t pos0[kBatch];
};

/// The decode window's scratch (kBatch rows).
struct WindowScratch {
    float *x = nullptr, *xn = nullptr, *tmp = nullptr;
    void* xq = nullptr;                          // the window gemm's activations (fp16 rows)
    void* x8 = nullptr;                          // MMQ's (q8_1 rows)
    kernels::wgemm::Workspace w8;                // the window gemm's split-K sums
    float* dacc = nullptr;                       // a batch window's block drafter features, packed
    float *z = nullptr;
    float *qfull = nullptr, *k = nullptr, *v = nullptr, *qo = nullptr, *ao = nullptr, *ag = nullptr;
    float *g = nullptr, *u = nullptr, *h = nullptr;
    float* attn = nullptr;                       // gqa_decode_attn's partial sums
    int32_t *tok = nullptr, *pos = nullptr;      // the window's tokens and [row][3] positions (device)
    float* cat = nullptr;                        // the drafter's [embedding | hidden] rows
    float *lp = nullptr, *lp_scratch = nullptr;  // the drafter's picks (row_logprobs rows) and their scratch
    float *y = nullptr, *hl = nullptr;           // the GDN output rows; the drafter's head rows
    int32_t *iota = nullptr, *bounds = nullptr;  // the products' row map and the window's rows
    int32_t* pos0 = nullptr;                     // the window's first cell (device), read by its attention
    // a batch window's: the packed GDN rows of the layer in hand (before they go to each session), its table
    float *rq = nullptr, *rh = nullptr, *rgate = nullptr, *rbeta = nullptr;
    BatchTable* btab = nullptr;
    // the block drafter's: its rows' q, k, v, attention and convolution coefficients, the selector's projection
    float *bq = nullptr, *bk = nullptr, *bv = nullptr, *bqo = nullptr, *bao = nullptr, *dyn = nullptr, *bh = nullptr;
    float *battn = nullptr, *blp = nullptr, *bsel = nullptr;
    int32_t* bpos = nullptr;
};

/// The prompt chunk's scratch (kChunk rows), its mixer and FFN buffers sharing memory.
struct ChunkScratch {
    float *x = nullptr, *xn = nullptr, *tmp = nullptr;
    void* xq = nullptr;                           // MMQ activations
    float *qkv = nullptr, *z = nullptr, *hc = nullptr, *gate = nullptr, *beta = nullptr, *ab = nullptr;
    float *y = nullptr, *yn = nullptr;
    void* work = nullptr;
    float *qfull = nullptr, *k = nullptr, *v = nullptr, *qo = nullptr, *ao = nullptr, *ag = nullptr;
    float* attn_split = nullptr;                  // gqa_prompt_attn's key-split scratch (kAttnSplits splits)
    float *g = nullptr, *u = nullptr, *h = nullptr;
    float *hn = nullptr, *cat = nullptr;          // the drafter's history: main's normed rows, its input rows
    float* dacc = nullptr;                        // the block drafter's projected features of the chunk's rows
    void* a4 = nullptr;                           // FP4 activations of a long chunk's NVFP4 products, their token scales
    float* gs = nullptr;
    int32_t* tile_off = nullptr;
    float* ffn_tail = nullptr;                    // per layer {s_gate, s_up, 1, 0}: the FP4 SwiGLU product's scales
    int32_t *tok = nullptr, *pos = nullptr, *iota = nullptr, *bounds = nullptr;
};

struct SessionState;

class Model {
public:
    Geometry g;
    Weights w;
    Options o;
    Arena dec_arena, pre_arena;
    WindowScratch ws;
    ChunkScratch cs;
    bool chunk_ready = false;
    std::unique_ptr<prefill::mmq::Context> mmq;
    std::vector<int> gdn_index, attn_index;      // per layer: its index among the GDN or attention layers (-1)
    // the batch windows' graphs, one per composition (the sessions in the window and their rows), the least recently used out
    using BatchKey = std::vector<std::pair<const SessionState*, int>>;
    struct BatchGraph {
        BatchKey key;
        cudaGraphExec_t exec = nullptr;
        uint64_t used = 0;
    };
    static constexpr size_t kBatchGraphs = 64;
    std::vector<BatchGraph> batch_graphs;
    uint64_t batch_clock = 0;
    std::unique_ptr<vision::Mmproj> mmproj;
    std::unique_ptr<vision::CudaEncoder> encoder;
    void* enc_arena = nullptr;
    size_t enc_bytes = 0;
    ~Model();
    bool ensure_chunk(std::string& err);
    /// llama.cpp's MMQ readied for every product of the model (its kernels loaded for each format and height).
    bool ready_products(std::string& err);
    /// The image encoder placed in device memory for pictures of up to `tokens` merged tokens.
    bool vision_attach(int tokens, std::string& err);
    /// Its memory goes back.
    void vision_detach();
};

/// One conversation's device state.
struct SessionState {
    cudaStream_t stream = nullptr;
    Arena arena;
    float* gdn = nullptr;                        // per GDN layer: state [S][h_v][S], then conv history [C][3]
    size_t gdn_floats = 0;                       // per layer
    std::vector<uint16_t*> kc, vc;               // per attention layer: [cell][n_kv][head_dim] fp16 (null with int8)
    std::vector<kernels::KvInt8> kq;             // per attention layer: the int8 cache ({} with fp16)
    // the last window's rows per GDN layer, until its commit
    float *rec_qkv = nullptr, *rec_h = nullptr, *rec_gate = nullptr, *rec_beta = nullptr;
    int32_t* n_keep = nullptr;                   // device
    float* logits = nullptr;                     // [kWindow][n_vocab]
    int32_t* picks = nullptr;                    // device [kWindow]
    int64_t rope_delta = 0;                      // rope position - cell index for the next text token
    cudaGraphExec_t wg[kWindow + 1] = {};        // the decode window of T rows, captured once per T
    // the drafter (null without one): its KV, the last window's normed hidden rows, and the row waiting for the
    // token after it (cell tail_cell, -1 none)
    uint16_t *mtp_kc = nullptr, *mtp_vc = nullptr;
    float *hid = nullptr, *tail = nullptr, *chain = nullptr;   // chain: [2][n_embd]
    float *dlogits = nullptr, *lp = nullptr;     // the drafter's head row and its picks [kWindow][kLogprobStride]
    int32_t* win_tok = nullptr;                  // the last window's tokens (device)
    int64_t tail_cell = -1;
    // the block drafter: its context K/V per layer (cyclic, dctx_cap cells), the last window's projected features,
    // its picks and their confidence
    std::vector<uint16_t*> dk, dv;
    int64_t dctx_cap = 0, dctx_from = 0;         // cells; the first position its context holds
    float* dacc = nullptr;
    int32_t* dids = nullptr;
    float* dconf = nullptr;
};

/// Runs a decode window of T rows at cells pos..pos+T-1 (`pos3` [T][3] their rope positions); leaves the logits in
/// st.logits and the GDN rows for the commit.
bool forward_window(Model& m, SessionState& st, const int32_t* tokens, int T, int64_t pos, const int32_t* pos3,
                    std::string& err);
/// One window of S sessions as one batch: segment k is segs[k].T rows (tokens and pos3 [row][3] packed in segment order, at
/// most kWindow rows in all) at cells segs[k].pos.. of its session.  Each session is left as after its own forward_window
/// (logits, the GDN rows for its commit, the drafter's rows); runs on the first session's stream.
struct BatchSeg {
    SessionState* st = nullptr;
    int T = 0;
    int64_t pos = 0;
};
bool forward_batch(Model& m, const BatchSeg* segs, int S, const int32_t* tokens, const int32_t* pos3, std::string& err);
/// Runs a prompt chunk; rows are token embeddings, or `rows` (fp32 [T][n_embd]) when given.  `pos3` [T][3].
bool forward_chunk(Model& m, SessionState& st, const int32_t* tokens, const float* rows, int T, int64_t pos,
                   const int32_t* pos3, std::string& err);
/// The commit of the last window's first n rows into the recurrent state.
void commit_window(Model& m, SessionState& st, int n);

/// The block drafter's context from the last window's first n rows (st.dacc) at positions pos.. .
bool block_ctx_window(Model& m, SessionState& st, int n, int64_t pos, std::string& err);
/// The block drafter's pass after `anchor` at `pos`: its picks for slots 1 .. block-1 into st.dids / st.dconf.
bool block_draft(Model& m, SessionState& st, int32_t anchor, int64_t pos, std::string& err);
/// The same pass for S sessions at once (S blocks of rows, at most kBatch rows), on the first session's stream.
bool block_draft_batch(Model& m, SessionState* const* sts, const int32_t* anchors, const int64_t* pos, int S, std::string& err);

/// The drafter over n rows of (hidden, next token) pairs at cells c0..c0+n-1: `hid` device rows [n][n_embd] (main's
/// normed hidden, or the drafter's own output), `tok` device ids of the tokens after them.  Writes its KV; with
/// `out` (device [n][n_embd]) also its normed output rows and, for the last row, the head's pick and probability
/// into lp_row (row_logprobs layout, device).  n <= kWindow.
bool mtp_window(Model& m, SessionState& st, const float* hid, const int32_t* tok, int n, int64_t c0, float* out,
                float* lp_row, std::string& err);
/// The drafter's history over a prompt chunk's pairs (n <= kChunk), KV only: hidden rows `hid`, next tokens `tok`
/// (device ids) or next rows `emb` (device [n][n_embd]).
bool mtp_chunk(Model& m, SessionState& st, const float* hid, const int32_t* tok, const float* emb, int n, int64_t c0,
               std::string& err);

}  // namespace basalt::dense
