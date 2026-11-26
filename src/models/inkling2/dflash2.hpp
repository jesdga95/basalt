// src/models/inkling2/dflash2.hpp - Inkling 2's embedded DFlash 2 block-diffusion drafter (`dflash2.*` tensors,
// `inkling2.dflash2.*` metadata), replacing `dspark.*` in packs built with `checkpoint_inkling2.py --dflash2 <dir>`.
//
// Target taps: identical to DSpark (visits 5, 23, 41 inputs and the final normed state at visit 48).
//   context   h_ctx = rms_norm(fc([f_0 | f_1 | f_2 | f_3]), hidden_norm); per layer l in 0..4:
//             k = rope(rms_head(wk h_ctx) * k_norm, p, theta = 1e4), v = wv h_ctx into ring cell p % cap (window 2048).
//   block     anchor `a` at p, rows p .. p + 7: x_0 = rms_norm(embed(a)) * embed_norm, x_{1..7} = rms_norm(embed(mask)) *
//             embed_norm. Each of the 5 non-causal sliding-window layers wraps attention and SwiGLU MLP in 2-tap group-16
//             dynamic causal convolutions (pre-conv on normed input, post-conv fused with the next sublayer's RMSNorm).
//   head      draft rows 1..n normed with `norm`: NVFP4 draft head scores all n_vocab columns on Tensor Cores, the top-64
//             candidates per row are rescored with exact Q8 target head dots to yield exact Top-16 raw logits + full-vocab
//             softcapped tail LSE, and the 16-warp rank-256 bilinear candidate selector walks steps 1..n.
#pragma once

#include "basalt/core/checkpoint_regions.hpp"
#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/models/inkling2/dflash2_kernels.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "models/inkling2/dspark.hpp"
#include "models/inkling2/pack.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::inkling2 {

struct Dflash2Layer {
    void *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;          ///< NVFP4 planes (commit & block wgemm)
    void *gate = nullptr, *up = nullptr, *down = nullptr;                      ///< NVFP4 planes (block wgemm)
    float* fp4_scale = nullptr;                                                ///< [7]: q, k, v, o, gate, up, down
    void *wk_q8 = nullptr, *wv_q8 = nullptr;                                   ///< Q8_0 (prompt MMQ)
    void *conv_a = nullptr, *conv_f = nullptr;                                 ///< Q8 planes [576][2304]
    float *base_a = nullptr, *base_f = nullptr;                                ///< F32 [2][2][2304]
    float *attn_norm = nullptr, *post_norm = nullptr;                          ///< F32 [2304]
    float *q_norm = nullptr, *k_norm = nullptr;                                ///< F32 [128]
};

class Dflash2Weights {
public:
    ~Dflash2Weights();
    model::Result finish(const void* head, const uint16_t* embedding, const float* embed_norm, float target_eps, void* stream);
    uint64_t bytes() const { return bytes_; }

    Dflash2Geometry geo;
    int n_vocab = 0;
    float target_eps = 1e-6f;
    std::vector<Dflash2Layer> layer;
    void* fc_q8 = nullptr;               ///< 4 contiguous [2304][2304] Q8_0 slices after finish() (prompt MMQ)
    void* fc_p = nullptr;                ///< [2304][9216] Q8 planes (Tensor Core wgemm decode/batch commit)
    float *hidden_norm = nullptr, *norm = nullptr, *mask = nullptr;
    uint16_t *sel_h = nullptr, *pred = nullptr, *succ = nullptr;
    void* head_fp4 = nullptr;            ///< NVFP4 planes [n_vocab][2304] (when n_vocab % 128 == 0; else Q8 head wgemm)
    float* head_fp4_scale = nullptr;
    const void* head = nullptr;          ///< target's Q8 head (q8_planes at runtime after InklingModel::finish)
    const uint16_t* embedding = nullptr; ///< target's BF16 embedding rows (mapped host memory)
    const float* embed_norm = nullptr;   ///< target's F32 embedding RMSNorm weights

    struct SharedScratch {
        int64_t att_cells = 0;
        void* tap_bf16 = nullptr;        ///< BF16 [wgemm::kRows][9216]
        float *c_raw = nullptr, *kb = nullptr, *vb = nullptr;
        int32_t* ring_bounds = nullptr;  ///< {0, ring_cells}
        float *x = nullptr, *xn = nullptr, *z = nullptr, *dyn = nullptr;
        float *q = nullptr, *qr = nullptr, *attn = nullptr, *att_scr = nullptr;
        void *act_a = nullptr, *act_q = nullptr, *act_h = nullptr;
        basalt::kernels::wgemm::Workspace ws;
        float *logits = nullptr, *g = nullptr, *tail_lse = nullptr;
        void *hq = nullptr, *top_scr = nullptr;
        int32_t* cand = nullptr;
    };
    const SharedScratch* ensure_shared_scratch(std::string& err) const;

private:
    mutable SharedScratch shared_scr_;
    friend class Dflash2Hook;
    friend model::Result dflash2_plan(const Pack&, core::LoadPlan&, int, int, std::unique_ptr<Dflash2Weights>&);
    void* dev(size_t bytes, bool& ok);
    std::vector<void*> allocs_;
    uint64_t bytes_ = 0;
    bool finished_ = false;
};

model::Result dflash2_plan(const Pack& pack, core::LoadPlan& reads, int file, int device, std::unique_ptr<Dflash2Weights>& out);
uint64_t dflash2_session_bytes(const Pack& pack);

class Dflash2Batch;

class Dflash2Hook {
public:
    ~Dflash2Hook();
    bool init(const Dflash2Weights* w, const Geometry& g, int64_t context, int tap_rows, float* tap, std::string& err);
    bool on() const { return w_ != nullptr; }
    int slot_of_visit(int visit) const;
    float* tap(int slot) const { return tap_ + (size_t) slot * tap_rows_ * kWidth; }
    float* tap_base() const { return tap_; }
    int tap_rows() const { return tap_rows_; }
    void use_tap(float* tap, int rows) {
        tap_ = tap;
        tap_rows_ = rows;
    }
    void commit(const int32_t* pos_d, int rows, void* stream, int row0 = 0);
    void commit_prompt(const int32_t* pos_d, int rows, basalt::prefill::mmq::Context& mmq,
                       const DsparkHook::PromptScratch& s, void* stream);
    void note(int64_t pos0, int rows);
    model::Result draft_begin(int32_t anchor, int64_t pos, int n, void* stream);
    model::Result draft_end(int32_t* ids, float* conf, int max_n, int& n, void* stream);

    static constexpr size_t kRunHeadBytes = 2 * sizeof(int64_t);
    size_t running_bytes() const;
    void running_state(int64_t pos, bool save, core::RunningState& st) const;
    bool running_restored(int64_t pos, const uint8_t* head);
    bool save_cells(int64_t positions, void* stream) const;
    bool restore_cells(size_t kv_size, size_t at, int64_t positions, void* stream);
    int64_t backed_cells() const { return context_ + kDf2Block; }

private:
    friend class Dflash2Batch;
    struct Cache {
        uint16_t *k = nullptr, *v = nullptr;
        int64_t cap = 0;
    };
    void* dev(size_t bytes, bool& ok);
    void stage(int k, int n, void* stream);
    bool stage_graph(int k, int n, void* stream);

    const Dflash2Weights* w_ = nullptr;
    std::vector<void*> allocs_, pinned_;
    std::vector<Cache> cache_;
    std::vector<int> slot_;
    int tap_rows_ = 0;
    int64_t context_ = 0;
    float* tap_ = nullptr;

    int32_t *meta_h_ = nullptr, *meta_d_ = nullptr;
    int32_t *ids_h_ = nullptr, *ids_d_ = nullptr;
    float *conf_h_ = nullptr, *conf_d_ = nullptr;
    cudaGraphExec_t g_stage_[kDf2MaxLayers] = {};
    cudaGraphExec_t g_head_[kDf2Block] = {};
    bool graphs_ = true;

    int64_t lo_ = 0, hi_ = 0;
    int pending_ = 0;
    Dflash2Batch* batch_owner_ = nullptr;
    int batch_slot_ = -1;
};

class Dflash2Batch {
public:
    ~Dflash2Batch();
    bool init(const Dflash2Weights* w, int max_sessions, int max_rows, int tap_rows, float* tap, std::string& err);
    bool on() const { return w_ != nullptr; }
    void commit(Dflash2Hook* const* hooks, int n_seg, const basalt::kernels::WindowSegs* at_d, const int32_t* pos_d,
                int rows, void* stream);
    model::Result draft(Dflash2Hook* const* hooks, const int32_t* anchors, const int64_t* pos, const int* n, int count,
                        void* stream);
    model::Result take(int slot, int pending, int32_t* ids, float* conf, int max_n, int& n);

private:
    void* dev(size_t bytes, bool& ok);
    const Dflash2Weights* w_ = nullptr;
    std::vector<void*> allocs_, pinned_;
    int max_sessions_ = 0, max_rows_ = 0, tap_rows_ = 0;
    int64_t att_cells_ = 0;
    float* tap_ = nullptr;

    // commit_batch scratch
    void *c_tap_bf16_ = nullptr, *c_act_ = nullptr;
    float *c_raw_ = nullptr, *c_kb_ = nullptr, *c_vb_ = nullptr;

    // block + head chunk scratch (up to kChunkSlots = 8 slots -> R <= 64 block rows, D <= 56 draft rows)
    static constexpr int kChunkSlots = 8;
    float *x_ = nullptr, *xn_ = nullptr, *z_ = nullptr, *dyn_ = nullptr, *q_ = nullptr, *kb_ = nullptr, *vb_ = nullptr;
    float *qr_ = nullptr, *attn_ = nullptr, *att_scr_ = nullptr;
    void *act_a_ = nullptr, *act_q_ = nullptr, *act_h_ = nullptr;
    basalt::kernels::wgemm::Workspace ws_;
    float *logits_ = nullptr, *g_ = nullptr, *tail_lse_ = nullptr;
    void *hq_ = nullptr, *top_scr_ = nullptr;
    int32_t *cand_ = nullptr, *meta_d_ = nullptr, *meta_h_ = nullptr;
    int32_t *ids_d_ = nullptr, *ids_h_ = nullptr;
    float *conf_d_ = nullptr, *conf_h_ = nullptr;
    void* sync_stream_ = nullptr;
    bool synced_ = true;
};

}  // namespace basalt::models::inkling2
