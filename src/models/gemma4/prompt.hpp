// src/models/gemma4/prompt.hpp - the prompt path of Gemma 4: a read of more rows than a decode window runs in chunks of up
// to kChunk rows, each one forward over chunk-wide row buffers. The projections and the MLP are W8A8 products on the FP8
// tensor cores (prefill/fp8_gemm.hpp), the router's scores one fp32 product over the chunk, the experts W4A4 (two FP4
// terms) on the FP4 block-scaled tensor cores grouped by expert and read in place (prefill/moe_fp4.hpp) with the GELU
// gate between the two products, the attention flash style over the caches (attn_prompt); the rest are the decode
// forward's kernels over all the rows. A chunk leaves the caches as decode windows of the same tokens would: a window
// layer's ring holds the window and a chunk, so the chunk's keys go straight to their cells.
#pragma once

#include "basalt/models/gemma4/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::gemma4 {

class Gemma4Model;
struct Geometry;

/// The prompt path's buffers, shared by the model's sessions (one read at a time), and its chunk forward.
class PromptPath {
public:
    ~PromptPath();
    bool init(const Gemma4Model& m, std::string& err);
    /// The device bytes init takes for a model of geometry `g` and MLP width `n_ff_pad`.
    static uint64_t bytes(const Geometry& g, int64_t n_ff_pad);
    /// One chunk: `rows` (at most kChunk) tokens at positions pos0.. through every layer, their keys and values written to
    /// `caches` ([layer]); no head. Queued on `st`; `tokens` are host memory, staged before the call returns.
    void chunk(const int32_t* tokens, int rows, int64_t pos0, const std::vector<SegKv>& caches, cudaStream_t st);

private:
    struct Carve;
    void layout(Carve& c, const Geometry& g, int64_t n_ff_pad);

    const Gemma4Model* m_ = nullptr;
    void* base_ = nullptr;
    int32_t* io_h_ = nullptr;          // pinned: tokens [kChunk], positions [kChunk]
    cudaEvent_t staged_ = nullptr;     // the last chunk's staging read
    int32_t* io_ = nullptr;            // device: the same
    float *x_ = nullptr, *tmp_ = nullptr, *mlp_ = nullptr, *xn_ = nullptr;   // [C][E]
    float *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *qp_ = nullptr, *att_ = nullptr;
    float *g_ = nullptr, *u_ = nullptr;                                       // [C][n_ff_pad]
    void* act_ = nullptr;                                                     // FP8 activations
    float *rlog_ = nullptr, *rw_ = nullptr;
    int32_t *ids_ = nullptr, *bounds_ = nullptr, *tok_ = nullptr, *dst_ = nullptr, *exp_ = nullptr;
    int32_t *tile_off_ = nullptr, *blk_off_ = nullptr;
    void* sort_scr_ = nullptr;
    void *a4x_ = nullptr, *a4h_ = nullptr;
    float *gsx_ = nullptr, *gsh_ = nullptr;
    float* h_ = nullptr;                                                      // the experts' H [C k][n_ff_exp]
    float* ex_ = nullptr;                                                     // the experts' outputs [C k][E]
};

}  // namespace basalt::models::gemma4
