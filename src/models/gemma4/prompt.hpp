// src/models/gemma4/prompt.hpp - the prompt path of Gemma 4: a read of more rows than a decode window runs in chunks of up
// to kChunk rows, each one forward over chunk-wide row buffers. The projections and the MLP are W8A8 products on the FP8
// tensor cores (prefill/fp8_gemm.hpp), an NVFP4 MLP W4A4 on the FP4 tensor cores (prefill/fp4_dense.hpp), the router's scores one fp32 product over the chunk, the experts W4A4 (two FP4
// terms) on the FP4 block-scaled tensor cores grouped by expert and read in place (prefill/moe_fp4.hpp) with the GELU
// gate between the two products, the attention flash style over the caches (attn_prompt); the rest are the decode
// forward's kernels over all the rows. A full layer whose keys are its values has its keys rotated once a chunk into an fp16
// buffer the attention reads in place. A chunk leaves the caches as decode windows of the same tokens would: a window
// layer's ring holds the window and a chunk, so the chunk's keys go straight to their cells. A picture's rows take the place
// of the token embeddings and are read as one chunk whose window layers attend both ways among its rows.
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
    /// [vision] The buffers go back to the driver while a request's pictures are read (no chunk runs then: ENC comes between
    /// calls, each of which syncs its stream), and come back before the next chunk; `unpark` fails when the card no longer
    /// has their room.
    void park();
    bool unpark(std::string& err);
    /// Before a call's chunks (or picture), with experts: the experts' tables as the tier places them, the ones off the
    /// card in a staging that each layer's RAM experts are copied into while the layer before the one before it runs; `end`
    /// gives the staging back.
    bool begin(cudaStream_t st, std::string& err);
    void end();
    /// The device bytes init takes for a model of geometry `g`, MLP width `n_ff_pad` (NVFP4 or FP8), `context` positions and
    /// caches of `kv`; `mxfp8`: some layer's experts are MXFP8 (their buffers are carved too).
    static uint64_t bytes(const Geometry& g, int64_t n_ff_pad, bool mlp_nvfp4, int64_t context, KvType kv, bool mxfp8);
    /// One chunk: `rows` (at most kChunk) tokens at positions pos0.. through every layer, their keys and values written to
    /// `caches` ([layer]); no head. Queued on `st`; `tokens` are host memory, staged before the call returns.
    void chunk(const int32_t* tokens, int rows, int64_t pos0, const std::vector<SegKv>& caches, cudaStream_t st);
    /// One picture: its `rows` (at most kChunk) embedding rows (host [rows][n_embd]) at positions pos0.., as chunk() reads
    /// tokens but the rows as they are and the window layers' attention both ways among them.
    void picture(const float* emb, int rows, int64_t pos0, const std::vector<SegKv>& caches, cudaStream_t st);

private:
    // positions pos0.. (and the tokens, when given) to the card
    void stage(const int32_t* tokens, int rows, int64_t pos0, cudaStream_t st);
    // the layers over the chunk's rows in x_, the positions staged
    void layers(int rows, int64_t pos0, bool span, const std::vector<SegKv>& caches, cudaStream_t st);
    struct Carve;
    void layout(Carve& c, const Geometry& g, int64_t n_ff_pad, bool mlp_nvfp4, int64_t context, KvType kv, bool mxfp8);
    // layer l's RAM experts into staging buffer l % 2, once the layer that read it last is done
    void issue(int l);

    const Gemma4Model* m_ = nullptr;
    void* base_ = nullptr;
    int32_t* io_h_ = nullptr;          // pinned: tokens [kChunk], positions [kChunk]
    cudaEvent_t staged_ = nullptr;     // the last chunk's staging read
    int32_t* io_ = nullptr;            // device: the same
    float *x_ = nullptr, *tmp_ = nullptr, *mlp_ = nullptr, *xn_ = nullptr;   // [C][E]
    float *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *qp_ = nullptr, *att_ = nullptr;
    float *g_ = nullptr, *u_ = nullptr;                                       // [C][n_ff_pad]
    void* act_ = nullptr;                                                     // FP8 activations
    void* a4m_ = nullptr;                                                     // an NVFP4 MLP's FP4 activations
    float* gsm_ = nullptr;                                                    // and their token scales [C]
    float *rlog_ = nullptr, *rw_ = nullptr;
    int32_t *ids_ = nullptr, *bounds_ = nullptr, *tok_ = nullptr, *dst_ = nullptr, *exp_ = nullptr;
    int32_t *tile_off_ = nullptr, *blk_off_ = nullptr;
    void* sort_scr_ = nullptr;
    void *a4x_ = nullptr, *a4h_ = nullptr;
    float *gsx_ = nullptr, *gsh_ = nullptr;
    void *mxx_ = nullptr, *mxh_ = nullptr;                                    // MXFP8 activation planes: tokens, H
    uint16_t* h16_ = nullptr;                                                 // H as fp16 [C k][n_ff_exp rounded up to 128]
    int32_t* mx_blk_ = nullptr;
    float* h_ = nullptr;                                                      // the experts' H [C k][n_ff_exp]
    float* ex_ = nullptr;                                                     // the experts' outputs [C k][E]
    void *k16_ = nullptr, *v16_ = nullptr;   // a full layer's keys (and an E4M3 cache's values) as fp16, [context][n_kv][hd]
    unsigned long long* tabs_ = nullptr;    // [layer][n_expert] each expert's blob for the products: its card slot or its staging
    // the staging of the experts off the card (begin .. end): two buffers of the most a layer has, a copy stream, and per
    // layer its RAM experts (staging index, host blob, bytes)
    struct Staged {
        int k;
        const uint8_t* host;
        uint64_t bytes;
    };
    std::vector<std::vector<Staged>> ram_;
    uint8_t* staging_ = nullptr;
    int stage_n_ = 0;                        // experts a buffer holds
    bool began_ = false;                     // begin ran (end gives the copies beside the homes back)
    cudaStream_t copy_ = nullptr;
    cudaEvent_t copied_[2] = {}, done_[2] = {};
};

}  // namespace basalt::models::gemma4
