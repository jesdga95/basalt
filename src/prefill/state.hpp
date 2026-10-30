#pragma once

#include "common.hpp"
#include "staging.hpp"
#include "peer.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/kv_stream.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/core/peer_experts.hpp"
#include <algorithm>
#include <vector>
#include <memory>
#include <future>

namespace basalt::prefill {
using namespace detail;

struct Prefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    core::ExpertSource* src = nullptr;
    const core::ExpertCache* cache = nullptr;
    const int32_t* host_res = nullptr;
    int64_t T = 0, T_max = 0;
    bool borrowed = false;
    uint32_t* hist = nullptr;                // routed-entry counts per (layer, expert); see route_hist_reset
    int64_t hist_n = 0, hist_tokens = 0;
    bool hist_on = false;
    cudaStream_t cs = nullptr, copy = nullptr;
    Gemm gemm;
    std::vector<void*> owned;
    // chunk buffers
    float *emb = nullptr, *R = nullptr, *xn = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
    float* grs = nullptr;                    // the hyper-connection read's row scales (T x 4)
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
    float* mixed = nullptr;
    uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
    float* bo = nullptr;
    // GDN
    float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *hbuf = nullptr, *y = nullptr;
    uint16_t* y_h = nullptr;
    // QSA
    float *Kc = nullptr, *Vc = nullptr, *Qf = nullptr, *q = nullptr, *idx_raw = nullptr, *q_idx = nullptr, *attn = nullptr;
    uint16_t* attn_h = nullptr;
    int32_t* steps_dev = nullptr;
    std::vector<int32_t> steps_host;
    int32_t* sel_ids = nullptr;
    float* sel_scores = nullptr;          // [sel_batch, max_blocks]
    int64_t sel_batch = 256, max_blocks = 0;
    float* attn_scratch = nullptr;
    int64_t attn_batch = 32, cap = 0;
    // MoE
    float *logits = nullptr, *w = nullptr, *GU = nullptr, *Dm = nullptr, *sgate = nullptr, *sup = nullptr,
          *shared = nullptr, *sg = nullptr;
    int32_t *ids = nullptr, *slot_dev = nullptr, *src_dev = nullptr;
    uint16_t *Xs = nullptr, *Hh = nullptr, *sh_h = nullptr;
    // MMQ: the activations quantized per layer, H in FP32 and its group's quantized rows, the identity
    // row map, the group bounds, the group buffers of gathered experts
    void *Xq = nullptr, *Hq = nullptr;
    float* H = nullptr;
    int32_t *ids_identity = nullptr, *bounds_dev = nullptr;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    std::vector<int32_t> bounds_host;
    std::unique_ptr<mmq::Context> mmq_ctx;
    std::vector<int32_t> ids_host, slot_host, src_host, cnt, off;
    // The CPU share of a small chunk's experts: the layer's normalized activations and the pool's output rows (pinned),
    // the pool's per-token activations and jobs, the stream and events that bring the activations to the host, and the
    // running per-expert costs the share is balanced from.
    float *cpu_x = nullptr, *cpu_rows = nullptr;
    size_t cpu_x_n = 0, cpu_rows_n = 0;
    std::vector<uint8_t> cpu_nact;
    std::vector<kernels::cpu::ActQ> cpu_actq;
    std::vector<kernels::cpu::ExpertJobMulti> cpu_jobs;
    std::vector<char> cpu_need;
    cudaStream_t cpu_s = nullptr;
    cudaEvent_t cpu_ev_in = nullptr, cpu_ev_x = nullptr, cpu_gev[2] = {};
    // The share of the streamed experts the CPU computes. Fixed, not timing-driven: the CPU and the GPU round
    // differently, so a share that moved with timing would change the output from run to run.
    double cpu_share_now = 0.6;
    // The grouping tables in mapped pinned memory, [ids | slot | src] of T_max * K each, then the MMQ bounds: kernels
    // read and write them in place.  A cudaMemcpyAsync of them queues behind the expert blobs the copy stream already
    // holds (up to `ring` of them), and the GPU idles meanwhile.
    int32_t* grp_host = nullptr;
    int32_t* grp_dev = nullptr;          // its device alias
    size_t grp_n = 0, grp_tk = 0;        // int32s allocated; T_max * K (the offset of slot, and of src past it)
    uint16_t* dq_gu[DQ] = {};
    uint16_t* dq_d[DQ] = {};
    uint8_t* stage_dev[RING_MAX] = {};
    int ring = STAGE;                        // the slots of this layout's ring (ring_slots)
    std::unique_ptr<Stager> stager;          // the unpinned experts' host copies
    cudaEvent_t copied[RING_MAX] = {}, used[RING_MAX] = {};
    bool stage_live[RING_MAX] = {};
    // the event that releases each ring slot: its own `used`, or - when an MMQ group is gathered in one launch - the
    // `used` of the last slot gathered with it, recorded once for all of them (a later record only waits longer)
    int used_of[RING_MAX] = {};
    // A chunk below stream_all_min(): the next layer's routing guessed from this layer's FFN rows (the next layer's router
    // on them; on the device and in mapped host memory), the rows it gives each expert, and per layer parity the
    // layer whose experts were copied ahead and each one's ring slot (-1: not copied ahead).
    int32_t *guess_ids = nullptr, *guess_host = nullptr, *guess_host_dev = nullptr;
    float* guess_w = nullptr;
    int64_t guess_cap = 0, guess_layer = -1;
    std::vector<int32_t> guess_rows;
    std::vector<int32_t> ahead_slot[2];
    int64_t ahead_layer[2] = {-1, -1};
    cudaEvent_t ahead_ev = nullptr;          // where on the compute stream the copies ahead may start
    // PLE
    float* ple_emb = nullptr;
    std::vector<float> ple_pageable[2];      // the fallback when no more RAM can be pinned
    float* ple_emb_host[2] = {};             // pinned, double-buffered: the next chunk's rows are read while this
    cudaEvent_t ple_copied[2] = {};          // one runs; the event marks that buffer's upload done
    std::vector<uint32_t> ple_rows[2];
    float* ple_norm = nullptr;
    uint8_t* region = nullptr;               // the attention/MoE scratch region (idle while the PLE block runs)
    uint64_t region_bytes = 0;
    PrefillStats* stats = nullptr;
    // KV streaming: one layer's whole K/V, staged from the host copy per layer and chunk (identity layout)
    basalt::kernels::KvHostPools stage;
    int32_t* ident_table = nullptr;
    // layer split: the device, and the hand-off to the next stage (two pinned chunk buffers, used in turn)
    int device = -1;
    float* hand[2] = {};
    int64_t hand_rows = 0;                   // the rows each hand-off buffer holds
    // the chunk's token ids on the device, for one batched embedding gather
    int32_t* tok_dev = nullptr;
    std::vector<int32_t> tok_host;
    std::unique_ptr<PeerPrefill> pp;         // multi-GPU: the peer GPU's expert share (set_peer)
    bool peer_share = true;                  // set_peer_share
    core::PeerExperts* ring_peer = nullptr;  // --peer-exclusive: experts held only there are copied from its memory
    // set_ring_remote: another card's cache, read for the experts without a pinned host copy
    const int32_t* remote_res = nullptr;
    const core::ExpertCache* remote_cache = nullptr;
    int remote_dev = -1;
    PeerStage peer_stage;                    // the asynchronous host-staged copy of those experts
    // row-cache warm-up of the prompt's later parts (set_ple_warm)
    struct PleWarm {
        std::vector<int64_t> tok;            // tokens[from - 2, to)
        int64_t run_pos0 = 0, to = 0;
        bool pending = false;
        std::future<void> fut;
    } warm;
};

}  // namespace basalt::prefill
