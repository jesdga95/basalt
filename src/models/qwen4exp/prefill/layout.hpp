#pragma once

#include "common.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/kv_stream.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"
#include <vector>

namespace basalt::prefill::detail {

// Either cudaMalloc (owned, freed with the object) or a bump allocation from a borrowed region; with no base and
// no region it only counts, which is how `bytes_needed` sizes the region.
struct Alloc {
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
    bool count_only = false;
    std::vector<void*>* owned = nullptr;
    template <typename T> T* take(size_t n, bool& ok) {
        const uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        if (count_only) { used += bytes; return nullptr; }
        if (base != nullptr) {
            if (used + bytes > cap) { ok = false; return nullptr; }
            T* p = (T*) (base + used);
            used += bytes;
            return p;
        }
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; return nullptr; }
        owned->push_back(p);
        used += bytes;
        return (T*) p;
    }
};

constexpr int64_t GEMM_SCRATCH = 32ll << 20;        // FP16 elements for the largest dequantized dense weight
constexpr size_t GEMM_WS = 32u << 20;               // cuBLAS workspace

// Which layers' experts go through MMQ (both weight types covered; the Q2_0 pack always - its blob
// is converted to GGUF Q2_0 blocks on the gather), whether any layer keeps the FP16 path (IQ1_M), and the largest
// gate/up and down matrices a group buffer slot holds.
constexpr int MMQ_GROUP = 16;                  // experts per MMQ launch (the gather is per expert, as blobs arrive)
static_assert(STAGE >= 2 * MMQ_GROUP, "the staging ring holds two groups");
// MMQ reads up to one 256-value tile past a matrix's last row when the row length is not a multiple of it (the down
// product: 640 values).  Those bytes meet zero activations, which is harmless only if they decode to finite numbers -
// llama.cpp zero-pads after every tensor, and so does a group buffer: this many zeroed bytes follow its last expert.
constexpr size_t MMQ_TAIL = 4096;
// A group buffer's down part ends in MMQ_GROUP 16-byte slots after its zeroed tail: each NVFP4 expert's FP32 scale
// tail {s_gate, s_up, s_down, 0}, gathered with it and applied in mmq::swiglu.
constexpr size_t MMQ_SCALES = (size_t) MMQ_GROUP * 16;
struct MmqPlan {
    bool any = false, fallback = true;
    std::vector<char> layer;                   // per layer: MMQ
    size_t gu_max = 0, d_max = 0;
};

// A chunk shorter than this leaves most expert tiles nearly empty and the per-expert launches dominate: MMQ is faster.
constexpr int64_t FP4_MIN_TOKENS = 1024;
// The peer computes its experts' rows of a streamed chunk of up to this many
// tokens (PeerPrefill::T_fp4).  The layer's activations go to it and its rows come back through host memory, and the
// layer's end waits for that round trip; it grows with the chunk while the blobs it saves do not, so a longer chunk
// streams the blobs instead.
constexpr int64_t kPeerFp4Tokens = 5120;
// A peer card slower than the main one takes the rows of routed-only chunks up to this size, no more.
constexpr int64_t kPeerSlowRows = 2559;
// In such a chunk the issuing thread keeps at most this many expert copies queued on the primary's link (the ring still
// runs as far ahead): copies on one link are served in the order they were queued, and the peer's result rows, which
// the layer's end waits for, would otherwise queue behind the copies for the layers after it.
constexpr int kPeerInFlight = 16;
constexpr size_t al256(size_t v) { return (v + 255) & ~(size_t) 255; }

struct Fp4Ptrs {
    uint8_t *a4x, *a4h;
    float *gsx, *gsh;
    int32_t *toh, *rel;
};
// Where the FP4 path's tables sit in Xq (the layer's token planes) and Hq (a group's H), as moe_bufs sized them.

const MmqPlan& mmq_plan();
float* grp_scales(uint8_t* grp_d);
void take_stage(Alloc& o, const core::SessionState& ss, const basalt::kernels::QsaShapes& s,
                basalt::kernels::KvHostPools& st, bool& ok);
basalt::kernels::QsaAttnPools pools_of(const basalt::kernels::KvHostPools& h, const int32_t* table);
uint64_t gdn_qkv_bytes(size_t T);
uint64_t gdn_set_bytes(size_t T);
uint64_t hc_set_bytes(size_t T);
uint64_t qsa_set_bytes(size_t T, int64_t cap, int64_t max_blocks, int64_t sel_batch, int64_t attn_batch,
                       const basalt::kernels::QsaShapes& s);
uint64_t moe_set_bytes(size_t T, int64_t n_expert, bool fused);
bool fused_layout(size_t T, bool src);
bool fp4_pack();
size_t fp4_x_bytes(size_t rows, int64_t ne);
size_t fp4_h_bytes(size_t rows);
bool fp4_every_layer();
size_t group_rows_cap(size_t T);
Fp4Ptrs fp4_ptrs(void* xq, void* hq, int64_t T, int64_t n_expert);

}  // namespace basalt::prefill::detail
