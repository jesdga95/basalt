#pragma once

#include "common.hpp"
#include "cpu_warm.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/dec_moe.hpp"
#include <vector>

namespace basalt::core {

struct DecMoe {
    int64_t rings = 0;                       ///< rings per window (one per layer)
    int32_t* res0 = nullptr;                 ///< CUDA0: [n_layers * n_expert] CUDA0 slot or -1 (host_res mirrored)
    int32_t* res1 = nullptr;                 ///< CUDA0: the same for the peer's slots
    std::vector<int32_t> shadow0, shadow1;   ///< what res0 / res1 hold
    CpuWarm* warm = nullptr;                 ///< the CPU tier's warm-up (null: off)
    basalt::kernels::DmPlan* plan = nullptr;
    unsigned int* ctr = nullptr;
    unsigned long long* slot_off = nullptr;  ///< CUDA0's per-slot offsets (null: slot * blob)
    int vram = -1;   // the VRAM share's kernels (dec_moe_vram's kind; -1 grouped)
    void* vscr = nullptr;                    ///< its gate/up rows
    void* xq_pub = nullptr;   // the side stream's q8_1 rows for the ring
    unsigned int* vsync = nullptr;           ///< its counters
    unsigned long long* wstat = nullptr;   // the per-layer wait counters (device), with --wait-stats
    unsigned long long* ptab0 = nullptr;   // [n_layers * n_expert] CUDA0 blob address or 0
    unsigned int* pub = nullptr;           // [rings][4] the FFN read's counts for the ring's block (HcFusedArgs::pub_ctr)
    unsigned int* cseen = nullptr;         // the combine's last ring whose CPU word it saw (dec_moe_combine's cpu_seen)
    // [rings + 1]: per layer the last adaptive round whose copies into its slots are in, then the round the window's
    // gate/up launches wait for (DmGlueArgs::landed)
    unsigned int* landed = nullptr;
    uint32_t* landed_pin = nullptr;        // pinned: the need copies' sources, a ring of kPinSlots
    int landed_i = 0;
    int64_t wstat_windows = 0, wstat_t = 0;
    // mapped: per ring header, FFN input and q8_1 rows; [0] the peer's ring, [16] its done ring, [32] its abort
    // record (kDmDiagWords); the peer's rows
    basalt::kernels::DmRing* h_ring = nullptr; basalt::kernels::DmRing* m_ring = nullptr;
    float* h_x = nullptr; float* m_x = nullptr;
    uint8_t* h_xq = nullptr; uint8_t* m_xq = nullptr;
    uint32_t* h_w = nullptr; uint32_t* m_w = nullptr;
    float* h_rows = nullptr; float* m_rows = nullptr;
    bool tag = false;       ///< the combine polls the peer's rows (each dword not yet written holds the tag word)
    bool tags = false;      ///< the combine waits for the CPU pool's word itself: no wait launch (without --wait-stats)
    size_t rows_words = 0;
    bool tag_dirty = false; ///< a released window may have left written rows behind
    size_t x_stride = 0, xq_stride = 0;      ///< per ring, floats / bytes
    // the peer card
    int pdev = -1;
    cudaStream_t ps = nullptr;
    unsigned long long* ptab = nullptr;      ///< [n_layers * n_expert] blob address or 0
    basalt::kernels::NativeExpertLayout* lay = nullptr;
    void* pscr = nullptr;
    unsigned int* parr = nullptr;
    int pblocks = 0;
    int8_t* rq = nullptr;   ///< the warm-up's predictor: every layer's router in int8, one scale per 32 values (rs)
    float* rs = nullptr;
    bool pq2 = false;   // a layer's gate/up is Q2_0: the peer kernel's Q2_0 build
    bool peer_live = false;                  ///< a peer window was launched and not yet waited for
    /// a further peer card: its own residency, ring, doorbell and done word ([0], [16], [32] of h_w as the first's)
    /// and its own persistent window; the FFN input's q8_1 rows and the rows it writes are the first peer's.  With
    /// further peers the combine ends its row poll on the first peer's word 60, raised only by a release.
    struct XPeer {
        PeerExperts* peer = nullptr;
        int dev = -1;
        cudaStream_t ps = nullptr;
        int32_t* res = nullptr;                  ///< CUDA0: [n_layers * n_expert] its slot or -1
        std::vector<int32_t> shadow;             ///< what res holds
        unsigned long long* ptab = nullptr;      ///< its card: blob address or 0
        basalt::kernels::NativeExpertLayout* lay = nullptr;
        void* pscr = nullptr;
        unsigned int* parr = nullptr;
        int pblocks = 0;
        int8_t* rq = nullptr;
        float* rs = nullptr;
        basalt::kernels::DmRing* h_ring = nullptr; basalt::kernels::DmRing* m_ring = nullptr;
        uint32_t* h_w = nullptr; uint32_t* m_w = nullptr;
        bool live = false;
    };
    std::vector<XPeer> xp;
    // the staged share (a card without peers): every expert's host-RAM address (device alias) or 0, its host copy and
    // the source's places it was read at, the staged count per count of distinct CPU misses, the copy's own stream
    unsigned long long* stage_src = nullptr;
    unsigned long long* stage_host = nullptr;
    uint64_t stage_gen = 0;
    int32_t* stage_k = nullptr;
    cudaStream_t st_s = nullptr;
    cudaEvent_t st_fork = nullptr, st_join = nullptr;
    int64_t stage_stride = 0;
    int stage_blocks = 0;
    int stage_min = 0;   // the fewest distinct CPU misses a layer stages a share of
    // pinned staging of the residency uploads: the copies take their source from here, so the call does
    // not wait for the stream; a ring of slots, each reused only after the window's stream sync
    uint8_t* pin = nullptr;
    size_t pin_stride = 0;
    int pin_i = 0;
    static constexpr int kPinSlots = 8;
    std::vector<uint32_t> put_i;             ///< dm_sync: the peer entries that changed
};

}  // namespace basalt::core
