#pragma once

#include "common.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>
#include <memory>

namespace basalt::prefill {
using namespace detail;

// multi-GPU: the peer GPU's share of a prompt chunk's experts.  Per MoE layer the primary copies its normed
// activations through host memory, the peer quantizes the rows routed to the experts it holds, runs the same MMQ products the
// primary would (gathered from its own slots), and copies the result rows back into the primary's Dm rows - the rows
// are laid out local-first, so the peer's are one contiguous block at the end.
struct PeerPrefill {
    core::PeerExperts* peer = nullptr;
    int dev = -1;
    int64_t cap_rows = 0, T_max = 0;
    cudaStream_t s = nullptr;
    cudaEvent_t ev_in = nullptr;    // on the primary: the activations and the row tables are ready
    cudaEvent_t ev_done = nullptr;  // on the peer: its rows have landed in the primary's Dm
    // multi-GPU: the result rows go back group by group on a second stream while the next group computes
    cudaStream_t s_out = nullptr;
    static constexpr int kGrpEv = NE / 16 + 2;
    cudaEvent_t ev_grp[kGrpEv] = {};
    bool out_pending = false;       // s_out may still read Dm (the next layer's products wait for it)
    // the peer's per-row buffers hold one GROUP's rows (a group = up to 16 experts and at most G rows), the result
    // rows double-buffered, and there is no row cap (every peer-held expert's rows go to the peer)
    int64_t G = 0;
    void *Xq_g = nullptr, *Hq_g = nullptr;
    float *GU_g = nullptr, *H_g = nullptr, *Dm_b[2] = {};
    cudaEvent_t ev_dm[2] = {};
    bool dm_live[2] = {};
    std::vector<std::pair<size_t, size_t>> groups;
    float* mixed = nullptr;
    int32_t *src = nullptr, *bounds = nullptr, *ident = nullptr;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    std::unique_ptr<mmq::Context> ctx;
    mmq::Context* run_ctx = nullptr;      // the context its products run in: ctx, or a layer split helper stage's
    std::vector<int32_t> bounds_host;
    std::vector<void*> owned;
    // WITHOUT DIRECT ACCESS, a streamed chunk of up to T_fp4 tokens: the primary leaves the experts the peer holds out
    // of its stream, and the peer computes their rows with the primary's FP4 kernels (the same bits) from its own copy
    // of the activations, in pieces of at most G rows (fp4::cut_pieces).  The activations and the result rows cross
    // through pinned host buffers on streams of their own, so neither card's issuing thread waits for a staged copy.
    int prim = -1;                        // the primary's device
    int64_t T_fp4 = 0;
    void* f_a4x = nullptr;                // the layer's token planes and token scales (fp4::quantize_tokens)
    float* f_gsx = nullptr;
    void* f_a4h = nullptr;                // a run's H planes, token scales, tile table and relative bounds
    float* f_gsh = nullptr;
    int32_t* f_toh = nullptr;
    int32_t* f_rel = nullptr;             // every group's piece bounds of a layer, relative to the group's first row
    size_t f_rel_cap = 0;
    std::vector<int32_t> f_rel_host;
    std::vector<size_t> f_rel_at;
    fp4::Pieces pieces;
    float* hx = nullptr;                  // pinned: the layer's activations on their way to the peer
    static constexpr int kRes = 4;
    float* hres[kRes] = {};               // pinned: a group's result rows on their way back
    cudaStream_t x_out = nullptr, res_in = nullptr;   // on the primary: the activations' download, the rows' upload
    cudaEvent_t ev_hx = nullptr;          // primary: hx holds the layer's activations
    cudaEvent_t ev_hx_free = nullptr;     // peer: its upload from hx is done
    cudaEvent_t ev_d2h[kRes] = {};        // peer: hres[b] holds a group's rows
    cudaEvent_t ev_h2d[kRes] = {};        // primary: hres[b]'s rows are in Dm
    cudaEvent_t ev_res = nullptr;         // primary: the layer's peer rows are in Dm
    bool hx_live = false, res_live[kRes] = {};
    ~PeerPrefill() {
        if (dev < 0) return;
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(dev);
        if (s) cudaStreamSynchronize(s);
        ctx.reset();
        for (void* p : owned) cudaFree(p);
        if (ev_done) cudaEventDestroy(ev_done);
        if (s_out) cudaStreamSynchronize(s_out);
        for (cudaEvent_t e : ev_grp) if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : ev_dm) if (e) cudaEventDestroy(e);
        if (ev_hx_free) cudaEventDestroy(ev_hx_free);
        for (cudaEvent_t e : ev_d2h) if (e) cudaEventDestroy(e);
        if (s_out) cudaStreamDestroy(s_out);
        if (s) cudaStreamDestroy(s);
        cudaSetDevice(prim >= 0 ? prim : prev);
        for (cudaStream_t st : {x_out, res_in})
            if (st) { cudaStreamSynchronize(st); cudaStreamDestroy(st); }
        for (cudaEvent_t e : {ev_hx, ev_res}) if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : ev_h2d) if (e) cudaEventDestroy(e);
        if (ev_in) cudaEventDestroy(ev_in);
        if (hx) cudaFreeHost(hx);
        for (float* h : hres) if (h) cudaFreeHost(h);
        cudaSetDevice(prev);
    }
};

// Copies a blob from the peer card's memory to the primary through pinned host buffers with only asynchronous calls:
// the peer's device-to-host copies run ahead of the primary's host-to-device ones, and no host thread blocks in a
// synchronous staged peer copy.
struct PeerStage {
    // a failed call here would let a buffer be reused early and corrupt weights silently: stop instead
    static void must(cudaError_t e, const char* what) {
        if (e != cudaSuccess) {
            basalt::log::error("basalt", "prefill peer stage: %s: %s", what, cudaGetErrorString(e));
            std::exit(1);
        }
    }
    static constexpr int kBufs = 16;
    uint8_t* buf[kBufs] = {};
    cudaEvent_t d2h[kBufs] = {}, h2d[kBufs] = {};
    cudaStream_t stream = nullptr;
    int dev = -1, n = 0;
    bool ok = false;
    size_t bytes = 0;
    bool setup(int peer_dev, size_t blob_bytes) {
        if (ok && dev == peer_dev && bytes >= blob_bytes) return true;
        release();
        dev = peer_dev;
        bytes = blob_bytes;
        for (int i = 0; i < kBufs; ++i)
            if (cudaHostAlloc((void**) &buf[i], bytes, cudaHostAllocDefault) != cudaSuccess) { cudaGetLastError(); release(); return false; }
        // h2d is recorded on the primary's copy stream, so it belongs to the current (primary) device; the rest to the peer
        for (int i = 0; i < kBufs; ++i)
            if (cudaEventCreateWithFlags(&h2d[i], cudaEventDisableTiming) != cudaSuccess) { release(); return false; }
        const core::OnDevice od(dev);
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) { release(); return false; }
        for (int i = 0; i < kBufs; ++i)
            if (cudaEventCreateWithFlags(&d2h[i], cudaEventDisableTiming) != cudaSuccess) { release(); return false; }
        ok = true;
        return true;
    }
    // queue `src` (on the peer) into `dst` (on the current device) behind `copy_stream`'s earlier work
    void copy(void* dst, const void* src, size_t nbytes, cudaStream_t copy_stream) {
        const int b = n++ % kBufs;
        {
            const core::OnDevice od(dev);
            must(cudaStreamWaitEvent(stream, h2d[b], 0), "wait for the buffer's previous host-to-device copy");
            must(cudaMemcpyAsync(buf[b], src, nbytes, cudaMemcpyDeviceToHost, stream), "device-to-host copy");
            must(cudaEventRecord(d2h[b], stream), "record the device-to-host copy");
        }
        must(cudaStreamWaitEvent(copy_stream, d2h[b], 0), "wait for the device-to-host copy");
        must(cudaMemcpyAsync(dst, buf[b], nbytes, cudaMemcpyHostToDevice, copy_stream), "host-to-device copy");
        must(cudaEventRecord(h2d[b], copy_stream), "record the host-to-device copy");
    }
    void release() {
        ok = false;
        for (int i = 0; i < kBufs; ++i) {
            if (buf[i]) cudaFreeHost(buf[i]);
            buf[i] = nullptr;
            if (d2h[i]) cudaEventDestroy(d2h[i]);
            if (h2d[i]) cudaEventDestroy(h2d[i]);
            d2h[i] = h2d[i] = nullptr;
        }
        if (stream) { const core::OnDevice od(dev); cudaStreamDestroy(stream); }
        stream = nullptr;
    }
    ~PeerStage() { release(); }
};

}  // namespace basalt::prefill
