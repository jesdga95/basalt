// src/models/inkling2/kernels/experts.cu - Inkling 2's routed experts on tensor cores (the tile kernels of
// kernels/cuda/expert_tile.cuh) with the experts' finish in the down's blocks: each segment's experts' sum, its
// convolution over the rows before it in the window (the ring before the window) and the residual, run by the down block
// that completes the segment's ticket for a range of channels: the same arithmetic whichever block runs it.
#include "basalt/kernels/pdl.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "kernels/cuda/expert_tile.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;
namespace xt = basalt::kernels::xt;

constexpr int kFf = 1152;   // the experts' width the kernels are built for

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 ") + what + ": " + cudaGetErrorString(e));
}

struct Finish {
    MoeFinish a;
    int rows = 0;
    int* tickets = nullptr;

    __device__ void span(int s, int& r0, int& r1) const {
        const bk::WindowSegs& at = seg_table(a.seg);
        r0 = at.n > 0 ? at.row0[s] : 0;
        r1 = at.n > 0 ? at.row0[s + 1] : rows;
    }
    __device__ int seg(int r) const { return bk::seg_of(seg_table(a.seg), r); }
    __device__ int target(int s) const {
        int r0, r1;
        span(s, r0, r1);
        return (r1 - r0) * a.k;
    }
    // rows r0 .. r1 - 1 of segment s, channels c0 .. c0 + kTicketCh - 1, each row's slots as one fma chain in slot
    // order: the rows' pairs were written by other blocks of the call (or before it), so they are read through L2
    __device__ void finish(int s, int c0, int tid, int nt) const {
        int r0, r1;
        span(s, r0, r1);
        const bk::WindowSegs& at = seg_table(a.seg);
        float* const ring = at.n > 0 ? a.seg.c[s].m_ring : a.ring;
        for (int idx = tid; idx < (r1 - r0) * xt::kTicketCh; idx += nt) {
            const int r = r0 + idx / xt::kTicketCh, c = c0 + idx % xt::kTicketCh;
            float acc = 0.f, x0 = 0.f;
#pragma unroll
            for (int j = 0; j < kConv; ++j) {
                const float tap = a.taps[c * kConv + (kConv - 1 - j)];
                float x;
                if (r - j >= r0) {
                    x = 0.f;
                    for (int q = 0; q < a.k; ++q)
                        x = fmaf(a.w[(r - j) * a.k + q], __ldcg(a.rows + ((size_t) (r - j) * a.k + q) * kWidth + c), x);
                } else {
                    const int p = a.pos[r] - j;
                    x = p < 0 ? 0.f : ring[(size_t) (p % kConvRing) * kWidth + c];
                }
                if (j == 0) x0 = x;
                acc = fmaf(tap, x, acc);
            }
            const size_t i = (size_t) r * kWidth + c;
            const float y = __fmul_rn(acc + x0, rsqrtf((float) kConv + 1.f)) + a.resid[i];
            a.out[i] = y;
            if (a.tap) a.tap[i] = y;
            ring[(size_t) (a.pos[r] % kConvRing) * kWidth + c] = x0;   // a window's slots: none its rows read before it
        }
    }
};

// The same finish, a thread a channel through every row of segment s: each row's experts' sum is formed once and kept
// for the rows after it in the window (finish forms it again for each tap); the same operations in the same order.
__global__ void __launch_bounds__(256) finish_rows_kernel(Finish f, int n_segs) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int s = (int) blockIdx.x, c = (int) blockIdx.y * 256 + (int) threadIdx.x;
    if (s >= n_segs || c >= kWidth) return;
    const MoeFinish& a = f.a;
    int r0, r1;
    f.span(s, r0, r1);
    const bk::WindowSegs& at = seg_table(a.seg);
    float* const ring = at.n > 0 ? a.seg.c[s].m_ring : a.ring;
    float tap[kConv];
#pragma unroll
    for (int j = 0; j < kConv; ++j) tap[j] = a.taps[c * kConv + (kConv - 1 - j)];
    float xs[kConv] = {};   // xs[j]: row r - j's sum, once the window holds it
    for (int r = r0; r < r1; ++r) {
#pragma unroll
        for (int j = kConv - 1; j > 0; --j) xs[j] = xs[j - 1];
        float x = 0.f;
        for (int q = 0; q < a.k; ++q) x = fmaf(a.w[r * a.k + q], __ldcg(a.rows + ((size_t) r * a.k + q) * kWidth + c), x);
        xs[0] = x;
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < kConv; ++j) {
            float xj;
            if (r - j >= r0) {
                xj = xs[j];
            } else {
                const int p = a.pos[r] - j;
                xj = p < 0 ? 0.f : ring[(size_t) (p % kConvRing) * kWidth + c];
            }
            acc = fmaf(tap[j], xj, acc);
        }
        const size_t i = (size_t) r * kWidth + c;
        const float y = __fmul_rn(acc + x, rsqrtf((float) kConv + 1.f)) + a.resid[i];
        a.out[i] = y;
        if (a.tap) a.tap[i] = y;
        ring[(size_t) (a.pos[r] % kConvRing) * kWidth + c] = x;   // a window's slots: none its rows read before it
    }
}

// The tile shapes by the call's pairs: few pairs (a single conversation's window) take 8-entry tiles of 64 rows and
// three stages; more take 16-entry tiles of 128 rows; a wide batch 32-entry tiles of 128 rows NVFP4 (an item reads its
// entries' activations once for twice the weight rows), 64 MXFP8. Every shape gives an entry the same bits. MXFP8 rows
// are twice the bytes: two stages for the wider tiles.
constexpr int kSmallPairs = 128, kWidePairs = 1024;
template<int TY, int K> using Small = xt::Cfg<TY, K, 8, 64, 3>;
template<int TY, int K> using Mid = xt::Cfg<TY, K, 16, 128, 2>;
template<int TY, int K> using Wide = xt::Cfg<TY, K, 32, TY == xt::kNvfp4 ? 128 : 64, 2>;

xt::Plan plan_of(const ExpertCall& e) {
    xt::Plan p;
    p.ptr = e.ptr;
    p.start = e.start;
    p.n_groups = e.n_groups;
    p.dst = e.dst;
    p.tok = e.tok;
    return p;
}

template<int TY, class C>
void gate_up_one(const ExpertCall& e, cudaStream_t s) {
    auto* kern = xt::gate_up_kernel<TY, kWidth, C>;
    static thread_local int set_dev = -1;
    int dev = 0;
    cudaGetDevice(&dev);
    if (set_dev != dev) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
        set_dev = dev;
    }
    const int tiles = xt::max_tiles(e.pairs, e.n_expert, C::TT);
    const dim3 grid((unsigned) (tiles * (2 * kFf / C::ROWS)));
    bk::launch_pdl(kern, grid, dim3(C::NT), (size_t) C::SMEM, s, plan_of(e), e.L, static_cast<const uint16_t*>(e.x),
                   static_cast<uint16_t*>(e.h));
}

template<int TY, class C>
void down_one(const ExpertCall& e, const Finish& f, bool fin, cudaStream_t s) {
    auto* kern = xt::down_kernel<TY, kFf, C, Finish>;
    static thread_local int set_dev = -1;
    int dev = 0;
    cudaGetDevice(&dev);
    if (set_dev != dev) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
        set_dev = dev;
    }
    const int tiles = xt::max_tiles(e.pairs, e.n_expert, C::TT);
    const dim3 grid((unsigned) (tiles * (kWidth / C::ROWS)));
    bk::launch_pdl(kern, grid, dim3(C::NT), (size_t) C::SMEM, s, plan_of(e), e.L, static_cast<const uint16_t*>(e.h),
                   const_cast<float*>(f.a.rows), f, fin);
}

void check_call(const ExpertCall& e, const char* what) {
    const bk::NativeExpertLayout& L = e.L;
    const bool fmt = (L.gu_type == xt::kNvfp4 || L.gu_type == xt::kMxfp8) && L.d_type == L.gu_type && L.up_type == L.gu_type;
    if (!fmt || L.n_embd != kWidth || L.n_ff != kFf || e.n_expert > xt::kMaxGroups || e.pairs <= 0 || e.pairs > kMaxPairs)
        throw std::runtime_error(std::string("inkling2 ") + what + ": the experts' formats or shape");
}

}  // namespace

size_t expert_h_bytes(int pairs, int64_t n_ff) { return (size_t) pairs * (size_t) n_ff * 2; }

size_t expert_tickets() { return (size_t) bk::kWindowSegsMax * (kWidth / xt::kTicketCh); }

void experts_gate_up(const ExpertCall& e, void* stream) {
    check_call(e, "experts_gate_up");
    cudaStream_t s = (cudaStream_t) stream;
    const bool mx = e.L.gu_type == xt::kMxfp8;
    if (e.pairs < kSmallPairs) {
        if (mx) gate_up_one<xt::kMxfp8, Small<xt::kMxfp8, kWidth>>(e, s);
        else gate_up_one<xt::kNvfp4, Small<xt::kNvfp4, kWidth>>(e, s);
    } else if (e.pairs < kWidePairs) {
        if (mx) gate_up_one<xt::kMxfp8, Mid<xt::kMxfp8, kWidth>>(e, s);
        else gate_up_one<xt::kNvfp4, Mid<xt::kNvfp4, kWidth>>(e, s);
    } else {
        if (mx) gate_up_one<xt::kMxfp8, Wide<xt::kMxfp8, kWidth>>(e, s);
        else gate_up_one<xt::kNvfp4, Wide<xt::kNvfp4, kWidth>>(e, s);
    }
    check("experts_gate_up");
}

namespace {
Finish finish_of(const ExpertCall& e, const MoeFinish& mf, int rows) {
    Finish f;
    f.a = mf;
    f.rows = rows;
    f.tickets = e.tickets;
    return f;
}

}  // namespace

void experts_finish(const ExpertCall& e, const MoeFinish& mf, int rows, void* stream) {
    if (rows <= 0 || rows > kBatchRows || mf.k * rows != e.pairs) throw std::runtime_error("inkling2 experts_finish: rows");
    const int n_segs = mf.seg.at.n > 0 ? mf.seg.at.n : mf.seg.n_segs > 0 ? mf.seg.n_segs : 1;
    bk::launch_pdl(finish_rows_kernel, dim3((unsigned) n_segs, kWidth / 256), dim3(256), 0, (cudaStream_t) stream,
                   finish_of(e, mf, rows), n_segs);
    check("experts_finish");
}

namespace {
void down_any(const ExpertCall& e, const MoeFinish& mf, int rows, bool fin, void* stream) {
    check_call(e, "experts_down");
    if (rows <= 0 || rows > kBatchRows || mf.k * rows != e.pairs) throw std::runtime_error("inkling2 experts_down: rows");
    cudaStream_t s = (cudaStream_t) stream;
    const Finish f = finish_of(e, mf, rows);
    const bool mx = e.L.gu_type == xt::kMxfp8;
    if (e.pairs < kSmallPairs) {
        if (mx) down_one<xt::kMxfp8, Small<xt::kMxfp8, kFf>>(e, f, fin, s);
        else down_one<xt::kNvfp4, Small<xt::kNvfp4, kFf>>(e, f, fin, s);
    } else if (e.pairs < kWidePairs) {
        if (mx) down_one<xt::kMxfp8, Mid<xt::kMxfp8, kFf>>(e, f, fin, s);
        else down_one<xt::kNvfp4, Mid<xt::kNvfp4, kFf>>(e, f, fin, s);
    } else {
        if (mx) down_one<xt::kMxfp8, Wide<xt::kMxfp8, kFf>>(e, f, fin, s);
        else down_one<xt::kNvfp4, Wide<xt::kNvfp4, kFf>>(e, f, fin, s);
    }
    check("experts_down");
}
}  // namespace

// A call of few pairs runs the finish in the down's blocks; a wider one in its own kernel after them (the blocks that
// complete the tickets come last and run the finishes at the down's occupancy: the tail would cost more than a launch)
void experts_down(const ExpertCall& e, const MoeFinish& mf, int rows, void* stream) {
    if (e.pairs < kSmallPairs) {
        down_any(e, mf, rows, true, stream);
        return;
    }
    down_any(e, mf, rows, false, stream);
    experts_finish(e, mf, rows, stream);
}
void experts_down_rows(const ExpertCall& e, const MoeFinish& mf, int rows, void* stream) { down_any(e, mf, rows, false, stream); }

}  // namespace basalt::models::inkling2
