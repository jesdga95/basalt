// src/models/deepseek41/peer_tier.cpp - see peer_tier.hpp.
#include "models/deepseek41/peer_tier.hpp"

#include "basalt/core/on_device.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/native_mmvq.hpp"

namespace basalt::models::deepseek41 {
namespace {

constexpr int kRows = 8;   // entries a share holds: the router's top-k fits

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("deepseek41: second card: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

}  // namespace

PeerTier::~PeerTier() {
    if (device_ < 0) return;
    const core::OnDevice on(device_);
    if (stream_) cudaStreamSynchronize(stream_);
    if (read_) cudaStreamSynchronize(read_);
    cache_.close();
    for (void* p : {(void*) d_x_, (void*) d_out_, (void*) d_plan_, d_xq_, d_scr_})
        if (p) cudaFree(p);
    if (h_plan_) cudaFreeHost(h_plan_);
    if (stream_) cudaStreamDestroy(stream_);
    if (read_) cudaStreamDestroy(read_);
}

bool PeerTier::open(int device, int n_layers, int n_expert, int n_embd, int n_ff, const float* h_x, float* h_rows,
                    std::string& err) {
    const core::OnDevice on(device);
    device_ = device;
    n_embd_ = n_embd;
    n_expert_ = n_expert;
    h_x_ = h_x;
    res_.assign((size_t) n_layers * n_expert, -1);
    const size_t row_bytes = (size_t) n_embd * sizeof(float);
    return ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err) &&
           ck(cudaStreamCreateWithFlags(&read_, cudaStreamNonBlocking), "read stream", err) &&
           ck(cudaHostGetDevicePointer((void**) &d_rows_, h_rows, 0), "mapped rows", err) &&
           ck(cudaHostAlloc((void**) &h_plan_, sizeof(Plan), cudaHostAllocPortable), "plan staging", err) &&
           ck(cudaMalloc((void**) &d_plan_, sizeof(Plan)), "plan", err) &&
           ck(cudaMalloc((void**) &d_x_, row_bytes), "activation", err) &&
           ck(cudaMalloc((void**) &d_out_, row_bytes * kRows), "rows", err) &&
           ck(cudaMalloc(&d_xq_, kernels::native_q8_1_bytes(n_embd, 1)), "q8_1 rows", err) &&
           ck(cudaMalloc(&d_scr_, kernels::native_expert_scratch_bytes(kRows, n_ff)), "scratch", err);
}

uint64_t PeerTier::room() const {
    const core::OnDevice on(device_);
    size_t free_b = 0, total_b = 0;
    return cudaMemGetInfo(&free_b, &total_b) == cudaSuccess ? (uint64_t) free_b : 0;
}

bool PeerTier::slots(const std::vector<int64_t>& slot_bytes, std::string& err) {
    const core::OnDevice on(device_);
    const int64_t n_layers = n_expert_ ? (int64_t) res_.size() / n_expert_ : 0;
    if (!cache_.open_sized(slot_bytes, n_layers, n_expert_, err)) return false;
    return ck(cudaDeviceSynchronize(), "slots", err);
}

bool PeerTier::fill(int32_t slot, const uint8_t* src, uint64_t bytes) {
    const core::OnDevice on(device_);
    return cudaMemcpy(cache_.device_slot(slot), src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool PeerTier::launch(const kernels::NativeExpertLayout& L, float limit, const Share& share, std::string& err) {
    if (share.n == 0) return true;
    const core::OnDevice on(device_);
    Plan& pl = *h_plan_;
    for (int i = 0; i < share.n; ++i) {
        pl.ptr[i] = (unsigned long long) share.blob[i];
        pl.start[i] = i;
        pl.dst[i] = i;
        pl.tok[i] = 0;
        pl.row[i] = share.row[i];
    }
    pl.start[share.n] = share.n;
    pl.ng[0] = share.n;
    if (!ck(cudaMemcpyAsync(d_x_, h_x_, (size_t) n_embd_ * sizeof(float), cudaMemcpyHostToDevice, stream_), "activation", err) ||
        !ck(cudaMemcpyAsync(d_plan_, h_plan_, sizeof(Plan), cudaMemcpyHostToDevice, stream_), "plan", err))
        return false;
    kernels::quantize_q8_1_rows(d_x_, 1, n_embd_, d_xq_, stream_);
    kernels::native_expert_grouped_clamped(L, limit, d_plan_->ptr, d_plan_->start, d_plan_->ng, d_plan_->dst, d_plan_->tok,
                                           share.n, share.n, d_xq_, d_scr_, d_out_, stream_);
    kernels::scatter_rows_f32(d_out_, d_rows_, d_plan_->row, share.n, n_embd_, stream_);
    launched_ = true;
    return true;
}

bool PeerTier::finish(std::string& err) {
    if (!launched_) return true;
    launched_ = false;
    const core::OnDevice on(device_);
    // spin: a blocking wait sleeps this thread and wakes it late
    cudaError_t e;
    while ((e = cudaStreamQuery(stream_)) == cudaErrorNotReady) {
    }
    return ck(e, "a layer's share", err);
}

void PeerTier::read(const uint8_t* blob, uint8_t* dst, uint64_t bytes) {
    const core::OnDevice on(device_);
    cudaMemcpyAsync(dst, blob, bytes, cudaMemcpyDeviceToHost, read_);
}

bool PeerTier::read_wait(std::string& err) {
    const core::OnDevice on(device_);
    return ck(cudaStreamSynchronize(read_), "reading experts", err);
}

}  // namespace basalt::models::deepseek41
