// src/models/glm5next/cache_buf.cpp - allocating and addressing the three caches.
#include "models/glm5next/cache_buf.hpp"

#include "basalt/log.hpp"

namespace basalt::models::glm5next {

namespace {
bool alloc_f32(float** out, size_t floats, const char* what, std::string& err) {
    const cudaError_t e = cudaMalloc(reinterpret_cast<void**>(out), floats * sizeof(float));
    if (e != cudaSuccess) {
        err = std::string("out of device memory for the ") + what + " cache";
        return false;
    }
    return true;
}
}  // namespace

GlmCaches::~GlmCaches() { free(); }

bool GlmCaches::alloc(const CacheDims& d, uint64_t context, int n_seq, std::string& err) {
    free();
    if (n_seq <= 0 || context == 0 || (d.n_kda == 0 && d.n_mla == 0)) {
        err = "glm5next: cache alloc with an empty geometry";
        return false;
    }
    dims_ = d;
    context_ = context;
    n_seq_ = n_seq;

    state_stride_ = (size_t) d.n_head * d.head_dim * d.head_dim;
    conv_stride_ = 3ull * (d.conv > 0 ? d.conv - 1 : 0) * d.n_head * d.head_dim;
    mla_stride_ = (size_t) context * d.kv_lora;
    // one pool slot at least, so the accessor is never a null pointer on a context shorter than a pool
    const size_t pools = d.n_pools(context) ? (size_t) d.n_pools(context) : 1;
    pooled_stride_ = pools * d.idx_dim;
    pend_stride_ = (size_t) kIndexerPendingRows * (d.kpool > 1 ? d.kpool - 1 : 0) * d.idx_dim;

    const size_t n = (size_t) n_seq;
    const size_t f_state = n * d.n_kda * state_stride_;
    const size_t f_conv = n * d.n_kda * conv_stride_;
    const size_t f_latent = n * d.n_mla * mla_stride_;
    dlat_stride_ = mla_stride_;   // same rows, its own block
    const size_t f_dlat = n * (size_t) d.n_mla_draft * dlat_stride_;
    const size_t f_pooled = n * d.n_idx * pooled_stride_;
    const size_t f_pend = n * d.n_idx * pend_stride_;
    if (!alloc_f32(&state_, f_state, "KDA state", err)) goto fail;
    if (!alloc_f32(&conv_, f_conv, "KDA conv", err)) goto fail;
    if (!alloc_f32(&latent_, f_latent, "MLA latent", err)) goto fail;
    if (f_dlat && !alloc_f32(&dlat_, f_dlat, "drafter latent", err)) goto fail;
    if (!alloc_f32(&pooled_, f_pooled, "indexer pooled keys", err)) goto fail;
    if (f_pend && !alloc_f32(&pend_, f_pend, "indexer pending rows", err)) goto fail;

    bytes_ = (uint64_t)(f_state + f_conv + f_latent + f_dlat + f_pooled + f_pend) * sizeof(float);

    // a fresh sequence: the recurrent state and conv history start empty
    cudaMemset(state_, 0, f_state * sizeof(float));
    if (conv_stride_) cudaMemset(conv_, 0, f_conv * sizeof(float));
    return true;

fail:
    free();
    return false;
}

void GlmCaches::free() {
    cudaFree(state_);
    cudaFree(conv_);
    cudaFree(latent_);
    cudaFree(dlat_);
    cudaFree(pooled_);
    cudaFree(pend_);
    state_ = conv_ = latent_ = dlat_ = pooled_ = pend_ = nullptr;
    context_ = 0;
    n_seq_ = 0;
    bytes_ = 0;
}

void GlmCaches::reset(int seq, cudaStream_t stream) {
    if (!state_ || seq < 0 || seq >= n_seq_) return;
    const size_t f_state = (size_t) dims_.n_kda * state_stride_;    // one sequence's state, all its KDA layers
    const size_t f_conv = (size_t) dims_.n_kda * conv_stride_;
    if (cudaMemsetAsync(state_ + (size_t) seq * f_state, 0, f_state * sizeof(float), stream) != cudaSuccess ||
        (f_conv && cudaMemsetAsync(conv_ + (size_t) seq * f_conv, 0, f_conv * sizeof(float), stream) != cudaSuccess)) {
        basalt::log::warn("basalt glm5next caches", "reset of sequence %d did not run: %s", seq,
                          cudaGetErrorString(cudaGetLastError()));
    }
}

float* GlmCaches::kda_state(int seq, int kda_layer) const {
    return state_ + ((size_t) seq * dims_.n_kda + kda_layer) * state_stride_;
}
float* GlmCaches::kda_conv(int seq, int kda_layer) const {
    return conv_ + ((size_t) seq * dims_.n_kda + kda_layer) * conv_stride_;
}
float* GlmCaches::mla_latent(int seq, int mla_layer) const {
    return latent_ + ((size_t) seq * dims_.n_mla + mla_layer) * mla_stride_;
}
float* GlmCaches::draft_latent(int seq, int draft_mla_layer) const {
    return dlat_ + ((size_t) seq * (size_t) dims_.n_mla_draft + draft_mla_layer) * dlat_stride_;
}
float* GlmCaches::idx_pooled(int seq, int idx_layer) const {
    return pooled_ + ((size_t) seq * dims_.n_idx + idx_layer) * pooled_stride_;
}
float* GlmCaches::idx_pending(int seq, int idx_layer) const {
    return pend_ + ((size_t) seq * dims_.n_idx + idx_layer) * pend_stride_;
}

}  // namespace basalt::models::glm5next
