// src/vision/encoder_region.cpp - see basalt/vision/encoder_region.hpp.
#include "basalt/vision/encoder_region.hpp"

#include <cuda_runtime.h>

namespace basalt::vision {

bool EncoderRegion::open(CudaEncoder& enc, int tokens, std::string& err) {
    drop();
    enc_ = &enc;
    const size_t bytes = enc.arena_bytes(tokens);
    if (lender_.room) {
        lent_ = true;
        if (!lender_.room(bytes, err)) return false;
    }
    if (cudaMalloc(&region_, bytes) != cudaSuccess) {
        cudaGetLastError();
        region_ = nullptr;
        err = "no device memory for the image encoder (" + std::to_string((unsigned long long) (bytes >> 20)) + " MiB)";
        return false;
    }
    if (!enc.attach(region_, bytes, tokens, err)) {
        drop();
        return false;
    }
    tokens_ = tokens;
    return true;
}

void EncoderRegion::drop() {
    if (!region_) return;
    enc_->detach();
    cudaFree(region_);
    region_ = nullptr;
    tokens_ = 0;
}

void EncoderRegion::release() {
    drop();
    if (!lent_) return;
    lent_ = false;
    lender_.give_back();
}

}  // namespace basalt::vision
