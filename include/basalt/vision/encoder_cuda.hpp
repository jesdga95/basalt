// basalt/vision/encoder_cuda.hpp - the vision encoder on one CUDA device.
//
// BF16 weights stay on the device as stored; the activations between the products are fp32 (the residual stream) or
// bf16 (the product inputs), attention runs on fp16 tensor cores with fp32 softmax state.  Work buffers are sized
// for `max_tokens` merged tokens at init, so no allocation happens after the encoder is ready.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "basalt/vision/encoder_cpu.hpp"

namespace basalt::vision {

class CudaEncoder {
public:
    CudaEncoder();
    ~CudaEncoder();
    CudaEncoder(const CudaEncoder&) = delete;
    CudaEncoder& operator=(const CudaEncoder&) = delete;

    /// Uploads the weights of `mp` to the current device and sizes the buffers for `max_tokens` merged tokens.
    bool init(const Mmproj& mp, int max_tokens, std::string& err);

    /// The lent-memory form of `init`: `stage` copies the weights into one host image (page-locked when `pinned`, so
    /// the upload runs at link speed), `create` makes the stream and product handle on the current device, and
    /// `attach` places the weights and the work buffers for `max_tokens` merged tokens in the device region
    /// [arena, arena + bytes) (at least `arena_bytes(max_tokens)`) and uploads the weights.  `detach` waits for the
    /// stream and forgets the region; the encoder can attach to another region (or this one again) afterwards.
    bool stage(const Mmproj& mp, bool pinned, std::string& err);
    bool create(std::string& err);
    size_t weights_bytes() const;
    size_t arena_bytes(int max_tokens) const;
    bool attach(void* arena, size_t bytes, int max_tokens, std::string& err);
    void detach();

    /// As CpuEncoder::encode; the taps copy to the host and are for parity runs only.
    bool encode(const std::vector<float>& patches, Size canvas, std::vector<float>& out, const TapFn& tap = {});

    /// Device bytes held by the weights and the work buffers.
    size_t device_bytes() const;
    std::string last_error() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

/// Launches one empty kernel on the current device. False with CUDA's own text when this binary holds no code for
/// the card it runs on: the way a build made for another compute capability reports itself at startup.
bool probe_device(std::string& err);

}  // namespace basalt::vision
