#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/qwen4exp/kernels/q6k_head.hpp"
#include "basalt/models/qwen4exp/kernels/q80_vhead.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda_runtime.h>
#include <climits>
#include <cstdio>
#include <cstring>
#include <exception>
#include "basalt/log.hpp"

namespace basalt::core {

NativeHead::~NativeHead() {
    if (vproxy_) cudaFree(vproxy_);
    if (scratch_) cudaFree(scratch_);
    if (weights_) cudaFree(weights_);
}

bool NativeHead::plan_bytes(const std::vector<std::string>& shards, uint64_t& out, std::string& err) {
    try {
        const basalt::GgufModel model(shards);
        const basalt::TensorInfo* tensor = model.find("output.weight");
        if (!tensor || !basalt::kernels::native_mmvq_supported((int) tensor->type) || tensor->shape.size() != 2) {
            err = "native head: expected a natively supported output.weight";
            return false;
        }
        out = basalt::kernels::native_mmvq_weight_bytes((int) tensor->type, (int) tensor->shape[0],
                                                        (int) tensor->shape[1]);
        if (tensor->type == 14 && basalt::kernels::q6k_vhead_supported((int) tensor->shape[0], (int) tensor->shape[1], 1))
            out += basalt::kernels::q6k_vproxy_bytes((int64_t) tensor->shape[1]);
        return true;
    } catch (const std::exception& error) {
        err = std::string("native head: ") + error.what();
        return false;
    }
}

bool NativeHead::load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out, LoadPlan& reads, std::string& err) {
    if (loaded()) { err = "native head is already loaded"; return false; }
    if (n_in <= 0 || n_out <= 0 || n_in > INT_MAX || n_out > INT_MAX || n_in % 256) {
        err = "native head requires positive int32 dimensions and whole 256-value rows";
        return false;
    }
    try {
        // The architecture is the metadata shard's; output.weight comes from whichever shard holds it (shard 2
        // of Unsloth's UD-Q4_K_XL, whose shard 1 holds no tensor).  GgufModel refuses a duplicate across shards.
        const basalt::GgufModel model(shards);
        err = basalt::model::check_architecture(model.meta());
        if (!err.empty()) return false;
        size_t at = 0;
        const basalt::TensorInfo* tensor = model.find("output.weight", &at);
        const basalt::GgufFile& gguf = model.shard(at);
        if (!tensor || !basalt::kernels::native_mmvq_supported((int) tensor->type) || tensor->shape.size() != 2 ||
            tensor->shape[0] != (uint64_t) n_in || tensor->shape[1] != (uint64_t) n_out) {
            err = "native head: expected a natively supported output.weight with the canonical head dimensions";
            return false;
        }
        const uint64_t bytes = basalt::kernels::native_mmvq_weight_bytes((int) tensor->type, (int) n_in, (int) n_out);
        const uint64_t payload = gguf.file_size() - gguf.data_start();
        if (tensor->offset > payload || bytes > payload - tensor->offset) {
            err = "native head: truncated output.weight payload";
            return false;
        }
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
        void* weights = nullptr;
        void* scratch = nullptr;
        cudaError_t status = cudaMalloc(&weights, bytes);
        if (status == cudaSuccess)
            status = cudaMalloc(&scratch, basalt::kernels::native_q8_1_bytes((int) n_in, 1));
        if (status != cudaSuccess) {
            if (scratch) cudaFree(scratch);
            if (weights) cudaFree(weights);
            err = std::string("native head upload: ") + cudaGetErrorString(status);
            return false;
        }
        LoadItem it;   // read in the caller's pass; finish() rewrites the rows once they have landed
        it.file = reads.file(gguf.path());
        it.off = gguf.data_start() + tensor->offset;
        it.bytes = bytes;
        it.dev = weights;
        it.device = dev;
        reads.add(it);
        weights_ = weights;
        scratch_ = scratch;
        bytes_ = bytes;
        n_in_ = (int) n_in;
        n_out_ = (int) n_out;
        type_ = (int) tensor->type;
        // the verify head's 2-bit copy (packed by finish(); without it the exact head runs), held now with the rows
        if (type_ == 14 && basalt::kernels::q6k_vhead_supported((int) n_in, (int) n_out, 1)) {
            if (cudaMalloc(&vproxy_, basalt::kernels::q6k_vproxy_bytes(n_out)) != cudaSuccess) vproxy_ = nullptr;
            cudaGetLastError();
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native head: ") + error.what();
        return false;
    }
}

bool NativeHead::finish(std::string& err) {
    if (!loaded()) return true;
    // a Q8_0 head in bit planes, rewritten in place: the same bytes per row, read by every kernel of this head
    if (type_ == 8 && basalt::kernels::q80p_supported(n_in_, 1)) {
        basalt::kernels::q80p_pack(weights_, n_out_, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            err = "native head: the bit planes failed";
            return false;
        }
        planes_ = true;
    }
    if (vproxy_ != nullptr) {   // packed once from the landed rows
        basalt::kernels::q6k_vproxy_pack(weights_, n_out_, vproxy_, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(vproxy_);
            vproxy_ = nullptr;
        }
        cudaGetLastError();
    }
    return true;
}

bool NativeHead::run(const float* mixed, float* logits, void* stream, std::string& err) const {
    if (!loaded() || !mixed || !logits || !stream) {
        err = "native head requires loaded weights, device buffers and an explicit stream";
        return false;
    }
    try {
        if (type_ == 13) {
            basalt::kernels::native_q5_k_f32(weights_, mixed, scratch_, logits, n_in_, n_out_, 1, stream);
        } else if (planes_) {
            basalt::kernels::native_quantize_q8_1(mixed, scratch_, n_in_, 1, stream);
            basalt::kernels::q80p_mmvq(weights_, scratch_, logits, n_in_, n_out_, 1, stream);
        } else {
            basalt::kernels::native_quantize_q8_1(mixed, scratch_, n_in_, 1, stream);
            basalt::kernels::native_mmvq(type_, weights_, scratch_, logits, n_in_, n_out_, 1, stream);
        }
    } catch (const std::exception& error) {
        err = std::string("native head launch: ") + error.what();
        return false;
    }
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        err = std::string("native head launch: ") + cudaGetErrorString(status);
        return false;
    }
    return true;
}

// ================================ THE NATIVE EMBEDDING ================================

namespace {
const NativeEmbed* g_embed = nullptr;
}
void set_native_embed(const NativeEmbed* e) { g_embed = e; }
const NativeEmbed* native_embed() { return g_embed; }

NativeEmbed::~NativeEmbed() {
    if (hot_) cudaFree(hot_);
    if (host_) cudaFreeHost(host_);
    else if (dev_) cudaFree(const_cast<void*>(dev_));   // the VRAM fallback below
}

bool NativeEmbed::load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab, LoadPlan& reads, std::string& err) {
    try {
        const basalt::GgufModel model(shards);
        err = basalt::model::check_architecture(model.meta());
        if (!err.empty()) { err = "native embedding: " + err; return false; }
        size_t at = 0;
        const basalt::TensorInfo* t = model.find("token_embd.weight", &at);
        const basalt::GgufFile& gguf = model.shard(at);
        if (!t || t->shape.size() != 2 || t->shape[0] != (uint64_t) n_embd || t->shape[1] != (uint64_t) n_vocab ||
            !basalt::kernels::embed_type_supported((int) t->type) || n_embd % 256) {
            err = "native embedding: token_embd.weight is absent, of another shape, or of a type without a GPU "
                  "dequantizer";
            return false;
        }
        row_ = basalt::kernels::iq_row_bytes((int) t->type, n_embd);
        bytes_ = (uint64_t) row_ * (uint64_t) n_vocab;
        // the table is copied out of the mapping below: a truncated shard must be an error, not a read past EOF
        if (!model.in_bounds(*t, at) || basalt::tensor_payload_bytes(*t) != bytes_) {
            err = "native embedding: token_embd.weight's payload is truncated or not " + std::to_string(bytes_) +
                  " B (" + gguf.path() + ")";
            bytes_ = 0;
            return false;
        }
        LoadItem it;   // the table, read in the caller's pass
        it.file = reads.file(gguf.path());
        it.off = gguf.data_start() + t->offset;
        it.bytes = bytes_;
        if (cudaHostAlloc(&host_, bytes_, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            // Under WSL2 the driver's pinned/mapped host budget (~1 GiB) can be spent by the GPU contexts
            // themselves (three cards). The table is only gathered from, so keep it in the current device's VRAM
            // instead: it costs its size there and reads faster than over PCIe.
            cudaGetLastError();
            host_ = nullptr;
            void* d = nullptr;
            if (cudaMalloc(&d, bytes_) != cudaSuccess) {
                cudaGetLastError();
                err = "native embedding: cannot pin " + std::to_string(bytes_ >> 20) + " MiB, nor place it in VRAM";
                return false;
            }
            it.dev = d;
            cudaGetDevice(&it.device);
            reads.add(it);
            basalt::log::warn("basalt", "native embedding: cannot pin %llu MiB, kept in VRAM instead",
                         (unsigned long long) (bytes_ >> 20));
            dev_ = d;
        } else {
            it.host = (uint8_t*) host_;
            reads.add(it);
            void* d = nullptr;
            if (cudaHostGetDevicePointer(&d, host_, 0) != cudaSuccess) {
                err = "native embedding: no device alias for the mapped table";
                return false;
            }
            dev_ = d;
            // the hot rows: without the VRAM for them every row stays on the mapped path
            const int64_t n_hot = kHotRows < n_vocab ? kHotRows : n_vocab;
            if (cudaMalloc(&hot_, (size_t) n_hot * row_) == cudaSuccess) {
                n_hot_ = n_hot;
                cudaGetDevice(&hot_device_);
                LoadItem hot;   // copied from the table once it has landed
                hot.from = (const uint8_t*) host_;
                hot.bytes = (uint64_t) n_hot * row_;
                hot.dev = hot_;
                hot.device = hot_device_;
                hot.count = 0;
                reads.add(hot);
            } else {
                if (hot_) cudaFree(hot_);
                hot_ = nullptr;
                cudaGetLastError();
            }
        }
        type_ = (int) t->type;
        n_embd_ = n_embd;
        n_vocab_ = n_vocab;
        return true;
    } catch (const std::exception& e) {
        err = std::string("native embedding: ") + e.what();
        return false;
    }
}

void NativeEmbed::gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const {
    int dev = 0;
    cudaGetDevice(&dev);
    const bool hot = dev == hot_device_;
    basalt::kernels::iq_embed_rows(type_, dev_, row_, tokens, n_tok, n_embd_, out, stream, hot ? hot_ : nullptr,
                                   hot ? n_hot_ : 0);
}

void NativeEmbed::gather_one(int64_t token, float* out, void* stream) const {
    int dev = 0;
    cudaGetDevice(&dev);
    const uint8_t* table = (const uint8_t*) (token < n_hot_ && dev == hot_device_ ? hot_ : dev_);
    basalt::kernels::iq_dequant_f32(type_, table + (size_t) token * row_, n_embd_, out, stream);
}

}  // namespace basalt::core
