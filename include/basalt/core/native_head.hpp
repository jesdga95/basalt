#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

/// Opt-in native output projection. Uploads unchanged GGUF blocks once at startup and owns its
/// activation scratch. The normal canonical head remains available for numerical A/B comparisons.
/// Plan v0.3 P6: any type the native MMVQ takes (Q5_K in the Q2_0 / IQ3_XXS files, IQ4_XS in IQ2_XS).
/// `load` takes the model's shards (basalt::gguf_split_paths): the architecture is checked on the first, the
/// tensor is read from the shard that holds it.
class NativeHead {
public:
    NativeHead() = default;
    ~NativeHead();
    NativeHead(const NativeHead&) = delete;
    NativeHead& operator=(const NativeHead&) = delete;

    bool load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out, std::string& err);
    /// The bytes `load` uploads for `output.weight`, from the GGUF headers only.
    static bool plan_bytes(const std::vector<std::string>& shards, uint64_t& out, std::string& err);
    bool run(const float* mixed, float* logits, void* stream, std::string& err) const;
    uint64_t weight_bytes() const { return bytes_; }
    bool loaded() const { return weights_ != nullptr; }
    /// Plan v0.3 P6: the GGUF blocks, for the verify window's multi-column head.
    const void* weights() const { return weights_; }
    int type() const { return type_; }
    /// Bytes of one vocabulary row.
    size_t row_bytes() const { return n_out_ > 0 ? (size_t) (bytes_ / (uint64_t) n_out_) : 0; }
    /// A Q6_K head's 2-bit copy for the verify window's two-stage head (q6k_vhead); null for other types or when it
    /// did not fit.
    const void* vproxy() const { return vproxy_; }

private:
    void* weights_ = nullptr;
    void* vproxy_ = nullptr;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    int n_in_ = 0, n_out_ = 0;
    int type_ = -1;
};

/// Plan v0.3 P6: `token_embd.weight` in its GGUF form (the IQ model files), in mapped pinned host memory: a row
/// is read over PCIe per token.  The rows of the lowest ids (the byte-level BPE gives the most frequent tokens the
/// lowest ids) also have a VRAM copy, `kHotRows` of them, which the gathers read instead: it is allocated before the
/// expert cache is sized, so it comes out of that cache.  `embed_row`, the verify window and the MTP drafter use it
/// instead of the canonical table when it is set (`set_native_embed`).
class NativeEmbed {
public:
    NativeEmbed() = default;
    ~NativeEmbed();
    NativeEmbed(const NativeEmbed&) = delete;
    NativeEmbed& operator=(const NativeEmbed&) = delete;
    bool load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab, std::string& err);
    /// Rows for device token ids.
    void gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const;
    /// One row for a host token id.
    void gather_one(int64_t token, float* out, void* stream) const;
    uint64_t bytes() const { return bytes_; }
    int type() const { return type_; }
    /// Rows with a VRAM copy (0: none), and its bytes.
    int64_t hot_rows() const { return n_hot_; }
    uint64_t hot_bytes() const { return (uint64_t) n_hot_ * row_; }
    static constexpr int64_t kHotRows = 8192;

private:
    void* host_ = nullptr;
    const void* dev_ = nullptr;
    void* hot_ = nullptr;
    int hot_device_ = -1;   ///< the device that holds hot_; the other devices read every row from the mapped table
    int64_t n_hot_ = 0;
    uint64_t bytes_ = 0;
    size_t row_ = 0;
    int64_t n_embd_ = 0, n_vocab_ = 0;
    int type_ = -1;
};
void set_native_embed(const NativeEmbed* e);
const NativeEmbed* native_embed();

}  // namespace basalt::core
