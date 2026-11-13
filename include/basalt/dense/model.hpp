// include/basalt/dense/model.hpp - a dense decoder of hybrid layers: each layer a token mixer (gated delta-net or
// gated grouped-query attention) and a dense SwiGLU FFN, with offset RMS norms, an fp32 residual and an untied head.
//
// Shared code: a model folder reads its pack into `Geometry` and `Weights` (device pointers it owns) and gets
// sessions from `make_session`, which the shared session engine serves (parallel slots, conversation cache,
// drafting, grammar, logprobs, images).  Nothing here names a model; every width comes from the geometry.
//
// Products: decode windows (up to 8 rows) on q8_1 activations through the shared MMVQ (Q8_0, NVFP4); prompt chunks
// through llama.cpp's MMQ on int8 tensor cores.  An NVFP4 matrix's fp32 tensor scale is applied after its product.
#pragma once

#include "basalt/core/model_file.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/gdn_window.hpp"
#include "basalt/model/adapter.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::dense {

enum class Mixer { Gdn, Attention };

/// One projection: `rows` outputs of `cols` inputs, ggml blocks of `type` (Q8_0 8, NVFP4 40) on the device; `scale`
/// an NVFP4 matrix's fp32 tensor scale (device), null for 1.
struct Linear {
    int type = -1;
    const void* w = nullptr;
    int64_t rows = 0, cols = 0;
    const float* scale = nullptr;
};

struct Layer {
    Mixer kind = Mixer::Gdn;
    const float* attn_norm = nullptr;   ///< the mixer's input norm, offset folded (1 + w)
    const float* ffn_norm = nullptr;    ///< the FFN's
    // gated delta-net: value heads in the order where value head h meets key head h % k_heads
    Linear qkv, z, out;
    const uint16_t* w_ab = nullptr;     ///< bf16 [2 v_heads][n_embd]: the alpha rows, then the beta rows
    const float *dt = nullptr, *ssm_a = nullptr, *conv = nullptr, *gamma = nullptr;
    // attention: q gives [query | gate] per head
    Linear q, k, v, o;
    const float *q_norm = nullptr, *k_norm = nullptr;
    // FFN
    Linear gate, up, down;
};

/// The drafter layer a pack may carry (multi-token prediction): fc(concat(norm_e(embed(next)), norm_h(hidden))),
/// one attention layer with its FFN, a final norm, the main head.
struct Mtp {
    bool present = false;
    Linear fc;
    const float *norm_e = nullptr, *norm_h = nullptr, *norm = nullptr;
    Layer layer;
};

struct Geometry {
    int n_layers = 0, n_embd = 0, n_ff = 0, n_vocab = 0;
    int n_head = 0, n_kv = 0, head_dim = 0;
    kernels::RopeSpec rope;
    int gdn_state = 0, gdn_k_heads = 0, gdn_v_heads = 0, gdn_conv = 0;
    kernels::GdnGate gdn_gate = kernels::GdnGate::Silu;
    float eps = 1e-6f;
    std::vector<Mixer> kinds;
    int64_t context_length = 0;          ///< the longest context the model supports
    int gdn_layers() const;
    int attn_layers() const;
    int conv_channels() const { return 2 * gdn_k_heads * gdn_state + gdn_v_heads * gdn_state; }
    int value_width() const { return gdn_v_heads * gdn_state; }
    /// Empty when the shared kernels take this geometry, else why not.
    std::string check() const;
};

struct Weights {
    std::vector<Layer> layers;
    const uint16_t* embd = nullptr;     ///< bf16 [n_vocab][n_embd], device-visible (may be mapped host memory)
    const float* out_norm = nullptr;
    Linear head;
    Mtp mtp;
};

struct Options {
    int64_t max_context = 0;            ///< cells each session holds
    int sessions = 1;
    int spec = 0;                       ///< rows a decode window may hold (0 or 1: one)
    bool draft = false;                 ///< run the pack's drafter
    bool vision = false;                ///< requests may carry images
    core::FileRange mmproj;             ///< the vision projector (vision/mmproj)
    std::string arch;                   ///< the architecture name, for messages
};

class Model;
void destroy(Model* m);
struct ModelDeleter {
    void operator()(Model* m) const { destroy(m); }
};
using ModelPtr = std::unique_ptr<Model, ModelDeleter>;

/// The model's shared state: geometry, weights, and what its sessions use one at a time (scratch, the image encoder).
/// Sessions run one after another on the serving thread; each call leaves its stream idle.
ModelPtr make_model(const Geometry& g, const Weights& w, const Options& o, std::string& err);
std::unique_ptr<model::ModelSession> make_session(Model& m, std::string& err);
/// Device bytes one session takes (its KV cells, recurrent states and window rows).
uint64_t session_bytes(const Geometry& g, int64_t max_context, bool mtp);
/// Device bytes the model's shared scratch takes.
uint64_t shared_bytes(const Geometry& g, int64_t max_context);

}  // namespace basalt::dense
