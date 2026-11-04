// include/basalt/model/adapter.hpp - the seam between the shared engine and one model.
//
// Shared code (core, kernels, tiers, cache, server) sees a model only through `ModelDesc` and this interface.
// Each model lives in src/models/<arch>/ and registers an adapter under its architecture name. Shared code
// never includes a model's headers, and models never include each other (tools/check_model_layering.sh).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace basalt {
class GgufFile;
}

namespace basalt::model {

/// Geometry of the model as shared code needs it. A model fills what it has; the rest stays zero.
struct ModelDesc {
    std::string arch;
    int64_t n_layers = 0;
    int64_t n_embd = 0;
    int64_t n_vocab = 0;
    int64_t n_expert = 0;        ///< routed experts per layer
    int64_t n_expert_used = 0;   ///< router top-k
    int64_t n_ff_exp = 0;        ///< routed expert hidden width
    int64_t n_expert_shared = 0;
    int64_t context_length = 0;  ///< the longest context the model supports (0: not stated)
    bool rope_from_model = false; ///< the rotary scaling comes from the model file only: the --rope-* flags are refused
    /// ggml type id of the gate/up/down blobs of one routed expert, per layer (empty: read from the tensors).
    std::vector<int32_t> expert_gate_up_type;
    std::vector<int32_t> expert_down_type;
};

/// A per-token side table read from storage (a per-layer embedding or an Engram table).
struct SideTableSpec {
    enum class Kind { Ple, Engram };
    Kind kind = Kind::Ple;
    std::string tensor;          ///< tensor name in the pack
    int64_t row_values = 0;
    int64_t rows = 0;
};

/// Tokenizer source: where the vocabulary lives and which chat template family the server uses.
struct TokenizerSpec {
    std::string vocab_key;       ///< metadata key holding the token list
    std::string chat_format;     ///< server-side chat format name
};

/// Outcome of a model operation. `UseEnginePath` means the call is served by the built-in engine path that
/// predates the adapter seam; the caller continues there unchanged.
enum class Status { Ok, NotImplemented, UseEnginePath, Error };

struct Result {
    Status status = Status::Ok;
    std::string message;
    bool ok() const { return status == Status::Ok; }
};

struct LoadRequest {
    std::vector<std::string> paths;   ///< pack or GGUF shards
    std::string diag_fixture;         ///< diagnostic: recorded outputs to replay (model-specific, empty = off)
    std::string diag_dump;            ///< diagnostic: directory for per-layer tensor dumps (empty = off)
    std::string diag_mmproj;          ///< diagnostic: image projector file used instead of the pack's vision section
    int diag_tier_ram = -1;           ///< diagnostic: fixed RAM-tier experts per layer (-1 = sized from free memory)
    int diag_tier_vram = -1;          ///< diagnostic: fixed VRAM-tier experts per layer (-1 = sized from free memory)
    std::string diag_dspark;          ///< diagnostic: drafter GGUF to load on the second GPU and run beside decode (empty = off)
    int64_t max_context = 0;          ///< --max-context (used when max_context_set; else the model picks its own)
    bool max_context_set = false;
    int spec = 0;                     ///< --spec: decode window size; 2 or more asks the model to load its drafter
};

struct DecodeRequest {
    const int32_t* tokens = nullptr;  ///< window of tokens, row 0 is the next position
    int n_tokens = 0;
    int64_t pos = 0;
    int32_t* next_token = nullptr;    ///< greedy token after the window's last row (optional)
    float* logits = nullptr;          ///< n_vocab logits after the window's last row (optional); a window of several rows fills n_tokens rows
    int32_t* row_tokens = nullptr;    ///< greedy token after each of the n_tokens rows (optional)
};

struct PrefillRequest {
    const int32_t* tokens = nullptr;
    int n_tokens = 0;
    int64_t pos = 0;
    std::function<void(int)> progress;   ///< optional: called with the tokens of this request read so far
    std::function<void(int)> live;       ///< optional: an estimate of the same, from a driver thread, inside a chunk
};

/// Model-owned per-conversation state (recurrent state, KV, windows).
class ModelSession {
public:
    virtual ~ModelSession() = default;
    virtual Result decode_window(const DecodeRequest& r) = 0;
    virtual Result prefill_chunk(const PrefillRequest& r) = 0;
    /// Drafts the tokens after `anchor`, the next token to feed at `pos` (up to `max_n`, with the draft's confidence in
    /// [0,1] each). Features of the rows of a decode window before `pos` are folded into the drafter first.
    virtual Result draft(int32_t anchor, int64_t pos, int32_t* ids, float* conf, int max_n, int* n_out) {
        (void) anchor; (void) pos; (void) ids; (void) conf; (void) max_n; (void) n_out;
        return {Status::NotImplemented, "this model has no drafter"};
    }
    /// An encoded picture: `n_rows` embedding rows of the model's width (the encoder's whole block). Rows with no
    /// vision support leave the default.
    virtual Result encode_image(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows) {
        (void) data; (void) bytes; (void) rows; (void) n_rows;
        return {Status::NotImplemented, "this model takes no images"};
    }
    /// Feeds `n_rows` embedding rows (from encode_image) at positions pos.. instead of token embeddings.
    virtual Result prefill_embeddings(const float* rows, int n_rows, int64_t pos) {
        (void) rows; (void) n_rows; (void) pos;
        return {Status::NotImplemented, "this model takes no images"};
    }
};

/// A loaded model: weights resolved, ready to make sessions.
class LoadedModel {
public:
    virtual ~LoadedModel() = default;
    virtual const ModelDesc& desc() const = 0;
    /// The context the loaded model runs with, when it sizes that itself (0: the caller's --max-context stands).
    virtual int64_t context() const { return 0; }
    virtual Result make_session(std::unique_ptr<ModelSession>& out) = 0;
};

/// Optional drafter (speculative decoding). Opaque to shared code until a second model needs one.
class Drafter {
public:
    virtual ~Drafter() = default;
};

class ModelAdapter {
public:
    virtual ~ModelAdapter() = default;
    /// Architecture name: the pack `meta.arch` and the GGUF `general.architecture`.
    virtual const char* arch() const = 0;
    /// Refuses a file that is not this model; empty string when it is.
    virtual std::string validate(const GgufFile& meta) const = 0;
    /// Geometry from the file's metadata.
    virtual ModelDesc describe(const GgufFile& meta) const = 0;
    virtual Result load(const LoadRequest& req, std::unique_ptr<LoadedModel>& out) const = 0;
    /// Null when the model has no drafter.
    virtual std::unique_ptr<Drafter> drafter() const { return nullptr; }
    virtual std::vector<SideTableSpec> side_tables(const ModelDesc&) const { return {}; }
    virtual TokenizerSpec tokenizer_spec() const = 0;
};

/// Registry of adapters by architecture name. Registration is explicit (`models::register_builtin_models()`
/// from the program's main), so linking order never decides what exists.
void register_adapter(std::unique_ptr<ModelAdapter> a);
/// Null when no adapter has that architecture.
const ModelAdapter* find_adapter(const std::string& arch);
std::vector<std::string> adapter_names();
/// The adapter for a file's `general.architecture`, or null with `err` set.
const ModelAdapter* adapter_for(const GgufFile& meta, std::string& err);
/// Replaces the old architecture guard: empty when the file belongs to a registered model and validates.
std::string check_architecture(const GgufFile& meta);

}  // namespace basalt::model
