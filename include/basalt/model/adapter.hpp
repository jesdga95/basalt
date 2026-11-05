// include/basalt/model/adapter.hpp - the seam between the shared engine and one model.
//
// Shared code (core, kernels, tiers, cache, server) sees a model only through `ModelDesc` and this interface.
// Each model lives in src/models/<arch>/ and registers an adapter under its architecture name. Shared code
// never includes a model's headers, and models never include each other (tools/check_model_layering.sh).
#pragma once

#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/program/grammar/tool_form.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace basalt {
class GgufFile;
}
namespace basalt::core {
struct ConversationKv;
}
namespace basalt::program {
struct CommonOptions;
struct GpuCard;
struct ProbeEnv;
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

/// Outcome of a model operation.
enum class Status { Ok, NotImplemented, Error };

struct Result {
    Status status = Status::Ok;
    std::string message;
    bool ok() const { return status == Status::Ok; }
};

struct LoadRequest {
    std::vector<std::string> paths;   ///< pack or GGUF shards
    std::string diag_dump;            ///< diagnostic: directory for per-layer tensor dumps (empty = off)
    int diag_tier_ram = -1;           ///< diagnostic: fixed RAM-tier experts per layer (-1 = sized from free memory)
    int diag_tier_vram = -1;          ///< diagnostic: fixed VRAM-tier experts per layer (-1 = sized from free memory)
    std::vector<int> peer_devices;    ///< CUDA devices of the expert tier cards after the main one
    int64_t max_context = 0;          ///< --max-context, or the default every model shares
    int sessions = 1;                 ///< sessions the model will be asked for (--batch)
    std::string expert_profile;       ///< the routing order of the experts (empty: the file's own order)
    int expert_cache = 0;             ///< VRAM expert slots over all layers (0 or -1: from the free memory)
    uint64_t resident_budget = 0;     ///< bytes the RAM tier may take (0: from the free memory)
    int pool_workers = 0;             ///< CPU pool workers (0: the machine's physical cores)
    kernels::cpu::PoolAffinity pool_affinity = kernels::cpu::PoolAffinity::All;
    bool pool_affinity_given = false;
    int peer_reserve_mib = 600;       ///< VRAM a second card leaves free
    bool prompt_skip = true;          ///< a prompt's far rows may stop at the layer a model says is the last they matter to
    int vram_reserve_mib = 0;         ///< VRAM the main card leaves free beyond what the model needs (0: none)
    int spec = 0;                     ///< rows a decode window may hold (0 or 1: one)
    bool draft = false;               ///< run the drafter the pack carries, if any
    int drafter_device = -1;          ///< diagnostic: the CUDA device it runs on (-1: the main card)
    std::string profile_dir;          ///< the engine's data/profiles, where a model finds the files its name keys
    int draft_experts = 32;           ///< the drafter's routed experts each stage keeps (0: all)
    bool diag_tier_frozen = false;    ///< diagnostic: the VRAM tier keeps what it holds, no bus copy and no promotion
};

struct DecodeRequest {
    const int32_t* tokens = nullptr;  ///< window of tokens, row 0 is the next position
    int n_tokens = 0;
    int64_t pos = 0;
    int32_t* next_token = nullptr;    ///< greedy token after the window's last row (optional)
    /// The pick after each row (optional, [n_tokens]). The window's rows then stay until a later call starts at an earlier
    /// position, which drops those past it.
    int32_t* row_tokens = nullptr;
    float* logits = nullptr;          ///< n_vocab logits after the window's last row (optional)
    const kernels::SamplerParams* sampling = nullptr;   ///< how next_token is drawn (null: greedy)
    const int32_t* history = nullptr; ///< per row, the last `history_len` tokens the pick follows (host), for the penalties
    int history_len = 0;
};

struct PrefillRequest {
    const int32_t* tokens = nullptr;
    int n_tokens = 0;
    int64_t pos = 0;
    /// Positions between these tokens and the read's stop, -1 unknown. State is saved or resumed only at that stop: a model
    /// may leave rows further back short of its last layers.
    int64_t rows_after = -1;
    std::function<void(int)> progress;   ///< optional: called with the tokens of this request read so far
    std::function<void(int)> live;       ///< optional: an estimate of the same, from a driver thread, inside a chunk
    std::function<bool()> stop;          ///< optional: true ends the read at a chunk's end; nothing can continue from there
    int total_tokens = 0;                ///< what the whole prompt still has to read, this part included: sizes the workspaces
    bool more = false;                   ///< another part follows: the workspaces stay until end_prompt()
};

/// Model-owned per-conversation state (recurrent state, KV, windows).
class ModelSession {
public:
    virtual ~ModelSession() = default;
    virtual Result decode_window(const DecodeRequest& r) = 0;
    virtual Result prefill_chunk(const PrefillRequest& r) = 0;
    /// Takes the memory the session will hold (expert tiers), so the first request does not wait for it.
    virtual Result prepare() { return {}; }
    /// The prompt is read: workspaces a part kept with `more` go back.
    virtual void end_prompt() {}
    /// A logit bias (device copy, n_vocab floats) for the next draws; null clears it.
    virtual Result set_logit_bias(const float* bias) { (void) bias; return {Status::NotImplemented, "this model takes no logit bias"}; }
    /// Draws the token of row `row` of the last decoded window again under `bias` (n_vocab floats): the grammar's mask.
    virtual Result resample_row(int row, const float* bias, int32_t& token) { (void) row; (void) bias; (void) token; return {Status::NotImplemented, "this model cannot draw again"}; }
    /// The rows a decode window may hold; 1 when the session drafts nothing.
    virtual int max_window() const { return 1; }
    /// Starts the drafts after `anchor`, the token at position `pos`. The rows of the last window before `pos` were accepted.
    virtual Result draft_begin(int32_t anchor, int64_t pos) { (void) anchor; (void) pos; return {Status::NotImplemented, "this model has no drafter"}; }
    /// The drafts draft_begin started: `n` tokens and each one's estimated chance of acceptance, at most `max_n`.
    virtual Result draft_end(int32_t* ids, float* conf, int max_n, int& n) { (void) ids; (void) conf; (void) max_n; (void) n; return {Status::NotImplemented, "this model has no drafter"}; }
    /// The last decoded row's logits (n_vocab floats).
    virtual Result copy_logits(float* out) { (void) out; return {Status::NotImplemented, "this model keeps no logits"}; }
    /// low: the VRAM the session can give back goes back to the driver; else it is taken again. The slots it holds now.
    virtual Result set_low_memory(bool low, int64_t& expert_slots) { (void) low; (void) expert_slots; return {Status::NotImplemented, "this model cannot give VRAM back"}; }
    /// What the session leaves free on the main card from now on (`mib` < 0: the start's reserve), through prompts and
    /// images too; its VRAM use is resized to that now. The slots it holds now.
    virtual Result set_vram_reserve(int64_t mib, int64_t& expert_slots) { (void) mib; (void) expert_slots; return {Status::NotImplemented, "this model cannot give VRAM back"}; }
    /// The running state at position `pos`: what cannot be rewound. Positional state stays valid below it.
    virtual size_t running_bytes() const { return 0; }
    virtual Result save_running(uint8_t* host, int64_t pos) { (void) host; (void) pos; return {Status::NotImplemented, "this model keeps no checkpoint"}; }
    virtual Result restore_running(const uint8_t* host, int64_t pos) { (void) host; (void) pos; return {Status::NotImplemented, "this model keeps no checkpoint"}; }
    /// The positional state below `positions`, one entry per layer that holds any, in the store's K/V form: a layer's
    /// `page_size` is the positions behind one of its rows, `cells` its rows times that.  What a conversation parked
    /// out of VRAM takes along; a store cuts it at any position, so restore takes what a cut left.
    virtual size_t cells_bytes(int64_t positions) const { (void) positions; return 0; }
    virtual Result save_cells(std::vector<core::ConversationKv>& kv, int64_t positions) { (void) kv; (void) positions; return {Status::NotImplemented, "this model keeps no cells"}; }
    virtual Result restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions) { (void) kv; (void) positions; return {Status::NotImplemented, "this model keeps no cells"}; }
    /// The tokens the session holds from now on, position by position (a restored conversation's).
    virtual Result set_history(const int32_t* tokens, int64_t n) { (void) tokens; (void) n; return {}; }
    /// What `PROBE rates` measures this model's paths with (after prepare).
    virtual Result probe_env(program::ProbeEnv& env) { (void) env; return {Status::NotImplemented, "this model has no probe"}; }
    /// Conversations shorter than this are cheaper to read again than to keep on disk.
    virtual int64_t disk_min_tokens() const { return 8192; }
    /// The routed experts of decode so far: `lookups` the main card or the CPU served, `hits` the main card's share of
    /// those, `offloaded` what a bus copy or another card served, `files` those of `lookups`
    /// read from the file.
    struct ExpertCounts { int64_t hits = 0, lookups = 0, offloaded = 0, files = 0; };
    virtual ExpertCounts expert_counts() const { return {}; }
    /// Facts for the engine's INFO lines once the session is prepared: `key=value` words.
    virtual std::string info_words() const { return {}; }
    /// An encoded picture: `n_rows` embedding rows of the model's width (the encoder's whole block). Rows with no
    /// vision support leave the default.
    virtual Result encode_image(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows) {
        (void) data; (void) bytes; (void) rows; (void) n_rows;
        return {Status::NotImplemented, "this model takes no images"};
    }
    /// A request's pictures are encoded: the encoder gives its memory back.
    virtual void images_done() {}
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

/// The texts of the model's own tokens in the pack's vocabulary: the defaults of --eos-ids, --turn-token, --image-pad-id.
struct TokenNames {
    std::vector<std::string> eos;
    std::string turn;
    std::string image_pad;
};

class FlagArgs;

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
    virtual std::vector<SideTableSpec> side_tables(const ModelDesc&) const { return {}; }
    virtual TokenizerSpec tokenizer_spec() const = 0;
    /// A model with its own start and serving runs them here to the exit code; nullopt: the shared server runs load()'s sessions.
    virtual std::optional<int> run(program::CommonOptions& o, std::vector<program::GpuCard>& cards) const {
        (void) o; (void) cards;
        return std::nullopt;
    }
    /// The verify window a start with these flags runs (0: plain decode), once finish_flags() has run.
    virtual int planned_spec(const program::CommonOptions& o) const;
    /// The flags the quality gate starts this model with, space separated.
    virtual const char* quality_flags() const { return ""; }
    /// The widest window the gate scores on this model; 0 takes what the request asks.
    virtual int quality_window() const { return 1; }
    /// The cards that may hold experts beside the main one.
    virtual int max_expert_cards() const = 0;
    virtual TokenNames token_names() const = 0;
    /// The pre-tokenizer's name in the pack's tokenizer.json and its rule.
    struct PreTokenizer {
        const char* name = "";
        void (*split)(const char* text, size_t size, std::vector<size_t>& ends) = nullptr;
    };
    virtual PreTokenizer pretokenizer() const = 0;
    virtual program::grammar::ToolForm tool_form() const = 0;
    /// The options object of this model: the common flags and its own.
    virtual std::unique_ptr<program::CommonOptions> make_options() const;
    /// Takes a flag outside the common set; false when this model has no such flag. `code`: an exit code for a bad value.
    virtual bool parse_flag(const std::string& name, FlagArgs& args, program::CommonOptions& o, int& code) const {
        (void) name; (void) args; (void) o; (void) code;
        return false;
    }
    /// After the last flag: what the model's flags imply; an exit code when they conflict.
    virtual std::optional<int> finish_flags(program::CommonOptions& o) const { (void) o; return std::nullopt; }
};

/// Registry of adapters by architecture name. Registration is explicit (`models::register_builtin_models()`
/// from the program's main), so linking order never decides what exists.
void register_adapter(std::unique_ptr<ModelAdapter> a);
/// Null when no adapter has that architecture.
const ModelAdapter* find_adapter(const std::string& arch);
std::vector<std::string> adapter_names();
/// The most expert cards any registered model takes: a card plan before a model is known.
int max_expert_cards();
/// The adapter for a file's `general.architecture`, or null with `err` set.
const ModelAdapter* adapter_for(const GgufFile& meta, std::string& err);
/// Replaces the old architecture guard: empty when the file belongs to a registered model and validates.
std::string check_architecture(const GgufFile& meta);

}  // namespace basalt::model
