// include/basalt/core/weights.hpp - the dense weights, loaded into VRAM in ENGINE form.
//
// The dense section of the model file is the ARTIFACT's business; this is the ENGINE's.  The differences are
// deliberate, and they run in BOTH directions:
//
//   * SHRINKING.  A tensor whose source type is BF16 is stored in the file promoted to 32 bits, because that is
//     lossless and it keeps the canonical decoder simple, but a running engine re-rounds it to 16 - which costs
//     nothing at all, because the value CAME from 16 bits.  That saving is the difference between the dense
//     weights fitting in VRAM beside the expert cache and not fitting at all.
//
//   * GROWING.  Some quantized tensors (all Q2_0 expert tensors, plus Q4_0, Q5_0, Q8_0 and IQ4_NL) carry fp16
//     SCALES in the file because that is what the source block holds.  Every kernel here takes
//     `const float* scales`, and fp16 is a subset of f32, so this loader WIDENS them - exactly, changing no
//     value.  Assuming 4 bytes per scale instead of reading `scales_fp16` makes the plane arithmetic disagree
//     with the file.
//
// WHAT THE LOADER DOES NOT DO: it does not know what any tensor MEANS.  It reads a flat index (written when
// the model file is built), copies each tensor into one arena at a recorded
// offset - as a list of PLANE SEGMENTS, because a scale plane can change width and the rest cannot - and hands
// back a name -> WeightRef map.  Geometry, forms and semantics belong to the caller.
//
// WHY AN INDEX FILE AND NOT A JSON PARSER: a JSON parser in the engine would be a new, unaudited component
// whose failure mode is a wrong byte offset - which decodes to a plausible weight and produces plausible
// logits.  Python wrote the manifest and already parses it correctly.
#pragma once

#include "basalt/core/model_file.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace basalt::core {

/// What the loader had to DO for a tensor, decided when the model file is built and written into the index
/// rather than inferred here.  The four cases are distinguishable in the manifest only by combining
/// `source_type` with `values_fp16`, and getting that combination wrong yields a plausible tensor.
enum class WeightKind : int {
    Verbatim = 0,   ///< quantized planes, or F16 already stored as 2 bytes: copy straight through
    Bf16InF32 = 1,  ///< the file holds the bf16 value promoted to f32: take the HIGH 16 BITS (exact)
    F32 = 2,        ///< copy 4 bytes per element
    F16InF32 = 3,   ///< an f16 value promoted to f32: a real f32->f16 conversion, not a truncation
};

struct WeightRef {
    const void* data = nullptr;   ///< device pointer inside the arena
    uint64_t bytes = 0;
    int64_t ne0 = 0;              ///< the manifest's own shape, ne0 the CONTIGUOUS axis
    int64_t ne1 = 0;
    int64_t elements = 0;
    WeightKind kind = WeightKind::Verbatim;
    /// The canonical-form attributes, valid only for a quantized VERBATIM tensor (`code_bits != 0`).
    int code_bits = 0;
    int code_bias = 0;
    int group_elems = 0;
    bool codebook_iq4nl = false;
    bool has_offset = false;

    /// THE THREE PLANE SIZES, IN ENGINE FORM, in the order they sit inside `data`.  Recorded by the loader
    /// from what the index TOLD it rather than re-derived by each caller: the one time this layout was
    /// re-derived independently it assumed 4-byte scales for tensors whose file scales are fp16, and the
    /// only reason that was not a silent wrong answer is that the re-derivation checked itself and refused.
    /// Two places computing one layout is two places to disagree.
    uint64_t codes_bytes = 0;
    uint64_t scales_bytes = 0;    ///< always 4 B per scale in the arena, whatever the file held
    uint64_t offset_bytes = 0;    ///< 0 when the form has no offset plane

    /// WHERE THIS TENSOR CAME FROM, so a caller can check the loader's work against the file's own bytes
    /// instead of against the loader's opinion of them.  Without `src_off` a test can only compare two
    /// derived numbers; with it, the widening of an fp16 scale plane can be verified value by value against
    /// the source bytes - and `file_id` says which of the section's files it came from.
    uint64_t src_off = 0;
    uint64_t src_bytes = 0;
    int file_id = 0;
    bool scales_fp16 = false;     ///< the FILE's width, before the loader widened it

    /// WHICH ACTIVATION THIS WEIGHT WANTS - 0 for Q8_0, 1 for Q8_K.  Decided when the model file is built from
    /// source_type and carried here, because **the S-form CANNOT express it**: Q5_0 and Q5_K are both 8-bit
    /// with bias -16, and IQ4_NL and IQ4_XS differ in nothing the S-form holds.
    int act_kind = 0;
    bool wants_q8k() const { return act_kind == 1; }

    bool quantized() const { return code_bits != 0; }

    // Optional native GGUF projection, owned by NativeDense. All references in
    // one table share its scratch and must execute on one ordered session stream.
    const void* native_data = nullptr;
    void* native_q8_1 = nullptr;
    int native_type = -1;
    /// False when the loader SKIPPED this tensor's canonical bytes because another form serves it
    /// (native GGUF projections, the native head).  The metadata above stays valid; `data` is null.
    bool resident = true;
};

struct LoadReport {
    size_t tensors = 0;
    uint64_t arena_bytes = 0;
    size_t re_rounded = 0;        ///< tensors written as 16 bits out of a 32-bit container
    uint64_t bytes_saved = 0;     ///< what those conversions saved against the file
    double read_ms = 0.0;
    double upload_ms = 0.0;
};

/// The ranges the dense index's rows refer to: the index and the byte ranges of its files, by the index's file id
/// (0 the dense pool).  Offsets in the index are relative to the start of the file's range.
struct PackFiles {
    FileRange index;
    FileRange files[4];
    /// The dense section and its index, from a .basalt (or any ModelSources).
    static PackFiles from(const ModelSources& s);
};

class WeightTable {
public:
    /// The arena size the index asks for, readable WITHOUT loading anything - so a caller can size its
    /// `DeviceArena` before committing to the load, and a plan that cannot fit is refused at startup rather
    /// than halfway through a 5 GB upload.
    /// With `skip`, the size of the compacted arena that holds every tensor EXCEPT the named ones.
    static bool pool_bytes(const PackFiles& pack, uint64_t& out, std::string& err,
                           const std::set<std::string>* skip = nullptr);
    /// The compacted arena of `pool_bytes` split by block: `per_layer[l]` is the bytes of the `blk.<l>.` tensors that
    /// are not in `skip`, `shared` those of every other tensor.
    static bool pool_bytes_by_layer(const PackFiles& pack, int64_t n_layers, std::vector<uint64_t>& per_layer,
                                    uint64_t& shared, std::string& err, const std::set<std::string>* skip = nullptr);

    /// The `code_bits` field of one row of the dense index, readable WITHOUT loading anything (0 = the
    /// file stores the tensor unquantized; -1 = no such row).  The loader needs this
    /// before it builds the skip set.
    static bool index_code_bits(const PackFiles& pack, const std::string& name, int& code_bits, std::string& err);

    /// Load every tensor in the dense index into `arena_base`.
    ///
    /// `arena_bytes` MUST be at least `pool_bytes`; the loader checks rather than trusting the caller,
    /// because the failure mode otherwise is a device write past the end of the arena.
    /// With `skip`, the named tensors are not read: their rows keep their metadata with `data == nullptr` and
    /// `resident == false`, and the other tensors are packed into the compacted arena `pool_bytes` sized.
    /// `report_progress == false` keeps a temporary load out of the startup progress lines.
    bool load(const PackFiles& pack, void* arena_base, uint64_t arena_bytes, std::string& err,
              const std::set<std::string>* skip = nullptr, bool report_progress = true);

    const WeightRef* find(const std::string& name) const;
    /// Point a row at another device form that stands for the tensor (an 8-bit copy keyed by its own address,
    /// hc_fused.hpp's hc8_adopt); its readers resolve that form.  False when there is no such row.
    bool rebind(const std::string& name, const void* data);
    const std::map<std::string, WeightRef>& all() const { return table_; }
    const LoadReport& report() const { return report_; }

private:
    friend class NativeDense;
    friend struct WeightTableFixture;   // builds a table row by row for the shape-check test
    std::map<std::string, WeightRef> table_;
    LoadReport report_;
};

}  // namespace basalt::core
