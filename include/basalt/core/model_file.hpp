#pragma once
// Reader for the .basalt model file (the layout is in model_format.hpp), and the byte ranges the loaders read.
//
// Every loader takes a FileRange: a path, an offset and a size; `--model` gives the sections of one file.
// A section is 2 MiB aligned in the file, so a range can be mapped, and read with O_DIRECT at its own offsets.
//
// The source tensors in the `gguf` section are served through the GGUF reader (basalt/artifact/gguf_reader.hpp):
// `ModelFile::register_gguf_view` makes `GgufFile(model path)` answer with `meta.gguf_kv` and the tensors of
// `meta.tensors`, whose offsets are absolute file offsets, so the loaders that read GGUF tensors in place read them
// from the .basalt unchanged.
//
// The `hc8` section is a table of the 8-bit hyper-connection copies, little-endian:
//   Hc8Header                 magic "BASALTH8", version, n_entries
//   n_entries x Hc8Entry      name (NUL-padded), rows, cols, srow, offset (from the start of the section), bytes
//   data                      at each entry's offset (a multiple of 256): rows x cols int8 codes, then
//                             rows x srow f16 scales (a row's cols / 32 scales first, the rest zero), `bytes` in all
// The names are the weight table's tensor names (e.g. blk.0.hc_attn_down.weight); the draft layer's matrices are
// prefixed "mtp." and use the drafter's own tensor names.  tools/convert/hc8.py computes them from the file's bf16
// matrices, bit for bit as the kernels' quantizer (hc8_adopt) does.
#include "basalt/core/model_format.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace basalt::core {

/// A byte range of a file.  `bytes == 0` means to the end of the file.
struct FileRange {
    std::string path;
    uint64_t offset = 0;
    uint64_t bytes = 0;

    FileRange() = default;
    explicit FileRange(std::string p, uint64_t off = 0, uint64_t n = 0) : path(std::move(p)), offset(off), bytes(n) {}

    bool empty() const { return path.empty(); }
    /// Whether the file exists and, for a bounded range, holds the range.
    bool exists() const;
    /// The range's size (the file's size past `offset` when unbounded); 0 when the file cannot be read.
    uint64_t size() const;
    /// Whether this is a whole file rather than a section.
    bool whole() const { return offset == 0 && bytes == 0; }
};

/// The range's bytes.  False with `err` when it cannot be read in full.
bool read_range(const FileRange& r, std::vector<uint8_t>& out, std::string& err);
/// Reads `n` bytes at `off` inside the range into `dst` (bounds-checked against a bounded range).
bool read_range_at(const FileRange& r, uint64_t off, void* dst, uint64_t n, std::string& err);

/// A parsed JSON value.  Integers keep their exact value; an array keeps its first `kJsonArrayKeep` items and its
/// full length in `count` (the source GGUF's tokenizer arrays are large and nothing reads past a sample).
struct Json {
    enum class Type { Null, Bool, Int, Float, String, Array, Object };
    static constexpr size_t kJsonArrayKeep = 64;
    Type type = Type::Null;
    bool b = false;
    bool neg = false;          ///< Int: the value is -u
    uint64_t u = 0;            ///< Int: magnitude
    double f = 0;              ///< Float
    std::string s;
    std::vector<Json> items;   ///< Array (first kJsonArrayKeep)
    uint64_t count = 0;        ///< Array: the real length
    std::map<std::string, Json> members;

    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }
    bool is_string() const { return type == Type::String; }
    bool is_number() const { return type == Type::Int || type == Type::Float; }
    /// A member of an object, or nullptr.
    const Json* get(const std::string& key) const;
    double num() const;
    int64_t i64() const;
    uint64_t u64() const { return u; }
};

/// Parses `text`.  False with `err` (and the position) on a syntax error.
bool parse_json(const char* text, size_t n, Json& out, std::string& err);

class ModelFile {
public:
    struct Section {
        std::string name;
        uint64_t offset = 0;
        uint64_t bytes = 0;
    };

    /// Opens `path`: checks the magic, the version, the size and every section's bounds, and reads the table.
    bool open(const std::string& path, std::string& err);
    const std::string& path() const { return path_; }
    uint64_t file_bytes() const { return file_bytes_; }
    const std::vector<Section>& sections() const { return sections_; }
    const Section* find(const std::string& name) const;
    /// The section as a range of the file; an empty range when there is no such section.
    FileRange range(const std::string& name) const;
    bool read(const std::string& name, std::vector<uint8_t>& out, std::string& err) const;

    /// Parses the `meta` section (once).
    bool load_meta(std::string& err);
    const Json& meta() const { return meta_; }

    /// After load_meta: makes `GgufFile(path())` serve meta.gguf_kv and the tensors of the `gguf` section.
    bool register_gguf_view(std::string& err) const;

private:
    std::string path_;
    uint64_t file_bytes_ = 0;
    std::vector<Section> sections_;
    Json meta_;
    bool meta_loaded_ = false;
};

/// Where each loader reads from: the sections of the .basalt.
struct ModelSources {
    FileRange dense, dense_index, experts, experts_layout, hc8;
    FileRange mtp_dense, mtp_dense_index, mtp_experts, mtp_draft_vocab;
    /// The file that serves the source GGUF tensors and metadata when they come from a .basalt, else empty.
    std::string model_path;
};

/// The ranges of a .basalt's sections (those that exist).
void model_sources(const ModelFile& m, ModelSources& s);

// ---- the hc8 table
struct Hc8Header {
    char magic[8];             ///< "BASALTH8"
    uint32_t version;          ///< 1
    uint32_t n_entries;
};
static_assert(sizeof(Hc8Header) == 16);

struct Hc8Entry {
    char name[128];
    uint32_t rows, cols, srow, reserved;
    uint64_t offset;
    uint64_t bytes;
};
static_assert(sizeof(Hc8Entry) == 160);

/// One copy in memory: the int8 codes then the f16 scales, as Hc8Entry describes them.
struct Hc8Copy {
    std::string name;
    uint32_t rows = 0, cols = 0, srow = 0;
    std::vector<uint8_t> data;
    const int8_t* q() const { return (const int8_t*) data.data(); }
    const uint16_t* s() const { return (const uint16_t*) (data.data() + (size_t) rows * cols); }
};

/// The table the engine loads copies from at startup (process-wide).
void hc8_set_table(std::map<std::string, Hc8Copy> table);
/// The loaded copy of `name` when its shape is rows x cols, else nullptr.
const Hc8Copy* hc8_find(const std::string& name, int rows, int cols);

/// Reads a table from a range (the `hc8` section of a .basalt).
bool hc8_read(const FileRange& r, std::map<std::string, Hc8Copy>& out, std::string& err);

}  // namespace basalt::core
