#pragma once
// The .basalt model file: one file holds everything the engine, the server and the vision sidecar load.
//
// Layout (little-endian):
//   [0, 64 KiB)          FileHeader, then n_sections SectionEntry records (the table fits in the first 64 KiB)
//   every section        starts at a multiple of kSectionAlign, so it can be mapped, read with O_DIRECT and
//                        backed by huge pages; the bytes between sections are zero
//
// A section's bytes are opaque to the container; what each one holds is listed below.  Text sections are UTF-8.
// tools/model_file.py is the Python side of this layout and must stay byte-compatible with it.
//
// Sections (v1):
//   meta                     JSON: format/version, model name, geometry, the source GGUF metadata the engine reads
//                            (`gguf_kv`), the type, shape, size and offset in `gguf` of every source tensor (`tensors`), the engine's
//                            default sampling (`sampling`), provenance (source
//                            files, sizes, sha256, converter version) and the license / attribution text
//   tokenizer/<file>         the tokenizer directory's files (tokenizer.json, vocab.json, merges.txt,
//                            token_type.json, chat_template.jinja)
//   dense                    the dense weight pool
//   dense.index              its tensor index (text, first line "# dense index v3")
//   experts                  the routed experts' blobs
//   experts.layout           their per-layer layout (text, first line "# experts v3"; offsets relative to `experts`)
//   experts.hadamard         Hadamard-folded experts only: the rotation's block, sign vectors and folded roles per
//                            layer (text, first line "# basalt hadamard v1"; kernels/cpu/hadamard.hpp)
//   gguf                     every non-expert tensor of the source GGUFs (token_embd.weight,
//                            per_layer_token_embd.weight, ...), raw bytes, each starting at a multiple of
//                            kTensorAlign within the section; meta.tensors[name].offset is relative to the section
//   mtp/dense, mtp/dense.index, mtp/experts, mtp/draft_vocab
//                            the drafter (the files of an mtp runtime directory)
//   profile                  the default expert profile
//   hc8                      the 8-bit copies of the hyper-connection matrices, as `basalt --dump-hc8` writes them
//   vision/mmproj            the vision projector GGUF, whole
//   vision/<name>            anything else the vision sidecar reads
#include <cstdint>

namespace basalt::core {

constexpr char kModelMagic[8] = {'B', 'A', 'S', 'A', 'L', 'T', 'M', 'F'};
constexpr uint32_t kModelVersion = 1;
constexpr uint64_t kSectionAlign = 2ull << 20;
constexpr uint64_t kHeaderBytes = 64ull << 10;
constexpr uint64_t kTensorAlign = 4096;

struct FileHeader {
    char magic[8];           ///< kModelMagic
    uint32_t version;        ///< kModelVersion
    uint32_t n_sections;
    uint64_t file_bytes;     ///< the whole file, for a truncation check
    uint8_t reserved[40];
};
static_assert(sizeof(FileHeader) == 64);

struct SectionEntry {
    char name[64];           ///< NUL-padded
    uint64_t offset;         ///< from the start of the file, a multiple of kSectionAlign
    uint64_t bytes;
    uint8_t sha256[32];
    uint8_t reserved[16];
};
static_assert(sizeof(SectionEntry) == 128);

constexpr uint32_t kMaxSections = (uint32_t) ((kHeaderBytes - sizeof(FileHeader)) / sizeof(SectionEntry));

}  // namespace basalt::core
