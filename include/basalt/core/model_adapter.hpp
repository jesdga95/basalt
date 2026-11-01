#pragma once
// Standalone .basalt adapters: validated whole-tensor replacements, without a base model or runtime binding.
#include "basalt/core/model_file.hpp"

namespace basalt::core {

struct AdapterTensor {
    std::string tensor;
    std::vector<uint64_t> shape;  ///< GGUF order: the fastest axis comes first.
    uint32_t type_id = 0;
    FileRange data;              ///< Absolute file offset and exact packed bytes; never an unbounded range.
};

// Region and storage kinds are semantic extension points. Only whole/replace is accepted, never delta math.
class AdapterFile {
public:
    /// Clears previous replacements, then publishes only a completely validated manifest.
    bool open(const std::string& path, std::string& err);
    const std::map<std::string, AdapterTensor>& replacements() const { return replacements_; }
    const AdapterTensor* find(const std::string& tensor) const;
    /// Returns the encoded replacement unchanged; clears out on failure.
    bool read_tensor(const std::string& tensor, std::vector<uint8_t>& out, std::string& err) const;

private:
    std::map<std::string, AdapterTensor> replacements_;
};

}  // namespace basalt::core
