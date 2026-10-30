#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

const float* MtpDrafter::f32(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "f32") return (const float*) (dense_ + t.off);
    return nullptr;
}
const uint16_t* MtpDrafter::bf16(const char* name) const {
    for (const auto& t : tensors_)
        if (t.name == name && t.kind == "bf16") return t.dev ? (const uint16_t*) t.dev : (const uint16_t*) (dense_ + t.off);
    return nullptr;
}
const void* MtpDrafter::q8(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "q8_0") return dense_ + t.off;
    return nullptr;
}

bool MtpDrafter::parse_index(const FileRange& index, std::vector<Tensor>& out, std::string& err) {
    std::vector<uint8_t> idx_text;
    if (!read_file(index, idx_text)) { err = "mtp: cannot open " + index.path; return false; }
    std::istringstream idx(std::string(idx_text.begin(), idx_text.end()));
    std::string line;
    while (std::getline(idx, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        Tensor t;
        is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
        if (!is) { err = "mtp: malformed dense index line: " + line; return false; }
        out.push_back(t);
    }
    return true;
}

bool MtpDrafter::is_hc8_matrix(const Tensor& t) {
    return t.kind == "bf16" && t.name.find("hyper_connection") != std::string::npos &&
           t.name.find("input_mix_weight_") != std::string::npos && (uint64_t) t.rows * (uint64_t) t.cols * 2 == t.bytes;
}

bool MtpDrafter::plan_bytes(const MtpFiles& files, const ModelGeometry& g, uint64_t& out, std::string& err) {
    uint64_t dense = files.dense.size();
    if (basalt::kernels::hc8_only()) {   // the dense blob without the matrices that are held as 8-bit copies only
        std::vector<Tensor> tensors;
        if (!parse_index(files.index, tensors, err)) return false;
        uint64_t kept = 0;
        for (const Tensor& t : tensors) {
            if (is_hc8_matrix(t)) continue;
            kept = ((kept + 255) & ~(uint64_t) 255) + t.bytes;
        }
        dense = kept;
    }
    out = dense + (uint64_t) g.n_expert * basalt::kernels::cpu::BLOB;
    return true;
}

std::vector<std::pair<std::string, const void*>> MtpDrafter::hc8_keys() const {
    std::vector<std::pair<std::string, const void*>> out;
    for (const Tensor& t : tensors_)
        if (t.kind == "bf16" && t.name.find("hyper_connection") != std::string::npos)
            out.emplace_back("mtp." + t.name, t.dev ? t.dev : (const void*) (dense_ + t.off));
    return out;
}


}
