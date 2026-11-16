// src/model/control_vectors.cpp - see basalt/model/control_vectors.hpp.
#include "basalt/model/control_vectors.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/kernels/cvec.hpp"
#include "basalt/log.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace basalt::model {

bool parse_cvec_flag(const std::string& a, FlagArgs& args, CvecFlags& f, int& code) {
    if (a == "--control-vector") f.files.push_back({args.next("--control-vector"), 1.0f});
    else if (a == "--control-vector-scaled") {
        // FILE:SCALE, comma-separated; the LAST colon splits
        std::stringstream list(args.next("--control-vector-scaled"));
        std::string item;
        while (std::getline(list, item, ',')) {
            const size_t colon = item.rfind(':');
            char* end = nullptr;
            const float sc = colon == std::string::npos ? 0.0f : std::strtof(item.c_str() + colon + 1, &end);
            if (colon == std::string::npos || colon == 0 || end == item.c_str() + colon + 1 || *end != '\0') {
                basalt::log::error("basalt", "--control-vector-scaled: expected FILE:SCALE, got '%s'", item.c_str());
                code = 2;
                return true;
            }
            f.files.push_back({item.substr(0, colon), sc});
        }
    }
    else if (a == "--control-vector-layer-range") {
        f.first = std::atoi(args.next("--control-vector-layer-range"));
        f.last = std::atoi(args.next("--control-vector-layer-range"));
    }
    else if (a == "--cvec-mode") {
        const std::string m = args.next("--cvec-mode");
        if (m == "project") f.mode = 0;
        else if (m == "add") f.mode = 1;
        else {
            basalt::log::error("basalt", "--cvec-mode: add or project, got '%s'", m.c_str());
            code = 2;
            return true;
        }
    }
    else if (a == "--cvec-dir") {
        const std::string d = args.next("--cvec-dir");
        if (d == "per-layer") f.single = -1;
        else if (d.rfind("single:", 0) == 0) f.single = std::atoi(d.c_str() + 7);
        else {
            basalt::log::error("basalt", "--cvec-dir: per-layer or single:L, got '%s'", d.c_str());
            code = 2;
            return true;
        }
    }
    else return false;
    return true;
}

bool load_control_vectors(const CvecFlags& cf, int64_t n_layers, int64_t n_embd, int64_t hc, const char* arch,
                          std::string& summary, std::string& err) {
    const int64_t L = n_layers, N = n_embd;
    std::vector<float> data((size_t) (L * N), 0.0f);
    std::vector<bool> have((size_t) L, false);
    for (const auto& [path, scale] : cf.files) {
        try {
            basalt::GgufFile f(path);
            const basalt::MetaValue* a = f.get("general.architecture");
            if (a == nullptr || a->s != "controlvector") {
                err = path + ": not a control vector GGUF (general.architecture is not 'controlvector')";
                return false;
            }
            const basalt::MetaValue* hint = f.get("controlvector.model_hint");
            if (hint != nullptr && hint->s != arch)
                basalt::log::warn("basalt cvec", "%s was made for '%s', not %s", path.c_str(), hint->s.c_str(), arch);
            int found = 0;
            for (const basalt::TensorInfo& t : f.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
                if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                if (t.type != 0 || t.elements() != (uint64_t) N) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) N) + " f32";
                    return false;
                }
                const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
                for (int64_t j = 0; j < N; ++j) data[(size_t) (l * N + j)] += scale * src[j];
                have[(size_t) l] = true;
                ++found;
            }
            if (found == 0) { err = path + ": no direction.<layer> tensors"; return false; }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
    const int first = cf.first <= 0 ? 1 : cf.first;
    const int last = (cf.last <= 0 || cf.last >= L) ? (int) L - 1 : cf.last;
    const int single = cf.mode == 0 ? cf.single : -1;
    if (single >= 0 && (single >= L || !have[(size_t) single])) {
        err = "--cvec-dir single:" + std::to_string(single) + ": the vector has no direction for that layer";
        return false;
    }
    std::vector<float> dir((size_t) (L * N), 0.0f), s((size_t) L, 0.0f);
    int steered = 0;
    for (int64_t l = first; l <= last; ++l) {
        const int64_t src = single >= 0 ? single : l;
        if (!have[(size_t) src]) continue;
        const float* d = data.data() + (size_t) (src * N);
        if (cf.mode == 0) {   // project: the unit direction, its length as the scale
            double nrm = 0.0;
            for (int64_t j = 0; j < N; ++j) nrm += (double) d[j] * d[j];
            nrm = std::sqrt(nrm);
            if (nrm <= 0.0) continue;
            s[(size_t) l] = (float) nrm;
            for (int64_t j = 0; j < N; ++j) dir[(size_t) (l * N + j)] = (float) (d[j] / nrm);
        } else {
            s[(size_t) l] = 1.0f;
            std::copy(d, d + N, dir.begin() + (size_t) (l * N));
        }
        ++steered;
    }
    if (steered == 0) {
        err = "the control vector has no direction in layers " + std::to_string(first) + ".." + std::to_string(last);
        return false;
    }
    if (!basalt::kernels::cvec_upload(dir, s, cf.mode, first, last, N, hc, err)) return false;
    summary = std::string(cf.mode == 0 ? "project" : "add") + ":" + std::to_string(first) + "-" + std::to_string(last) +
              (single >= 0 ? ":single" + std::to_string(single) : "");
    // the line llama.cpp's patched build prints, so a log shows the same thing
    basalt::log::info("basalt cvec", "control vector mode = %s, dir = %s, layers %d..%d (%d steered)",
                      cf.mode == 0 ? "project" : "add", single >= 0 ? "single" : "per-layer", first, last, steered);
    return true;
}

}  // namespace basalt::model
