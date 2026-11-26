#include "basalt/models/qwen4exp/shared_names.hpp"
#include "control_vectors.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/models/qwen4exp/decode/layout.hpp"
#include "basalt/models/qwen4exp/kernels/cvec.hpp"
#include "basalt/log.hpp"
#include "basalt/models/qwen4exp/options.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace basalt::models::qwen4exp::cli {

// --control-vector-scaled: llama.cpp's `common_control_vector_load` (every file's `direction.<l>` times its scale,
// summed; layer 0 has none) and `llama_adapter_cvec::apply` with the projection-mode patch (project: the unit
// direction and its norm as the scale), into the tables `cvec_upload` takes.  `summary` is what INFO reports.
bool load_control_vectors(const Options& o, const basalt::core::ModelGeometry& g, std::string& summary, std::string& err) {
    const int64_t L = g.n_layers, N = g.n_embd;
    if (L <= 0 || N <= 0) {
        err = "control vector: invalid model geometry";
        return false;
    }
    std::vector<float> data((size_t) (L * N), 0.0f);
    std::vector<float> vec((size_t) N);
    std::vector<bool> have((size_t) L, false);
    const uint64_t need_bytes = (uint64_t) N * sizeof(float);
    for (const auto& [path, scale] : o.cvec_files) {
        if (!std::isfinite(scale)) {
            err = path + ": non-finite control vector scale";
            return false;
        }
        try {
            basalt::GgufFile f(path);
            const basalt::MetaValue* arch = f.get("general.architecture");
            if (arch == nullptr || arch->s != "controlvector") {
                err = path + ": not a control vector GGUF (general.architecture is not 'controlvector')";
                return false;
            }
            const basalt::MetaValue* hint = f.get("controlvector.model_hint");
            if (hint != nullptr && hint->s != "qwen4exp")
                basalt::log::warn("basalt generate", "%s was made for '%s', not qwen4exp", path.c_str(), hint->s.c_str());
            const uint64_t payload = f.file_size() >= f.data_start() ? f.file_size() - f.data_start() : 0;
            int found = 0;
            for (const basalt::TensorInfo& t : f.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const char* start = t.name.c_str() + 10;
                if (*start < '0' || *start > '9') continue;
                char* end = nullptr;
                errno = 0;
                const long l = std::strtol(start, &end, 10);
                if (errno != 0 || end == start || *end != '\0') continue;
                if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                uint64_t elems = 1;
                bool shape_ok = !t.shape.empty();
                for (uint64_t dim : t.shape) {
                    if (dim == 0 || elems > (std::numeric_limits<uint64_t>::max)() / dim) {
                        shape_ok = false;
                        break;
                    }
                    elems *= dim;
                }
                if (t.type != 0 || !shape_ok || elems != (uint64_t) N) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) N) + " f32";
                    return false;
                }
                if (t.offset > payload || need_bytes > payload - t.offset) {
                    err = path + ": " + t.name + " payload is truncated";
                    return false;
                }
                std::memcpy(vec.data(), f.tensor_data(t), (size_t) need_bytes);
                for (int64_t j = 0; j < N; ++j) data[(size_t) (l * N + j)] += scale * vec[(size_t) j];
                have[(size_t) l] = true;
                ++found;
            }
            if (found == 0) { err = path + ": no direction.<layer> tensors"; return false; }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
    const int first = o.cvec_first <= 0 ? 1 : o.cvec_first;
    const int last = (o.cvec_last <= 0 || o.cvec_last >= L) ? (int) L - 1 : o.cvec_last;
    const int single = o.cvec_mode == 0 ? o.cvec_single : -1;
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
        if (o.cvec_mode == 0) {
            double nrm = 0.0;
            for (int64_t j = 0; j < N; ++j) nrm += (double) d[j] * d[j];
            nrm = std::sqrt(nrm);
            if (!std::isfinite(nrm) || nrm <= 0.0 || !std::isfinite((float) nrm)) continue;
            s[(size_t) l] = (float) nrm;
            for (int64_t j = 0; j < N; ++j) dir[(size_t) (l * N + j)] = (float) (d[j] / nrm);
        } else {
            bool finite = true;
            for (int64_t j = 0; j < N; ++j)
                if (!std::isfinite(d[j])) { finite = false; break; }
            if (!finite) continue;
            s[(size_t) l] = 1.0f;
            std::copy(d, d + N, dir.begin() + (size_t) (l * N));
        }
        ++steered;
    }
    if (steered == 0) { err = "the control vector has no direction in layers " + std::to_string(first) + ".." + std::to_string(last); return false; }
    if (!basalt::kernels::cvec_upload(dir, s, o.cvec_mode, first, last, N, g.hc, err)) return false;
    summary = std::string(o.cvec_mode == 0 ? "project" : "add") + ":" + std::to_string(first) + "-" + std::to_string(last) +
              (single >= 0 ? ":single" + std::to_string(single) : "");
    // the line llama.cpp's patched build prints, so a log shows the same thing
    basalt::log::info("basalt generate", "control vector mode = %s, dir = %s, layers %d..%d (%d steered)",
                 o.cvec_mode == 0 ? "project" : "add", single >= 0 ? "single" : "per-layer", first, last, steered);
    return true;
}

}  // namespace basalt::models::qwen4exp::cli
