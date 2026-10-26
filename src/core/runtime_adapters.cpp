// Fixed-address tensor replacement between drained requests. Sources retain their GGUF encoding;
// floating-point engine forms use the same dequantization and rounding as model conversion.

#include "basalt/core/runtime_adapters.hpp"

#include "basalt/core/direct_upload.hpp"
#include "basalt/artifact/dequant.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/bf16_bits.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace basalt::core {
namespace {

constexpr uint64_t kChunk = DirectUploader::kChunk;   // source bytes per staging round
constexpr uint32_t kTypeBf16 = 30;

bool source_geometry(uint32_t type, int& elements, int& bytes) {
    if (!basalt::block_geometry(type, elements, bytes)) return false;
    if (type == 0 || type == 1 || type == kTypeBf16 || type == 2 || type == 3 || type == 10) return true;
    const auto* traits = ggml_get_type_traits((ggml_type) type);
    return traits != nullptr && traits->to_float != nullptr;
}

void decode_source(uint32_t type, const uint8_t* input, float* output, size_t elements) {
    if (type == 0) basalt::dequantize_f32(input, output, (int) elements);
    else if (type == 1) basalt::dequantize_f16(input, output, (int) elements);
    else if (type == kTypeBf16) basalt::dequantize_bf16(input, output, (int) elements);
    else if (type == 2) {
        for (size_t at = 0; at < elements; at += 32, input += 18)
            basalt::dequantize_q4_0(input, output + at);
    } else if (type == 3) {
        for (size_t at = 0; at < elements; at += 32, input += 20) {
            const float scale = basalt::fp16_to_fp32(basalt::read_u16(input));
            const float offset = basalt::fp16_to_fp32(basalt::read_u16(input + 2));
            for (size_t i = 0; i < 16; ++i) {
                output[at + i] = scale * (input[4 + i] & 15) + offset;
                output[at + 16 + i] = scale * (input[4 + i] >> 4) + offset;
            }
        }
    } else if (type == 10) {
        for (size_t at = 0; at < elements; at += 256, input += 84) {
            const float scale = basalt::fp16_to_fp32(basalt::read_u16(input + 80));
            const float minimum = basalt::fp16_to_fp32(basalt::read_u16(input + 82));
            for (size_t i = 0; i < 256; ++i) {
                const uint8_t group = input[i / 16];
                const int code = (input[16 + i / 128 * 32 + i % 32] >> (2 * (i % 128 / 32))) & 3;
                output[at + i] = (scale * (group & 15)) * code - minimum * (group >> 4);
            }
        }
    } else {
        ggml_get_type_traits((ggml_type) type)->to_float(input, output, (int64_t) elements);
    }
}

bool json_uint(const Json* v, uint64_t& out) {
    if (!v || v->type != Json::Type::Int || (v->neg && v->u != 0)) return false;
    out = v->u;
    return true;
}

struct BaseTensor {
    std::vector<uint64_t> shape;  ///< GGUF order
    uint32_t type_id = 0;
    uint64_t offset = 0;          ///< relative to the gguf section
    uint64_t bytes = 0;
};

}  // namespace

struct RuntimeAdapters::Impl {
    std::map<std::string, BaseTensor> base;      ///< the base model's meta.tensors
    std::map<std::string, uint64_t> native_bytes;  ///< "<tensor>.native" meta entries: native projection sizes
    std::set<std::string> names;                 ///< the union of replacement targets over all groups
    std::vector<std::map<std::string, AdapterTensor>> models;  ///< per model: source per name; map nodes are stable
    size_t current = 0;

    /// One bound weight row on one device: everything needed to replace it in place.
    struct Row {
        std::string name;
        int device = 0;
        void* target = nullptr;       ///< device bytes target
        uint64_t bytes = 0;           ///< engine bytes at `target` (canonical) / native byte size (native rows)
        uint32_t type_id = 0;         ///< floating-point engine format; native rows retain their source type
        uint64_t elements = 0;
        bool native = false;
        const void* hc8_key = nullptr;  ///< the bf16 matrix the HC8 copy is keyed by
        int hc8_rows = 0, hc8_cols = 0;
    };
    std::vector<Row> rows;
    std::map<int, std::pair<void*, size_t>> staging;  ///< device -> one reusable HC8 staging buffer

    std::vector<uint8_t> input;
    std::vector<float> values;
    std::vector<uint16_t> rounded;

    bool upload(const Row& row, const AdapterTensor& source, void* target, std::string& err);
};

RuntimeAdapters::RuntimeAdapters() : impl_(new Impl) {}

RuntimeAdapters::~RuntimeAdapters() {
    int saved = 0;
    cudaGetDevice(&saved);
    for (auto& [dev, buf] : impl_->staging) {
        if (cudaSetDevice(dev) == cudaSuccess && buf.first != nullptr) cudaFree(buf.first);
        buf.first = nullptr;
    }
    cudaSetDevice(saved);
}

bool RuntimeAdapters::open(const ModelFile& base, const std::vector<std::vector<std::string>>& groups,
                           std::string& err) {
    err.clear();
    const std::string& base_path = base.path();
    const Json& meta = base.meta();
    const std::string prefix = base_path + ": adapters: ";
    const auto fail = [&](const std::string& message) {
        err = prefix + message;
        return false;
    };
    if (const Json* kind = meta.get("kind"); kind && (!kind->is_string() || kind->s != "model")) {
        return fail("base artifact is not a model");
    }
    const Json* tensors = meta.get("tensors");
    if (!tensors || !tensors->is_object()) return fail("base metadata has no tensors");
    const ModelFile::Section* gguf = base.find("gguf");
    if (!gguf) return fail("base model has no gguf section");

    impl_->base.clear();
    impl_->native_bytes.clear();
    for (const auto& [name, t] : tensors->members) {
        BaseTensor bt;
        const Json* shape = t.get("shape");
        uint64_t type_id = 0;
        if (!json_uint(t.get("type_id"), type_id) || !json_uint(t.get("offset"), bt.offset) ||
            !json_uint(t.get("bytes"), bt.bytes) || !shape || !shape->is_array() ||
            shape->count != shape->items.size() || shape->items.empty() || shape->items.size() > 4) {
            return fail("base tensor " + name + " is malformed");
        }
        bt.type_id = (uint32_t) type_id;
        for (const Json& d : shape->items) bt.shape.push_back(d.u64());
        if (const std::string suffix = ".native"; name.size() > suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            impl_->native_bytes[name.substr(0, name.size() - suffix.size())] = bt.bytes;
        }
        impl_->base.emplace(name, std::move(bt));
    }

    // Every adapter of every group is validated before anything is published; a compatible entry never masks
    // an incompatible one, and nothing is stored on a failed open.
    std::map<std::string, AdapterFile> parsed;
    std::set<std::string> names;
    for (const auto& group : groups) {
        for (const std::string& path : group) {
            auto it = parsed.find(path);
            if (it == parsed.end()) {
                AdapterFile file;
                if (!file.open(path, err)) return false;  // AdapterFile.open carries the path
                it = parsed.emplace(path, std::move(file)).first;
            }
            for (const auto& [id, rep] : it->second.replacements()) {
                if (rep.tensor == "per_layer_token_embd.weight") {
                    return fail(path + ": entry " + id + ": PLE table replacement is not supported");
                }
                const auto bt = impl_->base.find(rep.tensor);
                if (bt == impl_->base.end()) {
                    return fail(path + ": entry " + id + ": tensor " + rep.tensor + " is not in the base model");
                }
                if (bt->second.shape != rep.shape) {
                    return fail(path + ": entry " + id + ": shape does not match base tensor " + rep.tensor);
                }
                if (bt->second.type_id != rep.type_id) {
                    return fail(path + ": entry " + id + ": type " + std::to_string(rep.type_id) +
                                " does not match base tensor " + rep.tensor);
                }
                names.insert(rep.tensor);
            }
        }
    }

    const uint64_t gguf_offset = gguf->offset;
    const auto base_source = [&](const std::string& name) {
        const BaseTensor& bt = impl_->base.at(name);
        AdapterTensor t;
        t.tensor = name;
        t.shape = bt.shape;
        t.type_id = bt.type_id;
        t.data = FileRange(base_path, gguf_offset + bt.offset, bt.bytes);
        return t;
    };
    std::vector<std::map<std::string, AdapterTensor>> models;
    models.reserve(groups.size() + 1);
    {
        std::map<std::string, AdapterTensor> m;
        for (const std::string& name : names) m.emplace(name, base_source(name));
        models.push_back(std::move(m));
    }
    for (const auto& group : groups) {
        std::map<std::string, AdapterTensor> m;
        for (const std::string& name : names) {
            const AdapterTensor* win = nullptr;
            for (const std::string& path : group) {  // the LAST replacement in the group wins
                if (const AdapterTensor* rep = parsed.at(path).find(name)) win = rep;
            }
            m.emplace(name, win ? *win : base_source(name));
        }
        models.push_back(std::move(m));
    }
    impl_->names = std::move(names);
    impl_->models = std::move(models);
    impl_->current = 0;
    return true;
}

size_t RuntimeAdapters::model_count() const { return impl_->models.size(); }

const std::set<std::string>& RuntimeAdapters::names() const { return impl_->names; }

const AdapterTensor* RuntimeAdapters::source(size_t model, const std::string& name) const {
    if (model >= impl_->models.size()) return nullptr;
    const auto it = impl_->models[model].find(name);
    return it == impl_->models[model].end() ? nullptr : &it->second;
}

bool RuntimeAdapters::changed(size_t from, size_t to, const std::string& name) const {
    const AdapterTensor* a = source(from, name);
    const AdapterTensor* b = source(to, name);
    if (a == nullptr || b == nullptr) return a != b;
    return a->data.path != b->data.path || a->data.offset != b->data.offset || a->data.bytes != b->data.bytes;
}

bool RuntimeAdapters::bind_weights(WeightTable& weights, const PackFiles& pack, int device, std::string& err) {
    (void) pack;  // replacement bytes come from the resolved FileRanges, never the dense pack
    err.clear();
    int saved = 0;
    cudaGetDevice(&saved);
    if (cudaSetDevice(device) != cudaSuccess) {
        err = "adapters: cannot use CUDA" + std::to_string(device);
        return false;
    }
    size_t hc8_stage = 0;
    const auto fail = [&](const std::string& message) {
        cudaSetDevice(saved);
        err = message;
        return false;
    };
    for (const std::string& name : impl_->names) {
        if (name == "output.weight" || name == "token_embd.weight") {
            continue;  // the caller swaps the head and the embedding
        }
        if (name == "per_layer_token_embd.weight") {
            return fail(name + ": PLE table replacement is not supported");
        }
        const WeightRef* w = weights.find(name);
        if (w == nullptr) continue;  // a target the dense table does not serve binds nothing
        const bool planar = w->code_bits != 0 || w->codes_bytes != 0 || w->scales_bytes != 0 || w->offset_bytes != 0;
        const bool native = w->native_data != nullptr && w->native_type >= 0;
        if (planar && !native) return fail(name + ": canonical planar binding is not supported");
        if (w->resident && w->data != nullptr && !planar) {
            Impl::Row row;
            row.name = name;
            row.device = device;
            row.target = const_cast<void*>(w->data);
            row.bytes = w->bytes;
            row.elements = static_cast<uint64_t>(w->elements);
            row.type_id = w->kind == WeightKind::Bf16InF32 ? kTypeBf16 :
                          w->kind == WeightKind::F32 ? 0 : 1;
            const uint64_t width = row.type_id == 0 ? 4 : 2;
            const BaseTensor& source = impl_->base.at(name);
            int block_elements = 0, block_bytes = 0;
            if (w->elements <= 0 || row.bytes / width != row.elements || row.bytes % width != 0 ||
                !source_geometry(source.type_id, block_elements, block_bytes) ||
                row.elements % block_elements != 0 ||
                source.bytes != row.elements / block_elements * block_bytes) {
                return fail(name + ": unsupported source-to-engine conversion");
            }
            kernels::Hc8View view;
            if (kernels::hc8_view(w->data, view)) {
                row.hc8_key = w->data;  // the 8-bit copy is keyed by the bf16 matrix address
                row.hc8_rows = view.rows;
                row.hc8_cols = view.cols;
                if (view.owned) {
                    row.target = nullptr;  // w->data is the owned 8-bit copy, not a bf16 arena buffer
                }
                if (row.type_id != kTypeBf16 || row.elements != (uint64_t) view.rows * view.cols)
                    return fail(name + ": HC8 shape or engine format differs");
                if (view.owned)
                    hc8_stage = std::max(hc8_stage, static_cast<size_t>(view.rows) * view.cols * 2);
            }
            impl_->rows.push_back(std::move(row));
        }
        if (native) {
            uint64_t bytes = 0;
            const auto nb = impl_->native_bytes.find(name);
            if (nb != impl_->native_bytes.end()) {
                bytes = nb->second;
            } else {
                bytes = kernels::native_mmvq_weight_bytes(w->native_type, (int) w->ne0, (int) w->ne1);
            }
            Impl::Row row;
            row.name = name;
            row.device = device;
            row.target = const_cast<void*>(w->native_data);
            row.bytes = bytes;
            row.native = true;
            impl_->rows.push_back(std::move(row));
        }
    }
    auto& slot = impl_->staging[device];
    if (hc8_stage > slot.second) {
        void* staging = nullptr;
        if (cudaMalloc(&staging, hc8_stage) != cudaSuccess)
            return fail("adapters: HC8 staging allocation failed on CUDA" + std::to_string(device));
        if (slot.first != nullptr) cudaFree(slot.first);
        slot = {staging, hc8_stage};
    }
    if (!impl_->rows.empty()) {
        impl_->input.resize(kChunk);
        impl_->values.resize(kChunk / sizeof(float));
        impl_->rounded.resize(kChunk / sizeof(float));
    }
    cudaSetDevice(saved);
    basalt::log::info("basalt adapters", "bound %zu replacement rows on CUDA%d", impl_->rows.size(), device);
    return true;
}
bool RuntimeAdapters::Impl::upload(const Row& row, const AdapterTensor& source, void* target, std::string& err) {
    const auto fail = [&](const char* message) {
        err = row.name + ": " + message;
        return false;
    };
    if (row.native || source.type_id == row.type_id) {
        if (source.data.bytes != row.bytes) return fail("source size differs from engine storage");
        for (uint64_t offset = 0; offset < row.bytes;) {
            const uint64_t bytes = std::min<uint64_t>(input.size(), row.bytes - offset);
            if (!read_range_at(source.data, offset, input.data(), bytes, err)) return false;
            if (cudaMemcpy((uint8_t*) target + offset, input.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
                return fail("upload failed");
            offset += bytes;
        }
        return true;
    }
    int block_elements = 0, block_bytes = 0;
    if (!source_geometry(source.type_id, block_elements, block_bytes) ||
        row.elements % block_elements != 0 ||
        source.data.bytes != row.elements / block_elements * block_bytes)
        return fail("source geometry differs from engine storage");
    const uint64_t blocks_per_chunk = std::min<uint64_t>(input.size() / block_bytes, values.size() / block_elements);
    const uint64_t width = row.type_id == 0 ? 4 : 2;
    for (uint64_t done = 0; done < row.elements;) {
        const uint64_t blocks = std::min(blocks_per_chunk, (row.elements - done) / block_elements);
        const uint64_t count = blocks * block_elements;
        if (!read_range_at(source.data, done / block_elements * block_bytes, input.data(),
                           blocks * block_bytes, err)) return false;
        decode_source(source.type_id, input.data(), values.data(), (size_t) count);
        const void* host = values.data();
        if (row.type_id != 0) {
            for (size_t i = 0; i < count; ++i) {
                if (!std::isfinite(values[i])) return fail("non-finite source cannot be narrowed");
                rounded[i] = row.type_id == kTypeBf16 ? kernels::bf16_from_f32(values[i]) :
                                                      kernels::f16_from_f32(values[i]);
                const uint16_t exponent = row.type_id == kTypeBf16 ? 0x7f80 : 0x7c00;
                if ((rounded[i] & exponent) == exponent) return fail("source overflows engine storage");
            }
            host = rounded.data();
        }
        if (cudaMemcpy((uint8_t*) target + done * width, host, count * width, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("upload failed");
        done += count;
    }
    return true;
}

bool RuntimeAdapters::select(size_t model, std::string& err) {
    err.clear();
    if (model >= impl_->models.size()) {
        err = "adapters: model " + std::to_string(model) + " does not exist";
        return false;
    }
    if (model == impl_->current) return true;
    std::vector<std::string> set;
    for (const std::string& name : impl_->names) {
        if (changed(impl_->current, model, name)) set.push_back(name);
    }
    for (const std::string& name : set) {
        if (name == "output.weight" || name == "token_embd.weight") continue;
        const bool bound = std::any_of(impl_->rows.begin(), impl_->rows.end(),
                                       [&](const Impl::Row& row) { return row.name == name; });
        if (!bound) {
            err = name + ": tensor is not bound on any device";
            return false;
        }
    }
    if (set.empty()) {
        impl_->current = model;
        basalt::log::info("basalt adapters", "selected model %zu, 0 tensors replaced", model);
        return true;
    }
    const int in_dev = [] { int d = 0; cudaGetDevice(&d); return d; }();
    // A failed switch may have uploaded only part of the model; the caller must stop serving.
    int cur = in_dev;
    const auto use_device = [&](int d) {
        if (d == cur) return true;
        if (cudaSetDevice(d) != cudaSuccess) return false;
        cur = d;
        return true;
    };
    const auto fail = [&](const std::string& message) {
        cudaSetDevice(in_dev);
        err = message;
        return false;
    };

    for (const Impl::Row& row : impl_->rows) {
        if (std::find(set.begin(), set.end(), row.name) == set.end()) continue;
        if (!use_device(row.device)) return fail(row.name + ": device unavailable");
        void* target = row.target;
        if (target == nullptr) {
            const auto staging = impl_->staging.find(row.device);
            if (staging == impl_->staging.end() || staging->second.second < row.bytes)
                return fail(row.name + ": no HC8 staging on CUDA" + std::to_string(row.device));
            target = staging->second.first;
        }
        if (!impl_->upload(row, impl_->models[model].at(row.name), target, err)) return fail(err);
        if (row.hc8_key != nullptr &&
            !kernels::hc8_refresh(row.hc8_key, (const uint16_t*) target, row.hc8_rows, row.hc8_cols))
            return fail(row.name + ": HC8 refresh failed");
    }

    cudaSetDevice(in_dev);
    impl_->current = model;
    basalt::log::info("basalt adapters", "selected model %zu, %zu tensors replaced", model, set.size());
    return true;
}

size_t RuntimeAdapters::current() const { return impl_->current; }

}  // namespace basalt::core
