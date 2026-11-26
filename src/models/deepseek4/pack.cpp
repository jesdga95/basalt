#include "models/deepseek4/pack.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>

namespace basalt::models::deepseek4 {
namespace {

using core::Json;

const Json* member(const Json& object, const char* name) { return object.get(name); }

double number(const Json* value, double fallback = 0.0) {
    if (!value || !value->is_number()) return fallback;
    const double result = value->num();
    return std::isfinite(result) ? result : fallback;
}

int integer(const Json* value, int fallback = 0) {
    const double result = number(value, std::numeric_limits<double>::quiet_NaN());
    if (!std::isfinite(result) || std::trunc(result) != result || result < std::numeric_limits<int>::min() ||
        result > std::numeric_limits<int>::max()) return fallback;
    return (int) result;
}

bool read_geometry(const Json& json, Geometry& g, std::string& err) {
    g.n_layers = integer(member(json, "block_count"));
    g.n_embd = integer(member(json, "embedding_length"));
    g.n_head = integer(member(json, "attention.head_count"));
    g.head_dim = integer(member(json, "attention.key_length"));
    g.rope_dim = integer(member(json, "rope.dimension_count"));
    g.q_lora = integer(member(json, "attention.q_lora_rank"));
    g.window = integer(member(json, "attention.sliding_window"));
    g.o_groups = integer(member(json, "attention.output_group_count"));
    g.o_lora = integer(member(json, "attention.output_lora_rank"));
    g.n_head_kv = integer(member(json, "attention.head_count_kv"));
    g.n_expert = integer(member(json, "expert_count"));
    g.n_used = integer(member(json, "expert_used_count"));
    g.router_func = integer(member(json, "expert_gating_func"), -1);
    g.output_embd = integer(member(json, "embedding_length_out"));
    if (const Json* norm = member(json, "expert_weights_norm")) g.route_norm = norm->type == Json::Type::Bool && norm->b;
    g.n_ff_exp = integer(member(json, "expert_feed_forward_length"));
    g.n_shared = integer(member(json, "expert_shared_count"));
    g.hc_mult = integer(member(json, "hyper_connection.count"));
    g.sinkhorn_iters = integer(member(json, "hyper_connection.sinkhorn_iterations"));
    g.hash_layers = integer(member(json, "hash_layer_count"));
    g.idx_heads = integer(member(json, "attention.indexer.head_count"));
    g.idx_dim = integer(member(json, "attention.indexer.key_length"));
    g.idx_top_k = integer(member(json, "attention.indexer.top_k"));
    g.rms_eps = (float) number(member(json, "attention.layer_norm_rms_epsilon"));
    g.hc_eps = (float) number(member(json, "hyper_connection.epsilon"));
    g.route_scale = (float) number(member(json, "expert_weights_scale"));
    g.rope_base = (float) number(member(json, "rope.freq_base"));
    g.comp_rope_base = (float) number(member(json, "attention.compress_rope_freq_base"));
    g.context_length = (int64_t) number(member(json, "context_length"));
    g.yarn_factor = (float) number(member(json, "rope.scaling.factor"));
    g.yarn_orig_ctx = integer(member(json, "rope.scaling.original_context_length"));
    g.yarn_beta_fast = (float) number(member(json, "rope.scaling.yarn_beta_fast"));
    g.yarn_beta_slow = (float) number(member(json, "rope.scaling.yarn_beta_slow"));

    const Json* ratios = member(json, "attention.compress_ratios");
    if (ratios && !parse_compression_ratios(*ratios, g.compress_ratio, err)) {
        err = "meta.geometry: " + err;
        return false;
    }

    if (g.n_layers <= 0 || g.n_embd <= 0 || g.n_head <= 0 || g.n_head_kv != 1 || g.head_dim <= 0 ||
        g.rope_dim <= 0 || g.q_lora <= 0 || g.o_groups <= 0 || g.o_lora <= 0 || g.n_expert <= 0 ||
        g.n_used <= 0 || g.n_ff_exp <= 0 || g.hc_mult != 4 || g.idx_heads <= 0 || g.idx_dim <= 0 ||
        g.idx_top_k <= 0 || g.output_embd <= 0 || !g.route_norm || g.context_length <= 0 || g.rms_eps <= 0.f ||
        g.hc_eps <= 0.f || g.rope_base <= 0.f || g.comp_rope_base <= 0.f || g.yarn_factor < 1.f ||
        g.yarn_orig_ctx <= 0 || g.yarn_beta_fast <= 0.f || g.yarn_beta_slow <= 0.f || g.n_used != 6 ||
        g.hash_layers != 3 || g.router_func != 4 || g.route_scale != 1.5f ||
        g.output_embd != (int64_t) g.n_embd * g.hc_mult) {
        err = "meta.geometry is incomplete or outside the supported DeepSeek-V4 geometry";
        return false;
    }
    if (g.n_used > g.n_expert || g.n_layers > 64 || g.hash_layers < 0 || g.hash_layers > g.n_layers ||
        g.compress_ratio.size() != (size_t) g.n_layers + 3) {
        err = "meta.geometry has an invalid expert count, layer count, or compression schedule";
        return false;
    }
    for (int i = 0; i < g.n_layers; ++i) {
        const int ratio = g.compress_ratio[(size_t) i];
        if (ratio != 0 && ratio != 4 && ratio != 128) {
            err = "meta.geometry has unsupported compression ratio " + std::to_string(ratio) +
                  " at layer " + std::to_string(i);
            return false;
        }
    }
    for (size_t i = (size_t) g.n_layers; i < g.compress_ratio.size(); ++i) {
        if (g.compress_ratio[i] != 0) {
            err = "meta.geometry has a nonzero compression ratio after the transformer layers";
            return false;
        }
    }
    g.compress_ratio.resize((size_t) g.n_layers);
    return validate_supported_geometry(g, err);
}

bool tensor_shape(const Pack& pack, const std::string& name, std::initializer_list<uint64_t> shape, int type = -1) {
    const TensorRef* tensor = pack.find(name);
    if (!tensor || tensor->shape != std::vector<uint64_t>(shape) || (type >= 0 && tensor->type != type)) return false;
    return true;
}

bool validate_tensor_layout(const Pack& pack, std::string& err) {
    const Geometry& g = pack.geo;
    if (!tensor_shape(pack, "token_embd.weight", {(uint64_t) g.n_embd, (uint64_t) g.n_vocab}, 30) ||
        !tensor_shape(pack, "output.weight", {(uint64_t) g.n_embd, (uint64_t) g.n_vocab}, 30) ||
        !tensor_shape(pack, "output_norm.weight", {(uint64_t) g.n_embd})) {
        err = "token embedding or output tensor has an unexpected shape or type";
        return false;
    }
    for (int layer = 0; layer < g.n_layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const auto has_shape = [&](const char* suffix, std::initializer_list<uint64_t> shape, int type = -1) {
            return tensor_shape(pack, prefix + suffix, shape, type);
        };
        if (!has_shape("attn_q_a.weight", {(uint64_t) g.n_embd, (uint64_t) g.q_lora}) ||
            !has_shape("attn_q_a_norm.weight", {(uint64_t) g.q_lora}, 0) ||
            !has_shape("attn_q_b.weight", {(uint64_t) g.q_lora, (uint64_t) g.n_head * g.head_dim}) ||
            !has_shape("attn_kv.weight", {(uint64_t) g.n_embd, (uint64_t) g.head_dim}) ||
            !has_shape("attn_kv_a_norm.weight", {(uint64_t) g.head_dim}, 0) ||
            !has_shape("attn_sinks.weight", {(uint64_t) g.n_head}, 0) ||
            !has_shape("attn_output_a.weight", {(uint64_t) g.n_head * g.head_dim / g.o_groups,
                                                 (uint64_t) g.o_groups * g.o_lora}) ||
            !has_shape("attn_output_b.weight", {(uint64_t) g.o_groups * g.o_lora, (uint64_t) g.n_embd}) ||
            !has_shape("attn_norm.weight", {(uint64_t) g.n_embd}, 0) ||
            !has_shape("ffn_norm.weight", {(uint64_t) g.n_embd}, 0) ||
            !has_shape("ffn_gate_inp.weight", {(uint64_t) g.n_embd, (uint64_t) g.n_expert}, 30) ||
            !has_shape("ffn_gate_shexp.weight", {(uint64_t) g.n_embd, (uint64_t) g.n_ff_exp}) ||
            !has_shape("ffn_up_shexp.weight", {(uint64_t) g.n_embd, (uint64_t) g.n_ff_exp}) ||
            !has_shape("ffn_down_shexp.weight", {(uint64_t) g.n_ff_exp, (uint64_t) g.n_embd}) ||
            !has_shape("exp_probs_b_vl.bias", {(uint64_t) g.n_expert}, 0) ||
            !has_shape("hc_attn_base.weight", {24}, 0) ||
            !has_shape("hc_attn_fn.weight", {(uint64_t) g.output_embd, 24}, 0) ||
            !has_shape("hc_attn_scale.weight", {3}, 0) || !has_shape("hc_ffn_base.weight", {24}, 0) ||
            !has_shape("hc_ffn_fn.weight", {(uint64_t) g.output_embd, 24}, 0) ||
            !has_shape("hc_ffn_scale.weight", {3}, 0) ||
            (layer >= g.hash_layers && !has_shape("exp_probs_b.bias", {(uint64_t) g.n_expert}, 0))) {
            err = "layer " + std::to_string(layer) + " has a missing tensor or unsupported attention, expert, or mHC shape/type";
            return false;
        }
        const int ratio = g.compress_ratio[(size_t) layer];
        if (ratio == 4) {
            if (!tensor_shape(pack, prefix + "attn_compressor_ape.weight", {(uint64_t) 2 * g.head_dim, 4}, 0) ||
                !tensor_shape(pack, prefix + "attn_compressor_norm.weight", {(uint64_t) g.head_dim}, 0) ||
                !tensor_shape(pack, prefix + "attn_compressor_gate.weight", {(uint64_t) g.n_embd, (uint64_t) 2 * g.head_dim}, 30) ||
                !tensor_shape(pack, prefix + "attn_compressor_kv.weight", {(uint64_t) g.n_embd, (uint64_t) 2 * g.head_dim}, 30) ||
                !tensor_shape(pack, prefix + "indexer_compressor_ape.weight", {(uint64_t) 2 * g.idx_dim, 4}, 0) ||
                !tensor_shape(pack, prefix + "indexer_compressor_norm.weight", {(uint64_t) g.idx_dim}, 0) ||
                !tensor_shape(pack, prefix + "indexer_compressor_gate.weight", {(uint64_t) g.n_embd, (uint64_t) 2 * g.idx_dim}, 30) ||
                !tensor_shape(pack, prefix + "indexer_compressor_kv.weight", {(uint64_t) g.n_embd, (uint64_t) 2 * g.idx_dim}, 30) ||
                !tensor_shape(pack, prefix + "indexer.proj.weight", {(uint64_t) g.n_embd, (uint64_t) g.idx_heads}, 30) ||
                !tensor_shape(pack, prefix + "indexer.attn_q_b.weight", {(uint64_t) g.q_lora,
                                                                           (uint64_t) g.idx_heads * g.idx_dim}, 8)) {
                err = "layer " + std::to_string(layer) + " has invalid or missing CSA tensors";
                return false;
            }
        } else if (ratio == 128) {
            if (!tensor_shape(pack, prefix + "attn_compressor_ape.weight", {512, 128}, 0) ||
                !tensor_shape(pack, prefix + "attn_compressor_norm.weight", {512}, 0) ||
                !tensor_shape(pack, prefix + "attn_compressor_gate.weight", {4096, 512}, 30) ||
                !tensor_shape(pack, prefix + "attn_compressor_kv.weight", {4096, 512}, 30)) {
                err = "layer " + std::to_string(layer) + " has invalid or missing HCA tensors";
                return false;
            }
        }
    }
    for (int layer = 0; layer < g.hash_layers; ++layer) {
        if (!tensor_shape(pack, "blk." + std::to_string(layer) + ".ffn_gate_tid2eid.weight",
                          {6, (uint64_t) g.n_vocab}, 26)) {
            err = "hash-routed layer " + std::to_string(layer) + " has an invalid token-to-expert tensor";
            return false;
        }
    }
    return true;
}

bool validate_hash_tables(const Pack& pack, std::string& err) {
    const Geometry& g = pack.geo;
    if (g.hash_layers == 0) return true;
    const core::FileRange gguf = pack.file.range("gguf");
    if (gguf.empty()) {
        err = "GGUF tensor section is missing for hash routing";
        return false;
    }
    constexpr uint64_t kHashWidth = 6;
    const uint64_t row_bytes = kHashWidth * sizeof(int32_t);
    for (int layer = 0; layer < g.hash_layers; ++layer) {
        const std::string name = "blk." + std::to_string(layer) + ".ffn_gate_tid2eid.weight";
        const TensorRef* tensor = pack.find(name);
        if (!tensor || tensor->bytes != (uint64_t) g.n_vocab * row_bytes || tensor->offset < gguf.offset ||
            tensor->offset - gguf.offset > gguf.bytes || tensor->bytes > gguf.bytes - (tensor->offset - gguf.offset)) {
            err = "hash-routed layer " + std::to_string(layer) + " has an invalid token-to-expert data range";
            return false;
        }
        constexpr int kChunkTokens = 4096;
        std::vector<int32_t> ids((size_t) kChunkTokens * kHashWidth);
        for (int first_token = 0; first_token < g.n_vocab; first_token += kChunkTokens) {
            const int token_count = std::min(kChunkTokens, g.n_vocab - first_token);
            const uint64_t chunk_bytes = (uint64_t) token_count * row_bytes;
            const uint64_t chunk_offset = tensor->offset - gguf.offset + (uint64_t) first_token * row_bytes;
            if (!core::read_range_at(gguf, chunk_offset, ids.data(), chunk_bytes, err)) {
                err = "cannot read hash-routed layer " + std::to_string(layer) + ": " + err;
                return false;
            }
            for (int row_index = 0; row_index < token_count; ++row_index) {
                const int token = first_token + row_index;
                const int32_t* row = ids.data() + (size_t) row_index * kHashWidth;
                for (uint64_t i = 0; i < kHashWidth; ++i) {
                    if (row[i] < 0 || row[i] >= g.n_expert) {
                        err = "hash-routed layer " + std::to_string(layer) + " has an out-of-range expert id at token " +
                              std::to_string(token);
                        return false;
                    }
                    for (uint64_t j = 0; j < i; ++j) {
                        if (row[i] == row[j]) {
                            err = "hash-routed layer " + std::to_string(layer) + " has duplicate expert ids at token " +
                                  std::to_string(token);
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool read_expert_layout(const core::ModelFile& file, int layers, std::vector<ExpertLayer>& out, std::string& err) {
    const core::ModelFile::Section* section = file.find("experts");
    if (!section) {
        err = "experts section is missing";
        return false;
    }
    std::vector<uint8_t> bytes;
    if (!file.read("experts.layout", bytes, err)) return false;
    std::istringstream in(std::string(bytes.begin(), bytes.end()));
    std::string line;
    out.assign((size_t) layers, {});
    std::vector<bool> seen((size_t) layers, false);
    int version = 0;
    while (std::getline(in, line)) {
        if (line.rfind("# experts v", 0) == 0) version = std::atoi(line.c_str() + 11);
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        long long layer = -1, gate_up = -1, down = -1, up = -1;
        unsigned long long offset = 0, size = 0;
        if (!(row >> layer >> gate_up >> down >> offset >> size) || layer < 0 || layer >= layers ||
            gate_up < 0 || down < 0 || size == 0 || offset > section->bytes || size > section->bytes - offset ||
            (version >= 7 && !(row >> up))) {
            err = "experts.layout: malformed row '" + line + "'";
            return false;
        }
        if (version >= 7 && up != gate_up) {
            err = "experts.layout: gate/up storage types differ at layer " + std::to_string(layer);
            return false;
        }
        if (seen[(size_t) layer]) {
            err = "experts.layout: duplicate layer " + std::to_string(layer);
            return false;
        }
        seen[(size_t) layer] = true;
        out[(size_t) layer] = {(int) gate_up, (int) down, offset, size};
    }
    for (int layer = 0; layer < layers; ++layer) {
        if (!seen[(size_t) layer]) {
            err = "experts.layout is missing layer " + std::to_string(layer);
            return false;
        }
    }
    return true;
}

}  // namespace

bool validate_supported_geometry(const Geometry& g, std::string& err) {
    if (g.n_layers != 43 || g.n_embd != 4096 || g.n_head != 64 || g.n_head_kv != 1 || g.head_dim != 512 ||
        g.rope_dim != 64 || g.q_lora != 1024 || g.window != 128 || g.o_groups != 8 || g.o_lora != 1024 ||
        g.n_expert != 256 || g.n_used != 6 || g.n_ff_exp != 2048 || g.n_shared != 1 || g.hc_mult != 4 ||
        g.sinkhorn_iters != 20 || g.hash_layers != 3 || g.router_func != 4 || !g.route_norm ||
        g.route_scale != 1.5f || g.output_embd != 16384 || g.idx_heads != 64 || g.idx_dim != 128 ||
        g.idx_top_k != 512 || g.context_length != 1048576 || g.yarn_factor != 16.f || g.yarn_orig_ctx != 65536 ||
        g.yarn_beta_fast != 32.f || g.yarn_beta_slow != 1.f || g.rope_base != 10000.f || g.comp_rope_base != 160000.f) {
        err = "meta.geometry is outside the supported DeepSeek-V4-0731 geometry";
        return false;
    }
    err.clear();
    return true;
}

bool parse_compression_ratios(const core::Json& value, std::vector<int>& ratios, std::string& err) {
    if (!value.is_array() || value.items.size() != value.count) {
        err = "compression ratios must be a complete JSON array";
        return false;
    }
    std::vector<int> parsed;
    parsed.reserve(value.items.size());
    for (size_t i = 0; i < value.items.size(); ++i) {
        const int ratio = integer(&value.items[i], -1);
        if (ratio < 0) {
            err = "compression ratio at index " + std::to_string(i) + " is not a nonnegative integer";
            return false;
        }
        parsed.push_back(ratio);
    }
    ratios = std::move(parsed);
    err.clear();
    return true;
}

bool open_pack(const std::string& path, Pack& pack, std::string& err) {
    if (!pack.file.open(path, err) || !pack.file.load_meta(err)) return false;
    const Json& meta = pack.file.meta();
    const Json* arch = member(meta, "arch");
    const Json* geometry = member(meta, "geometry");
    const Json* tensors = member(meta, "tensors");
    if (!arch || !arch->is_string() || arch->s != "deepseek4" || !geometry || !geometry->is_object() ||
        !tensors || !tensors->is_object()) {
        err = path + ": not a complete deepseek4 pack";
        return false;
    }
    if (!read_geometry(*geometry, pack.geo, err)) {
        err = path + ": " + err;
        return false;
    }

    const Json* kv = member(meta, "gguf_kv");
    if (kv) {
        const Json* vocab = member(*kv, "tokenizer.ggml.tokens");
        if (vocab && vocab->is_array()) pack.geo.n_vocab = (int) vocab->count;
    }
    if (pack.geo.n_vocab <= 0) {
        err = path + ": tokenizer.ggml.tokens is missing from gguf_kv";
        return false;
    }

    for (const auto& [name, value] : tensors->members) {
        TensorRef tensor;
        tensor.type = integer(member(value, "type_id"), -1);
        tensor.bytes = (uint64_t) number(member(value, "bytes"));
        const Json* shape = member(value, "shape");
        const Json* offset = member(value, "offset");
        if (tensor.type < 0 || tensor.bytes == 0 || !shape || !shape->is_array() || !offset) {
            err = path + ": invalid tensor record for " + name;
            return false;
        }
        for (const Json& dim : shape->items) tensor.shape.push_back(dim.u64());
        std::string section = "gguf";
        if (const Json* section_value = member(value, "section"); section_value && section_value->is_string())
            section = section_value->s;
        const core::ModelFile::Section* source = pack.file.find(section);
        if (!source || offset->u64() > source->bytes || tensor.bytes > source->bytes - offset->u64()) {
            err = path + ": tensor " + name + " has an invalid section offset";
            return false;
        }
        tensor.offset = source->offset + offset->u64();
        pack.tensors.emplace(name, std::move(tensor));
    }

    if (!validate_tensor_layout(pack, err)) {
        err = path + ": " + err;
        return false;
    }
    if (!validate_hash_tables(pack, err)) {
        err = path + ": " + err;
        return false;
    }
    if (!pack.file.find("experts") || !pack.file.find("experts.layout")) {
        err = path + ": routed expert sections are missing";
        return false;
    }
    if (!read_expert_layout(pack.file, pack.geo.n_layers, pack.experts, err)) return false;
    pack.vision = pack.file.range("vision/mmproj");
    if (pack.vision.empty()) {
        err = path + ": vision/mmproj section is missing";
        return false;
    }
    return true;
}

bool read_embedding_row(const Pack& pack, int32_t token, std::vector<float>& row, std::string& err) {
    const Geometry& g = pack.geo;
    const TensorRef* tensor = pack.find("token_embd.weight");
    const core::FileRange gguf = pack.file.range("gguf");
    if (!tensor || tensor->type != 30 || tensor->shape != std::vector<uint64_t>{(uint64_t) g.n_embd, (uint64_t) g.n_vocab} ||
        g.n_embd <= 0 || g.n_vocab <= 0 || token < 0 || token >= g.n_vocab || gguf.empty()) {
        err = "deepseek4 token embedding: invalid tensor, token id, or GGUF section";
        return false;
    }
    const uint64_t n_embd = (uint64_t) g.n_embd;
    if (n_embd > std::numeric_limits<uint64_t>::max() / (uint64_t) g.n_vocab / 2) {
        err = "deepseek4 token embedding: tensor size overflows";
        return false;
    }
    const uint64_t row_bytes = n_embd * sizeof(uint16_t);
    const uint64_t tensor_bytes = row_bytes * (uint64_t) g.n_vocab;
    if (tensor->bytes != tensor_bytes || tensor->offset < gguf.offset ||
        tensor->offset - gguf.offset > gguf.bytes || tensor_bytes > gguf.bytes - (tensor->offset - gguf.offset) ||
        (uint64_t) token > std::numeric_limits<uint64_t>::max() / row_bytes) {
        err = "deepseek4 token embedding: invalid tensor bounds";
        return false;
    }
    std::vector<uint8_t> bytes((size_t) row_bytes);
    if (!core::read_range_at(gguf, tensor->offset - gguf.offset + (uint64_t) token * row_bytes,
                             bytes.data(), row_bytes, err)) {
        err = "deepseek4 token embedding: " + err;
        return false;
    }
    std::vector<float> decoded((size_t) n_embd);
    for (size_t i = 0; i < decoded.size(); ++i) {
        const uint32_t bits = ((uint32_t) bytes[2 * i] | ((uint32_t) bytes[2 * i + 1] << 8)) << 16;
        std::memcpy(&decoded[i], &bits, sizeof(bits));
    }
    row = std::move(decoded);
    err.clear();
    return true;
}

bool load_hash_route_table(const Pack& pack, int layer, std::vector<int32_t>& ids, std::string& err) {
    const Geometry& g = pack.geo;
    if (layer < 0 || layer >= g.hash_layers || g.n_vocab <= 0 || g.n_used != 6) {
        err = "deepseek4 hash route table: invalid layer or routing geometry";
        return false;
    }
    const std::string name = "blk." + std::to_string(layer) + ".ffn_gate_tid2eid.weight";
    const TensorRef* tensor = pack.find(name);
    const core::FileRange gguf = pack.file.range("gguf");
    constexpr uint64_t row_bytes = 6 * sizeof(int32_t);
    const uint64_t bytes = (uint64_t) g.n_vocab * row_bytes;
    if (!tensor || tensor->type != 26 || tensor->shape != std::vector<uint64_t>{6, (uint64_t) g.n_vocab} ||
        tensor->bytes != bytes || tensor->offset < gguf.offset ||
        tensor->offset - gguf.offset > gguf.bytes || bytes > gguf.bytes - (tensor->offset - gguf.offset) ||
        bytes > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "deepseek4 hash route table: invalid tensor bounds or shape for layer " + std::to_string(layer);
        return false;
    }
    std::vector<int32_t> loaded((size_t) g.n_vocab * 6);
    if (!core::read_range_at(gguf, tensor->offset - gguf.offset, loaded.data(), bytes, err)) {
        err = "deepseek4 hash route table: " + err;
        return false;
    }
    for (size_t i = 0; i < loaded.size(); ++i) {
        if (loaded[i] < 0 || loaded[i] >= g.n_expert) {
            err = "deepseek4 hash route table: expert id out of range at flat entry " + std::to_string(i);
            return false;
        }
    }
    ids = std::move(loaded);
    err.clear();
    return true;
}

}  // namespace basalt::models::deepseek4
