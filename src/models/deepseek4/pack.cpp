#include "models/deepseek4/pack.hpp"

#include <cstdlib>
#include <sstream>

namespace basalt::models::deepseek4 {
namespace {

using core::Json;

const Json* member(const Json& object, const char* name) { return object.get(name); }

double number(const Json* value, double fallback = 0.0) {
    return value && value->is_number() ? value->num() : fallback;
}

int integer(const Json* value, int fallback = 0) {
    return (int) number(value, fallback);
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
    g.n_expert = integer(member(json, "expert_count"));
    g.n_used = integer(member(json, "expert_used_count"));
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

    const Json* ratios = member(json, "attention.compress_ratios");
    if (ratios && ratios->is_array())
        for (const Json& value : ratios->items) g.compress_ratio.push_back((int) value.num());

    if (g.n_layers <= 0 || g.n_embd <= 0 || g.n_head <= 0 || g.head_dim <= 0 || g.rope_dim <= 0 ||
        g.q_lora <= 0 || g.o_groups <= 0 || g.o_lora <= 0 || g.n_expert <= 0 || g.n_used <= 0 ||
        g.n_ff_exp <= 0 || g.hc_mult != 4 || g.context_length <= 0 || g.rms_eps <= 0.f || g.hc_eps <= 0.f) {
        err = "meta.geometry is incomplete or outside the supported DeepSeek-V4 geometry";
        return false;
    }
    if (g.n_used > g.n_expert || g.n_layers > 64 || g.hash_layers < 0 || g.hash_layers > g.n_layers ||
        g.compress_ratio.size() < (size_t) g.n_layers) {
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
    return true;
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
        if (!pack.find(prefix + "attn_q_a.weight") || !pack.find(prefix + "attn_q_b.weight") ||
            !pack.find(prefix + "attn_kv.weight") || !pack.find(prefix + "attn_output_a.weight") ||
            !pack.find(prefix + "attn_output_b.weight") || !pack.find(prefix + "ffn_gate_inp.weight") ||
            !pack.find(prefix + "ffn_gate_shexp.weight") || !pack.find(prefix + "ffn_up_shexp.weight") ||
            !pack.find(prefix + "ffn_down_shexp.weight") || !pack.find(prefix + "hc_attn_base.weight") ||
            !pack.find(prefix + "hc_attn_fn.weight") || !pack.find(prefix + "hc_ffn_base.weight") ||
            !pack.find(prefix + "hc_ffn_fn.weight")) {
            err = "layer " + std::to_string(layer) + " is missing a required attention, expert, or hyper-connection tensor";
            return false;
        }
        const int ratio = g.compress_ratio[(size_t) layer];
        if (ratio == 4) {
            if (!tensor_shape(pack, prefix + "attn_compressor_ape.weight", {1024, 4}, 0) ||
                !tensor_shape(pack, prefix + "attn_compressor_norm.weight", {512}, 0) ||
                !tensor_shape(pack, prefix + "attn_compressor_gate.weight", {4096, 1024}, 30) ||
                !tensor_shape(pack, prefix + "attn_compressor_kv.weight", {4096, 1024}, 30) ||
                !tensor_shape(pack, prefix + "indexer_compressor_ape.weight", {256, 4}, 0) ||
                !tensor_shape(pack, prefix + "indexer_compressor_norm.weight", {128}, 0) ||
                !tensor_shape(pack, prefix + "indexer_compressor_gate.weight", {4096, 256}, 30) ||
                !tensor_shape(pack, prefix + "indexer_compressor_kv.weight", {4096, 256}, 30) ||
                !pack.find(prefix + "indexer.proj.weight") || !pack.find(prefix + "indexer.attn_q_b.weight")) {
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

}  // namespace basalt::models::deepseek4
