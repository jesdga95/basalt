#include "basalt/core/loader.hpp"
#include "basalt/core/expert_tier.hpp"
#include "models/deepseek4/model_weights.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace basalt::models::deepseek4;

int main(int argc, char** argv) {
    if (argc != 2) return 77;

    Pack source;
    std::string err;
    if (!open_pack(argv[1], source, err)) {
        std::cerr << err << '\n';
        return 1;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return 77;
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return 77;
    (void) total_bytes;

    uint64_t expected_bytes = 0;
    size_t expected_items = 0;
    bool saw_hash_table = false;
    constexpr char kHashSuffix[] = "ffn_gate_tid2eid.weight";
    for (const auto& [name, tensor] : source.tensors) {
        if (name == "token_embd.weight") continue;
        if (name.size() >= sizeof(kHashSuffix) - 1 &&
            name.compare(name.size() - (sizeof(kHashSuffix) - 1), sizeof(kHashSuffix) - 1, kHashSuffix) == 0) {
            saw_hash_table = true;
            continue;
        }
        const uint64_t bytes = tensor.bytes +
            (tensor.type == 30 && name.find("ffn_gate_inp.weight") != std::string::npos ? tensor.bytes : 0);
        if (bytes > std::numeric_limits<uint64_t>::max() - expected_bytes) return 1;
        expected_bytes += bytes;
        ++expected_items;
    }
    constexpr uint64_t kReserveBytes = 1ull << 30;
    if (!saw_hash_table || expected_items == 0 || free_bytes < kReserveBytes ||
        expected_bytes > (uint64_t) free_bytes - kReserveBytes) {
        std::cout << "skip: dense weight plan needs " << (expected_bytes >> 20) << " MiB; free "
                  << (free_bytes >> 20) << " MiB\n";
        return 77;
    }

    basalt::core::LoadPlan reads;
    ModelWeights weights;
    if (!weights.plan(argv[1], device, reads, err)) {
        std::cerr << err << '\n';
        return 1;
    }
    if (weights.geo().n_layers != source.geo.n_layers || reads.items().size() != expected_items ||
        reads.counted() != expected_bytes || weights.tensor("token_embd.weight") ||
        weights.tensor(0, "ffn_gate_tid2eid.weight")) {
        std::cerr << "DeepSeek4 weight plan does not match pack tensor policy\n";
        return 1;
    }
    basalt::core::ExpertTierSpec tier;
    basalt::model::LoadRequest request;
    if (!weights.expert_tier_spec(request, tier, err) || tier.n_layers != source.geo.n_layers ||
        tier.n_expert != source.geo.n_expert || tier.top_k != 6 || tier.layers.size() != source.experts.size() ||
        tier.layers.front().gu_type != 39 || tier.layers.front().d_type != 39 || tier.fd < 0) {
        std::cerr << (err.empty() ? "DeepSeek4 expert tier spec does not match the real pack" : err) << '\n';
        return 1;
    }

    const basalt::core::FileRange experts = source.file.range("experts");
    std::vector<uint8_t> expert_blob((size_t) source.experts.front().bytes);
    std::vector<uint8_t> expert_reference(expert_blob.size());
    if (experts.empty() || !weights.read_expert_blob(0, 0, expert_blob.data(), expert_blob.size(), err) ||
        !basalt::core::read_range_at(experts, source.experts.front().offset, expert_reference.data(),
                                     expert_reference.size(), err) || expert_blob != expert_reference) {
        std::cerr << (err.empty() ? "DeepSeek4 staged expert blob differs from its pack bytes" : err) << '\n';
        return 1;
    }

    const DeviceTensor* output = weights.tensor("output.weight");
    const DeviceTensor* router = weights.tensor(0, "ffn_gate_inp.weight");
    const TensorRef* output_ref = source.find("output.weight");
    const TensorRef* router_ref = source.find(0, "ffn_gate_inp.weight");
    if (!output || !router || !output_ref || !router_ref || !output->ptr || output->type != output_ref->type ||
        output->bytes != output_ref->bytes || router->type != 0 || router->bytes != router_ref->bytes * 2) {
        std::cerr << "DeepSeek4 weight plan has an invalid output or router tensor\n";
        return 1;
    }

    basalt::core::Loader loader(8);
    if (!loader.prepare({device}, err) || !loader.run(reads, err)) {
        std::cerr << (err.empty() ? "DeepSeek4 one-pass weight placement failed" : err) << '\n';
        return 1;
    }
    const basalt::core::FileRange gguf = source.file.range("gguf");
    uint8_t source_bf16[2]{};
    if (gguf.empty() || router_ref->offset < gguf.offset ||
        !basalt::core::read_range_at(gguf, router_ref->offset - gguf.offset, source_bf16, sizeof(source_bf16), err)) {
        std::cerr << (err.empty() ? "Cannot read the source router weight" : err) << '\n';
        return 1;
    }
    const uint32_t expected_bits = ((uint32_t) source_bf16[0] | ((uint32_t) source_bf16[1] << 8)) << 16;
    float expected_router = 0.f, actual_router = 0.f;
    std::memcpy(&expected_router, &expected_bits, sizeof(expected_router));
    if (cudaMemcpy(&actual_router, router->ptr, sizeof(actual_router), cudaMemcpyDeviceToHost) != cudaSuccess ||
        actual_router != expected_router) {
        std::cerr << "DeepSeek4 BF16 router gate widening does not match its source value\n";
        return 1;
    }
    uint16_t source_output = 0, loaded_output = 0;
    if (output_ref->offset < gguf.offset ||
        !basalt::core::read_range_at(gguf, output_ref->offset - gguf.offset, &source_output, sizeof(source_output), err) ||
        cudaMemcpy(&loaded_output, output->ptr, sizeof(loaded_output), cudaMemcpyDeviceToHost) != cudaSuccess ||
        source_output != loaded_output) {
        std::cerr << (err.empty() ? "DeepSeek4 BF16 output tensor placement changed its bytes" : err) << '\n';
        return 1;
    }

    std::vector<float> embedding;
    if (!weights.embedding_row(0, embedding, err) || embedding.size() != (size_t) source.geo.n_embd) {
        std::cerr << (err.empty() ? "DeepSeek4 embedding row has an invalid width" : err) << '\n';
        return 1;
    }
    std::vector<int32_t> routes;
    if (!weights.hash_route_table(0, routes, err) || routes.size() != (size_t) source.geo.n_vocab * 6) {
        std::cerr << (err.empty() ? "DeepSeek4 hash table has an invalid width" : err) << '\n';
        return 1;
    }

    std::cout << "DeepSeek4 weights placed: " << reads.items().size() << " device tensors, "
              << (reads.counted() >> 20) << " MiB; expert=" << source.geo.n_embd << 'x' << source.geo.n_ff_exp
              << " MXFP4; BF16 router widening verified, embedding and hash routes file-backed\n";
    return 0;
}
