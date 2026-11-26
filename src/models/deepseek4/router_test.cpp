#include "basalt/models/deepseek4/router.hpp"
#include "basalt/log.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

bool require(bool condition, const char* what) {
    if (!condition) basalt::log::error("basalt deepseek4 router test", "%s", what);
    return condition;
}

}  // namespace

int main() {
    using basalt::models::deepseek4::RouteRows;
    using basalt::models::deepseek4::build_route_rows;
    using basalt::models::deepseek4::router_mixed_args;

    constexpr int vocab = 129280;
    const int32_t tokens[] = {42, vocab + 1, 128, vocab + 17};
    RouteRows rows;
    std::string err;
    if (!build_route_rows(tokens, 4, vocab, 1, 43, 3, rows, err)) {
        basalt::log::error("basalt deepseek4 router test", "valid hash-layer inputs rejected: %s", err.c_str());
        return 1;
    }
    if (!require(rows.fixed_rows == std::vector<int32_t>{42, 0, 128, 0}, "hash layer fixed rows differ") ||
        !require(rows.fixed_mask == std::vector<uint8_t>{1, 0, 1, 0}, "hash layer fixed mask differs") ||
        !require(rows.alternate_mask == std::vector<uint8_t>{0, 1, 0, 1}, "image selector mask differs")) return 1;

    if (!build_route_rows(tokens, 4, vocab, 3, 43, 3, rows, err)) {
        basalt::log::error("basalt deepseek4 router test", "valid scored-layer inputs rejected: %s", err.c_str());
        return 1;
    }
    if (!require(rows.fixed_rows == std::vector<int32_t>{0, 0, 0, 0}, "scored layer retained fixed rows") ||
        !require(rows.fixed_mask == std::vector<uint8_t>{0, 0, 0, 0}, "scored layer retained fixed mask") ||
        !require(rows.alternate_mask == std::vector<uint8_t>{0, 1, 0, 1}, "scored layer image selector mask differs")) return 1;

    const auto* table = reinterpret_cast<const int32_t*>(uintptr_t{1});
    const auto* fixed_rows = reinterpret_cast<const int32_t*>(uintptr_t{2});
    const auto* fixed_mask = reinterpret_cast<const uint8_t*>(uintptr_t{3});
    const auto* alternate_mask = reinterpret_cast<const uint8_t*>(uintptr_t{4});
    const auto* bias = reinterpret_cast<const float*>(uintptr_t{5});
    const auto* alternate_bias = reinterpret_cast<const float*>(uintptr_t{6});
    const auto args = router_mixed_args(256, vocab, table, fixed_rows, fixed_mask, alternate_mask, bias, alternate_bias);
    if (!require(args.n_expert == 256 && args.k == 6 && args.fixed_table_rows == vocab,
                 "mixed selector geometry differs from the V4 pack") ||
        !require(args.weight_scale == 1.5f && args.gate == basalt::kernels::RouterGate::SqrtSoftplus,
                 "mixed selector scoring differs from the V4 pack") ||
        !require(args.fixed_table == table && args.fixed_rows == fixed_rows && args.fixed_mask == fixed_mask &&
                     args.alternate_mask == alternate_mask && args.bias == bias && args.alternate_bias == alternate_bias,
                 "mixed selector inputs were not forwarded")) return 1;

    const int32_t bad_tokens[] = {7, -1};
    if (!require(!build_route_rows(bad_tokens, 2, vocab, 0, 43, 3, rows, err) && err.find("negative token") != std::string::npos,
                 "negative token ID was not refused") ||
        !require(!build_route_rows(tokens, 4, vocab, 43, 43, 3, rows, err) && err.find("invalid") != std::string::npos,
                 "out-of-range layer was not refused") ||
        !require(!build_route_rows(tokens, 4, vocab, 0, 43, 44, rows, err) && err.find("invalid") != std::string::npos,
                 "excess hash-layer count was not refused")) return 1;

    std::puts("deepseek4 route rows: hash routing, image selection, scored layers, and invalid inputs passed");
    return 0;
}
