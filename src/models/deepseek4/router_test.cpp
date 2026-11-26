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
