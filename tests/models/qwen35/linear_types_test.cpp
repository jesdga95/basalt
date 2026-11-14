// tests/models/qwen35/linear_types_test.cpp - which tensor types and shapes a 27B matrix may have, and the refusals
// for the rest: a pack from before NVFP4_PLANES (NVFP4 in ggml blocks) is told to re-convert. Host only.
#include "models/qwen35/meta.hpp"

#include <cstdio>
#include <string>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
}  // namespace

int main() {
    using namespace basalt::models::qwen35;
    const std::string w = "blk.0.ffn_gate";
    check(linear_type_error(w, kTypeQ8_0, 17408, 5120).empty(), "Q8_0 runs");
    check(linear_type_error(w, kTypeNvfp4Planes, 17408, 5120).empty(), "NVFP4_PLANES of 128-row blocks runs");
    check(linear_type_error(w, kTypeF8, 6144, 5120).empty(), "F8_E4M3 with 256-value rows runs");

    const std::string old = linear_type_error(w, kTypeNvfp4, 17408, 5120);
    check(has(old, "blk.0.ffn_gate.weight") && has(old, "older than NVFP4_PLANES") && has(old, "re-convert"),
          "an old pack's ggml-block NVFP4 matrix is told to re-convert, by name");

    const std::string f32 = linear_type_error(w, kTypeF32, 17408, 5120);
    check(has(f32, "blk.0.ffn_gate.weight is F32") && has(f32, "Q8_0, NVFP4_PLANES and F8_E4M3"),
          "another type is named, with the types the kernels read");
    check(has(linear_type_error(w, kTypeNvfp4Planes, 100, 5120), "rows in 128s"), "planes of 100 rows are refused");
    check(has(linear_type_error(w, kTypeNvfp4Planes, 128, 96), "64-value groups"), "planes of 96 values are refused");
    check(has(linear_type_error(w, kTypeF8, 6144, 5000), "multiples of 256"), "F8 rows of 5000 values are refused");

    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
