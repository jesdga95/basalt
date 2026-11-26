// tests/models/qwen4exp/cli/kv_batch_test.cpp - which --kv formats run with batch slots, and the refusal for the rest
// (it names the format and the one that works). Host only.
#include "models/qwen4exp/cli/kv_batch.hpp"

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
    using basalt::models::qwen4exp::cli::kv_batch_refusal;
    check(kv_batch_refusal("int8", 8).empty(), "int8 runs with slots");
    for (const char* kv : {"fp16", "q4_0", "k8v4"}) {
        check(kv_batch_refusal(kv, 1).empty(), "one slot takes any format");
        check(kv_batch_refusal(kv, 0).empty(), "no batch takes any format");
    }
    const std::string r = kv_batch_refusal("k8v4", 8);
    check(has(r, "--kv k8v4") && has(r, "8 slots") && has(r, "parallel") && has(r, "--kv int8") && has(r, "resident --kv fp16"),
          "k8v4 with slots is refused by name, with the formats that work");
    check(kv_batch_refusal("fp16", 2).empty(), "a resident fp16 runs with slots");
    check(!kv_batch_refusal("q4_0", 2).empty() && !kv_batch_refusal("k8v4", 2).empty(), "the other formats are refused too");
    return failures == 0 ? 0 : 1;
}
