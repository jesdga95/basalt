// src/models/ernie55/pretok_test.cpp - unit test for ERNIE-5.5 pre-tokenizer (right-aligned 3-digit grouping, CJK, words).
#include "models/ernie55/adapter.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void check(const char* name, const std::string& input, const std::vector<std::string>& want) {
    std::vector<size_t> ends;
    basalt::models::ernie55::pretok_ernie55(input.data(), input.size(), ends);
    std::vector<std::string> got;
    size_t prev = 0;
    for (size_t e : ends) {
        got.push_back(input.substr(prev, e - prev));
        prev = e;
    }
    if (got != want) {
        ++g_fail;
        std::fprintf(stderr, "FAIL %s: input='%s'\n  got (%zu):", name, input.c_str(), got.size());
        for (const auto& s : got) std::fprintf(stderr, " [%s]", s.c_str());
        std::fprintf(stderr, "\n  want(%zu):", want.size());
        for (const auto& s : want) std::fprintf(stderr, " [%s]", s.c_str());
        std::fprintf(stderr, "\n");
    }
}

}  // namespace

int main() {
    check("empty", "", {});
    check("digits_1", "7", {"7"});
    check("digits_3", "123", {"123"});
    check("digits_4_right_aligned", "1234", {"1", "234"});
    check("digits_5_right_aligned", "12345", {"12", "345"});
    check("digits_6_right_aligned", "123456", {"123", "456"});
    check("digits_7_right_aligned", "1234567", {"1", "234", "567"});
    check("mixed_text_digits", "ERNIE 5.5 has 280000 params", {"ERNIE", " ", "5", ".", "5", " has", " ", "280", "000", " params"});
    check("cjk_isolated", "Hello世界12345!", {"Hello", "世界", "12", "345", "!"});
    check("punct_letter", "foo.bar", {"foo", ".bar"});
    std::printf("ernie55_pretok_test: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
