// tests/program/split_cuts_test.cpp - the cut points of --layer-split: the parse and the ordering rules.
#include "basalt/program/split_cuts.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace basalt::program;

static int g_fail = 0;
#define CHECK(c)                                                                                     \
    do {                                                                                             \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; }              \
    } while (0)

static std::vector<int64_t> parse(const std::string& s) {
    std::vector<int64_t> v;
    if (!parse_split_cuts(s, v)) return {};
    return v;
}

int main() {
    // a plain list, the shipped shape
    CHECK(parse("8,20,34") == (std::vector<int64_t>{8, 20, 34}));
    CHECK(parse("24") == (std::vector<int64_t>{24}));

    // rejects: empty, a hole, a non-digit, a sign (the field is a layer index)
    CHECK(parse("").empty());
    CHECK(parse("8,,20").empty());
    CHECK(parse("8,").empty());
    CHECK(parse(",8").empty());
    CHECK(parse("8,x").empty());
    CHECK(parse("-8").empty());
    CHECK(parse("8, 20").empty());
    CHECK(parse("auto").empty());

    // the ordering rules: >= 2 before the first cut, strictly rising
    CHECK(split_cuts_ordered({}));               // an empty list has nothing to violate
    CHECK(split_cuts_ordered({8}));
    CHECK(split_cuts_ordered({8, 20, 34}));
    CHECK(!split_cuts_ordered({1}));             // stage 0 must own a layer
    CHECK(!split_cuts_ordered({0}));
    CHECK(!split_cuts_ordered({8, 8}));          // stage 1 would own nothing
    CHECK(!split_cuts_ordered({20, 8}));         // a falling cut
    CHECK(!split_cuts_ordered({8, 20, 1}));      // rising then falling
    CHECK(split_cuts_ordered({2, 3}));           // the tightest legal cut

    if (g_fail == 0) std::printf("split_cuts_test: ok\n");
    return g_fail == 0 ? 0 : 1;
}
