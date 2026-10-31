// tests/program/cli_arguments_test.cpp - the engine's command line: the control-vector numbers are checked here.
#include "arguments.hpp"

#include "basalt/program/options.hpp"

#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

using namespace basalt::program;

static int g_fail = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

static std::optional<int> parse(std::vector<std::string> words, Options& o) {
    std::vector<char*> argv;
    argv.reserve(words.size());
    for (auto& word : words) argv.push_back(word.data());
    std::string ppl_base, ppl_out;
    return cli::parse_options((int) argv.size(), argv.data(), o, ppl_base, ppl_out);
}

int main() {
    {
        Options o;
        CHECK(!parse({"basalt", "--control-vector-layer-range", "4", "44"}, o));
        CHECK(o.cvec_first == 4 && o.cvec_last == 44);
        CHECK(!parse({"basalt", "--control-vector-layer-range", "4", "44", "--cvec-mode", "project"}, o));
        CHECK(o.cvec_mode == 0);
    }
    {
        Options o;
        CHECK(!parse({"basalt", "--cvec-dir", "single:7"}, o));
        CHECK(o.cvec_single == 7);
        CHECK(!parse({"basalt", "--cvec-dir", "per-layer"}, o));
        CHECK(o.cvec_single == -1);
        CHECK(!parse({"basalt", "--control-vector-scaled", "x.gguf:1.5"}, o));
        CHECK(o.cvec_files.size() == 1 && std::fabs(o.cvec_files[0].second - 1.5f) < 1e-6f);
    }
    // A value that is not a number is refused here, instead of atoi() reading the next flag as 0.
    for (const auto& bad : {std::vector<std::string>{"basalt", "--control-vector-layer-range", "4", "--cvec-mode"},
                            std::vector<std::string>{"basalt", "--control-vector-layer-range", "four", "44"},
                            std::vector<std::string>{"basalt", "--control-vector-layer-range", "-1", "44"},
                            std::vector<std::string>{"basalt", "--control-vector-layer-range", "4", "2147483648"},
                            std::vector<std::string>{"basalt", "--control-vector-layer-range", "44", "4"},
                            std::vector<std::string>{"basalt", "--cvec-dir", "single:foo"},
                            std::vector<std::string>{"basalt", "--cvec-dir", "single:-1"},
                            std::vector<std::string>{"basalt", "--cvec-dir", "single:0"},
                            std::vector<std::string>{"basalt", "--cvec-dir", "single:"},
                            std::vector<std::string>{"basalt", "--cvec-dir", "single:2147483648"},
                            std::vector<std::string>{"basalt", "--control-vector-scaled", "x.gguf:nan"},
                            std::vector<std::string>{"basalt", "--control-vector-scaled", "x.gguf:inf"},
                            std::vector<std::string>{"basalt", "--control-vector-scaled", "x.gguf:1e999"}}) {
        Options o;
        CHECK(parse(bad, o) == 2);
    }
    if (g_fail == 0) std::puts("cli_arguments_test: passed");
    return g_fail == 0 ? 0 : 1;
}
