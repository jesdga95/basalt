// tests/program/tokenizer_bench.cpp - cold and warm encode time of the C++ tokenizer on text files.
// usage: tokenizer_bench <pack tokenizer dir> <text file>...   (each file is encoded in a fresh tokenizer, then again)
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <string>

#include "basalt/program/tokenizer.hpp"

namespace {

std::string read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: tokenizer_bench <tokenizer dir> <text file>...\n"); return 2; }
    const std::string dir = argv[1];
    const std::string vocab = read_all(dir + "/vocab.json"), merges = read_all(dir + "/merges.txt"),
                      types = read_all(dir + "/token_type.json");
    for (int a = 2; a < argc; ++a) {
        const std::string text = read_all(argv[a]);
        basalt::program::Tokenizer t;
        std::string err;
        auto t0 = std::chrono::steady_clock::now();
        if (!t.load(vocab, merges, types, err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
        const double load_ms = ms_since(t0);
        std::vector<int32_t> ids;
        t0 = std::chrono::steady_clock::now();
        if (!t.encode(text, true, {}, ids, err)) { std::fprintf(stderr, "encode: %s\n", err.c_str()); return 1; }
        const double cold = ms_since(t0);
        double warm = 1e30;
        for (int r = 0; r < 5; ++r) {
            t0 = std::chrono::steady_clock::now();
            t.encode(text, true, {}, ids, err);
            warm = std::min(warm, ms_since(t0));
        }
        std::printf("tokenizer_bench %s: %zu bytes, %zu tokens, load %.1f ms, cold %.2f ms (%.1f MB/s, %.2f Mtok/s), warm %.2f ms\n",
                    argv[a], text.size(), ids.size(), load_ms, cold, (double) text.size() / 1e3 / cold,
                    (double) ids.size() / 1e3 / cold, warm);
    }
    return 0;
}
