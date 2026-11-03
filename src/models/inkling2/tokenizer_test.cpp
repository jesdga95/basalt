// src/models/inkling2/tokenizer_test.cpp - the engine's tokenizer on the Inkling 2 pack against the official tokenizer:
// the 337 vocabulary entries the merges cannot build (the checkpoint's tokenizer.json sets ignore_merges, so each is one
// token) and two sentences that use them, with the ids Hugging Face tokenizers gives (tests/data/inkling2/).
//
//   inkling2_tokenizer_test <ignore_merges.tsv> <Inkling-2-Mini pack>      exit 77 when the pack is not there
#include "models/inkling2/adapter.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "basalt/core/model_file.hpp"
#include "basalt/program/json_scan.hpp"
#include "basalt/program/tokenizer.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: inkling2_tokenizer_test <ignore_merges.tsv> <pack>\n");
        return 2;
    }
    if (!std::ifstream(argv[2]).good()) {
        std::printf("no pack at %s: skipped\n", argv[2]);
        return 77;
    }
    basalt::core::ModelFile mf;
    std::string err;
    std::vector<uint8_t> files[3];
    const char* names[3] = {"tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/token_type.json"};
    if (!mf.open(argv[2], err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    for (int i = 0; i < 3; ++i)
        if (!mf.read(names[i], files[i], err)) { std::fprintf(stderr, "%s: %s\n", names[i], err.c_str()); return 1; }
    const auto view = [](const std::vector<uint8_t>& b) { return std::string_view((const char*) b.data(), b.size()); };
    basalt::program::Tokenizer t;
    t.set_pretokenize(basalt::models::inkling2::pretok_inkling2);
    t.set_ignore_merges(basalt::models::inkling2::kIgnoreMerges);
    if (!t.load(view(files[0]), view(files[1]), view(files[2]), err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }

    std::ifstream g(argv[1], std::ios::binary);
    std::string line;
    int cases = 0, bad = 0;
    while (std::getline(g, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t tab = line.find('\t');
        basalt::program::json::Scan sc{std::string_view(line).substr(0, tab)};
        std::string text;
        if (tab == std::string::npos || !sc.str(text)) { std::fprintf(stderr, "bad line: %s\n", line.c_str()); return 1; }
        std::vector<int32_t> want;
        for (const char* p = line.c_str() + tab + 1; *p;) {
            char* e = nullptr;
            want.push_back((int32_t) std::strtol(p, &e, 10));
            p = *e == ',' ? e + 1 : e;
        }
        std::vector<int32_t> got;
        ++cases;
        if (!t.encode(text, false, {}, got, err) || got != want) {
            ++bad;
            std::fprintf(stderr, "MISMATCH %s: %zu ids, want %zu\n", line.substr(0, tab).c_str(), got.size(), want.size());
        }
    }
    std::printf("inkling2_tokenizer_test: %d cases, %d mismatches\n", cases, bad);
    return cases > 0 && bad == 0 ? 0 : 1;
}
