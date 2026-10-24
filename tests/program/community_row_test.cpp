// tests/program/community_row_test.cpp - adding a row to docs/community/benchmarks.json: the exact text for a populated
// and an empty runs list, the refusal of a non-canonical tail, and string escaping. CPU only.
//   community_row_test                         the built-in cases
//   community_row_test --apply IN OUT CASE     writes IN with the case row added to OUT (community_row_fix_test.py
//                                              compares it with what tools/check_benchmarks.py --fix writes)
#include "basalt/program/community_row.hpp"

#include <cstdio>
#include <cstring>
#include <string>

using namespace basalt::program;

namespace {
int failures = 0;
void check(bool ok, const std::string& what, const std::string& got = "") {
    std::printf("community_row %-56s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    if (!ok) std::printf("  got: [%s]\n", got.c_str());
    failures |= !ok;
}

CommunityRow plain_row() {
    CommunityRow r;
    r.submitter = "tester"; r.date = "2026-10-24"; r.commit = "abc1234"; r.preset = "community-1";
    r.gpus = {{"RTX 5090", 32, 400}, {"RTX 5060 Ti", 16, 180}};
    r.ram_gib = 64; r.os = "Linux"; r.quant = "NVFP4";
    r.prefill_16k_tps = 1234.5; r.prefill_64k_tps = 2345.6; r.decode_tps = 345.7;
    return r;
}

// quotes, a backslash, a control character, 0x7f and non-ASCII text
CommunityRow odd_row() {
    CommunityRow r = plain_row();
    r.submitter = "\xC3\x9Cn\xC3\xAF \"q\" \\ \xE2\x98\x83 \xE6\x97\xA5\xE6\x9C\xAC";
    r.gpus = {{"Card \"X\"\tTi", 24, 300}};
    r.notes = "line\nbreak \xE2\x82\xAC \x01\x7f";
    return r;
}

bool read_all(const char* path, std::string& s) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    char b[4096];
    size_t n;
    while ((n = std::fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
    std::fclose(f);
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 5 && std::strcmp(argv[1], "--apply") == 0) {
        std::string err;
        const CommunityRow r = std::strcmp(argv[4], "odd") == 0 ? odd_row() : plain_row();
        std::string in, out;
        if (!read_all(argv[2], in) || !community_add_row(in, r, out, err)) {
            std::printf("community_row: %s\n", err.c_str());
            return 1;
        }
        std::FILE* f = std::fopen(argv[3], "wb");
        if (!f || std::fwrite(out.data(), 1, out.size(), f) != out.size()) return 1;
        std::fclose(f);
        return 0;
    }

    const std::string one =
        "    {\n      \"submitter\": \"tester\",\n      \"date\": \"2026-10-24\",\n      \"commit\": \"abc1234\",\n"
        "      \"preset\": \"community-1\",\n      \"gpus\": [\n        {\n          \"name\": \"RTX 5090\",\n"
        "          \"vram_gib\": 32,\n          \"power_w\": 400\n        },\n        {\n          \"name\": \"RTX 5060 Ti\",\n"
        "          \"vram_gib\": 16,\n          \"power_w\": 180\n        }\n      ],\n      \"ram_gib\": 64,\n"
        "      \"os\": \"Linux\",\n      \"quant\": \"NVFP4\",\n      \"prefill_16k_tps\": 1234.5,\n"
        "      \"prefill_64k_tps\": 2345.6,\n      \"decode_tps\": 345.7\n    }";
    std::string out, err;
    check(community_row_text(plain_row()) == one, "the row text", community_row_text(plain_row()));

    const std::string head = "{\n  \"preset\": \"basalt-bench --community\",\n  \"runs\": [\n";
    const std::string prev = "    {\n      \"submitter\": \"a\"\n    }";
    check(community_add_row(head + prev + "\n  ]\n}", plain_row(), out, err) && out == head + prev + ",\n" + one + "\n  ]\n}",
          "a row is added after the last one with a comma", out);
    check(community_add_row("{\n  \"preset\": \"p\",\n  \"runs\": []\n}", plain_row(), out, err) &&
              out == "{\n  \"preset\": \"p\",\n  \"runs\": [\n" + one + "\n  ]\n}",
          "an empty runs list gets its first row", out);

    const std::string bad[] = {"", "{}", head + prev + "\n  ]\n}\n", head + prev + "\n  ]\n}   ", "{\"runs\": [{\"a\": 1}]}",
                         "{\n  \"runs\": [],\n  \"x\": 1\n}", head + "    {\"submitter\": \"a\"}\n  ]\n}"};
    int refused = 0;
    for (const std::string& b : bad) {
        std::string e;
        refused += !community_add_row(b, plain_row(), out, e) && !e.empty();
    }
    check(refused == (int) (sizeof bad / sizeof *bad), "a non-canonical tail is an error with a reason");

    check(json_quote("a\"b\\c\n\t\r\x01\x7f\xC3\xA9") == "\"a\\\"b\\\\c\\n\\t\\r\\u0001\x7f\xC3\xA9\"", "json_quote escapes like json.dumps",
          json_quote("a\"b\\c\n\t\r\x01\x7f\xC3\xA9"));
    check(community_row_text(odd_row()).find("\"notes\": \"line\\nbreak \xE2\x82\xAC \\u0001\x7f\"\n    }") != std::string::npos, "notes close the row");

    std::string missing;
    check(!community_add_row_file("/nonexistent-dir/benchmarks.json", plain_row(), missing) && !missing.empty(), "a missing file is an error");
    return failures ? 1 : 0;
}
