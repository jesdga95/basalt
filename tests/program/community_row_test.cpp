// tests/program/community_row_test.cpp - adding a row to docs/community/benchmarks.json: the exact text for a populated
// and an empty runs list, the refusal of a non-canonical tail, and string escaping. CPU only.
//   community_row_test                         the built-in cases
//   community_row_test --apply IN OUT CASE     writes IN with the case row added to OUT (community_row_fix_test.py
//                                              compares it with what tools/check_benchmarks.py --fix writes)
//   community_row_test --dup FILE              prints "I J" for each row I of FILE that has the setup of an earlier row J
//                                              (the first such J), for community_row_fix_test.py to compare with the checker
#include "basalt/program/community_row.hpp"
#include "basalt/program/host_rates.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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
    r.model = "Qwen3.8-Flash-Next"; r.kv = "int8";
    r.gpus = {{"RTX 5090", 32, 400, "5.0 x16", 55.3}, {"RTX 5060 Ti", 16, 180, "5.0 x8", 27.9}};
    r.cpu = "Test CPU 9, 16 cores";
    r.ram_gib = 64; r.ram_gbps = 78.2; r.os = "Linux"; r.quant = "NVFP4";
    r.prefill_16k_tps = 1234.5; r.prefill_64k_tps = 2345.6; r.decode_tps = 345.7;
    r.tests[0] = {400.1, 62.5, 3.2}; r.tests[1] = {500.3, 80.0, 4.1}; r.tests[2] = {300.2, 0, 0};   // prose did not draft
    r.engine_args = "--gpu 0,1 --max-context 8192 --kv int8";
    return r;
}

// quotes, a backslash, a control character, 0x7f and non-ASCII text
CommunityRow odd_row() {
    CommunityRow r = plain_row();
    r.submitter = "\xC3\x9Cn\xC3\xAF\"q\"\\\xE2\x98\x83\xE6\x97\xA5\xE6\x9C\xAC";
    r.gpus = {{"Card \"X\"\tTi", 24, 300, "", 0}};   // no link figures: the optional keys are left out
    r.cpu.clear(); r.ram_gbps = 0;
    for (CommunityTest& t : r.tests) t = CommunityTest();   // no per-test figures, no engine arguments
    r.engine_args.clear();
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
    if (argc == 3 && std::strcmp(argv[1], "--dup") == 0) {
        std::string text, err;
        std::vector<CommunityRow> rows;
        if (!read_all(argv[2], text) || !community_parse_rows(text, rows, err)) {
            std::printf("community_row: %s\n", err.c_str());
            return 1;
        }
        for (size_t i = 0; i < rows.size(); ++i)
            for (size_t j = 0; j < i; ++j)
                if (community_same_setup(rows[j], rows[i])) {
                    std::printf("%zu %zu\n", i, j);
                    break;
                }
        return 0;
    }
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

    {
        const std::vector<int> none;
        auto str = [](const std::vector<int>& v) { std::string t; for (int d : v) t += (t.empty() ? "" : ",") + std::to_string(d); return t; };
        check(str(engine_gpu_order(-1, none, none)) == "0", "one card");
        check(str(engine_gpu_order(1, {1}, none)) == "0,1", "two cards, one tier");
        check(str(engine_gpu_order(1, {1, 2, 3}, none)) == "0,1,2,3", "four cards, three tiers", str(engine_gpu_order(1, {1, 2, 3}, none)));
        check(str(engine_gpu_order(-1, none, {1, 2})) == "0,1,2", "layer split stages");
        check(str(engine_gpu_order(1, {1, 1, 2}, {2, 3})) == "0,1,2,3", "repeats listed once, in order");
        const auto ids = split_gpu_ids("0000:01:00.0,0000:02:00.0,0000:03:00.0,0000:04:00.0");
        check(ids.size() == 4 && ids[0] == "0000:01:00.0" && ids[3] == "0000:04:00.0", "a four-card gpus= field gives four ids");
        check(split_gpu_ids("").empty() && split_gpu_ids("a,,b,").size() == 2, "empty fields are dropped");
    }

    const std::string one =
        "    {\n      \"submitter\": \"tester\",\n      \"date\": \"2026-10-24\",\n      \"commit\": \"abc1234\",\n"
        "      \"preset\": \"community-1\",\n      \"model\": \"Qwen3.8-Flash-Next\",\n      \"gpus\": [\n        {\n          \"name\": \"RTX 5090\",\n"
        "          \"vram_gib\": 32,\n          \"power_w\": 400,\n          \"pcie\": \"5.0 x16\",\n          \"h2d_gbps\": 55.3\n        },\n        {\n          \"name\": \"RTX 5060 Ti\",\n"
        "          \"vram_gib\": 16,\n          \"power_w\": 180,\n          \"pcie\": \"5.0 x8\",\n          \"h2d_gbps\": 27.9\n        }\n      ],\n      \"ram_gib\": 64,\n"
        "      \"cpu\": \"Test CPU 9, 16 cores\",\n      \"ram_gbps\": 78.2,\n"
        "      \"os\": \"Linux\",\n      \"quant\": \"NVFP4\",\n      \"kv\": \"int8\",\n      \"prefill_16k_tps\": 1234.5,\n"
        "      \"prefill_64k_tps\": 2345.6,\n      \"decode_tps\": 345.7,\n"
        "      \"decode_conv_tps\": 400.1,\n      \"decode_struct_tps\": 500.3,\n      \"decode_prose_tps\": 300.2,\n"
        "      \"accept_conv_pct\": 62.5,\n      \"accept_struct_pct\": 80.0,\n      \"draft_len_conv\": 3.2,\n      \"draft_len_struct\": 4.1,\n"
        "      \"engine_args\": \"--gpu 0,1 --max-context 8192 --kv int8\"\n    }";
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
    check(json_quote("\b\f") == "\"\\b\\f\"", "json_quote spells backspace and form feed like json.dumps", json_quote("\b\f"));
    check(community_row_text(odd_row()).find("\"notes\": \"line\\nbreak \xE2\x82\xAC \\u0001\x7f\"\n    }") != std::string::npos, "notes close the row");

    check(cpu_label("AMD Ryzen 9 9950X 16-Core Processor", 16) == "AMD Ryzen 9 9950X, 16 cores", "cpu_label: AMD name");
    check(cpu_label("13th Gen Intel(R) Core(TM) i9-13900K", 24) == "13th Gen Intel Core i9-13900K, 24 cores", "cpu_label: marks");
    check(cpu_label("Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz", 1) == "Intel Xeon Gold 6130, 1 core", "cpu_label: clock and CPU");
    check(pack_quant_label("Qwen3.8-Flash-Next NVIDIA-NVFP4") == "NVFP4" && pack_quant_label("DeepSeek-V4.1-Flash-1017 UD-IQ1_M") == "IQ1_M" &&
              pack_quant_label("") == "unknown", "pack_quant_label");
    check(cpu_label("", 8).empty() && cpu_label("X", 0) == "X", "cpu_label: empty name, unknown cores");

    const auto links = parse_link_rates("noise\npcie_h2d 0000:01:00.0 55.3\npcie_h2d 0000:02:00.0 bad\npcie_h2d 0000:03:00.0 27.9\n");
    check(links.size() == 2 && links[0].second == 55.3 && normalize_bus_id(links[1].first) == normalize_bus_id("00000000:03:00.0"),
          "parse_link_rates keeps the good lines, bus ids compare across domain widths");

    // a row of the same setup replaces the earlier one in its place and keeps its notes; another setup is appended
    {
        CommunityRow a = plain_row(), b = plain_row(), c = plain_row();
        a.submitter = "first"; a.notes = "hand \"written\"\n\x08\x0c \xE2\x82\xAC\xF0\x9F\x98\x80";
        b.submitter = "second"; b.notes.clear();
        c.submitter = "third";
        std::string f = "{\n  \"preset\": \"p\",\n  \"runs\": []\n}", next;
        for (const CommunityRow* r : {&a, &b, &c})
            if (!community_add_row(f, *r, next, err)) { check(false, "building the update fixture", err); break; }
            else f = next;
        CommunityRow upd = plain_row();   // the first row's setup again, other figures, its submitter in other case
        upd.submitter = "FIRST"; upd.kv = "fp16"; upd.decode_tps = 99.9; upd.cpu.clear();
        CommunityEdit e;
        const bool ok = community_add_row(f, upd, out, err, &e);
        std::vector<CommunityRow> rows;
        const bool parsed = ok && community_parse_rows(out, rows, err);
        check(ok && e.replaced && e.index == 0 && e.notes == a.notes && parsed && rows.size() == 3 && rows[0].submitter == "FIRST" &&
                  rows[1].submitter == "second" && rows[2].submitter == "third" && rows[0].notes == a.notes,
              "a row of the same setup replaces the earlier one in place, notes kept", out);
        check(ok && out.find("\"decode_tps\": 99.9") != std::string::npos && e.row_text.find("\"notes\": ") != std::string::npos &&
                  out.find(e.row_text) != std::string::npos, "the replaced row is the new figures and the edit reports its text");
        CommunityRow other = plain_row();
        other.submitter = "FIRST"; other.gpus[0].power_w = 401;
        CommunityEdit e2;
        check(community_add_row(f, other, out, err, &e2) && !e2.replaced && community_parse_rows(out, rows, err) && rows.size() == 4 &&
                  rows[3].gpus[0].power_w == 401, "another power limit is another setup and is appended");
        upd.ram_gib = 64.4;   // the file holds 64 and 400
        upd.gpus[0].power_w = 400.4;
        CommunityEdit e3;
        check(community_add_row(f, upd, out, err, &e3) && e3.replaced, "figures compare as the file spells them (rounded)");
        // CRLF files are edited the same way
        std::string crlf;
        for (char ch : f) { if (ch == '\n') crlf += '\r'; crlf += ch; }
        CommunityEdit e4;
        check(community_add_row(crlf, upd, out, err, &e4) && e4.replaced && out.find("\r\n") != std::string::npos &&
                  out.find("\"decode_tps\": 99.9") != std::string::npos, "a CRLF file is updated in place too");
        // a row with no setup figures of its own matches (an old row lacks cpu), a different system does not
        CommunityRow old = plain_row();
        old.submitter = "first"; old.cpu.clear();
        check(community_same_setup(old, plain_row()) == false, "same_setup: another submitter does not match");
        old.submitter = "tester";
        check(community_same_setup(old, plain_row()) && community_same_setup(plain_row(), old), "same_setup: a missing cpu matches");
        old.os = "WSL2";
        check(!community_same_setup(old, plain_row()), "same_setup: another system does not match");
    }

    std::string missing;
    check(!community_add_row_file("/nonexistent-dir/benchmarks.json", plain_row(), missing) && !missing.empty(), "a missing file is an error");
    return failures ? 1 : 0;
}
