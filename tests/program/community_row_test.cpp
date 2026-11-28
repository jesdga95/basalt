// tests/program/community_row_test.cpp - a community run's file: the exact row text, the id and folder, the write and its overwrite,
// and string escaping. CPU only.
//   community_row_test                    the built-in cases
//   community_row_test --apply DIR CASE   writes the case row's run file under DIR and prints its path (community_row_fix_test.py
//                                         compares the file and its path with what tools/check_benchmarks.py spells)
//   community_row_test --ids FILE...      prints the id of each row file (the checker's row_id must agree)
//   community_row_test --dup FILE...      prints "I J" for each row file I that has the setup of an earlier one J (the first such J),
//                                         for community_row_fix_test.py to compare with the checker
#include "basalt/program/community_row.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/program/host_rates.hpp"

#include <cstdio>
#include <cstdlib>
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
    if (argc >= 2 && (std::strcmp(argv[1], "--dup") == 0 || std::strcmp(argv[1], "--ids") == 0)) {
        std::vector<CommunityRow> rows;
        for (int k = 2; k < argc; ++k) {
            std::string text, err;
            rows.emplace_back();
            if (!read_all(argv[k], text) || !community_parse_row(text, rows.back(), err)) {
                std::printf("community_row: %s: %s\n", argv[k], err.c_str());
                return 1;
            }
        }
        for (size_t i = 0; i < rows.size(); ++i) {
            if (std::strcmp(argv[1], "--ids") == 0) {
                std::printf("%s\n", community_run_id(rows[i]).c_str());
                continue;
            }
            for (size_t j = 0; j < i; ++j)
                if (community_same_setup(rows[j], rows[i])) {
                    std::printf("%zu %zu\n", i, j);
                    break;
                }
        }
        return 0;
    }
    if (argc == 4 && std::strcmp(argv[1], "--apply") == 0) {
        std::string err;
        const CommunityRow r = std::strcmp(argv[3], "odd") == 0 ? odd_row() : plain_row();
        CommunityEdit edit;
        if (!community_write_run(argv[2], r, err, &edit)) {
            std::printf("community_row: %s\n", err.c_str());
            return 1;
        }
        std::printf("%s\n", edit.path.c_str());
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
        "{\n  \"submitter\": \"tester\",\n  \"date\": \"2026-10-24\",\n  \"commit\": \"abc1234\",\n"
        "  \"preset\": \"community-1\",\n  \"model\": \"Qwen3.8-Flash-Next\",\n  \"gpus\": [\n    {\n      \"name\": \"RTX 5090\",\n"
        "      \"vram_gib\": 32,\n      \"power_w\": 400,\n      \"pcie\": \"5.0 x16\",\n      \"h2d_gbps\": 55.3\n    },\n    {\n      \"name\": \"RTX 5060 Ti\",\n"
        "      \"vram_gib\": 16,\n      \"power_w\": 180,\n      \"pcie\": \"5.0 x8\",\n      \"h2d_gbps\": 27.9\n    }\n  ],\n  \"ram_gib\": 64,\n"
        "  \"cpu\": \"Test CPU 9, 16 cores\",\n  \"ram_gbps\": 78.2,\n"
        "  \"os\": \"Linux\",\n  \"quant\": \"NVFP4\",\n  \"kv\": \"int8\",\n  \"prefill_16k_tps\": 1234.5,\n"
        "  \"prefill_64k_tps\": 2345.6,\n  \"decode_tps\": 345.7,\n"
        "  \"decode_conv_tps\": 400.1,\n  \"decode_struct_tps\": 500.3,\n  \"decode_prose_tps\": 300.2,\n"
        "  \"accept_conv_pct\": 62.5,\n  \"accept_struct_pct\": 80.0,\n  \"draft_len_conv\": 3.2,\n  \"draft_len_struct\": 4.1,\n"
        "  \"engine_args\": \"--gpu 0,1 --max-context 8192 --kv int8\"\n}";
    std::string out, err;
    check(community_row_text(plain_row()) == one, "the row text", community_row_text(plain_row()));

    check(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" &&
              sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" &&
              sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
                  "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" &&
              sha256_hex(std::string(1000, 'a')) == "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3",
          "sha256_hex matches the published vectors");
    {
        const CommunityRow a = plain_row();
        CommunityRow b = a, c = a, d = a;
        b.submitter = "TESTER";                    // the submitter in other case: one setup, one id
        b.kv = "fp16"; b.decode_tps = 1; b.notes = "x"; b.gpus[0].pcie = "4.0 x8";   // figures the setup does not include
        c.gpus[1].power_w = 181;                   // another power limit: another setup
        d.os = "WSL2";
        const std::string id = community_run_id(a);
        check(id.size() == 12 && id.find_first_not_of("0123456789abcdef") == std::string::npos, "the id is 12 hex digits", id);
        check(community_run_id(b) == id, "the id ignores the submitter's case and the figures");
        check(community_run_id(c) != id && community_run_id(d) != id, "another power limit or system is another id");
        check(community_model_slug("Qwen3.8-Flash-Next") == "qwen3.8-flash-next" && community_model_slug("Hy4-flash") == "hy4-flash", "model slug");
    }

    // writing a run: a file under runs/<slug>/<id>.json; the same setup again overwrites it and keeps its notes
    {
        const std::string dir = basalt::platform::temp_dir() + "/community_row_test_" + community_run_id(plain_row());
        CommunityRow r = plain_row();
        std::remove((dir + "/runs/qwen3.8-flash-next/" + community_run_id(r) + ".json").c_str());   // a leftover of an earlier run
        CommunityEdit e;
        const bool ok = community_write_run(dir, r, err, &e);
        std::string text;
        check(ok && !e.replaced && e.path == dir + "/runs/qwen3.8-flash-next/" + community_run_id(r) + ".json" && read_all(e.path.c_str(), text) &&
                  text == one + "\n", "a run is written as its own file with a final newline", err + text);
        CommunityRow back;
        check(community_parse_row(text, back, err) && back.submitter == "tester" && back.model == "Qwen3.8-Flash-Next" && back.gpus.size() == 2 &&
                  back.gpus[1].power_w == 180 && back.ram_gib == 64 && back.cpu == "Test CPU 9, 16 cores" && back.notes.empty() &&
                  community_run_id(back) == community_run_id(r), "the file reads back to the same setup and id", err);
        r.notes = "hand \"written\"\n\x08\x0c \xE2\x82\xAC\xF0\x9F\x98\x80";
        CommunityEdit e2;
        check(community_write_run(dir, r, err, &e2) && e2.replaced && e2.notes.empty(), "overwriting a run reports it", err);
        CommunityRow again = plain_row();
        again.submitter = "TESTER"; again.decode_tps = 99.9;
        CommunityEdit e3;
        text.clear();
        check(community_write_run(dir, again, err, &e3) && e3.replaced && e3.notes == r.notes && e3.path == e2.path && read_all(e3.path.c_str(), text) &&
                  text.find("\"decode_tps\": 99.9") != std::string::npos && text.find("\"submitter\": \"TESTER\"") != std::string::npos &&
                  text == e3.row_text + "\n" && community_parse_row(text, back, err) && back.notes == r.notes,
              "the same setup again overwrites the file with the new figures and keeps the notes", err + text);
        std::remove(e3.path.c_str());
    }
    {
        CommunityRow x;
        const std::string bad[] = {"", "{}", "{\n  \"submitter\": \"a\"\n", "{\"submitter\": \"a\"}\n", "{\n  \"submitter\": \"a\"\n}\nextra\n",
                                   "{\n  \"submitter\": \"\\q\"\n}\n", "{\n  \"gpus\": [\n    x\n  ]\n}\n"};
        int refused = 0;
        for (const std::string& b : bad) {
            std::string e;
            refused += !community_parse_row(b, x, e) && !e.empty();
        }
        check(refused == (int) (sizeof bad / sizeof *bad), "a row not in the house form is an error with a reason");
    }

    check(json_quote("a\"b\\c\n\t\r\x01\x7f\xC3\xA9") == "\"a\\\"b\\\\c\\n\\t\\r\\u0001\x7f\xC3\xA9\"", "json_quote escapes like json.dumps",
          json_quote("a\"b\\c\n\t\r\x01\x7f\xC3\xA9"));
    check(json_quote("\b\f") == "\"\\b\\f\"", "json_quote spells backspace and form feed like json.dumps", json_quote("\b\f"));
    check(community_row_text(odd_row()).find("\"notes\": \"line\\nbreak \xE2\x82\xAC \\u0001\x7f\"\n}") != std::string::npos, "notes close the row");

    check(cpu_label("AMD Ryzen 9 9950X 16-Core Processor", 16) == "AMD Ryzen 9 9950X, 16 cores", "cpu_label: AMD name");
    check(cpu_label("13th Gen Intel(R) Core(TM) i9-13900K", 24) == "13th Gen Intel Core i9-13900K, 24 cores", "cpu_label: marks");
    check(cpu_label("Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz", 1) == "Intel Xeon Gold 6130, 1 core", "cpu_label: clock and CPU");
    check(cpu_label("AMD Ryzen Threadripper PRO 9955WX 16-Cores", 16) == "AMD Ryzen Threadripper PRO 9955WX, 16 cores",
          "cpu_label: plural cores tag");
    check(cpu_label("AMD Ryzen Threadripper PRO 7995WX 96-Cores Foo", 96) == "AMD Ryzen Threadripper PRO 7995WX Foo, 96 cores",
          "cpu_label: plural cores tag before more text");
    check(cpu_label("AMD Ryzen Threadripper PRO 7995WX 96-Cores (rev2)", 96) == "AMD Ryzen Threadripper PRO 7995WX (rev2), 96 cores",
          "cpu_label: plural cores tag before punctuation");
    check(cpu_label("Foo 16-Coresx", 16) == "Foo 16-Coresx, 16 cores", "cpu_label: tag kept when the word continues");
    check(cpu_label("Foo 16-Cores2", 16) == "Foo 16-Cores2, 16 cores", "cpu_label: tag kept when a digit follows the s");
    check(cpu_label("Intel Pentium Dual-Cores", 2) == "Intel Pentium Duals, 2 cores", "cpu_label: plural s kept without a count");
    check(cpu_label("AMD Phenom II X6 Hexa-Core", 6) == "AMD Phenom II X6 Hexa, 6 cores" &&
              cpu_label("Foo Hexa-Corex", 6) == "Foo Hexax, 6 cores", "cpu_label: tag without a count");
    check(pack_quant_label("Qwen3.8-Flash-Next NVIDIA-NVFP4") == "NVFP4" && pack_quant_label("DeepSeek-V4.1-Flash-1017 UD-IQ1_M") == "IQ1_M" &&
              pack_quant_label("") == "unknown", "pack_quant_label");
    check(pack_quant_label("Qwen3.8-Flash-Next") == "unknown" && pack_quant_label("Qwen3.8-Flash-Next UD-Q4_K_XL") == "Q4_K_XL" &&
              pack_quant_label("Qwen3.8-Flash-Next UD-IQ3_XXS") == "IQ3_XXS", "pack_quant_label: a name without a space is unknown");
    check(pack_quant_label("Qwen3.8-Flash-Next ud-q4_k_m") == "q4_k_m" &&
              pack_quant_label("Qwen3.8-Flash-Next Q4_K_M ") == "Q4_K_M" &&
              pack_quant_label("Qwen3.8-Flash-Next nvidia-NVFP4-DQ8") == "NVFP4-DQ8" &&
              pack_quant_label(" Qwen3.8-Flash-Next Huihui-Abliterated-V2\t") == "Huihui-Abliterated-V2" &&
              pack_quant_label("  Qwen3.8-Flash-Next  ") == "unknown",
          "pack_quant_label: edge spaces trimmed, prefixes stripped in any case, the label keeps its own case");
    check(pack_quant_label("Qwen3.8-Flash-Next UD-NVIDIA-Q4_K_XL") == "Q4_K_XL" &&
              pack_quant_label("Qwen3.8-Flash-Next UD-UD-Q4") == "Q4",
          "pack_quant_label: prefixes stripped repeatedly in any order");
    check(pack_quant_label("Inkling 2 Mini") == "unknown" && pack_quant_label("Hy4-flash NVFP4") == "NVFP4" &&
              pack_quant_label("Foo Q8_0") == "Q8_0" && pack_quant_label("Foo BF16") == "BF16" && pack_quant_label("Foo UD-Mini") == "unknown",
          "pack_quant_label: a last word without a digit is no quant label");
    check(pack_quant_label("Gemma 4") == "unknown" && pack_quant_label("Gemma 4 12B") == "unknown" &&
              pack_quant_label("Gemma 4 31B") == "unknown" && pack_quant_label("Gemma 4 31B FP8") == "FP8" &&
              pack_quant_label("Gemma 4 26B A4B") == "unknown" && pack_quant_label("Gemma 4 26B A4B NVFP4") == "NVFP4" &&
              pack_quant_label("Foo 8bit") == "8bit",
          "pack_quant_label: a size or a version is no quant label");
    check(cpu_label("", 8).empty() && cpu_label("X", 0) == "X", "cpu_label: empty name, unknown cores");

    const auto links = parse_link_rates("noise\npcie_h2d 0000:01:00.0 55.3\npcie_h2d 0000:02:00.0 bad\npcie_h2d 0000:03:00.0 27.9\n");
    check(links.size() == 2 && links[0].second == 55.3 && normalize_bus_id(links[1].first) == normalize_bus_id("00000000:03:00.0"),
          "parse_link_rates keeps the good lines, bus ids compare across domain widths");

    // the setup rule: a row with no figure of its own matches (an old row lacks cpu), a different system does not
    {
        CommunityRow old = plain_row();
        old.submitter = "first"; old.cpu.clear();
        check(community_same_setup(old, plain_row()) == false, "same_setup: another submitter does not match");
        old.submitter = "tester";
        check(community_same_setup(old, plain_row()) && community_same_setup(plain_row(), old), "same_setup: a missing cpu matches");
        old.os = "WSL2";
        check(!community_same_setup(old, plain_row()), "same_setup: another system does not match");
    }

    std::string missing;
    // a folder under a plain file can be made on no system (a path from the root can be created on Windows)
    const std::string blocker = basalt::platform::temp_dir() + "/community_row_test_blocker";
    if (FILE* f = std::fopen(blocker.c_str(), "wb")) std::fclose(f);
    check(!community_write_run(blocker + "/x", plain_row(), missing) && !missing.empty(), "an unwritable folder is an error");
    std::remove(blocker.c_str());
    return failures ? 1 : 0;
}
