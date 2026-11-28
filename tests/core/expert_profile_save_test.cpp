// --expert-profile-save's profile - rank_learned_profile's order (resident first, then the counted routing, then
// the profile the engine started from) and write_expert_profile's file, byte for byte the profile format,
// read back by read_expert_profile.  CPU only: no device is touched.
#include "basalt/core/expert_cache.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}
}  // namespace

int main() {
    using Pair = std::pair<int32_t, int32_t>;
    const int64_t L = 3, E = 4;   // 12 pairs, index = layer * 4 + expert
    std::vector<uint8_t> resident(12, 0);
    resident[5] = 1;              // (1, 1)
    resident[2] = 1;              // (0, 2)
    std::vector<double> heat(12, 0.0);
    heat[2] = 1.0;
    heat[5] = 7.0;
    heat[11] = 9.0;               // (2, 3): the hottest, but not resident
    heat[0] = 3.0;
    const std::vector<Pair> prior = {{2, 0}, {1, 3}, {0, 1}};   // a profile that ranks three pairs
    const std::vector<Pair> r = basalt::core::rank_learned_profile(L, E, resident, heat, prior);
    check(r.size() == 12, "every pair is ranked");
    const std::vector<Pair> head = {{1, 1}, {0, 2}, {2, 3}, {0, 0}, {2, 0}, {1, 3}, {0, 1}, {0, 3}};
    for (size_t i = 0; i < head.size() && i < r.size(); ++i)
        check(r[i] == head[i], "resident by heat, then the rest by heat, then the prior, then the index");
    // no counts at all: the resident ones, then the prior's order
    const std::vector<Pair> r0 = basalt::core::rank_learned_profile(L, E, resident, {}, prior);
    check(r0.size() == 12 && r0[0] == Pair(0, 2) && r0[1] == Pair(1, 1) && r0[2] == Pair(2, 0) &&
          r0[3] == Pair(1, 3) && r0[4] == Pair(0, 1) && r0[5] == Pair(0, 0), "without heat: resident, then prior");

    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "profile_save_test";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "learned.bin").string();
    std::string err;
    check(basalt::core::write_expert_profile(path, L, E, r, err), "written");
    check(!std::filesystem::exists(path + ".tmp"), "no temporary file left");
    // the profile format's bytes for the same ranking
    std::vector<uint8_t> want;
    auto u32 = [&](uint32_t v) { for (int b = 0; b < 4; ++b) want.push_back((uint8_t) (v >> (8 * b))); };
    auto u16 = [&](uint32_t v) { want.push_back((uint8_t) v); want.push_back((uint8_t) (v >> 8)); };
    for (uint32_t v : {1u, 3u, 4u, 12u, 12u}) u32(v);
    for (const Pair& p : r) { u16((uint32_t) p.first); u16((uint32_t) p.second); }
    std::vector<int32_t> table(12, -1);
    for (size_t i = 0; i < r.size(); ++i) table[(size_t) (r[i].first * E + r[i].second)] = (int32_t) i;
    for (int32_t v : table) u32((uint32_t) v);
    std::vector<uint8_t> got(want.size() + 1);
    if (std::FILE* f = std::fopen(path.c_str(), "rb")) {
        got.resize(std::fread(got.data(), 1, got.size(), f));
        std::fclose(f);
    }
    check(got == want, "the file is the profile format, byte for byte");
    std::vector<Pair> back;
    int64_t slots = 0;
    check(basalt::core::read_expert_profile(basalt::core::FileRange(path), L, E, back, slots, err) && back == r && slots == 12,
          "read back by the loader");
    // a second save replaces the first (the rename over an existing file)
    check(basalt::core::write_expert_profile(path, L, E, r0, err), "written again");
    check(basalt::core::read_expert_profile(basalt::core::FileRange(path), L, E, back, slots, err) && back == r0, "replaced");
    check(!basalt::core::write_expert_profile(path, L, E, {{3, 0}}, err), "a pair out of range is refused");
    check(basalt::core::read_expert_profile(basalt::core::FileRange(path), L, E, back, slots, err) && back == r0,
          "a refused save keeps the file");
    // Corrupt profiles: duplicate pairs, n_ranked > n_layers * n_expert, and invalid geometry are refused without clobbering `back`.
    {
        const std::vector<Pair> saved_back = back;
        auto write_raw = [&](const std::string& p, const std::vector<uint32_t>& hdr, const std::vector<Pair>& pairs) {
            std::vector<uint8_t> bytes;
            auto pu32 = [&](uint32_t v) { for (int b = 0; b < 4; ++b) bytes.push_back((uint8_t) (v >> (8 * b))); };
            auto pu16 = [&](uint32_t v) { bytes.push_back((uint8_t) v); bytes.push_back((uint8_t) (v >> 8)); };
            for (uint32_t v : hdr) pu32(v);
            for (const Pair& pr : pairs) { pu16((uint32_t) pr.first); pu16((uint32_t) pr.second); }
            if (std::FILE* f = std::fopen(p.c_str(), "wb")) {
                std::fwrite(bytes.data(), 1, bytes.size(), f);
                std::fclose(f);
            }
        };
        const std::string dup_path = (dir / "dup.bin").string();
        write_raw(dup_path, {1u, 3u, 4u, 2u, 2u}, {{1, 2}, {1, 2}});
        check(!basalt::core::read_expert_profile(basalt::core::FileRange(dup_path), L, E, back, slots, err) &&
                  back == saved_back,
              "duplicate pair in expert profile is refused without clobbering output");

        const std::string over_path = (dir / "over.bin").string();
        std::vector<Pair> thirteen(13, {0, 0});
        write_raw(over_path, {1u, 3u, 4u, 13u, 13u}, thirteen);
        check(!basalt::core::read_expert_profile(basalt::core::FileRange(over_path), L, E, back, slots, err) &&
                  back == saved_back,
              "n_ranked > n_layers * n_expert is refused");

        check(!basalt::core::read_expert_profile(basalt::core::FileRange(path), 0, E, back, slots, err) &&
                  back == saved_back,
              "invalid model expert geometry is refused");

        // Header-only ExpertCache::replace bounds guard: safe no-op on empty/out-of-range/self-replace calls.
        basalt::core::ExpertCache empty_cache;
        empty_cache.replace(-1, 0, 1);
        empty_cache.replace(0, 0, 0);
        empty_cache.replace(0, -1, 1);
    }
    std::filesystem::remove_all(dir);
    if (fails == 0) std::puts("expert_profile_save_test: OK");
    return fails == 0 ? 0 : 1;
}
