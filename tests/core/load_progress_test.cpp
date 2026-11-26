// The startup bar: the total is printed once before any byte and never changes; done climbs to it and no further; a
// start that places more or less than it announced is reported once as a warning; bytes placed outside a start (no total
// announced) are not counted.  Each destination's share is announced with done 0 before any byte, climbs to its own
// total and no further, and ends once with done == total; the shares sum to the total.  The lines go to stderr, read
// back from a file here.
#include "basalt/core/load_progress.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
#include <vector>

using basalt::core::LoadProgress;

int main() {
    const std::string path = (std::filesystem::current_path() / "load_progress_test.txt").string();
    if (std::freopen(path.c_str(), "w", stderr) == nullptr) return 1;
    LoadProgress::placed(7);   // outside a start
    LoadProgress::announce_total(100);
    LoadProgress::shares({{"cuda0", 60}, {"cuda1", 0}, {"ram", 40}});
    LoadProgress::into("cuda0");
    LoadProgress::placed(30, 0);
    LoadProgress::placed(30, 0);
    LoadProgress::into("ram");
#ifdef NDEBUG
    LoadProgress::placed(60, 2);   // 20 past the total (a debug build stops there instead)
    LoadProgress::placed(5, 2);
    const int want_warnings = 1;
#else
    LoadProgress::placed(40, 2);
    const int want_warnings = 0;
#endif
    LoadProgress::complete();
    std::fflush(stderr);
    // let go of the file before reading and removing it (Windows cannot remove a file stderr still holds)
    const std::string after = path + ".after";
    if (std::freopen(after.c_str(), "w", stderr) == nullptr) return 1;
    std::ifstream in(path);
    std::vector<std::string> lines;
    for (std::string l; std::getline(in, l);) lines.push_back(l);
    in.close();
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(after, ec);   // still held on Windows: an empty file stays
    int totals = 0, warnings = 0, fails = 0;
    long long last_done = -1;
    bool total_first = false;
    std::map<std::string, std::vector<std::pair<unsigned long long, unsigned long long>>> shares;
    std::vector<std::string> share_order;
    for (const std::string& l : lines) {
        unsigned long long a = 0, b = 0;
        char target[32] = {};
        if (std::sscanf(l.c_str(), "basalt load: share %31s %llu %llu", target, &a, &b) == 3) {
            if (shares[target].empty()) {
                share_order.push_back(target);
                if (a != 0 || last_done > 0) { std::fprintf(stdout, "FAIL share %s announced at 0, before any byte\n", target); ++fails; }
            }
            shares[target].emplace_back(a, b);
        } else if (std::sscanf(l.c_str(), "basalt load: total %llu", &a) == 1) {
            ++totals;
            total_first = last_done < 0 && a == 100;
        } else if (std::sscanf(l.c_str(), "basalt load: weights %llu %llu", &a, &b) == 2) {
            if (b != 100 || (long long) a < last_done || a > b) ++fails;
            last_done = (long long) a;
        } else if (l.find("basalt load: warning: placed more than the announced total") == 0) {
            ++warnings;
        }
    }
    std::FILE* out = stdout;
    unsigned long long share_sum = 0;
    for (const std::string& t : share_order) {
        const auto& seen = shares[t];
        share_sum += seen.front().second;
        int ends = 0;
        unsigned long long prev = 0;
        for (const auto& [done, total] : seen) {
            if (total != seen.front().second || done > total || done < prev) {
                std::fprintf(out, "FAIL share %s climbs to its own total (%llu/%llu)\n", t.c_str(), done, total);
                ++fails;
            }
            prev = done;
            ends += done == total && total > 0;
        }
        if (seen.front().second > 0 && (ends != 1 || seen.back().first != seen.back().second)) {
            std::fprintf(out, "FAIL share %s ends once at its total (%d)\n", t.c_str(), ends);
            ++fails;
        }
    }
    if (share_order != std::vector<std::string>{"cuda0", "cuda1", "ram"} || shares["cuda0"].front().second != 60 ||
        shares["cuda1"].size() != 1 || shares["ram"].front().second != 40 || share_sum != 100) {
        std::fprintf(out, "FAIL the shares as announced, in order, summing to the total\n");
        ++fails;
    }
    if (totals != 1 || !total_first) { std::fprintf(out, "FAIL the total once, before any weights line, unchanged\n"); ++fails; }
    if (warnings != want_warnings) { std::fprintf(out, "FAIL one warning for the miscount (%d)\n", warnings); ++fails; }
    if (last_done != 100) { std::fprintf(out, "FAIL the last line at the total (%lld)\n", last_done); ++fails; }
    std::fprintf(out, "load_progress_test: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
