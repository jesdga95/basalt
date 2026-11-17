// The startup bar: the total is printed once before any byte and never changes; done climbs to it and no further; a
// start that places more or less than it announced is reported once as a warning; bytes placed outside a start (no total
// announced) are not counted.  The lines go to stderr, read back from a file here.
#include "basalt/core/load_progress.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

using basalt::core::LoadProgress;

int main() {
    const std::string path = (std::filesystem::current_path() / "load_progress_test.txt").string();
    if (std::freopen(path.c_str(), "w", stderr) == nullptr) return 1;
    LoadProgress::placed(7);   // outside a start
    LoadProgress::announce_total(100);
    LoadProgress::placed(60);
#ifdef NDEBUG
    LoadProgress::placed(60);   // 20 past the total (a debug build stops there instead)
    LoadProgress::placed(5);
    const int want_warnings = 1;
#else
    LoadProgress::placed(40);
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
    for (const std::string& l : lines) {
        unsigned long long a = 0, b = 0;
        if (std::sscanf(l.c_str(), "basalt load: total %llu", &a) == 1) {
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
    if (totals != 1 || !total_first) { std::fprintf(out, "FAIL the total once, before any weights line, unchanged\n"); ++fails; }
    if (warnings != want_warnings) { std::fprintf(out, "FAIL one warning for the miscount (%d)\n", warnings); ++fails; }
    if (last_done != 100) { std::fprintf(out, "FAIL the last line at the total (%lld)\n", last_done); ++fails; }
    std::fprintf(out, "load_progress_test: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
