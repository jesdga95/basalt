// The start's check for the expert profiles the engine ships beside its binary: a missing or empty file, or a missing
// data/profiles folder, gives an error naming the path and the install layout; a pack that names none is never refused.
#include "basalt/core/shipped_profiles.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
}  // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "basalt_shipped_profiles_test";
    fs::remove_all(root);
    const std::string dir = (root / "data" / "profiles").string();
    using basalt::core::missing_shipped_profile;

    check(missing_shipped_profile(dir, {}).empty(), "a pack that ships no profile is never refused, with or without the folder");
    const std::string no_dir = missing_shipped_profile(dir, {"m-1x2.prof"});
    check(!no_dir.empty(), "no data/profiles folder is refused");
    check(has(no_dir, (dir + "/m-1x2.prof").c_str()), "the error names the missing path");
    check(has(no_dir, "next to bin/"), "the error says data/ must sit next to bin/");

    fs::create_directories(dir);
    { std::ofstream f(dir + "/m-1x2.prof", std::ios::binary); f << "x"; }
    check(missing_shipped_profile(dir, {"m-1x2.prof"}).empty(), "a profile that is there passes");
    const std::string second = missing_shipped_profile(dir, {"m-1x2.prof", "m-1x2-dspark.prof"});
    check(has(second, "m-1x2-dspark.prof"), "the second profile missing is named");
    { std::ofstream f(dir + "/m-1x2-dspark.prof", std::ios::binary); }
    check(has(missing_shipped_profile(dir, {"m-1x2-dspark.prof"}), "m-1x2-dspark.prof"), "an empty profile is refused");

    fs::remove_all(root);
    std::printf(fails == 0 ? "shipped_profiles ok\n" : "shipped_profiles FAILED\n");
    return fails == 0 ? 0 : 1;
}
