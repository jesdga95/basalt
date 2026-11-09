// tests/program/engine_session_test.cpp - Engine::start against a stand-in engine (this binary run with --serve): an
// error-marked log line during a healthy start does not fail it, an engine that exits before READY does, with its
// error lines in the reason; an engine with lossy flags reports them. Also the default engine and fixtures paths and
// the lossy note. CPU only.
#include "basalt/program/engine_session.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string& what, const std::string& got = "") {
    std::printf("engine_session %-58s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    if (!ok) std::printf("  got: [%s]\n", got.c_str());
    failures |= !ok;
}

// The stand-in: `--serve ok` logs a warning and an error line, then serves; `--serve die` logs an error line and exits;
// `--serve lossy` starts as an engine with --route-tail-skip 7 does.
int stand_in(const std::string& mode) {
    if (mode == "lossy") {
        std::fprintf(stderr, "basalt: warning: lossy flags enabled, these flags will affect the accuracy of the served model\n");
        std::fprintf(stderr, "basalt: warning: --route-tail-skip 7 drops low-ranked missed experts\n");
        std::fflush(stderr);
        std::printf("INFO batch_slots=1\nINFO lossy=--route-tail-skip\nREADY 4096 stop\n");
        std::fflush(stdout);
        for (std::string line; std::getline(std::cin, line);)
            if (line.rfind("QUIT", 0) == 0) break;
        return 0;
    }
    std::fprintf(stderr, "basalt serve: warning: a copy was skipped\n");
    std::fprintf(stderr, "basalt serve: error: a swap failed (layer 3)\n");
    std::fflush(stderr);
    if (mode == "die") return 1;
    std::printf("INFO batch_slots=1\nREADY 4096 stop\n");
    std::fflush(stdout);
    for (std::string line; std::getline(std::cin, line);)
        if (line.rfind("QUIT", 0) == 0) break;
    return 0;
}

std::string read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--serve") == 0) return stand_in(argv[2]);

    const std::string self = std::filesystem::absolute(argv[0]).string();
    const std::string dir = std::filesystem::temp_directory_path().string();
    {
        const std::string log = dir + "/engine_session_test_ok.log";
        basalt::program::Engine e;
        std::string err;
        const bool up = e.start(self, {"ok"}, log, err);
        check(up, "an error line in a healthy start does not fail it", err);
        check(e.info["batch_slots"] == "1", "the INFO line is read", e.info["batch_slots"]);
        check(e.info.count("lossy") == 0, "a start without lossy flags reports none");
        e.stop();
        check(read_all(log).find("error: a swap failed") != std::string::npos, "the error line is in the log");
        std::filesystem::remove(log);
    }
    {
        const std::string log = dir + "/engine_session_test_die.log";
        basalt::program::Engine e;
        std::string err;
        const bool up = e.start(self, {"die"}, log, err);
        check(!up, "an engine that exits before READY fails the start");
        check(err.find("the engine exited before READY") == 0, "the reason says so", err);
        check(err.find("error: a swap failed (layer 3)") != std::string::npos, "the reason carries the error line", err);
        check(err.find("warning:") == std::string::npos, "the reason leaves the warning out", err);
        std::filesystem::remove(log);
    }
    {
        const std::string log = dir + "/engine_session_test_lossy.log";
        basalt::program::Engine e;
        std::string err;
        const bool up = e.start(self, {"lossy"}, log, err);
        check(up, "a start with lossy flags is a start", err);
        check(e.info["lossy"] == "--route-tail-skip", "the lossy flags are read from INFO", e.info["lossy"]);
        e.stop();
        std::filesystem::remove(log);
    }
    {   // the engine and fixtures a program uses without --engine / --fixtures
        namespace fs = std::filesystem;
        const fs::path top = fs::path(dir) / "engine_session_test_install";
        fs::remove_all(top);
        fs::create_directories(top / "bin");
        const std::string bench = (top / "bin" / "basalt-bench.exe").string(), built = "/build/bin/basalt";
        check(basalt::program::default_engine(bench, built) == built, "no engine beside the program: the build's");
        std::ofstream((top / "bin" / "basalt.exe").string()) << "x";
        fs::permissions(top / "bin" / "basalt.exe", fs::perms::owner_all);
        const std::string got = basalt::program::default_engine(bench, built);
        check(got == (top / "bin" / "basalt.exe").string(), "the engine beside the program, with its suffix", got);
        check(basalt::program::default_engine("", built) == built, "a program that cannot name itself: the build's");
        check(basalt::program::default_fixtures(top.string(), "/build/data/bench") == "/build/data/bench",
              "no data/bench in the install: the build's fixtures");
        fs::create_directories(top / "data" / "bench");
        check(basalt::program::default_fixtures(top.string(), "/build/data/bench") == (top / "data" / "bench").string(),
              "the install's data/bench");
        fs::remove_all(top);
    }
    {
        const std::string one = basalt::program::lossy_note("--route-tail-skip");
        check(one == "LOSSY: --route-tail-skip changes the model's output; these numbers do not compare with a run without it",
              "the lossy note names the flag", one);
        const std::string two = basalt::program::lossy_note("--route-prior,--route-tail-skip");
        check(two == "LOSSY: --route-prior, --route-tail-skip change the model's output; these numbers do not compare with a "
                     "run without them", "and every flag", two);
    }
    return failures ? 1 : 0;
}
