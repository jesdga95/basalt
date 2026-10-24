// tests/program/engine_session_test.cpp - Engine::start against a stand-in engine (this binary run with --serve): an
// error-marked log line during a healthy start does not fail it, an engine that exits before READY does, with its
// error lines in the reason. CPU only.
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

// The stand-in: `--serve ok` logs a warning and an error line, then serves; `--serve die` logs an error line and exits.
int stand_in(const std::string& mode) {
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
    return failures ? 1 : 0;
}
