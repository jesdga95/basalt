// tests/platform/process_pipe_test.cpp - platform::spawn_piped, the pipe calls and capture_output, with this program as
// the child. CPU only, no model. `--echo` is the child's mode: it answers each line of its input with the same line
// upper-cased on its output, reports on its error stream, and ends with code 7 when its input ends.
#include "basalt/core/progress.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/platform/system.hpp"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("process_pipe %-54s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}

int child_echo() {
    // the protocol is newline-framed bytes: no text-mode translation may touch it
    basalt::platform::set_stdio_binary();
    std::fprintf(stderr, "child ready\n");
    std::fflush(stderr);
    std::string line;
    while (std::getline(std::cin, line)) {
        for (char& c : line) c = (char) std::toupper((unsigned char) c);
        std::cout << line << "\n" << std::flush;
    }
    return 7;
}

std::string read_line(int fd) {
    std::string line;
    char c = 0;
    while (basalt::platform::pipe_read(fd, &c, 1) == 1 && c != '\n') line += c;
    return line;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--echo") == 0) return child_echo();
    if (argc > 1 && std::strcmp(argv[1], "--both") == 0) {
        std::fputs("to stdout\n", stdout);
        std::fflush(stdout);
        std::fputs("to stderr\n", stderr);
        return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "--crash-terminate") == 0) {
        basalt::platform::install_crash_handler();
        basalt::core::progress_at("decode", 17, 64);
        try {
            throw std::runtime_error("boom");
        } catch (...) {
            std::terminate();
        }
    }
    using namespace basalt::platform;
    const std::string self = module_path();
    check(!self.empty(), "the program knows its own path");
    check(process_id() > 0, "process_id");
    check(!python_interpreter(".").empty(), "python_interpreter names something");

    const std::string log = temp_dir() + "/basalt_process_pipe_test.err";
    PipedChild pc;
    std::string err;
    check(spawn_piped({self, "--echo"}, log, pc, err), "the child starts: " + err);
    if (pc.in_fd >= 0) {
        check(pc.child.id > 0 && pc.out_fd >= 0, "the child has an id and both pipes");
        check(pipe_write_all(pc.in_fd, "hello\n", 6), "a request is written");
        check(read_line(pc.out_fd) == "HELLO", "its reply comes back, no CR");
        std::string big(200000, 'x');
        big += "\n";
        check(pipe_write_all(pc.in_fd, big.data(), big.size()), "a request larger than a pipe buffer is written");
        std::string got = read_line(pc.out_fd);
        check(got.size() == 200000 && got[0] == 'X', "and answered in full");
        check(child_wait(pc.child, 0) == -1, "the child is still running while its input is open");
        close_fd(pc.in_fd);
        pc.in_fd = -1;
        char c;
        check(pipe_read(pc.out_fd, &c, 1) == 0, "the end of its output is seen once it ends");
        check(child_wait(pc.child, 10000) == 7, "its exit code is reported");
        close_fd(pc.out_fd);
        child_close(pc.child);
        std::ifstream f(log);
        std::string line;
        std::getline(f, line);
        check(line.rfind("child ready", 0) == 0, "its error stream went to the log file");
    }
    std::remove(log.c_str());

    // a program that cannot be started is refused, not crashed
    PipedChild none;
    std::string err2;
    const bool started = spawn_piped({temp_dir() + "/basalt_no_such_program"}, log, none, err2);
    if (started) {   // fork starts a child that fails at exec: its exit code says so
        check(child_wait(none.child, 10000) != 0, "a missing program ends non-zero");
        close_fd(none.in_fd);
        close_fd(none.out_fd);
        child_close(none.child);
    } else {
        check(!err2.empty(), "a missing program is refused with a reason");
    }
    std::remove(log.c_str());

    const auto both = capture_output({self, "--both"}, true);
    check(both && both->find("to stdout") != std::string::npos && both->find("to stderr") != std::string::npos,
          "capture_output with stderr mixed in");
    const auto only = capture_output({self, "--both"}, false);
    check(only && only->find("to stdout") != std::string::npos && only->find("to stderr") == std::string::npos,
          "capture_output without stderr");

    const std::string crash_log = temp_dir() + "/basalt_process_crash_test.err";
    PipedChild crashed;
    std::string crash_err;
    check(spawn_piped({self, "--crash-terminate"}, crash_log, crashed, crash_err), "crash child starts: " + crash_err);
    if (crashed.in_fd >= 0) {
        const int rc = child_wait(crashed.child, 10000);
        check(rc != 0, "crash child exits non-zero");
        close_fd(crashed.in_fd);
        close_fd(crashed.out_fd);
        child_close(crashed.child);
        std::ifstream cf(crash_log);
        std::string cline;
        std::getline(cf, cline);
        check(cline.find("basalt crash: error: unhandled exception (boom) in decode (17) chunk 64") != std::string::npos,
              "crash handler writes exception and stage to stderr");
    }
    std::remove(crash_log.c_str());
    return failures ? 1 : 0;
}
