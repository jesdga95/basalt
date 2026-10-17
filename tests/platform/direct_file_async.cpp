#include "basalt/platform/direct_file.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif
using basalt::platform::DirectFile;
using basalt::platform::Completion;

static void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }

/// Reports a call that names its own failure. Handing `error.c_str()` straight to `check` would read the string
/// before the call that fills it, and that write can move the buffer, so the message is taken after it returned.
static void call_ok(bool ok, const std::string& error, const char* what) {
    if (!ok) throw std::runtime_error(error.empty() ? std::string(what) + ": failed" : error);
}

/// A path no other run of this test holds. mkstemp is POSIX; a name unique to this process and this call is what
/// the test needs. The build directory is where CMake points it, since a temporary directory is not always
/// writable for a test run from a build tree.
static std::string temp_path() {
#if defined(BASALT_TEST_TMPDIR)
    const std::string base = BASALT_TEST_TMPDIR;
#elif defined(_WIN32)
    char dir[MAX_PATH];
    const DWORD n = GetTempPathA((DWORD) sizeof dir, dir);
    const std::string base = (n > 0 && n < (DWORD) sizeof dir) ? std::string(dir, n) : std::string(".");
#else
    const std::string base = "/tmp";
#endif
#if defined(_WIN32)
    const unsigned long owner = (unsigned long) GetCurrentProcessId();
#else
    const unsigned long owner = (unsigned long) getpid();
#endif
    static std::atomic<unsigned> seq{0};
    return base + "/basalt-direct-test-" + std::to_string(owner) + "-" + std::to_string(seq++) + ".bin";
}

int main() {
    const std::string name = temp_path();
    void* memory = DirectFile::alloc_aligned(4096 * 18);
    try {
        check(memory != nullptr, "buffer allocation");
        {
            std::ofstream f(name, std::ios::binary);
            const std::string bytes(4096 * 16 + 17, 'x');
            f.write(bytes.data(), (std::streamsize) bytes.size());
            check(f.is_open() && (bool) f, ("cannot write the test file to " + name).c_str());
        }
        DirectFile f;
        std::string error;
        call_ok(f.open(name, error), error, "open");
        check(!f.submit(1, memory, 4096, 0, error), "unaligned request accepted");
        for (unsigned i = 0; i < 17; ++i)
            call_ok(f.submit(i * 4096, static_cast<char*>(memory) + i * 4096, 4096, i, error), error, "submit");
        std::vector<bool> seen(17);
        unsigned count = 0;
        while (count < 17) {
            Completion c[17]; int n = f.wait(c, 17, 5000);
            check(n > 0, "completion timeout");
            for (int i = 0; i < n; ++i) {
                check(c[i].tag < 17 && !seen[c[i].tag], "wrong or repeated completion");
                seen[c[i].tag] = true; ++count;
                check(c[i].ok && c[i].bytes == (c[i].tag == 16 ? 17u : 4096u), "incorrect full/EOF completion");
                check(static_cast<char*>(memory)[c[i].tag * 4096] == 'x', "read data mismatch");
            }
        }
        std::jthread waker([&] { f.wake(); });
        Completion c; check(f.wait(&c, 1, 5000) == 1 && c.tag == DirectFile::WAKE_TAG, "wake lost");
        waker.join();
        call_ok(f.submit(0, memory, 4096, 99, error), error, "submit before close");
        f.close(); // Must drain outstanding buffer writes before returning.
        std::memset(memory, 0, 4096);
        call_ok(f.open(name, error), error, "reopen");
        check(f.wait(&c, 1, 0) == 0, "stale completion on reopen");
        call_ok(f.submit(0, memory, 4096, 100, error), error, "submit after reopen");
        check(f.wait(&c, 1, 5000) == 1 && c.ok && c.tag == 100, "reopened read failed");
        f.close();
        DirectFile::free_aligned(memory); std::remove(name.c_str());
        std::cout << "direct_file_async: passed queued reads, short EOF, wake, close drain, reopen\n";
        return 0;
    } catch (const std::exception& e) {
        DirectFile::free_aligned(memory); std::remove(name.c_str());
        std::cerr << e.what() << '\n'; return 1;
    }
}
