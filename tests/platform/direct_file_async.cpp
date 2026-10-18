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

#if defined(_WIN32)
// ----------------------------------------------------------------------------------- Windows stress tests
/// A test file of `page_count` alignment-sized pages and an aligned buffer with spare pages for reads that start
/// past the end of it. Both are released whichever way the test leaves.
struct Fixture {
    static constexpr unsigned spare = 8;
    std::string name;
    void* memory = nullptr;
    unsigned page_count = 0;

    Fixture(unsigned pages, char value) : page_count(pages) {
        const uint32_t page = DirectFile::alignment();
        name = temp_path();
        std::ofstream f(name, std::ios::binary);
        const std::string content((size_t) page_count * page, value);
        f.write(content.data(), (std::streamsize) content.size());
        check(f.is_open() && (bool) f, ("cannot write the test file to " + name).c_str());
        memory = DirectFile::alloc_aligned((size_t) (page_count + spare) * page);
        check(memory != nullptr, "buffer allocation");
    }
    ~Fixture() {
        DirectFile::free_aligned(memory);
        std::remove(name.c_str());
    }
    /// Page `i` of the buffer. Pages at or past `page_count` are scratch for reads beyond the end of the file, so
    /// a write that belongs to no completion cannot be mistaken for the data a real read must have left.
    char* at(size_t i) const { return static_cast<char*>(memory) + i * DirectFile::alignment(); }
};

/// `close` must not return while a read can still write: a caller that abandons a prefill releases its buffers as
/// soon as close returns. No read here is ever waited on, so all of them are still queued or in flight when close
/// runs.
static void test_close_with_reads_in_flight() {
    const uint32_t page = DirectFile::alignment();
    const unsigned pages = 128;   // more than the issuers can have finished by the time close runs
    Fixture fx(pages, 'x');
    DirectFile f;
    std::string error;
    call_ok(f.open(fx.name, error), error, "open");
    for (unsigned i = 0; i < pages; ++i)
        call_ok(f.submit((uint64_t) i * page, fx.at(i), page, i, error), error, "submit");
    f.close();
    check(!f.is_open(), "close left the file open");
    // No write, late completion or otherwise, may land in the caller's buffer once close has returned.
    std::memset(fx.memory, 0, (size_t) pages * page);
    Sleep(200);
    for (unsigned i = 0; i < pages; ++i)
        check(fx.at(i)[0] == 0 && fx.at(i)[page - 1] == 0, "a read wrote into the buffer after close");
    // close() reaped the requests back into the free list, so these cycles reuse them: a request handed out
    // twice would put two reads on one OVERLAPPED and lose or repeat a completion.
    for (int cycle = 0; cycle < 3; ++cycle) {
        call_ok(f.open(fx.name, error), error, "reopen");
        for (unsigned i = 0; i < pages; ++i)
            call_ok(f.submit((uint64_t) i * page, fx.at(i), page, i, error), error, "submit after reopen");
        std::vector<bool> seen(pages, false);
        unsigned got = 0;
        while (got < pages) {
            Completion c[16];
            const int n = f.wait(c, 16, 1000);
            check(n > 0, "completion timeout after reopen");
            for (int i = 0; i < n; ++i) {
                check(c[i].tag < pages && !seen[c[i].tag], "wrong or repeated completion after reopen");
                seen[c[i].tag] = true; ++got;
                check(c[i].ok && c[i].bytes == page, "incorrect completion after reopen");
                check(fx.at(c[i].tag)[0] == 'x' && fx.at(c[i].tag)[page - 1] == 'x', "read data mismatch after reopen");
            }
        }
        f.close();
        Completion extra;
        check(f.wait(&extra, 1, 0) == 0, "stale completion after close");
    }
}

/// Hand the completions of a mixed workload through an array of `max` slots followed by guard slots: nothing may
/// be written outside the slots, every completion arrives exactly once, and the caller sees only the wakes it
/// asked for, never the nudge left behind by a read that finished inside the backend.
static void test_wait_bounds_and_wakes(unsigned max) {
    const uint32_t page = DirectFile::alignment();
    const unsigned pages = 24, past_end = Fixture::spare, wakes = 3;
    Fixture fx(pages, 'y');
    DirectFile f;
    std::string error;
    call_ok(f.open(fx.name, error), error, "open");
    for (unsigned i = 0; i < pages; ++i)
        call_ok(f.submit((uint64_t) i * page, fx.at(i), page, i, error), error, "submit");
    // Reads starting past the end of the file finish inside the backend rather than as a packet of their own, and
    // each still owes exactly one completion.
    const uint64_t eof_tag = 900;
    for (unsigned i = 0; i < past_end; ++i)
        call_ok(f.submit((uint64_t) (pages + 4 + i) * page, fx.at(pages + i), page, eof_tag + i, error),
                error, "submit past the end of the file");
    for (unsigned i = 0; i < wakes; ++i) f.wake();

    constexpr unsigned slots = 32;
    Completion buf[slots];
    std::vector<bool> seen(pages, false), seen_end(past_end, false);
    unsigned real = 0, ended = 0, woken = 0, empty = 0;
    while (real < pages || ended < past_end || woken < wakes) {
        for (unsigned i = 0; i < slots; ++i) buf[i] = Completion{};
        const int n = f.wait(buf, (int) max, 1000);
        check(n >= 0 && n <= (int) max, "wait returned more completions than max");
        for (unsigned i = max; i < slots; ++i)
            check(buf[i].tag == 0 && buf[i].bytes == 0 && !buf[i].ok, "wait wrote past max");
        if (n == 0) ++empty; else empty = 0;
        check(empty < 10, "wait stopped making progress");
        for (int i = 0; i < n; ++i) {
            if (buf[i].tag == DirectFile::WAKE_TAG) {
                check(buf[i].ok && buf[i].bytes == 0, "a wake completion is wrong");
                ++woken;
                continue;
            }
            if (buf[i].tag >= eof_tag && buf[i].tag < eof_tag + past_end) {
                check(!seen_end[buf[i].tag - eof_tag], "a read past the end completed twice");
                seen_end[buf[i].tag - eof_tag] = true; ++ended;
                continue;
            }
            check(buf[i].tag < pages && !seen[buf[i].tag], "wrong or repeated completion");
            seen[buf[i].tag] = true; ++real;
            check(buf[i].ok && buf[i].bytes == page, "incorrect read completion");
            check(fx.at(buf[i].tag)[0] == 'y' && fx.at(buf[i].tag)[page - 1] == 'y', "read data mismatch");
        }
    }
    check(woken == wakes, "a wake was lost or an internal nudge reached the caller");
    check(ended == past_end, "a read past the end of the file did not complete exactly once");
    Completion extra;
    check(f.wait(&extra, 1, 0) == 0, "extra completion after every one was returned");
    f.close();
}
#endif

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
#if defined(_WIN32)
        test_close_with_reads_in_flight();
        const unsigned maxes[] = {1u, 2u, 3u, 7u, 17u};   // one slot and a batch across the boundary
        for (unsigned max : maxes) test_wait_bounds_and_wakes(max);
        std::cout << "direct_file_async: passed Windows close with reads in flight, waits bounded by max\n";
#endif
        DirectFile::free_aligned(memory); std::remove(name.c_str());
        std::cout << "direct_file_async: passed queued reads, short EOF, wake, close drain, reopen\n";
        return 0;
    } catch (const std::exception& e) {
        DirectFile::free_aligned(memory); std::remove(name.c_str());
        std::cerr << e.what() << '\n'; return 1;
    }
}
