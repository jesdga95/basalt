// tests/ngram/ple_reader_ram_test.cpp - CPU unit test for PleReader::open_ram (--ple-io ram):
// byte-for-byte parity with Direct reads across misaligned rows, straddles, duplicates, out-of-range
// zero-fill, empty tickets, concurrent issue/collect, mode switching, RAM accounting and error paths.
#include "basalt/ngram/ple_reader.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace ng = basalt::ngram;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                                     \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);     \
            std::fprintf(stderr, __VA_ARGS__);                            \
            std::fprintf(stderr, "\n");                                   \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

constexpr uint64_t kHeader = 192;   // misaligned offset so rows straddle 4 KiB pages

void expected_row(uint32_t row, uint32_t rb, uint8_t* out) {
    for (uint32_t b = 0; b < rb; ++b) out[b] = (uint8_t) ((row * 2654435761u + b * 97u) >> 7);
    if (rb >= 4) std::memcpy(out, &row, 4);
}

bool make_table(const std::filesystem::path& path, uint32_t rows, uint32_t rb) {
    std::ofstream f(path, std::ios::binary);
    std::vector<uint8_t> head(kHeader, 0xAB);
    f.write((const char*) head.data(), (std::streamsize) head.size());
    std::vector<uint8_t> r(rb);
    for (uint32_t i = 0; i < rows; ++i) {
        expected_row(i, rb, r.data());
        f.write((const char*) r.data(), (std::streamsize) rb);
    }
    return (bool) f;
}

bool check_batch(ng::PleReader& rd, const std::vector<uint32_t>& rows, uint32_t n_rows, uint32_t rb,
                 std::vector<uint8_t>* out_copy, const char* what) {
    std::vector<uint8_t> out(rows.size() * rb, 0xCC);
    std::string err;
    const auto t = rd.issue(rows.empty() ? nullptr : rows.data(), rows.size(), rows.empty() ? nullptr : out.data());
    if (!rd.collect(t, err)) {
        CHECK(false, "%s: collect failed: %s", what, err.c_str());
        return false;
    }
    std::vector<uint8_t> want(rb);
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] >= n_rows) std::memset(want.data(), 0, rb);
        else expected_row(rows[i], rb, want.data());
        if (std::memcmp(want.data(), &out[i * rb], rb) != 0) {
            CHECK(false, "%s: row %u (index %zu) differs", what, rows[i], i);
            return false;
        }
    }
    if (out_copy) *out_copy = std::move(out);
    return true;
}

void test_row_size(const std::filesystem::path& dir, uint32_t rb) {
    const uint32_t n_rows = 2000;   // ~180-352 KiB: spans dozens of 4 KiB pages
    const uint64_t expected_bytes = (uint64_t) n_rows * rb;
    const std::filesystem::path path = dir / ("ple_reader_ram_test_" + std::to_string(rb) + ".bin");
    struct Guard {
        std::filesystem::path p;
        ~Guard() {
            std::error_code ec;
            std::filesystem::remove(p, ec);
        }
    } guard{path};

    CHECK(make_table(path, n_rows, rb), "rb=%u: write temp table", rb);
    const std::string spath = path.string();

    // Build a comprehensive query covering boundaries, straddling pages, duplicates and out-of-range rows.
    std::vector<uint32_t> probe = {0, 1, n_rows - 1, n_rows, n_rows + 7, 0xFFFFFFFFu, 0, 1};
    for (uint32_t i = 0; i < n_rows; ++i) {
        const uint64_t off = kHeader + (uint64_t) i * rb;
        if (off / ng::PAGE != (off + rb - 1) / ng::PAGE) probe.push_back(i);
    }
    std::mt19937 rng(42 + rb);
    for (int i = 0; i < 128; ++i) probe.push_back(rng() % n_rows);

    // Direct mode reference (also tests Direct -> RAM -> Direct transition).
    ng::PleReader rd;
    std::string err;
    CHECK(rd.open(spath, kHeader, n_rows, 16, 256, err, rb), "rb=%u: direct open: %s", rb, err.c_str());
    CHECK(rd.is_open(), "rb=%u: direct is_open", rb);
    CHECK(!rd.locked(), "rb=%u: direct !locked", rb);
    CHECK(rd.ram_bytes() == 0, "rb=%u: direct ram_bytes == 0", rb);
    rd.lock_ram();   // no-op in direct mode
    CHECK(!rd.locked(), "rb=%u: direct lock_ram no-op", rb);
    std::vector<uint8_t> direct_bytes;
    check_batch(rd, probe, n_rows, rb, &direct_bytes, "direct");

    for (bool lock : {false, true}) {
        CHECK(rd.open_ram(spath, kHeader, n_rows, err, rb, lock), "rb=%u lock=%d: open_ram: %s", rb, (int) lock, err.c_str());
        CHECK(rd.is_open(), "rb=%u lock=%d: is_open", rb, (int) lock);
        CHECK(rd.row_bytes() == rb, "rb=%u lock=%d: row_bytes", rb, (int) lock);
        CHECK(rd.ram_bytes() == expected_bytes, "rb=%u lock=%d: ram_bytes=%llu want=%llu", rb, (int) lock,
              (unsigned long long) rd.ram_bytes(), (unsigned long long) expected_bytes);
        if (!lock) {
            CHECK(!rd.locked(), "rb=%u: lock=false must report !locked before lock_ram()", rb);
            // Deferred lock_ram() (the attach() path after expert-tier reads finish).
            rd.lock_ram();
        }

        // Keep-alive calls are safe no-ops in RAM mode.
        rd.set_keepalive(40.0, 60.0);
        rd.rearm_keepalive();

        // Byte-for-byte parity with Direct mode.
        std::vector<uint8_t> ram_bytes_out;
        check_batch(rd, probe, n_rows, rb, &ram_bytes_out, "ram");
        CHECK(ram_bytes_out == direct_bytes, "rb=%u lock=%d: RAM bytes differ from Direct bytes", rb, (int) lock);

        // Empty ticket (n == 0).
        check_batch(rd, {}, n_rows, rb, nullptr, "ram empty");

        // Stats contract: requests counted, zero SSD reads/bytes.
        const ng::ReaderStats st = rd.snapshot();
        CHECK(st.requests == probe.size(), "rb=%u lock=%d: requests=%llu want=%zu", rb, (int) lock,
              (unsigned long long) st.requests, probe.size());
        CHECK(st.reads == 0 && st.bytes == 0, "rb=%u lock=%d: expected zero SSD reads/bytes", rb, (int) lock);

        // Ticket validation: unknown ticket and double-collect must both fail.
        std::string terr;
        CHECK(!rd.collect(ng::PleReader::Ticket{0}, terr) && !terr.empty(), "rb=%u: collect(0) must fail", rb);
        std::vector<uint8_t> one(rb);
        const uint32_t row0 = 0;
        const auto tk = rd.issue(&row0, 1, one.data());
        CHECK(rd.collect(tk, terr), "rb=%u: first collect must succeed: %s", rb, terr.c_str());
        terr.clear();
        CHECK(!rd.collect(tk, terr) && !terr.empty(), "rb=%u: double collect must fail", rb);

        // Concurrent issue/collect across threads.
        std::atomic<int> bad{0};
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&, t] {
                std::mt19937 trng(1000u + (uint32_t) t * 97u + rb);
                std::vector<uint32_t> batch(32);
                std::vector<uint8_t> buf(batch.size() * rb);
                std::vector<uint8_t> want(rb);
                for (int iter = 0; iter < 40; ++iter) {
                    for (size_t i = 0; i < batch.size(); ++i)
                        batch[i] = (i % 7 == 0) ? (n_rows + (uint32_t) i) : (trng() % n_rows);
                    const auto ctk = rd.issue(batch.data(), batch.size(), buf.data());
                    std::string cerr;
                    if (!rd.collect(ctk, cerr)) { ++bad; return; }
                    for (size_t i = 0; i < batch.size(); ++i) {
                        if (batch[i] >= n_rows) std::memset(want.data(), 0, rb);
                        else expected_row(batch[i], rb, want.data());
                        if (std::memcmp(want.data(), &buf[i * rb], rb) != 0) { ++bad; return; }
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
        CHECK(bad.load() == 0, "rb=%u lock=%d: concurrent issue/collect had %d failures", rb, (int) lock, bad.load());
    }

    // Reopen in Direct mode directly from RAM mode without explicit close().
    CHECK(rd.open(spath, kHeader, n_rows, 8, 64, err, rb), "rb=%u: direct reopen: %s", rb, err.c_str());
    CHECK(rd.ram_bytes() == 0 && !rd.locked(), "rb=%u: direct reopen state", rb);
    check_batch(rd, probe, n_rows, rb, nullptr, "direct reopen");

    rd.close();
    CHECK(!rd.is_open() && rd.ram_bytes() == 0 && !rd.locked(), "rb=%u: closed state", rb);
}

void test_errors(const std::filesystem::path& dir) {
    const std::filesystem::path path = dir / "ple_reader_ram_test_err.bin";
    struct Guard {
        std::filesystem::path p;
        ~Guard() {
            std::error_code ec;
            std::filesystem::remove(p, ec);
        }
    } guard{path};
    CHECK(make_table(path, 16, 96), "write err table");

    ng::PleReader rd;
    std::string err;

    CHECK(!rd.open_ram((dir / "does_not_exist_ple_ram.bin").string(), 0, 1, err, 96, false) && !err.empty(),
          "missing file must fail");
    CHECK(!rd.is_open() && rd.ram_bytes() == 0, "failed open leaves reader closed");

    err.clear();
    CHECK(!rd.open_ram(path.string(), kHeader, 16, err, 0, false) && !err.empty(), "row_bytes == 0 must fail");
    err.clear();
    CHECK(!rd.open_ram(path.string(), kHeader, 16, err, ng::PAGE + 1, false) && !err.empty(), "row_bytes > PAGE must fail");
    err.clear();
    CHECK(!rd.open_ram(path.string(), kHeader, UINT64_MAX / 96 + 1, err, 96, false) && !err.empty(), "n_rows overflow must fail");
    err.clear();
    CHECK(!rd.open_ram(path.string(), kHeader, 17, err, 96, false) && !err.empty(), "table past EOF must fail");
    err.clear();
    CHECK(!rd.open_ram(path.string(), kHeader + 16 * 96 + 1, 0, err, 96, false) && !err.empty(), "offset past EOF must fail");
}

}  // namespace

int main() {
    const auto tag = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("basalt_ple_ram_" + tag);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "cannot create temp dir %s: %s\n", dir.string().c_str(), ec.message().c_str());
        return 2;
    }

    for (uint32_t rb : {ng::ROW_BYTES, 96u, 176u}) test_row_size(dir, rb);
    test_errors(dir);

    std::filesystem::remove_all(dir, ec);
    if (g_fail == 0) std::printf("ple_reader_ram_test: ok\n");
    return g_fail == 0 ? 0 : 1;
}
