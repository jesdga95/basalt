// tests/core/ple_ram_quant_test.cpp - background/demand conversion and owned-table lifetime.
#include "gguf_fixture.hpp"

#include "basalt/artifact/dequant.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/ngram.hpp"
#include "basalt/ngram/ple_ram_quant.hpp"

#include <algorithm>
#include <array>
#include <barrier>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using basalt::kernels::PleIo;
using basalt::kernels::PleIoOptions;
using basalt::kernels::PleTable;
using basalt::ngram::open_ple_q8_0;

namespace {
constexpr uint32_t kRows = 1031;
constexpr size_t kCols = 160, kQ8Bytes = 170, kBf16Bytes = 320;
constexpr size_t kHeads = 16;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

std::string utf8(const fs::path& path) {
    const auto text = path.u8string();
    return std::string(reinterpret_cast<const char*>(text.data()), text.size());
}

struct TempDir {
    fs::path path;
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned attempt = 0; attempt != 100; ++attempt) {
            const fs::path candidate = fs::temp_directory_path() /
                ("basalt-ple-ram-quant-" + std::to_string(stamp) + "-" + std::to_string(attempt));
            if (fs::create_directory(candidate)) { path = candidate; return; }
        }
        throw std::runtime_error("cannot create isolated temporary directory");
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

// All values are exactly BF16-representable. Each row has a different signed pattern, including
// both halves, so treating a BF16 row as 160 bytes cannot accidentally reproduce its oracle.
float expected_value(uint32_t row, size_t col) {
    const size_t block = col / 32, j = col % 32;
    if (row == 0 && block == 0) return 0.0f;
    constexpr int exponents[] = {-20, -8, 0, 8, 14};
    const float scale = std::ldexp(1.0f, exponents[(row + block) % 5]);
    if (j == 0) return 127.0f * scale;
    if (j == 1) return -127.0f * scale;
    if (j == 2) return 0.5f * scale;
    if (j == 3) return -0.5f * scale;
    if (j == 4) return 0.0f;
    uint32_t mixed = row * 0x9e3779b9u + static_cast<uint32_t>(col) * 0x85ebca6bu;
    mixed ^= mixed >> 16;
    mixed *= 0x7feb352du;
    mixed ^= mixed >> 15;
    return static_cast<float>(static_cast<int>(mixed % 253) - 126) * scale;
}

struct Oracle {
    std::vector<float> bf16 = std::vector<float>(kRows * kCols);
    std::vector<uint8_t> source = std::vector<uint8_t>(kRows * kBf16Bytes);
    std::vector<uint8_t> packed = std::vector<uint8_t>(kRows * kQ8Bytes);

    Oracle() {
        const auto* traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
        require(traits && traits->from_float_ref, "Q8_0 CPU reference encoder unavailable");
        for (uint32_t row = 0; row < kRows; ++row) {
            for (size_t col = 0; col < kCols; ++col) {
                const float value = expected_value(row, col);
                const uint32_t bits = std::bit_cast<uint32_t>(value);
                require((bits & 0xffffu) == 0, "fixture value is not exact BF16");
                bf16[row * kCols + col] = value;
                const uint16_t word = static_cast<uint16_t>(bits >> 16);
                std::memcpy(source.data() + row * kBf16Bytes + col * 2, &word, sizeof word);
            }
            // Independent of source reads, offsets, chunking and BF16 widening: the ordinary model-file
            // encoder sees the expected row directly. This tests the loader, not a second quantizer.
            traits->from_float_ref(bf16.data() + row * kCols, packed.data() + row * kQ8Bytes, kCols);
        }
    }
};

uint64_t write_table(const fs::path& path, uint32_t type, const std::vector<uint8_t>& payload) {
    const auto written = fixture::write(path, {}, {
        {"prefix.weight", {8}, 0, 93},
        {"per_layer_token_embd.weight", {kCols, kRows}, type, 71},
    });
    const uint64_t offset = written.data_start + written.offsets[1];
    require(written.offsets[1] != 0 && offset % 4096 != 0,
            "fixture must have a nonzero tensor offset unaligned to direct-I/O pages");
    require(written.size % 4096 != 0, "fixture must end in a partial page");
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(static_cast<std::streamoff>(offset));
    file.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    require(static_cast<bool>(file), "cannot populate GGUF fixture payload");
    return offset;
}

void compare_packed(const basalt::ngram::LockedPleBuffer& buffer, const Oracle& oracle,
                    const std::vector<float>& decoded, const std::string& label) {
    // Large pages are inherently resident; their explicit locked_bytes count is zero.
    require(buffer.data && buffer.mapping && buffer.mapping_bytes >= oracle.packed.size() &&
            (buffer.locked_bytes == 0 || buffer.locked_bytes >= oracle.packed.size()),
            label + ": output must own the full resident allocation");
    const auto* bytes = static_cast<const uint8_t*>(buffer.data);
    size_t bad_bytes = 0, bad_values = 0;
    float max_error = 0.0f;
    std::array<float, kCols> row_values{};
    for (uint32_t row = 0; row < kRows; ++row) {
        for (size_t b = 0; b < kQ8Bytes; ++b)
            bad_bytes += bytes[row * kQ8Bytes + b] != oracle.packed[row * kQ8Bytes + b];
        for (size_t block = 0; block < kCols / 32; ++block)
            basalt::dequantize_q8_0(bytes + row * kQ8Bytes + block * 34, row_values.data() + block * 32);
        for (size_t col = 0; col < kCols; ++col) {
            const float expected = decoded[row * kCols + col];
            bad_values += row_values[col] != expected;
            max_error = std::max(max_error, std::abs(row_values[col] - expected));
        }
    }
    std::printf("  %s: %u rows, %zu packed bytes, %zu decoded values; mismatches %zu/%zu, max error %.9g\n",
                label.c_str(), kRows, oracle.packed.size(), decoded.size(), bad_bytes, bad_values,
                static_cast<double>(max_error));
    require(bad_bytes == 0 && bad_values == 0, label + ": differs from CPU model-file Q8_0 oracle");
}

void exercise_partial(basalt::ngram::LockedPleBuffer& buffer, const Oracle& oracle) {
    std::array<int, 4> ok{1, 1, 1, 1};
    std::barrier start((ptrdiff_t) ok.size());
    std::vector<std::thread> readers;
    for (size_t worker = 0; worker < ok.size(); ++worker) readers.emplace_back([&, worker] {
        std::string err;
        start.arrive_and_wait();
        // Competing readers visit the tail first, cross every 128-row region boundary, and repeat rows.
        for (uint32_t i = 0; i < 2 * kRows; ++i) {
            const uint32_t row = kRows - 1 - ((i + (uint32_t) worker * 127) % kRows);
            if (!buffer.ensure_row(row, err) ||
                std::memcmp(buffer.data + row * kQ8Bytes, oracle.packed.data() + row * kQ8Bytes, kQ8Bytes) != 0)
                ok[worker] = 0;
        }
    });
    for (auto& thread : readers) thread.join();
    require(std::all_of(ok.begin(), ok.end(), [](int value) { return value != 0; }),
            "foreground/background ownership must preserve exact Q8 bytes for all concurrent readers");
    std::string err;
    require(!buffer.ensure_row(kRows, err) && !err.empty(), "ensure must reject invalid rows");
    require(buffer.ensure_row(kRows - 1, err), "invalid demand must not poison valid rows");
    require(buffer.wait(err), "background completion: " + err);
    require(buffer.wait(err), "completion wait must be repeatable: " + err);
}

void compare_all_rows(const PleTable& table, const std::vector<float>& expected, const std::string& label) {
    std::array<float, kCols> row_values{};
    size_t bad = 0;
    for (uint32_t row = 0; row < kRows; ++row) {
        table.read_row(row, row_values.data());
        for (size_t col = 0; col < kCols; ++col) bad += row_values[col] != expected[row * kCols + col];
    }
    std::printf("  %s: %u rows / %zu values, %zu mismatches\n", label.c_str(), kRows, expected.size(), bad);
    require(bad == 0, label + ": row mismatch");
}

std::array<uint32_t, 3 * kHeads> batch_indices() {
    constexpr uint32_t boundaries[] = {0, 1, 12, 13, 126, 127, 128, 508, 509, 510, 1015, 1016, 1017, 1018, 1029, 1030};
    std::array<uint32_t, 3 * kHeads> rows{};
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = boundaries[(i * 7 + i / kHeads) % kHeads];
    rows[17] = rows[16]; // duplicate rows must preserve output placement, not disappear during dedupe
    return rows;
}

bool rows_equal(const float* values, const uint32_t* rows, size_t count, const std::vector<float>& expected) {
    for (size_t i = 0; i < count; ++i)
        for (size_t col = 0; col < kCols; ++col)
            if (values[i * kCols + col] != expected[rows[i] * kCols + col]) return false;
    return true;
}

void exercise_apis(PleTable& table, const std::vector<float>& expected) {
    const auto rows = batch_indices();
    std::vector<float> out(rows.size() * kCols, -999.0f);
    std::string err;
    table.gather(rows.data(), out.data());
    require(rows_equal(out.data(), rows.data(), kHeads, expected), "gather ordering/parity");
    require(!table.collect(out.data(), err), "collect without issue must fail");
    require(table.issue(rows.data()), "issue converted rows");
    require(!table.gather_batch(rows.data(), 3, out.data(), err), "batch must not consume pending decode slot");

    // The independent prompt API must not overwrite the pending token, even with simultaneous readers.
    std::array<int, 3> thread_ok{1, 1, 1};
    std::vector<std::thread> readers;
    for (size_t worker = 0; worker < thread_ok.size(); ++worker) {
        readers.emplace_back([&, worker] {
            std::vector<float> values(rows.size() * kCols);
            std::string error;
            for (int generation = 0; generation < 4; ++generation) {
                if (!table.gather_rows(rows.data(), 3, values.data(), error) ||
                    !rows_equal(values.data(), rows.data(), rows.size(), expected)) thread_ok[worker] = 0;
            }
        });
    }
    for (auto& reader : readers) reader.join();
    require(std::all_of(thread_ok.begin(), thread_ok.end(), [](int ok) { return ok != 0; }),
            "concurrent gather_rows must preserve every prompt row");
    require(table.collect(out.data(), err) && rows_equal(out.data(), rows.data(), kHeads, expected),
            "collect must retain original token across concurrent prompt reads");
    require(!table.collect(out.data(), err), "collected ticket cannot be collected twice");
    require(table.gather_batch(rows.data(), 3, out.data(), err) &&
            rows_equal(out.data(), rows.data(), rows.size(), expected), "batch duplicates/order/tail parity");
    table.prefetch_rows(rows.data());
    table.wait_prefetches();
    require(table.warm_rows(rows.data(), 3, err), "warm converted rows");
    require(table.gather_rows(rows.data(), 3, out.data(), err) &&
            rows_equal(out.data(), rows.data(), rows.size(), expected), "prefetch/warm must preserve owned table");
    require(table.gather_batch(rows.data(), 0, out.data(), err), "empty batch succeeds");
    for (uint32_t invalid : {kRows, std::numeric_limits<uint32_t>::max()}) {
        std::array<float, kCols> missing;
        missing.fill(123.0f);
        table.read_row(invalid, missing.data());
        require(std::all_of(missing.begin(), missing.end(), [](float v) { return v == 0.0f; }),
                "out-of-bounds converted read_row must zero the result");
    }
}

PleIoOptions quant_options(int workers = 3) {
    PleIoOptions io;
    io.mode = PleIo::Mmap;
    io.lock = true;
    io.quantize_q8_0 = true;
    io.quant_workers = workers;
    return io;
}

void failures(const fs::path& source, uint64_t offset, const fs::path& q8, const fs::path& dir) {
    std::string err;
    const std::string path = utf8(source);
    auto refused = [&](uint64_t at, uint64_t rows, int workers, uint32_t chunk, const char* label) {
        err.clear();
        auto result = open_ple_q8_0(path, at, rows, err, workers, chunk);
        require(!result && !err.empty(), label);
    };
    refused(offset, kRows, 0, 127, "helper must reject zero workers");
    refused(offset, kRows, -1, 127, "helper must reject negative workers");
    refused(offset, kRows, 1, 0, "helper must reject zero chunk rows");
    refused(offset, 0, 1, 127, "helper must reject empty table");
    refused(std::numeric_limits<uint64_t>::max() - 100, kRows, 1, 127, "helper must reject overflowing offset");
    refused(offset, std::numeric_limits<uint64_t>::max(), 1, 127, "helper must reject overflowing row extent");
    refused(offset + 1, kRows, 1, 127, "helper must reject one-byte source overrun");

    const fs::path truncated = dir / "truncated.gguf";
    fs::copy_file(source, truncated);
    fs::resize_file(truncated, fs::file_size(truncated) - 1);
    err.clear();
    require(!open_ple_q8_0(utf8(truncated), offset, kRows, err, 3, 509) && !err.empty(),
            "helper must reject truncated final source row");
    err.clear();
    require(!open_ple_q8_0(utf8(dir / "absent.gguf"), offset, kRows, err, 1, 127) && !err.empty(),
            "helper must report missing source");

    PleTable table;
    auto reject_open = [&](const fs::path& file, const PleIoOptions& io, const char* label) {
        err.clear();
        require(!table.open(utf8(file), err, io) && !err.empty() && !table.is_open() && !table.locked(), label);
    };
    auto io = quant_options();
    io.mode = PleIo::Direct;
    reject_open(source, io, "quantized startup must reject direct mode");
    io = quant_options();
    io.lock = false;
    reject_open(source, io, "quantized startup must reject unlocked mmap");
    reject_open(q8, quant_options(), "quantized startup must reject non-BF16 source");
    reject_open(source, quant_options(0), "table must reject zero quant workers");
    reject_open(source, quant_options(-1), "table must reject negative quant workers");
    reject_open(truncated, quant_options(), "table must reject truncated BF16 source");
    reject_open(dir / "absent.gguf", quant_options(), "table must report absent BF16 source");
}
} // namespace

int main() {
    try {
        TempDir temp;
        const fs::path bf16_path = temp.path / fs::path(u8"source-\u03bb-\u6f22.gguf");
        const fs::path q8_path = temp.path / "oracle-q8.gguf";
        const Oracle oracle;
        const uint64_t offset = write_table(bf16_path, 30, oracle.source);
        write_table(q8_path, 8, oracle.packed);
        std::printf("ple_ram_quant_test: BF16 offset %llu, %u x %zu, %zu source bytes\n",
                    static_cast<unsigned long long>(offset), kRows, kCols, oracle.source.size());

        std::string err;
        PleTable reader;
        PleIoOptions ordinary;
        ordinary.cache_rows = 0; // exercise default Direct/BF16 semantics without allocating a production cache
        require(reader.open(utf8(q8_path), err, ordinary), "open ordinary Q8_0 reader: " + err);
        require(std::string(reader.format()) == "Q8_0" && !reader.locked(), "ordinary Q8_0 format/lifetime");
        std::vector<float> decoded(kRows * kCols);
        for (uint32_t row = 0; row < kRows; ++row) reader.read_row(row, decoded.data() + row * kCols);
        reader.close();

        for (const auto& [workers, chunk] : {std::pair{1, 127u}, std::pair{3, 509u}}) {
            auto buffer = open_ple_q8_0(utf8(bf16_path), offset, kRows, err, workers, chunk);
            require(static_cast<bool>(buffer), "CPU helper must allocate and lock small fixture: " + err);
            exercise_partial(*buffer, oracle);
            compare_packed(*buffer, oracle, decoded,
                           "helper " + std::to_string(workers) + " workers / " + std::to_string(chunk) + " rows");
        }
        // Immediate teardown must drain reads before freeing stages/destination; later opens still work.
        for (int generation = 0; generation < 8; ++generation) {
            auto cancelled = open_ple_q8_0(utf8(bf16_path), offset, kRows, err, 3, 128);
            require(static_cast<bool>(cancelled), "open before cancellation: " + err);
            cancelled.reset();
        }
        failures(bf16_path, offset, q8_path, temp.path);

        PleTable table;
        for (int generation = 0; generation < 2; ++generation) {
            require(table.open(utf8(bf16_path), err, quant_options(generation == 0 ? 1 : 3)),
                    "open CPU-converted table: " + err);
            require(table.is_open() && table.locked() && table.rows() == kRows &&
                    table.mode() == PleIo::Mmap && std::string(table.format()) == "Q8_0",
                    "converted table must expose locked Q8_0 RAM");
            exercise_apis(table, decoded);
            compare_all_rows(table, decoded, "converted generation " + std::to_string(generation));
        }
        err.clear();
        require(!table.open(utf8(q8_path), err, quant_options()) && !err.empty() &&
                !table.is_open() && !table.locked() && table.rows() == 0,
                "failed reopen must release prior converted storage rather than expose stale rows");
        require(table.open(utf8(bf16_path), err, ordinary), "reopen original BF16 after conversion: " + err);
        require(std::string(table.format()) == "BF16" && !table.locked() && table.mode() == PleIo::Direct,
                "ordinary reopen must discard converted storage/format");
        compare_all_rows(table, oracle.bf16, "ordinary BF16 source remains unchanged");
        require(table.open(utf8(q8_path), err, ordinary), "reopen ordinary Q8_0 after BF16: " + err);
        compare_all_rows(table, decoded, "ordinary Q8_0 reopen");
        require(table.open(utf8(bf16_path), err, quant_options()), "reopen converted table before close: " + err);
        auto retained = open_ple_q8_0(utf8(bf16_path), offset, kRows, err, 3, 509);
        require(static_cast<bool>(retained), "retain helper allocation: " + err);
        require(retained->wait(err), "wait before source deletion: " + err);

        const auto rows = batch_indices();
        require(table.issue(rows.data()), "issue before close");
        table.close();
        require(!table.is_open() && !table.locked() && table.rows() == 0, "close must release converted table");
        require(fs::remove(bf16_path), "completed/cancelled conversion must release source handles");
        require(retained->ensure_row(kRows - 1, err), "completed rows survive source deletion");
        require(retained->wait(err), "completed wait survives source deletion");
        compare_packed(*retained, oracle, decoded, "helper after source deletion");
        require(table.open(utf8(q8_path), err, ordinary), "reopen after closing pending conversion: " + err);
        std::array<float, kHeads * kCols> out{};
        require(!table.collect(out.data(), err), "reopen must not retain old pending token");
        compare_all_rows(table, decoded, "Q8_0 after owned-buffer close");
        std::printf("ple_ram_quant_test: PASS; exact packed and decoded CPU parity, no GPU required\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ple_ram_quant_test: FAIL: %s\n", error.what());
        return 1;
    }
}
