// tests/core/ple_nvfp4_test.cpp - an NVFP4 n-gram table read through `PleTable`: the rows it hands back are the
// rows the decoder means, the metadata table scale is the one applied, and a table without it is refused.
//
// The row decode itself is checked against a double-precision transcription in `ple_parity`; what is checked
// HERE is everything the table reader adds over it - the type dispatch, the 90-byte row stride, the F32
// metadata scale, the tensor-size check and the refusals.  Each of those fails *plausibly*: a wrong stride
// decodes shifted rows, a dropped table scale decodes every row by one constant factor, so both must be
// asserted rather than assumed.  The oracles are the standalone row decode and ggml's own block decode, on the
// very bytes the reader reads.
//
// Synthetic file at small geometry; no device, no model.
#include "gguf_fixture.hpp"

#include "basalt/artifact/dequant.hpp"
#include "basalt/kernels/ngram.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace k = basalt::kernels;
namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-78s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("basalt-ple-nvfp4-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

constexpr const char* SCALE_KEY = k::PLE_NVFP4_SCALE_KEY;
constexpr uint64_t ROWS = 32;                          // even: 160 values a row are whole 64-value NVFP4 blocks
constexpr float TABLE_SCALE = 3.324236240587197e-05f;  // the scale a real QAD checkpoint carries for its table
constexpr uint32_t TYPE_NVFP4 = 40, TYPE_Q8_0 = 8;

/// The table's bytes as `fixture::write` fills them, so the expected decode is computed from exactly the bytes
/// the reader will read.
std::vector<uint8_t> table_bytes() {
    std::vector<uint8_t> b(ROWS * (uint64_t) k::PLE_ROW_BYTES_NVFP4);
    for (uint64_t i = 0; i < b.size(); ++i) b[i] = fixture::pattern(7, i);
    return b;
}

const uint8_t* row_of(const std::vector<uint8_t>& t, uint64_t r) {
    return t.data() + r * k::PLE_ROW_BYTES_NVFP4;
}

/// A file holding the table alone, with `kv` as its metadata.
fs::path write_table(const TempDir& tmp, const std::string& tag, uint32_t type, const std::vector<fixture::Kv>& kv) {
    const fs::path p = tmp.path / ("ple-" + tag + ".gguf");
    fixture::write(p, kv, {{"per_layer_token_embd.weight", {(uint64_t) k::PLE_HEAD_DIM, ROWS}, type, 7}});
    return p;
}

}  // namespace

int main() {
    const std::vector<uint8_t> bytes = table_bytes();
    const std::vector<fixture::Kv> none = {fixture::str("general.architecture", "qwen4exp")};
    std::vector<fixture::Kv> scaled = none, unit = none;
    scaled.push_back(fixture::f32(SCALE_KEY, TABLE_SCALE));
    unit.push_back(fixture::f32(SCALE_KEY, 1.0f));
    TempDir tmp;

    std::printf("1. an NVFP4 table read through the table reader\n");
    {
        k::PleTable table;
        std::string err;
        if (!table.open(write_table(tmp, "nvfp4", TYPE_NVFP4, scaled).string(), err)) {
            check(false, "it opens: " + err);
            return 1;
        }
        check(std::strcmp(table.format(), "NVFP4") == 0, std::string("the format is NVFP4, not ") + table.format());
        check(table.rows() == ROWS, "the row count is the GGUF's shape[1]");
        bool rows_match = true, block_match = true, finite = true;
        int moved = 0;
        for (uint64_t r = 0; r < ROWS; ++r) {
            std::vector<float> got(k::PLE_HEAD_DIM, 0.0f), want(k::PLE_HEAD_DIM, 0.0f), unit(k::PLE_HEAD_DIM, 0.0f);
            table.read_row((uint32_t) r, got.data());
            k::nvfp4_dequant_row(row_of(bytes, r), want.data(), TABLE_SCALE);
            k::nvfp4_dequant_row(row_of(bytes, r), unit.data(), 1.0f);   // unscaled: a block decode carries no d
            // The experts' block decode over the row's groups, their scales hoisted to a block's front.  What the
            // two layouts share is a 16-value group: one UE4M3 scale and 8 code bytes, value j in the low nibble
            // of byte j and value j + 8 in its high nibble.  Four groups make a block; a row's last two groups are
            // the half block no block decoder can read, which is why the row stride is 90 and not 72+36.
            std::vector<float> blk(128, 0.0f), misread(64, 1.0f);
            for (int b = 0; b < 2; ++b) {
                uint8_t buf[36] = {};
                for (int g = 0; g < 4; ++g) {
                    const uint8_t* grp = row_of(bytes, r) + (b * 4 + g) * 9;
                    buf[g] = grp[0];
                    std::memcpy(buf + 4 + g * 8, grp + 1, 8);
                }
                basalt::dequantize_nvfp4(buf, blk.data() + b * 64);
            }
            basalt::dequantize_nvfp4(row_of(bytes, r), misread.data());   // the row read AS a block: wrong, on purpose
            for (int i = 0; i < k::PLE_HEAD_DIM; ++i) {
                if (!std::isfinite(got[i])) finite = false;
                if (got[i] != want[i]) rows_match = false;                // exact: same bytes, same decode
                if (i < 128 && unit[i] != blk[i]) block_match = false;    // exact: the shared group decode
            }
            for (int i = 0; i < 64; ++i)
                if (unit[i] != misread[i]) ++moved;
        }
        const int total = (int) (ROWS * 64);
        check(rows_match, "every row equals the standalone row decode of the same bytes");
        check(block_match, "the row's groups, hoisted into blocks, decode the same 128 values");
        check(moved >= total - total / 4, "a row read as a block is a different decode (moved " + std::to_string(moved) +
                                          " of " + std::to_string(total) + ")");
        check(finite, "no non-finite value in any decoded row");
        table.close();
    }

    std::printf("2. the metadata table scale is the one applied\n");
    {
        k::PleTable real, unscaled;
        std::string err;
        const bool ok = real.open(write_table(tmp, "nvfp4", TYPE_NVFP4, scaled).string(), err) &&
                        unscaled.open(write_table(tmp, "unit", TYPE_NVFP4, unit).string(), err);
        check(ok, "the table opens with its own scale and with a unit one: " + err);
        bool unscaled_ok = true, ratio_ok = true, distinct = false;
        if (ok) {
            std::vector<float> a(k::PLE_HEAD_DIM), b(k::PLE_HEAD_DIM), want(k::PLE_HEAD_DIM);
            const uint64_t pick[] = {0, 5, 17, ROWS - 1};
            for (uint64_t r : pick) {
                unscaled.read_row((uint32_t) r, a.data());
                real.read_row((uint32_t) r, b.data());
                k::nvfp4_dequant_row(row_of(bytes, r), want.data(), 1.0f);
                for (int i = 0; i < k::PLE_HEAD_DIM; ++i) {
                    if (a[i] != want[i]) unscaled_ok = false;
                    if (a[i] != b[i]) distinct = true;
                    const double expect = (double) b[i] / TABLE_SCALE;
                    if (std::fabs((double) a[i] - expect) > std::fabs(expect) * 1e-6 + 1e-30) ratio_ok = false;
                }
            }
        }
        check(unscaled_ok, "a unit table scale decodes the rows unscaled - the key is read, not decorative");
        check(ratio_ok, "the unit-scaled rows are the real rows divided by the table scale");
        check(distinct, "the table scale moves the values");
        real.close();
        unscaled.close();
    }

    std::printf("3. a gathered row is the row it is alone\n");
    {
        k::PleTable table;
        std::string err;
        if (!table.open(write_table(tmp, "nvfp4", TYPE_NVFP4, scaled).string(), err)) {
            check(false, "the table opens for the gather: " + err);
        } else {
            std::vector<uint32_t> rows16(16);
            for (int i = 0; i < 16; ++i) rows16[i] = (uint32_t) i;
            std::vector<float> batch(16 * k::PLE_HEAD_DIM), once(k::PLE_HEAD_DIM);
            table.gather(rows16.data(), batch.data());
            bool same = true;
            for (uint32_t r = 0; r < 16; ++r) {
                table.read_row(r, once.data());
                for (int i = 0; i < k::PLE_HEAD_DIM; ++i)
                    if (batch[r * k::PLE_HEAD_DIM + i] != once[i]) same = false;
            }
            check(same, "gather(16 rows) holds each row's own 160 values, head-slowest");
            std::vector<uint32_t> moved(16, 9);
            std::vector<float> again(16 * k::PLE_HEAD_DIM);
            table.gather(moved.data(), again.data());
            bool position_free = true;
            for (int p = 0; p < 16; ++p)
                for (int i = 0; i < k::PLE_HEAD_DIM; ++i)
                    if (again[p * k::PLE_HEAD_DIM + i] != again[9 * k::PLE_HEAD_DIM + i]) position_free = false;
            check(position_free, "a row decodes the same in every slot of the window");
            table.close();
        }
    }

    std::printf("4. a table whose scale is missing or unusable is refused\n");
    {
        struct Case { const char* tag; std::vector<fixture::Kv> kv; };
        const std::vector<Case> cases = {
            {"absent", none},
            {"zero", {fixture::str("general.architecture", "qwen4exp"), fixture::f32(SCALE_KEY, 0.0f)}},
            {"negative", {fixture::str("general.architecture", "qwen4exp"), fixture::f32(SCALE_KEY, -1.5f)}},
            {"NaN", {fixture::str("general.architecture", "qwen4exp"), fixture::f32(SCALE_KEY, std::nanf(""))}},
            {"a string", {fixture::str("general.architecture", "qwen4exp"), fixture::str(SCALE_KEY, "3.3e-05")}},
        };
        for (const auto& c : cases) {
            k::PleTable table;
            std::string err;
            const bool ok = table.open(write_table(tmp, c.tag, TYPE_NVFP4, c.kv).string(), err);
            check(!ok && err.find("nvfp4_scale_2") != std::string::npos,
                  std::string("a table with the scale ") + c.tag + " is refused, naming the key");
            table.close();
        }
        // The scale belongs to NVFP4 rows only: a Q8_0 table opens with it beside it and without it.
        for (const auto& kv : {scaled, none}) {
            k::PleTable table;
            std::string err;
            const bool ok = table.open(write_table(tmp, "q8", TYPE_Q8_0, kv).string(), err);
            check(ok && std::strcmp(table.format(), "Q8_0") == 0, "a Q8_0 table opens, with the key or without it");
            table.close();
        }
    }

    std::printf(g_fail ? "ple_nvfp4_test: %d FAILED\n" : "ple_nvfp4_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
