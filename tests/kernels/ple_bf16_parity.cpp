// A small synthetic BF16 PLE table through PleTable: every element against an independent decode.
// BF16 is the high half of f32, so the loader's shift and a from-scratch read of the sign, exponent and
// mantissa fields must agree everywhere, infinities and NaNs included.  Single reads, the batched
// issue/collect path and the Direct reader all read the same bits of the same one-tensor GGUF, whose
// first rows carry hand-picked bit patterns (zeros, infinities, two NaN payloads, the smallest
// denormal, the smallest normal, the largest finite values) and whose later rows are random.
#include "basalt/kernels/ngram.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

namespace k = basalt::kernels;

constexpr uint32_t kTypeBF16 = 30;             // the ggml type id this reader names "BF16"
constexpr uint32_t kRows = 4096;               // 1.3 MB of rows: a small table with a real header

template <typename T>
void put(std::vector<uint8_t>& b, T v) {
    const auto* p = (const uint8_t*) &v;
    b.insert(b.end(), p, p + sizeof(T));
}

uint16_t row_bits(uint32_t row, int i) {
    static const uint16_t edge[] = {0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC1, 0xFFC1,
                                    0x0001, 0x0041, 0x0080, 0x8080, 0x7F7F, 0xFF7F, 0x477E, 0xC77E};
    if (row < 16) return edge[(i + (int) row) % 16];
    std::mt19937 rng(row * 2654435761u + 11u);
    return (uint16_t) rng();
}

// An independent decode: the fields read out by hand, never a shift of the whole half word.
float bf16_value(uint16_t u) {
    const int e = (u >> 7) & 0xFF, m = u & 0x7F;
    const float sign = (u & 0x8000) ? -1.0f : 1.0f;
    if (e == 255)
        return m ? std::numeric_limits<float>::quiet_NaN() : sign * std::numeric_limits<float>::infinity();
    return e == 0 ? sign * std::ldexp((float) m, -133) : sign * std::ldexp((float) (m + 128), e - 134);
}

bool same(float got, float want) { return std::isnan(want) ? std::isnan(got) : got == want; }

bool write_gguf(const std::string& path) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);   // "GGUF"
    put<uint32_t>(b, 3);
    put<uint64_t>(b, 1);             // tensors
    put<uint64_t>(b, 0);             // metadata
    const std::string name = "per_layer_token_embd.weight";
    put<uint64_t>(b, name.size());
    b.insert(b.end(), name.begin(), name.end());
    put<uint32_t>(b, 2);
    put<uint64_t>(b, k::PLE_HEAD_DIM);
    put<uint64_t>(b, kRows);
    put<uint32_t>(b, kTypeBF16);     // type
    put<uint64_t>(b, 0);             // offset within the data section
    b.resize((b.size() + 31) / 32 * 32);
    for (uint32_t r = 0; r < kRows; ++r)
        for (int i = 0; i < k::PLE_HEAD_DIM; ++i) put<uint16_t>(b, row_bits(r, i));
    std::ofstream out(path, std::ios::binary);
    out.write((const char*) b.data(), (std::streamsize) b.size());
    return out.good();
}

void check_rows(const k::PleTable& table, const char* what, const std::vector<uint32_t>& rows,
                long long& checked, int& bad) {
    for (uint32_t r : rows) {
        float got[k::PLE_HEAD_DIM];
        table.read_row(r, got);
        for (int i = 0; i < k::PLE_HEAD_DIM; ++i) {
            ++checked;
            if (!same(got[i], bf16_value(row_bits(r, i)))) {
                if (bad < 8)
                    std::fprintf(stderr, "%s row %u element %d: got %g, want %g\n", what, r, i,
                                 (double) got[i], (double) bf16_value(row_bits(r, i)));
                ++bad;
            }
        }
    }
}

}  // namespace

int main() {
    const std::string path = (std::filesystem::temp_directory_path() / "ple_bf16_parity.gguf").string();
    if (!write_gguf(path)) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        return 2;
    }
    long long checked = 0;
    int bad = 0;

    std::vector<uint32_t> probes;
    for (uint32_t r = 0; r < 16; ++r) probes.push_back(r);   // every hand-built pattern row
    probes.push_back(kRows - 1);
    std::mt19937 rng(7);
    for (int i = 0; i < 64; ++i) probes.push_back(rng() % kRows);

    std::string err;
    k::PleTable table;
    k::PleIoOptions mo;
    mo.mode = k::PleIo::Mmap;
    if (!table.open(path, err, mo)) {
        std::fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    if (std::strcmp(table.format(), "BF16") != 0 || table.rows() != kRows) {
        std::fprintf(stderr, "open gave %s x %llu, expected BF16 x %u\n", table.format(),
                     (unsigned long long) table.rows(), kRows);
        return 1;
    }
    check_rows(table, "mapped", probes, checked, bad);

    // The batched path: one token's sixteen rows, edge and random alike, through issue/collect.
    uint32_t batch[k::PLE_N_HEADS];
    for (int h = 0; h < k::PLE_N_HEADS; ++h) batch[h] = probes[(h * 5) % probes.size()];
    std::vector<float> out(k::PLE_N_HEADS * k::PLE_HEAD_DIM);
    if (!table.issue(batch) || !table.collect(out.data(), err)) {
        std::fprintf(stderr, "issue/collect: %s\n", err.c_str());
        return 1;
    }
    for (int h = 0; h < k::PLE_N_HEADS; ++h)
        for (int i = 0; i < k::PLE_HEAD_DIM; ++i) {
            ++checked;
            if (!same(out[h * k::PLE_HEAD_DIM + i], bf16_value(row_bits(batch[h], i)))) {
                if (bad < 8) std::fprintf(stderr, "batch row %u element %d differs\n", batch[h], i);
                ++bad;
            }
        }
    table.close();

    // The Direct reader over the same file: the generic row_bytes path carrying BF16's 320-byte rows.
    k::PleTable direct;
    k::PleIoOptions dopt;
    dopt.mode = k::PleIo::Direct;
    dopt.cache_rows = 1u << 16;
    if (!direct.open(path, err, dopt)) {
        std::fprintf(stderr, "direct open: %s\n", err.c_str());
        return 1;
    }
    check_rows(direct, "direct", probes, checked, bad);
    direct.close();

    std::filesystem::remove(path);
    std::printf("BF16 PLE parity: %u-row synthetic table, %lld elements checked, %s\n", kRows, checked,
                bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
