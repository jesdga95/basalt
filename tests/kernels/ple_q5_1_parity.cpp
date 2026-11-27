// A small synthetic Q5_1 PLE table through PleTable: every element against an independent decode.
// Q5_1 is asymmetric - value = d * code + m with code 0..31 (no -16 offset) and m an fp16 min - so the loader and
// an oracle built from the logical codes on the d/m grid must agree everywhere.  Single reads, the batched
// issue/collect path and the Direct reader all read the same bits of the same one-tensor GGUF, whose first rows
// carry hand-picked patterns (all-zero codes, max codes 31, m = 0, negative d and m, d = 0, max finite d, the
// smallest normal d) and whose later rows are random.
#include "basalt/models/qwen4exp/kernels/ngram.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

namespace k = basalt::kernels;

constexpr uint32_t kTypeQ5_1 = 7;              // the ggml type id this reader names "Q5_1"
constexpr uint32_t kRows = 4096;               // 480 KB of rows: a small table with a real header
constexpr int kBlocks = k::PLE_HEAD_DIM / 32;  // 5 Q5_1 blocks of 32 values

template <typename T>
void put(std::vector<uint8_t>& b, T v) {
    const auto* p = (const uint8_t*) &v;
    b.insert(b.end(), p, p + sizeof(T));
}

// An independent read of an fp16 scale: the fields decoded by hand, never a shift of the whole half word.
float half_value(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const float sign = (h & 0x8000) ? -1.0f : 1.0f;
    return e == 0 ? sign * std::ldexp((float) m, -24) : sign * std::ldexp((float) (m + 1024), e - 25);
}

// The logical row: the per-block scale and min and the 160 codes.  The oracle value of element i is
// code[i] * d + m on these, so a byte-layout mistake in the writer or the reader shows as a mismatch.
struct RowSpec {
    uint16_t d[kBlocks];
    uint16_t m[kBlocks];
    int code[k::PLE_HEAD_DIM];
};

RowSpec row_spec(uint32_t row) {
    RowSpec s{};
    const uint16_t one = 0x3C00, neg1 = 0xBC00, half = 0x3800;
    auto fill_dm = [&](uint16_t d, uint16_t m) {
        for (int b = 0; b < kBlocks; ++b) {
            s.d[b] = d;
            s.m[b] = m;
        }
    };
    switch (row) {
    case 0: fill_dm(one, 0x0000); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = 0; break;          // all-zero codes
    case 1: fill_dm(one, 0x0000); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = 31; break;         // max codes
    case 2: fill_dm(one, 0x0000); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = i & 31; break;      // every code once
    case 3: fill_dm(one, one); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = 0; break;             // m = +1
    case 4: fill_dm(one, neg1); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = 31; break;           // negative m
    case 5: fill_dm(neg1, 0x0000); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = 0; break;         // negative d
    case 6: fill_dm(neg1, neg1); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = (i & 1) ? 31 : 0; break;  // d and m negative
    case 7: fill_dm(0x0400, 0x0000); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = 31; break;      // smallest normal d
    case 8: fill_dm(0x2400, 0x2400); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = (i * 7) & 31; break;  // small d and m
    case 9: fill_dm(one, 0x8000); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = i % 32; break;     // m = -0
    case 10: fill_dm(0x0000, one); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = 31; break;        // d = 0: the value is m
    case 11: fill_dm(0x0000, 0x0000); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = i & 31; break; // d = m = 0
    case 12: fill_dm(one, 0x0000); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = 16 + (i & 15); break;  // high bit only
    case 13: fill_dm(one, 0x0000); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = (i % 32) < 16 ? 31 : 0; break;  // split-half decode
    case 14: fill_dm(0x7BFF, 0xC800); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = 31; break;    // max finite d, m = -8
    case 15: fill_dm(0x0400, half); for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = (i & 1) ? 0 : 31; break;
    default: {
        std::mt19937 rng(row * 2654435761u + 11u);
        for (int b = 0; b < kBlocks; ++b) {
            s.d[b] = (uint16_t) (0x2000 + rng() % 0x2000);   // a finite normal scale
            s.m[b] = (uint16_t) (0x2000 + rng() % 0x2000);   // a finite normal min
            if (rng() & 1) s.d[b] |= 0x8000;
            if (rng() & 3) s.m[b] |= 0x8000;
        }
        for (int i = 0; i < k::PLE_HEAD_DIM; ++i) s.code[i] = (int) (rng() & 31);
        break;
    }
    }
    return s;
}

// The Q5_1 block as stored: {fp16 d, fp16 m, uint8 qh[4], uint8 qs[16]}.  qh is the high bits of elements 0..31
// little-endian, qs[j] carries element j in its low nibble and element j + 16 in its high nibble.
void put_row(std::vector<uint8_t>& b, const RowSpec& s) {
    for (int blk = 0; blk < kBlocks; ++blk) {
        put<uint16_t>(b, s.d[blk]);
        put<uint16_t>(b, s.m[blk]);
        uint32_t h = 0;
        for (int i = 0; i < 32; ++i)
            if (s.code[blk * 32 + i] & 16) h |= 1u << i;
        for (int k = 0; k < 4; ++k) put<uint8_t>(b, (uint8_t) (h >> (8 * k)));
        for (int j = 0; j < 16; ++j) {
            const uint8_t lo = (uint8_t) (s.code[blk * 32 + j] & 15);
            const uint8_t hi = (uint8_t) (s.code[blk * 32 + j + 16] & 15);
            put<uint8_t>(b, (uint8_t) (lo | (hi << 4)));
        }
    }
}

float want_value(const RowSpec& s, int i) {
    return (float) s.code[i] * half_value(s.d[i / 32]) + half_value(s.m[i / 32]);
}

bool write_gguf(const std::string& path, const std::vector<RowSpec>& specs) {
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
    put<uint32_t>(b, kTypeQ5_1);     // type
    put<uint64_t>(b, 0);             // offset within the data section
    b.resize((b.size() + 31) / 32 * 32);
    for (uint32_t r = 0; r < kRows; ++r) put_row(b, specs[r]);
    std::ofstream out(path, std::ios::binary);
    out.write((const char*) b.data(), (std::streamsize) b.size());
    return out.good();
}

// A relative tolerance of 1e-6: the decode here runs in another translation unit than the table's, so a fused
// multiply-add may differ in the last bit; a wrong code or min is orders of magnitude larger.
bool same(float got, float want) {
    if (std::isnan(want) || std::isnan(got)) return std::isnan(want) && std::isnan(got);
    return std::fabs((double) got - (double) want) <= 1e-6 * (std::fabs((double) want) + 1e-30);
}

void check_rows(const k::PleTable& table, const char* what, const std::vector<uint32_t>& rows,
                const std::vector<RowSpec>& specs, long long& checked, int& bad) {
    for (uint32_t r : rows) {
        float got[k::PLE_HEAD_DIM];
        table.read_row(r, got);
        for (int i = 0; i < k::PLE_HEAD_DIM; ++i) {
            ++checked;
            const float want = want_value(specs[r], i);
            if (!same(got[i], want)) {
                if (bad < 8)
                    std::fprintf(stderr, "%s row %u element %d: got %g, want %g\n", what, r, i,
                                 (double) got[i], (double) want);
                ++bad;
            }
        }
    }
}

}  // namespace

int main() {
    const std::string path = (std::filesystem::temp_directory_path() / "ple_q5_1_parity.gguf").string();
    std::vector<RowSpec> specs(kRows);
    for (uint32_t r = 0; r < kRows; ++r) specs[r] = row_spec(r);
    if (!write_gguf(path, specs)) {
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
    if (std::strcmp(table.format(), "Q5_1") != 0 || table.rows() != kRows) {
        std::fprintf(stderr, "open gave %s x %llu, expected Q5_1 x %u\n", table.format(),
                     (unsigned long long) table.rows(), kRows);
        return 1;
    }
    check_rows(table, "mapped", probes, specs, checked, bad);

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
            if (!same(out[h * k::PLE_HEAD_DIM + i], want_value(specs[batch[h]], i))) {
                if (bad < 8) std::fprintf(stderr, "batch row %u element %d differs\n", batch[h], i);
                ++bad;
            }
        }
    table.close();

    // The Direct reader over the same file: the generic row_bytes path carrying Q5_1's 120-byte rows.
    k::PleTable direct;
    k::PleIoOptions dopt;
    dopt.mode = k::PleIo::Direct;
    dopt.cache_rows = 1u << 16;
    if (!direct.open(path, err, dopt)) {
        std::fprintf(stderr, "direct open: %s\n", err.c_str());
        return 1;
    }
    check_rows(direct, "direct", probes, specs, checked, bad);
    direct.close();

    std::filesystem::remove(path);
    std::printf("Q5_1 PLE parity: %u-row synthetic table, %lld elements checked, %s\n", kRows, checked,
                bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
