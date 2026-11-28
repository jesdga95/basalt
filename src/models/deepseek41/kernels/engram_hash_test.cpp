// src/models/deepseek41/kernels/engram_hash_test.cpp - CPU unit tests for engram_hash bounds, null-prev,
// zero-prime, out-of-range pad_token_id, stack/heap buffer parity, and chunked vs whole-sequence equivalence.
#include "basalt/models/deepseek41/kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace basalt::models::deepseek41;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}

EngramHashParams make_params(int n_layers, int n_gram, int n_head) {
    EngramHashParams p;
    p.n_gram = n_gram;
    p.n_head = n_head;
    p.pad_token_id = 2;
    p.token_map.resize(256);
    for (int i = 0; i < 256; ++i) p.token_map[(size_t) i] = (i % 13 == 5) ? -1 : (i * 11 + 3) % 197;
    p.token_map[2] = 42;
    const int nc = p.n_cols();
    p.mult.resize((size_t) n_layers * (size_t) n_gram);
    for (size_t i = 0; i < p.mult.size(); ++i) p.mult[i] = 6364136223846793005ull * (i + 1) + 1442695040888963407ull;
    p.primes.resize((size_t) n_layers * (size_t) nc);
    p.offsets.resize((size_t) n_layers * (size_t) nc);
    for (size_t i = 0; i < p.primes.size(); ++i) {
        p.primes[i] = 1009 + 2 * i;
        p.offsets[i] = i * 2000;
    }
    return p;
}

void ref_engram_hash(const EngramHashParams& p, int e, const int32_t* tokens, int n, int64_t pos0, const int32_t* prev,
                     int32_t* out) {
    const int n_gram = p.n_gram, n_prev = n_gram - 1, n_head = p.n_head, n_cols = p.n_cols();
    const int64_t pad = p.pad_token_id >= 0 && (size_t) p.pad_token_id < p.token_map.size()
                            ? p.token_map[(size_t) p.pad_token_id]
                            : 0;
    auto cid = [&](int32_t t) -> int64_t {
        return t < 0 || (size_t) t >= p.token_map.size() ? -1 : p.token_map[(size_t) t];
    };
    std::vector<uint64_t> toks((size_t) n_gram);
    for (int i = 0; i < n; ++i) {
        int64_t c = cid(tokens[i]);
        bool blocked = c < 0;
        toks[0] = (uint64_t) (blocked ? pad : c);
        for (int s = 1; s < n_gram; ++s) {
            if (pos0 + i - s < 0) c = -1;
            else {
                const int j = i - s;
                c = j >= 0 ? cid(tokens[j]) : (prev ? cid(prev[n_prev + j]) : -1);
            }
            blocked = blocked || c < 0;
            toks[(size_t) s] = (uint64_t) (blocked ? pad : c);
        }
        uint64_t rolling = toks[0] * p.mult[(size_t) e * (size_t) n_gram];
        for (int o = 1; o < n_gram; ++o) {
            rolling ^= toks[(size_t) o] * p.mult[(size_t) e * (size_t) n_gram + (size_t) o];
            for (int h = 0; h < n_head; ++h) {
                const int col = (o - 1) * n_head + h;
                const uint64_t pr = p.primes[(size_t) e * (size_t) n_cols + (size_t) col];
                const uint64_t off = p.offsets[(size_t) e * (size_t) n_cols + (size_t) col];
                out[(size_t) i * (size_t) n_cols + (size_t) col] = (int32_t) ((pr ? rolling % pr : 0) + off);
            }
        }
    }
}

}  // namespace

int main() {
    // 1. Standard geometry (stack toks) and large n_gram > 16 (heap fallback): whole vs chunked vs reference.
    for (int n_gram : {4, 18}) {
        const EngramHashParams p = make_params(2, n_gram, 4);
        const int nc = p.n_cols();
        std::vector<int32_t> seq(40);
        for (int i = 0; i < 40; ++i) seq[(size_t) i] = (i == 12) ? -1 : (i == 25 ? 300 : (i * 29 + 7) % 256);

        for (int e = 0; e < 2; ++e) {
            std::vector<int32_t> whole((size_t) 40 * (size_t) nc), ref((size_t) 40 * (size_t) nc),
                chunk((size_t) 40 * (size_t) nc);
            engram_hash(p, e, seq.data(), 40, 0, nullptr, whole.data());
            ref_engram_hash(p, e, seq.data(), 40, 0, nullptr, ref.data());
            check(whole == ref, n_gram <= 16 ? "stack n_gram matches reference" : "heap n_gram matches reference");

            for (int s = 0; s < 40; s += 3) {
                const int n = s + 3 <= 40 ? 3 : 40 - s;
                std::vector<int32_t> prev((size_t) (n_gram - 1), 0);
                for (int j = 0; j < n_gram - 1; ++j)
                    if (s - (n_gram - 1) + j >= 0) prev[(size_t) j] = seq[(size_t) (s - (n_gram - 1) + j)];
                engram_hash(p, e, &seq[(size_t) s], n, s, s ? prev.data() : nullptr, &chunk[(size_t) s * (size_t) nc]);
            }
            check(whole == chunk, n_gram <= 16 ? "stack n_gram chunked == whole" : "heap n_gram chunked == whole");
        }
    }

    // 2. prev == nullptr when pos0 > 0 treats missing history as dead/blocked (no null dereference).
    {
        const EngramHashParams p = make_params(1, 4, 4);
        const int nc = p.n_cols();
        const int32_t toks[4] = {10, 11, 12, 13};
        const int32_t dead_prev[3] = {-1, -1, -1};
        std::vector<int32_t> got((size_t) 4 * (size_t) nc), want((size_t) 4 * (size_t) nc);
        engram_hash(p, 0, toks, 4, 5, nullptr, got.data());
        engram_hash(p, 0, toks, 4, 5, dead_prev, want.data());
        check(got == want, "pos0 > 0 with null prev matches dead prev");
    }

    // 3. Out-of-range pad_token_id falls back to pad=0 without OOB access.
    {
        EngramHashParams p = make_params(1, 4, 2);
        const int32_t toks[2] = {10, 11};
        const int nc = p.n_cols();
        for (int bad_pad : {-1, (int) p.token_map.size(), 9999}) {
            p.pad_token_id = bad_pad;
            std::vector<int32_t> got((size_t) 2 * (size_t) nc), ref((size_t) 2 * (size_t) nc);
            engram_hash(p, 0, toks, 2, 0, nullptr, got.data());
            ref_engram_hash(p, 0, toks, 2, 0, nullptr, ref.data());
            check(got == ref, "out-of-range pad_token_id falls back to 0");
        }
    }

    // 4. Zero prime avoids division by zero and emits offset verbatim.
    {
        EngramHashParams p = make_params(1, 4, 2);
        p.primes[0] = 0;
        p.primes[3] = 0;
        const int32_t toks[2] = {3, 4};
        const int nc = p.n_cols();
        std::vector<int32_t> got((size_t) 2 * (size_t) nc), ref((size_t) 2 * (size_t) nc);
        engram_hash(p, 0, toks, 2, 0, nullptr, got.data());
        ref_engram_hash(p, 0, toks, 2, 0, nullptr, ref.data());
        check(got == ref, "zero prime produces offset without divide-by-zero");
        check(got[0] == (int32_t) p.offsets[0] && got[3] == (int32_t) p.offsets[3], "zero prime equals offset");
    }

    // 5. Invalid / out-of-range parameters return early without touching out.
    {
        const EngramHashParams good = make_params(1, 4, 2);
        const int32_t toks[2] = {1, 2};
        std::vector<int32_t> out(16, -777);
        const std::vector<int32_t> sentinel = out;
        engram_hash(good, 0, toks, 0, 0, nullptr, out.data());
        engram_hash(good, 0, toks, -1, 0, nullptr, out.data());
        engram_hash(good, -1, toks, 2, 0, nullptr, out.data());
        engram_hash(good, 1, toks, 2, 0, nullptr, out.data());   // layer 1 out of bounds (only 1 layer)
        engram_hash(good, 0, nullptr, 2, 0, nullptr, out.data());
        engram_hash(good, 0, toks, 2, 0, nullptr, nullptr);
        EngramHashParams bad = good;
        bad.n_gram = 1;
        engram_hash(bad, 0, toks, 2, 0, nullptr, out.data());
        bad = good;
        bad.n_head = 0;
        engram_hash(bad, 0, toks, 2, 0, nullptr, out.data());
        bad = good;
        bad.primes.pop_back();
        engram_hash(bad, 0, toks, 2, 0, nullptr, out.data());
        check(out == sentinel, "invalid arguments leave output untouched");
    }

    if (g_fail == 0) std::printf("ds41_engram_hash_test: ok\n");
    return g_fail ? 1 : 0;
}
