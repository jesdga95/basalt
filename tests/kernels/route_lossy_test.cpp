// tests/kernels/route_lossy_test.cpp - the tail skip's rule (basalt/kernels/route_lossy.hpp) on the CPU: the weight rank,
// the residency check over several tables and the drop of an expert no GPU holds, against hand-made windows and a plain
// reference over random windows at each MoE model's geometry.
#include "basalt/kernels/route_lossy.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace basalt::kernels;

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++fails;
    }
}

// The window rewrite as a model's kernel does it, one entry after another.
int skip(std::vector<int32_t>& ids, std::vector<float>& w, const ResidentTabs& held, int k, int R) {
    const int n = (int) ids.size();
    std::vector<uint8_t> tail((size_t) n);
    std::vector<bool> miss((size_t) n);
    for (int i = 0; i < n; ++i) {
        tail[(size_t) i] = route_weight_rank(w.data(), i, k) >= R;
        miss[(size_t) i] = ids[(size_t) i] >= 0 && !held.held(ids[(size_t) i]);
    }
    std::vector<bool> drop((size_t) n);
    for (int i = 0; i < n; ++i) drop[(size_t) i] = route_tail_drop(ids.data(), tail.data(), n, i, miss[(size_t) i]);
    int dropped = 0;
    for (int i = 0; i < n; ++i)
        if (drop[(size_t) i]) {
            ids[(size_t) i] = -1;
            w[(size_t) i] = 0.0f;
            ++dropped;
        }
    return dropped;
}
}  // namespace

int main() {
    // ranks: heavier first, a tie to the lower slot, per token
    {
        const float w[8] = {0.1f, 0.4f, 0.4f, 0.1f, 0.9f, 0.05f, 0.03f, 0.02f};
        check(route_weight_rank(w, 1, 4) == 0 && route_weight_rank(w, 2, 4) == 1, "a tie ranks the lower slot first");
        check(route_weight_rank(w, 0, 4) == 2 && route_weight_rank(w, 3, 4) == 3, "the light ones last");
        check(route_weight_rank(w, 4, 4) == 0 && route_weight_rank(w, 7, 4) == 3, "the second token ranks on its own");
    }
    // residency: any table holds it
    {
        std::vector<int32_t> a(8, -1), b(8, -1);
        a[2] = 5;
        b[6] = 0;
        ResidentTabs h;
        h.add(a.data());
        h.add(nullptr);
        h.add(b.data());
        check(h.n == 2, "a null table is not added");
        check(h.held(2) && h.held(6) && !h.held(3), "held when any table has a slot");
        ResidentTabs none;
        check(!none.held(2), "no table holds nothing");
    }
    // the drop: two tokens, k = 3, R = 2 (rank 2 is the tail)
    {
        std::vector<int32_t> tab(16, -1);
        tab[1] = 0;   // only expert 1 is held
        ResidentTabs h;
        h.add(tab.data());
        // token 0: 7 (tail, not held) ; token 1: 7 at rank 0 keeps it; 9 is tail in both tokens and not held; 1 is held
        std::vector<int32_t> ids = {3, 4, 7, 7, 1, 9};
        std::vector<float> w = {0.5f, 0.3f, 0.2f, 0.6f, 0.3f, 0.1f};
        std::vector<int32_t> ids2 = ids;
        std::vector<float> w2 = w;
        const int d = skip(ids2, w2, h, 3, 2);
        check(d == 1 && ids2[5] == -1 && w2[5] == 0.0f, "the tail expert no GPU holds is dropped");
        check(ids2[2] == 7 && w2[2] == 0.2f, "an expert another token ranks above the tail is kept");
        check(ids2[0] == 3 && ids2[1] == 4 && w2[0] == 0.5f && w2[3] == 0.6f, "the others and their weights are kept");
        tab[9] = 3;
        ids2 = ids;
        w2 = w;
        check(skip(ids2, w2, h, 3, 2) == 0, "a held tail expert is kept");
    }
    // random windows against the plain rule
    std::mt19937 rng(20261119);
    const int geo[3][2] = {{512, 10}, {384, 6}, {512, 16}};
    long dropped = 0, bad = 0;
    for (int it = 0; it < 3000; ++it) {
        const int ne = geo[it % 3][0], k = geo[it % 3][1], nt = 1 + (int) (rng() % 8), n = nt * k;
        const int R = 1 + (int) (rng() % (k - 1));
        std::vector<int32_t> tab((size_t) ne);
        for (auto& t : tab) t = rng() % 3 == 0 ? 0 : -1;
        ResidentTabs h;
        h.add(tab.data());
        std::vector<int32_t> ids((size_t) n);
        std::vector<float> w((size_t) n);
        const int pool = k + 2 + (int) (rng() % 40);
        for (int t = 0; t < nt; ++t) {
            std::vector<int32_t> q((size_t) pool);
            for (int j = 0; j < pool; ++j) q[(size_t) j] = (int32_t) (j * 13 % ne);
            std::shuffle(q.begin(), q.end(), rng);
            for (int j = 0; j < k; ++j) {
                ids[(size_t) (t * k + j)] = q[(size_t) j];
                w[(size_t) (t * k + j)] = (float) (1 + rng() % 50) / 64.0f;
            }
        }
        std::vector<int32_t> want = ids;
        std::vector<float> wantw = w;
        for (int i = 0; i < n; ++i) {
            if (tab[(size_t) ids[(size_t) i]] >= 0) continue;
            bool keep = false;
            for (int j = 0; j < n && !keep; ++j) {
                if (ids[(size_t) j] != ids[(size_t) i]) continue;
                int r = 0;
                const int b = j - j % k;
                for (int m = 0; m < k; ++m) r += w[(size_t) (b + m)] > w[(size_t) j] || (w[(size_t) (b + m)] == w[(size_t) j] && b + m < j);
                keep = r < R;
            }
            if (!keep) {
                want[(size_t) i] = -1;
                wantw[(size_t) i] = 0.0f;
            }
        }
        dropped += skip(ids, w, h, k, R);
        bad += ids != want || w != wantw;
    }
    check(bad == 0, "random windows match the plain rule");
    check(dropped > 0, "random windows drop entries");
    std::printf("route_lossy_test: %ld entries dropped over random windows, %d failures\n", dropped, fails);
    return fails == 0 ? 0 : 1;
}
