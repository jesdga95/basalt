// tests/models/glm5next/kernels/cell_expand_parity.cpp - selected pools -> token-cell ids against a CPU oracle that
// mirrors the deterministic slot order (pool-slot-major, member-minor, then tail). Dead pool slots (past the token's
// visible count), cache-padded members and missing tail cells all become -1. Model kpool=4 plus a kpool=3 canary, tail
// both on and off.
#include "models/glm5next/kernels/cell_expand.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;
void check(bool ok, const char* what) { if (!ok) { std::printf("FAIL: %s\n", what); ++g_failures; } }

struct Lcg {
    unsigned long long s = 7095528748143395ull;
    unsigned int next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return (unsigned int) (s >> 33); }
};

void* d_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { std::printf("FAIL: cudaMalloc\n"); std::exit(1); }
    return p;
}

void run_case(const char* name, int n_tokens, int n_pool, int kpool, int n_top_pool, bool use_tail, Lcg& rng) {
    const int n_cells = n_pool * kpool;
    const int sentinel = n_cells;  // a value pool_idxs/tail_idxs use for a padded/missing entry
    const int n_sel = basalt::models::glm5next::cell_expand_n_sel(kpool, n_top_pool, use_tail);

    std::vector<int32_t> top_pools((size_t) n_tokens * n_top_pool, -1);
    std::vector<int32_t> nvisible(n_tokens);
    std::vector<int32_t> pool_idxs((size_t) kpool * n_pool);
    std::vector<int32_t> tail_idxs((size_t) (kpool - 1) * n_tokens);
    for (auto& v : pool_idxs) v = (sentinel == (int)(rng.next() % 11)) ? sentinel : (int) (rng.next() % n_cells);
    for (auto& v : tail_idxs) v = (n_cells == 0 || sentinel == (int) (rng.next() % 7)) ? sentinel : (int) (rng.next() % n_cells);
    for (int t = 0; t < n_tokens; ++t) {
        const int nv = 1 + (int) (rng.next() % (n_top_pool + 1));  // visible pools, may exceed top slots
        nvisible[t] = nv;
        for (int p = 0; p < n_top_pool; ++p)
            top_pools[(size_t) t * n_top_pool + p] = p < nv ? (int) (rng.next() % n_pool) : -1;
    }
    std::vector<int32_t> gpu((size_t) n_tokens * n_sel), cpu((size_t) n_tokens * n_sel, -2);

    int32_t* d_tp = (int32_t*) d_alloc(top_pools.size() * 4);
    int32_t* d_pi = (int32_t*) d_alloc(pool_idxs.size() * 4);
    int32_t* d_ti = (int32_t*) d_alloc(tail_idxs.size() * 4);
    int32_t* d_out = (int32_t*) d_alloc(gpu.size() * 4);
    cudaMemcpy(d_tp, top_pools.data(), top_pools.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pi, pool_idxs.data(), pool_idxs.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ti, tail_idxs.data(), tail_idxs.size() * 4, cudaMemcpyHostToDevice);
    // A slot the pass never writes is a bug, not a zero: poison the output so an early return cannot read as a match.
    cudaMemset(d_out, 0x5a, gpu.size() * 4);

    basalt::models::glm5next::cell_expand(d_tp, d_pi, d_ti, sentinel, d_out, n_tokens, n_pool, kpool, n_top_pool,
                                          use_tail, 0);
    check(cudaGetLastError() == cudaSuccess, "cell_expand launches");
    check(cudaDeviceSynchronize() == cudaSuccess, "cell_expand runs");
    cudaMemcpy(gpu.data(), d_out, gpu.size() * 4, cudaMemcpyDeviceToHost);

    for (int t = 0; t < n_tokens; ++t) {
        for (int j = 0; j < n_sel; ++j) {
            int32_t cell = -1;
            if (j < kpool * n_top_pool) {
                const int ps = j / kpool, s = j % kpool;
                const int32_t pool = top_pools[(size_t) t * n_top_pool + ps];
                if (pool >= 0 && pool < n_pool) {
                    const int32_t m = pool_idxs[(size_t) s * n_pool + pool];
                    cell = (m != sentinel) ? m : -1;
                }
            } else if (use_tail) {
                const int ts = j - kpool * n_top_pool;
                const int32_t m = tail_idxs[(size_t) ts * n_tokens + t];
                cell = (m != sentinel) ? m : -1;
                if (cell >= 0) {
                    const int32_t pool = cell / kpool;
                    for (int ps = 0; ps < n_top_pool; ++ps)
                        if (top_pools[(size_t) t * n_top_pool + ps] == pool) {
                            cell = -1;
                            break;
                        }
                }
            }
            cpu[(size_t) t * n_sel + j] = cell;
        }
    }
    int mismatch = 0;
    for (size_t i = 0; i < gpu.size(); ++i) mismatch += (gpu[i] != cpu[i]);
    std::printf("%-18s ntok=%d npool=%d kpool=%d top=%d tail=%d: mismatched=%d\n", name, n_tokens, n_pool, kpool,
                n_top_pool, (int) use_tail, mismatch);
    check(mismatch == 0, "cell ids match the oracle (expansion, sentinel->-1, tail)");

    // The same token inside a wider window, with one row's data duplicated at the end and the strides the wider call
    // reads them in: no row may move, and the duplicate must come back as the row it copies.
    {
        const int wide = n_tokens + 1;
        std::vector<int32_t> tp_w((size_t) wide * n_top_pool), ti_w((size_t) (kpool - 1) * wide), out_w((size_t) wide * n_sel, -2);
        for (int t = 0; t < wide; ++t) {
            const int src = t < n_tokens ? t : 0;
            for (int ps = 0; ps < n_top_pool; ++ps)
                tp_w[(size_t) t * n_top_pool + ps] = top_pools[(size_t) src * n_top_pool + ps];
            for (int ts = 0; ts < kpool - 1; ++ts)
                ti_w[(size_t) ts * wide + t] = tail_idxs[(size_t) ts * n_tokens + src];
        }
        int32_t* d_tp_w = (int32_t*) d_alloc(tp_w.size() * 4);
        int32_t* d_ti_w = (int32_t*) d_alloc(ti_w.size() * 4);
        int32_t* d_out_w = (int32_t*) d_alloc(out_w.size() * 4);
        cudaMemcpy(d_tp_w, tp_w.data(), tp_w.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ti_w, ti_w.data(), ti_w.size() * 4, cudaMemcpyHostToDevice);
        cudaMemset(d_out_w, 0x5a, out_w.size() * 4);
        basalt::models::glm5next::cell_expand(d_tp_w, d_pi, d_ti_w, sentinel, d_out_w, wide, n_pool, kpool,
                                              n_top_pool, use_tail, 0);
        check(cudaGetLastError() == cudaSuccess, "cell_expand launches in the wider window");
        check(cudaDeviceSynchronize() == cudaSuccess, "cell_expand runs in the wider window");
        cudaMemcpy(out_w.data(), d_out_w, out_w.size() * 4, cudaMemcpyDeviceToHost);
        bool same_rows = true;
        for (int t = 0; t < n_tokens && same_rows; ++t)
            for (int j = 0; j < n_sel; ++j)
                same_rows = same_rows && out_w[(size_t) t * n_sel + j] == gpu[(size_t) t * n_sel + j];
        for (int j = 0; j < n_sel && same_rows; ++j)
            same_rows = out_w[(size_t) n_tokens * n_sel + j] == gpu[j];
        check(same_rows, "a token's cells do not depend on the width of the window it ran in");
        cudaFree(d_tp_w); cudaFree(d_ti_w); cudaFree(d_out_w);
    }

    cudaFree(d_tp); cudaFree(d_pi); cudaFree(d_ti); cudaFree(d_out);
}

}  // namespace

/// The shape the model actually runs: the tail's cells always live in the token's own pool, so whenever that pool is
/// among the selected ones the pool expansion has already emitted every tail cell. Those slots have to come back masked
/// (the window's n_sel is fixed), or the attention counts the freshest rows of the context twice.
void run_overlap_case(const char* name, const std::vector<int32_t>& top, const std::vector<int32_t>& tail,
                      const std::vector<int32_t>& want) {
    const int kpool = 4, n_pool = 2, n_tokens = 1, sentinel = n_pool * kpool;
    const int n_top_pool = (int) top.size();
    const int n_sel = basalt::models::glm5next::cell_expand_n_sel(kpool, n_top_pool, true);
    std::vector<int32_t> pool_idxs((size_t) kpool * n_pool), ids((size_t) n_sel, -7);
    for (int p = 0; p < n_pool; ++p)
        for (int s = 0; s < kpool; ++s) pool_idxs[(size_t) s * n_pool + p] = (int32_t) (p * kpool + s);
    int32_t* d_tp = (int32_t*) d_alloc(top.size() * 4);
    int32_t* d_pi = (int32_t*) d_alloc(pool_idxs.size() * 4);
    int32_t* d_ti = (int32_t*) d_alloc(tail.size() * 4);
    int32_t* d_out = (int32_t*) d_alloc(ids.size() * 4);
    cudaMemcpy(d_tp, top.data(), top.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pi, pool_idxs.data(), pool_idxs.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ti, tail.data(), tail.size() * 4, cudaMemcpyHostToDevice);
    basalt::models::glm5next::cell_expand(d_tp, d_pi, d_ti, sentinel, d_out, n_tokens, n_pool, kpool, n_top_pool, true, 0);
    check(cudaGetLastError() == cudaSuccess, "cell_expand launches (overlap)");
    check(cudaDeviceSynchronize() == cudaSuccess, "cell_expand runs (overlap)");
    cudaMemcpy(ids.data(), d_out, ids.size() * 4, cudaMemcpyDeviceToHost);
    bool same = ids == want;
    std::printf("%-24s top=%2d n_sel=%2d  ids:", name, n_top_pool, n_sel);
    for (int32_t v : ids) std::printf(" %d", v);
    std::printf("  %s\n", same ? "ok" : "MISMATCH");
    check(same, name);
    cudaFree(d_tp); cudaFree(d_pi); cudaFree(d_ti); cudaFree(d_out);
}

int main() {
    int n_gpus = 0;
    if (cudaGetDeviceCount(&n_gpus) != cudaSuccess) { std::printf("SKIP: no CUDA device\n"); return 77; }
    bool have = false;
    for (int d = 0; d < n_gpus; ++d) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) continue;
        if (prop.major == 12 && cudaSetDevice(d) == cudaSuccess) { have = true; break; }
    }
    if (!have) { std::printf("SKIP: no sm_120 device\n"); return 77; }

    Lcg rng;
    run_case("k4-tail", 3, 40, 4, 12, true, rng);
    run_case("k4-notail", 2, 40, 4, 16, false, rng);
    run_case("k3-tail-canary", 4, 30, 3, 9, true, rng);
    run_case("k3-notail-canary", 4, 30, 3, 7, false, rng);
    // A sequence that has not filled its first pool: no pool slots at all, and the tail is the whole attended set. The
    // slots have to be written here or mla_sparse attends whatever the previous window left in them.
    run_case("k2-nopool-tail", 3, 0, 2, 0, true, rng);

    run_overlap_case("own-pool-selected", {0, 1}, {7, 6, 5}, {0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1});
    run_overlap_case("own-pool-unselected", {0}, {7, 6, 5}, {0, 1, 2, 3, 7, 6, 5});
    run_overlap_case("tail-in-selected-pool", {1}, {7, 6, 5}, {4, 5, 6, 7, -1, -1, -1});

    if (g_failures != 0) { std::printf("CELL_EXPAND PARITY: %d FAILURES\n", g_failures); return 1; }
    std::printf("CELL_EXPAND PARITY: ok\n");
    return 0;
}
