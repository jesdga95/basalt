// tests/models/glm5next/cache_buf_test.cpp - the three cache buffers: allocated from the tiny pack's CacheDims, laid out
// sequence-major then layer with a token stride inside, and zero-initialised. The total bytes must equal the independent
// cache_bytes() (allocator and sizing are separate code paths); the (sequence, layer) block strides and the fresh zero
// state are checked directly. GPU- and fixture-gated.
#include "models/glm5next/cache_buf.hpp"
#include "models/glm5next/pack.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace basalt::models::glm5next;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

bool file_exists(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (f) {
        std::fclose(f);
        return true;
    }
    return false;
}

std::string fixture() {
    if (const char* e = std::getenv("GLM5NEXT_TINY_PACK"); e && *e && file_exists(e)) return e;
#ifdef GLM_SRC_DIR
    const std::string out = "/tmp/glm5next_tiny_test.basalt";
    const std::string cmd = "python3 '" GLM_SRC_DIR "/tools/convert/glm5next_tiny_pack.py' '" + out + "' >/dev/null 2>&1";
    if (std::system(cmd.c_str()) == 0 && file_exists(out)) return out;
#endif
    return "";
}

bool have_sm120() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) continue;
        if (prop.major == 12 && cudaSetDevice(d) == cudaSuccess) return true;
    }
    return false;
}

}  // namespace

int main() {
    const std::string path = fixture();
    if (path.empty()) {
        std::printf("SKIP: no tiny pack\n");
        return 0;
    }
    if (!have_sm120()) {
        std::printf("SKIP: no sm_120 device\n");
        return 77;
    }

    Pack p;
    std::string err;
    if (!open_pack(path, p, err)) {
        std::printf("open_pack failed: %s\n", err.c_str());
        return 1;
    }
    const CacheDims dims = cache_dims(p.geo);

    const uint64_t context = 64;
    const int n_seq = 2;
    GlmCaches c;
    check(c.alloc(dims, context, n_seq, err), "caches allocate");
    if (!err.empty()) std::printf("alloc: %s\n", err.c_str());
    check(!c.empty(), "caches not empty");
    check(c.bytes() == cache_bytes(dims, context, n_seq), "allocated bytes == cache_bytes");

    const size_t state_stride = (size_t) dims.n_head * dims.head_dim * dims.head_dim;
    check(c.kda_state(0, 0) != nullptr, "kda state allocated");
    check(c.kda_conv(0, 0) != nullptr, "kda conv allocated");
    check(c.mla_latent(0, 0) != nullptr, "mla latent allocated");
    check(c.idx_pooled(0, 0) != nullptr, "indexer pooled keys allocated");
    check(c.idx_pending(0, 0) != nullptr, "indexer pending rows allocated");

    // layer stride within a sequence, and sequence stride across sequences
    check(c.kda_state(0, 1) - c.kda_state(0, 0) == (ptrdiff_t) state_stride, "KDA layer stride");
    check(c.kda_state(1, 0) - c.kda_state(0, 0) == (ptrdiff_t)((size_t) dims.n_kda * state_stride), "KDA sequence stride");
    check(c.mla_latent(1, 0) - c.mla_latent(0, 0) == (ptrdiff_t)((size_t) dims.n_mla * context * dims.kv_lora),
          "MLA sequence stride");
    const size_t pools = dims.n_pools(context) ? (size_t) dims.n_pools(context) : 1;
    check(c.idx_pooled(1, 0) - c.idx_pooled(0, 0) ==
              (ptrdiff_t)((size_t) dims.n_idx * pools * dims.idx_dim),
          "indexer pooled sequence stride");
    check(c.idx_pending(1, 0) - c.idx_pending(0, 0) ==
              (ptrdiff_t)((size_t) dims.n_idx * kIndexerPendingRows * (dims.kpool - 1) * dims.idx_dim),
          "indexer pending sequence stride");

    // a fresh sequence: the recurrent state reads back zero
    std::vector<float> head(4, -1.f);
    check(cudaMemcpy(head.data(), c.kda_state(0, 0), 4 * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess,
          "read state head");
    check(head[0] == 0.f && head[1] == 0.f && head[2] == 0.f && head[3] == 0.f, "fresh state zeroed");

    // free() leaves it empty
    c.free();
    check(c.empty() && c.bytes() == 0, "free empties the caches");

    if (g_failures != 0) {
        std::printf("CACHE BUF TEST: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("CACHE BUF TEST: ok\n");
    return 0;
}
