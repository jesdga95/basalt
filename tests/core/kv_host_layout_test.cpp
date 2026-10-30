// tests/core/kv_host_layout_test.cpp - the per-request host K/V's arithmetic, host only (CUDA is linked, never
// initialized): what a switchable QSA state carves, the host copy's size and layout, and the mode switch's refusals.
#include "basalt/core/kv_host_pool.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/session.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/kernels/kv_q8.hpp"
#include "basalt/kernels/kv_stream.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

namespace c = basalt::core;
namespace k = basalt::kernels;

static int g_fail = 0;
#define CHECK(x)                                                                        \
    do {                                                                                \
        if (!(x)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++g_fail; } \
    } while (0)

int main() {
    const c::ModelGeometry g{};
    const int64_t ctx = 262144, floor = 32768;
    for (const char* kv : {"int8", "fp16", "q4_0"}) {
        c::qsa_set_kv_int8(std::string(kv) == "int8");
        c::qsa_set_kv_q4(std::string(kv) == "q4_0");
        // off: a state's bytes are what they were, elastic or not
        c::qsa_set_kv_host(0);
        c::qsa_set_kv_elastic(0);
        const uint64_t whole = c::qsa_state_bytes(g, ctx);
        c::qsa_set_kv_elastic(floor);
        const uint64_t elastic = c::qsa_state_bytes(g, ctx), elastic_rest = c::qsa_state_bytes(g, ctx, false);
        // a host slot pool asked without the elastic KV carves nothing
        c::qsa_set_kv_elastic(0);
        c::qsa_set_kv_host(floor);
        CHECK(c::qsa_state_bytes(g, ctx) == whole);
        // on: nothing more in the session (the residency map is made by the switch to streaming)
        c::qsa_set_kv_elastic(floor);
        CHECK(c::qsa_state_bytes(g, ctx) == elastic);
        CHECK(c::qsa_state_bytes(g, ctx, false) == elastic_rest);
        c::qsa_set_kv_host(4 * floor);
        CHECK(c::qsa_state_bytes(g, ctx) == elastic);
        c::qsa_set_kv_host(0);
        c::qsa_set_kv_elastic(0);
    }
    // the host copy of one INT8 layer: 4,224 B per 4-cell page, the four arrays back to back (16-byte aligned)
    c::QsaState st;
    st.kv_int8 = true;
    CHECK(k::kv_block_bytes(k::qsa_real_shapes(), k::kKvInt8) == 4224);
    CHECK(c::qsa_kv_host_layer_bytes(st, g, 1000) == 250ull * 4224 + 1024);
    CHECK(c::qsa_kv_host_layer_bytes(st, g, 1001) == 251ull * 4224 + 1024);
    uint8_t* const base = reinterpret_cast<uint8_t*>(uint64_t{1} << 20);   // never dereferenced
    const k::KvHostPools h = c::qsa_kv_host_carve(st, g, base, 1000);
    const uint64_t rows = 250ull * g.n_head_kv * 4;
    CHECK((uint8_t*) h.k_q == base);
    CHECK((uint8_t*) h.v_q == base + rows * g.head_dim);
    CHECK((uint8_t*) h.k_scale == base + 2 * rows * g.head_dim);
    CHECK((uint8_t*) h.v_scale == base + 2 * rows * g.head_dim + rows * (g.head_dim / 64) * 2);
    CHECK((uint8_t*) h.v_scale + rows * (g.head_dim / 64) * 2 <= base + c::qsa_kv_host_layer_bytes(st, g, 1000));
    CHECK(h.present() && h.k_pool == nullptr && h.k_q4 == nullptr);
    // Q4_0 and FP16: two arrays each
    c::QsaState q4;
    q4.kv_q4 = true;
    const k::KvHostPools hq = c::qsa_kv_host_carve(q4, g, base, 1000);
    CHECK(hq.k_q4 == base && hq.v_q4 == base + rows * k::kv_q4_bytes_per_head((int) g.head_dim) && hq.k_q == nullptr);
    c::QsaState f16;
    const k::KvHostPools hf = c::qsa_kv_host_carve(f16, g, base, 1000);
    CHECK((uint8_t*) hf.k_pool == base && (uint8_t*) hf.v_pool == base + rows * g.head_dim * 2);
    CHECK(c::qsa_kv_host_layer_bytes(f16, g, 1000) == 250ull * k::kv_block_bytes(k::qsa_real_shapes(), k::kKvF16) + 1024);
    // a session's pool: one 256-aligned copy per allocated QSA layer
    c::QsaState layers[12];
    for (c::QsaState& x : layers) x.kv_int8 = true;
    c::SessionState ss;
    ss.qsa_states = layers;
    ss.qsa_ord0 = 0;
    ss.qsa_alloc = 12;
    const uint64_t per = (c::qsa_kv_host_layer_bytes(layers[0], g, 131072) + 255) & ~255ull;
    CHECK(c::KvHostPool::bytes_for(ss, g, 131072) == 12 * per);
    CHECK(c::KvHostPool::bytes_for(ss, g, 131072) > 131072ull * 12 * 1056);   // 1,056 B per cell and layer, plus slack
    // the switch refuses, before any device work, a state that cannot stream and a mode-1 switch without a host copy
    std::string err;
    CHECK(!c::session_kv_can_stream(ss));
    CHECK(!c::session_kv_set_mode(ss, g, 1, nullptr, 0, nullptr, err) && !err.empty());
    err.clear();
    CHECK(!c::qsa_state_set_mode(layers[0], g, 1, k::KvHostPools{}, 0, 0, nullptr, err) && !err.empty());
    CHECK(c::session_kv_mode(ss) == 0);
    layers[3].kv_mode = 1;
    CHECK(c::session_kv_mode(ss) == -1);
    if (g_fail) { std::printf("kv_host_layout_test: %d failures\n", g_fail); return 1; }
    std::printf("kv_host_layout_test: ok\n");
    return 0;
}
