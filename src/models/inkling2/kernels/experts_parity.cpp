// src/models/inkling2/kernels/experts_parity.cpp - the tensor-core routed experts (experts_gate_up, then experts_down with
// the finish in its blocks) at the real geometry (2304 x 1152, 128 experts, top 8, the softcapped SwiGLU), NVFP4 and MXFP8:
//   - against the plain CPU reference: the dequantized weights and the activation rows as the kernels read them (f16 /
//     bf16), double sums, the down's input rounded as the gate/up writes it; and against the kernels they replace
//     (q8_1 activations, native_expert_grouped, moe_combine and post_conv); max abs and rel error of both;
//   - bits: each conversation's rows alone (a single window, the small tile shape) against the same rows in a batch window
//     of 64 segments (the wide shape) and of 8 (the middle one); a call with experts off the card (the down writes the
//     card's rows, the others come in, then the finish) against the same call with every expert on it;
//   - --bench: time a call (gate/up + down) and bandwidth over the experts' weights the call reads, at 1, 4, 8, 64, 256 and 320 rows.
#include "basalt/core/device_copy.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/expert_rows.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace basalt::models::inkling2;
namespace bk = basalt::kernels;

namespace {

constexpr int E = kWidth, FF = 1152, NE = 128, K = 8;
constexpr int kBlobs = 16;   // distinct blobs behind the 128 experts' table entries (expert e reads blob e % kBlobs)
std::mt19937 rng(41);
int failures = 0;

template <class T> T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T) + 64);
    basalt::core::device_copy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    basalt::core::device_copy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
template <class T> T* zeros(size_t n) {
    T* p = nullptr;
    cudaMalloc(&p, n * sizeof(T) + 64);
    cudaMemset(p, 0, n * sizeof(T) + 64);
    return p;
}
std::vector<float> randn(size_t n, float s, float mean = 0.f) {
    std::normal_distribution<float> d(mean, s);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}
void report(const char* what, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
    double max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }
    const double rel = max_abs / std::max(max_ref, 1e-30);
    const bool ok = rel <= tol && std::isfinite(max_abs);
    std::printf("%-48s max abs %.3e  max rel %.3e  %s\n", what, max_abs, rel, ok ? "ok" : "FAIL");
    failures += !ok;
}
void same_bits(const char* what, const void* a, const void* b, size_t bytes) {
    const bool ok = std::memcmp(a, b, bytes) == 0;
    std::printf("%-48s %s\n", what, ok ? "bitwise" : "FAIL");
    failures += !ok;
}
double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// The experts of one format: kBlobs blobs of random codes with sane scales, the dequantized weights, the table.
struct Experts {
    int type = 40;
    bk::NativeExpertLayout L;
    std::vector<std::vector<float>> wg, wu, wd;   // [blob] dequantized, scales applied
    uint8_t* blobs = nullptr;
    std::vector<unsigned long long> table;        // [NE]
    unsigned long long* dtable = nullptr;
};

void random_rows(int type, uint8_t* dst, int rows, int k, size_t row_bytes) {
    std::uniform_int_distribution<int> byte(0, 255);
    for (int r = 0; r < rows; ++r) {
        uint8_t* row = dst + (size_t) r * row_bytes;
        if (type == 40) {   // blocks of 64: 4 UE4M3 scales (small, finite), 32 code bytes
            for (int b = 0; b < k / 64; ++b) {
                uint8_t* blk = row + (size_t) b * 36;
                for (int j = 0; j < 4; ++j) blk[j] = (uint8_t) (0x18 + byte(rng) % 16);
                for (int j = 4; j < 36; ++j) blk[j] = (uint8_t) byte(rng);
            }
        } else {            // blocks of 128: 4 UE8M0 scales near 2^-8, 128 E4M3 codes (no NaN)
            for (int b = 0; b < k / 128; ++b) {
                uint8_t* blk = row + (size_t) b * 132;
                for (int j = 0; j < 4; ++j) blk[j] = (uint8_t) (117 + byte(rng) % 4);
                for (int j = 4; j < 132; ++j) {
                    uint8_t c = (uint8_t) byte(rng);
                    if ((c & 0x7F) == 0x7F) c ^= 1;
                    blk[j] = c;
                }
            }
        }
    }
}

Experts make_experts(int type) {
    Experts X;
    X.type = type;
    X.L = bk::native_expert_layout(type, type, E, FF);
    X.L.gate_softcap = 4.f;
    X.L.up_softcap = 25.f;
    const bk::NativeExpertLayout& L = X.L;
    const auto* tr = ggml_get_type_traits((ggml_type) type);
    const size_t stride = (L.bytes + 255) / 256 * 256;
    std::vector<uint8_t> h((size_t) kBlobs * stride + 256, 0);
    X.wg.resize(kBlobs);
    X.wu.resize(kBlobs);
    X.wd.resize(kBlobs);
    for (int b = 0; b < kBlobs; ++b) {
        uint8_t* blob = h.data() + (size_t) b * stride;
        random_rows(type, blob, FF, E, L.gu_row);
        random_rows(type, blob + L.up_off, FF, E, L.up_row);
        random_rows(type, blob + L.down_off, E, FF, L.d_row);
        float sc[4] = {1.f, 1.f, 1.f, 0.f};
        if (type == 40) {
            sc[0] = 0.5f + 0.1f * (float) b;
            sc[1] = 0.75f;
            sc[2] = 1.25f;
            std::memcpy(blob + L.bytes - 16, sc, 16);
        }
        const auto deq = [&](std::vector<float>& w, const uint8_t* src, int rows, int k, size_t rb, float s) {
            w.resize((size_t) rows * k);
            for (int r = 0; r < rows; ++r) tr->to_float(src + (size_t) r * rb, &w[(size_t) r * k], k);
            for (float& v : w) v *= s;
        };
        deq(X.wg[b], blob, FF, E, L.gu_row, sc[0]);
        deq(X.wu[b], blob + L.up_off, FF, E, L.up_row, sc[1]);
        deq(X.wd[b], blob + L.down_off, E, FF, L.d_row, sc[2]);
    }
    X.blobs = dev(h);
    X.table.resize(NE);
    for (int e = 0; e < NE; ++e) X.table[(size_t) e] = (unsigned long long) (X.blobs + (size_t) (e % kBlobs) * stride);
    X.dtable = dev(X.table);
    return X;
}

uint16_t to_half_bits(float v, bool bf) {
    return bf ? __bfloat16_as_ushort(__float2bfloat16_rn(v)) : __half_as_ushort(__float2half_rn(v));
}
double from_half_bits(uint16_t v, bool bf) {
    return bf ? (double) __bfloat162float(__ushort_as_bfloat16(v)) : (double) __half2float(__ushort_as_half(v));
}

// One test call's inputs on the device and the scratch the calls write.
struct Call {
    int rows = 0;
    std::vector<int32_t> ids;
    std::vector<float> w, x, resid, taps;
    std::vector<int> pos;
    int32_t *d_ids = nullptr, *start = nullptr, *ng = nullptr, *dst = nullptr, *tok = nullptr, *tickets = nullptr, *d_pos = nullptr;
    unsigned long long* ptr = nullptr;
    float *d_w = nullptr, *d_resid = nullptr, *d_taps = nullptr, *part = nullptr, *out = nullptr;
    uint16_t* xh = nullptr;
    void* h = nullptr;
};

Call make_call(int rows, bool bf, int pos0) {
    Call c;
    c.rows = rows;
    std::uniform_int_distribution<int> ex(0, NE - 1);
    c.ids.resize((size_t) rows * K);
    for (int r = 0; r < rows; ++r) {
        std::set<int> used;
        for (int s = 0; s < K; ++s) {
            int e;
            do e = ex(rng);
            while (!used.insert(e).second);
            c.ids[(size_t) r * K + s] = e;
        }
    }
    c.w = randn((size_t) rows * K, 0.3f, 1.f);
    c.x = randn((size_t) rows * E, 1.0f);
    c.resid = randn((size_t) rows * E, 1.0f);
    c.taps = randn((size_t) E * kConv, 0.3f);
    c.pos.resize((size_t) rows);
    for (int r = 0; r < rows; ++r) c.pos[(size_t) r] = pos0 + r;
    std::vector<uint16_t> xh((size_t) rows * E);
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < E; ++i) xh[(size_t) r * E + bk::xt::xh_pos(i)] = to_half_bits(c.x[(size_t) r * E + i], bf);
    c.d_ids = dev(c.ids);
    c.start = zeros<int32_t>((size_t) rows * K + 2);
    c.ng = zeros<int32_t>(4);
    c.dst = zeros<int32_t>((size_t) rows * K);
    c.tok = zeros<int32_t>((size_t) rows * K);
    c.ptr = zeros<unsigned long long>((size_t) rows * K);
    c.tickets = zeros<int32_t>(expert_tickets());
    c.d_pos = dev(c.pos);
    c.d_w = dev(c.w);
    c.d_resid = dev(c.resid);
    c.d_taps = dev(c.taps);
    c.part = zeros<float>((size_t) rows * K * E);
    c.out = zeros<float>((size_t) rows * E);
    c.xh = dev(xh);
    c.h = zeros<uint8_t>(expert_h_bytes(rows * K, FF));
    return c;
}

ExpertCall expert_call(const Experts& X, const Call& c, const unsigned long long* table) {
    ExpertCall e;
    e.L = X.L;
    e.ptr = c.ptr;
    e.start = c.start;
    e.n_groups = c.ng;
    e.dst = c.dst;
    e.tok = c.tok;
    e.pairs = c.rows * K;
    e.n_expert = NE;
    e.x = c.xh;
    e.h = c.h;
    e.tickets = c.tickets;
    (void) table;
    return e;
}

MoeFinish finish_of(const Call& c, float* ring) {
    MoeFinish f;
    f.rows = c.part;
    f.w = c.d_w;
    f.k = K;
    f.pos = c.d_pos;
    f.taps = c.d_taps;
    f.ring = ring;
    f.resid = c.d_resid;
    f.out = c.out;
    return f;
}

// The call's rows [r0, r0 + n) of one conversation through the CPU reference: the outputs of the window's rows, its ring
// before the window `ring0` (positions before pos[r0]).
std::vector<double> reference(const Experts& X, const Call& c, int r0, int n, const std::vector<float>& ring0) {
    const bool bf = X.type == 46;
    std::vector<double> y((size_t) n * E, 0.0), out((size_t) n * E);
    for (int r = 0; r < n; ++r) {
        std::vector<double> xv(E);
        for (int i = 0; i < E; ++i) xv[(size_t) i] = from_half_bits(to_half_bits(c.x[(size_t) (r0 + r) * E + i], bf), bf);
        for (int s = 0; s < K; ++s) {
            const int b = c.ids[(size_t) (r0 + r) * K + s] % kBlobs;
            std::vector<double> h(FF);
            for (int f = 0; f < FF; ++f) {
                double g = 0, u = 0;
                for (int i = 0; i < E; ++i) {
                    g += (double) X.wg[(size_t) b][(size_t) f * E + i] * xv[(size_t) i];
                    u += (double) X.wu[(size_t) b][(size_t) f * E + i] * xv[(size_t) i];
                }
                const double hv = 4.0 * std::tanh(g / 4.0) * sig(g) * 25.0 * std::tanh(u / 25.0);
                h[(size_t) f] = from_half_bits(to_half_bits((float) hv, bf), bf);
            }
            for (int i = 0; i < E; ++i) {
                double o = 0;
                for (int f = 0; f < FF; ++f) o += (double) X.wd[(size_t) b][(size_t) i * FF + f] * h[(size_t) f];
                y[(size_t) r * E + i] += (double) c.w[(size_t) (r0 + r) * K + s] * o;
            }
        }
    }
    for (int r = 0; r < n; ++r)
        for (int i = 0; i < E; ++i) {
            double acc = 0;
            for (int j = 0; j < kConv; ++j) {
                double v;
                if (r - j >= 0) v = y[(size_t) (r - j) * E + i];
                else {
                    const int p = c.pos[(size_t) (r0 + r)] - j;
                    v = p < 0 ? 0.0 : ring0[(size_t) (p % kConvRing) * E + i];
                }
                acc += (double) c.taps[(size_t) i * kConv + (kConv - 1 - j)] * v;
            }
            out[(size_t) r * E + i] = (acc + y[(size_t) r * E + i]) / std::sqrt((double) kConv + 1.0) + c.resid[(size_t) (r0 + r) * E + i];
        }
    return out;
}

void run(const Experts& X, Call& c, const MoeFinish& f, const unsigned long long* table, cudaStream_t st) {
    bk::moe_plan(c.d_ids, c.rows, K, NE, table, c.ptr, c.start, c.ng, c.dst, c.tok, st);
    ExpertCall e = expert_call(X, c, table);
    experts_gate_up(e, st);
    experts_down(e, f, c.rows, st);
}

void test_format(int type) {
    const char* tag = type == 40 ? "NVFP4" : "MXFP8";
    const bool bf = type == 46;
    const Experts X = make_experts(type);
    cudaStream_t st;
    cudaStreamCreate(&st);
    char name[96];

    // a single conversation's window of 4 rows: the CPU reference and the kernels it replaces
    {
        Call c = make_call(4, bf, 300);
        const std::vector<float> ring0 = randn((size_t) kConvRing * E, 0.4f);
        float* ring = dev(ring0);
        run(X, c, finish_of(c, ring), X.dtable, st);
        cudaStreamSynchronize(st);
        const std::vector<float> got = host(c.out, (size_t) c.rows * E);
        const std::vector<double> ref = reference(X, c, 0, c.rows, ring0);
        std::snprintf(name, sizeof name, "experts %s: vs CPU (4 rows)", tag);
        report(name, got, ref, 3e-3);
        // the kernels it replaces: q8_1 activations, the grouped kernels, the combine and the post-convolution
        void* xq = zeros<uint8_t>(bk::native_q8_1_bytes(E, c.rows));
        float* dx = dev(c.x);
        void* scr = zeros<uint8_t>(bk::native_expert_scratch_bytes(c.rows * K, FF));
        float *rows_old = zeros<float>((size_t) c.rows * K * E), *y = zeros<float>((size_t) c.rows * E),
              *out_old = zeros<float>((size_t) c.rows * E), *ring_old = dev(ring0);
        bk::moe_plan(c.d_ids, c.rows, K, NE, X.dtable, c.ptr, c.start, c.ng, c.dst, c.tok, st);
        bk::native_quantize_q8_1(dx, xq, E, c.rows, st);
        bk::native_expert_grouped(X.L, c.ptr, c.start, c.ng, c.dst, c.tok, NE, c.rows * K, xq, scr, rows_old, st);
        moe_combine(rows_old, c.d_w, c.rows, K, y, st);
        post_conv(y, c.d_pos, c.d_taps, ring_old, c.d_resid, c.rows, out_old, st);
        cudaStreamSynchronize(st);
        const std::vector<float> old = host(out_old, (size_t) c.rows * E);
        std::snprintf(name, sizeof name, "  the replaced kernels vs CPU");
        report(name, old, ref, 2e-2);
        std::vector<double> oldd(old.begin(), old.end());
        std::snprintf(name, sizeof name, "  new vs the replaced kernels");
        report(name, got, oldd, 2e-2);
    }

    // bits: 64 conversations of 5 rows in one batch window (the wide tile shape) and 8 of them (the middle one) against
    // each alone (the small shape)
    for (const int S : {64, 8}) {
        const int T = 5, R = S * T;
        Call c = make_call(R, bf, 0);
        for (int sg = 0; sg < S; ++sg)
            for (int t = 0; t < T; ++t) c.pos[(size_t) (sg * T + t)] = 40 + 13 * sg + t;
        basalt::core::device_copy(c.d_pos, c.pos.data(), c.pos.size() * 4, cudaMemcpyHostToDevice);
        std::vector<std::vector<float>> rings(S);
        Segments segs;
        segs.at.n = S;
        for (int sg = 0; sg <= S; ++sg) segs.at.row0[sg] = sg * T;
        std::vector<float*> d_rings(S);
        for (int sg = 0; sg < S; ++sg) {
            rings[(size_t) sg] = randn((size_t) kConvRing * E, 0.4f);
            d_rings[(size_t) sg] = dev(rings[(size_t) sg]);
            segs.c[sg].m_ring = d_rings[(size_t) sg];
        }
        MoeFinish f = finish_of(c, d_rings[0]);
        f.seg = segs;
        run(X, c, f, X.dtable, st);
        cudaStreamSynchronize(st);
        const std::vector<float> batch = host(c.out, (size_t) R * E);
        bool same = true;
        for (int sg = 0; sg < S && same; ++sg) {
            Call one = make_call(T, bf, 0);
            std::copy(c.ids.begin() + (size_t) sg * T * K, c.ids.begin() + (size_t) (sg + 1) * T * K, one.ids.begin());
            std::copy(c.w.begin() + (size_t) sg * T * K, c.w.begin() + (size_t) (sg + 1) * T * K, one.w.begin());
            basalt::core::device_copy(one.d_ids, one.ids.data(), one.ids.size() * 4, cudaMemcpyHostToDevice);
            basalt::core::device_copy(one.d_w, one.w.data(), one.w.size() * 4, cudaMemcpyHostToDevice);
            basalt::core::device_copy(one.xh, c.xh + (size_t) sg * T * E, (size_t) T * E * 2, cudaMemcpyDeviceToDevice);
            basalt::core::device_copy(one.d_resid, c.d_resid + (size_t) sg * T * E, (size_t) T * E * 4, cudaMemcpyDeviceToDevice);
            basalt::core::device_copy(one.d_taps, c.d_taps, (size_t) E * kConv * 4, cudaMemcpyDeviceToDevice);
            basalt::core::device_copy(one.d_pos, c.d_pos + sg * T, (size_t) T * 4, cudaMemcpyDeviceToDevice);
            float* ring1 = dev(rings[(size_t) sg]);
            run(X, one, finish_of(one, ring1), X.dtable, st);
            cudaStreamSynchronize(st);
            same = !std::memcmp(host(one.out, (size_t) T * E).data(), batch.data() + (size_t) sg * T * E, (size_t) T * E * 4) &&
                   host(ring1, (size_t) kConvRing * E) == host(d_rings[(size_t) sg], (size_t) kConvRing * E);
        }
        std::snprintf(name, sizeof name, "experts %s: %d segments of 5 = each alone", tag, S);
        std::printf("%-48s %s\n", name, same ? "bitwise" : "FAIL");
        failures += !same;
    }

    // bits: experts off the card (their rows written before the down, as the CPU's merge does), the others on it
    {
        Call c = make_call(8, bf, 500);
        const std::vector<float> ring0 = randn((size_t) kConvRing * E, 0.4f);
        float* ring = dev(ring0);
        run(X, c, finish_of(c, ring), X.dtable, st);
        cudaStreamSynchronize(st);
        const std::vector<float> all_on = host(c.out, (size_t) c.rows * E);
        const std::vector<float> part = host(c.part, (size_t) c.rows * K * E);
        std::vector<unsigned long long> off = X.table;
        for (int e = 0; e < NE; e += 3) off[(size_t) e] = 0;
        unsigned long long* doff = dev(off);
        Call c2 = c;
        c2.part = zeros<float>((size_t) c.rows * K * E);
        c2.out = zeros<float>((size_t) c.rows * E);
        c2.tickets = zeros<int32_t>(expert_tickets());
        std::vector<float> pre((size_t) c.rows * K * E, 0.f);   // the off-card pairs' rows, as the merge leaves them
        for (int i = 0; i < c.rows * K; ++i)
            if (!off[(size_t) c.ids[(size_t) i]]) std::copy(part.begin() + (size_t) i * E, part.begin() + (size_t) (i + 1) * E, pre.begin() + (size_t) i * E);
        basalt::core::device_copy(c2.part, pre.data(), pre.size() * 4, cudaMemcpyHostToDevice);
        float* ring2 = dev(ring0);
        bk::moe_plan(c2.d_ids, c2.rows, K, NE, doff, c2.ptr, c2.start, c2.ng, c2.dst, c2.tok, st);
        ExpertCall e = expert_call(X, c2, doff);
        experts_gate_up(e, st);
        experts_down_rows(e, finish_of(c2, ring2), c2.rows, st);
        experts_finish(e, finish_of(c2, ring2), c2.rows, st);
        cudaStreamSynchronize(st);
        std::snprintf(name, sizeof name, "experts %s: a third off the card = all on it", tag);
        same_bits(name, host(c2.out, (size_t) c.rows * E).data(), all_on.data(), (size_t) c.rows * E * 4);
        // and every expert off the card: the down writes nothing, the finish reads the rows given
        std::vector<unsigned long long> none(NE, 0ull);
        unsigned long long* dnone = dev(none);
        Call c3 = c;
        c3.part = dev(part);
        c3.out = zeros<float>((size_t) c.rows * E);
        c3.tickets = zeros<int32_t>(expert_tickets());
        float* ring3 = dev(ring0);
        bk::moe_plan(c3.d_ids, c3.rows, K, NE, dnone, c3.ptr, c3.start, c3.ng, c3.dst, c3.tok, st);
        ExpertCall e3 = expert_call(X, c3, dnone);
        experts_gate_up(e3, st);
        experts_down_rows(e3, finish_of(c3, ring3), c3.rows, st);
        experts_finish(e3, finish_of(c3, ring3), c3.rows, st);
        cudaStreamSynchronize(st);
        std::snprintf(name, sizeof name, "experts %s: every expert off the card", tag);
        same_bits(name, host(c3.out, (size_t) c.rows * E).data(), all_on.data(), (size_t) c.rows * E * 4);
        const std::vector<int32_t> tk = host(c3.tickets, expert_tickets());
        bool zero = std::all_of(tk.begin(), tk.end(), [](int32_t v) { return v == 0; });
        std::printf("%-48s %s\n", "  the tickets after the calls", zero ? "zero" : "FAIL");
        failures += !zero;
    }
    cudaStreamDestroy(st);
}

// `same`: every row routes the same 8 experts (a batch of one conversation repeated: few groups, many entries each)
void bench(int type, bool same) {
    const bool bf = type == 46;
    const Experts X = make_experts(type);
    cudaStream_t st;
    cudaStreamCreate(&st);
    for (const int R : {1, 4, 8, 64, 256, 320}) {
        Call c = make_call(R, bf, 100);
        if (same) {
            for (int r = 0; r < R; ++r)
                for (int q = 0; q < K; ++q) c.ids[(size_t) r * K + q] = 3 + 7 * q;
            basalt::core::device_copy(c.d_ids, c.ids.data(), c.ids.size() * 4, cudaMemcpyHostToDevice);
        }
        float* ring = zeros<float>((size_t) kConvRing * E);
        MoeFinish f = finish_of(c, ring);
        if (R > kDecodeRows) {   // a batch window: conversations of 5 rows (the last one the rest)
            f.seg.at.n = (R + 4) / 5;
            for (int sg = 0; sg <= f.seg.at.n; ++sg) f.seg.at.row0[sg] = std::min(R, 5 * sg);
            f.seg.n_segs = f.seg.at.n;
            for (int sg = 0; sg < f.seg.at.n; ++sg) f.seg.c[sg].m_ring = zeros<float>((size_t) kConvRing * E);
        }
        std::set<int> distinct(c.ids.begin(), c.ids.end());
        const double bytes = (double) distinct.size() * (double) X.L.bytes;
        bk::moe_plan(c.d_ids, c.rows, K, NE, X.dtable, c.ptr, c.start, c.ng, c.dst, c.tok, st);
        ExpertCall e = expert_call(X, c, X.dtable);
        for (int i = 0; i < 5; ++i) {
            experts_gate_up(e, st);
            experts_down(e, f, R, st);
        }
        cudaStreamSynchronize(st);
        const int n = 50;
        cudaEvent_t a, b, m;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventCreate(&m);
        float gu = 0.f, dn = 0.f;
        for (int i = 0; i < n; ++i) {
            float t0 = 0.f, t1 = 0.f;
            cudaEventRecord(a, st);
            experts_gate_up(e, st);
            cudaEventRecord(m, st);
            experts_down(e, f, R, st);
            cudaEventRecord(b, st);
            cudaEventSynchronize(b);
            cudaEventElapsedTime(&t0, a, m);
            cudaEventElapsedTime(&t1, m, b);
            gu += t0;
            dn += t1;
        }
        const double ug = 1e3 * gu / n, ud = 1e3 * dn / n, flop = 2.0 * R * K * 3.0 * E * FF;
        std::printf("bench experts %s%s %3d rows: gate/up %7.1f us, down + finish %7.1f us, %3zu experts, %6.0f GB/s, %5.1f TFLOP/s\n",
                    type == 40 ? "NVFP4" : "MXFP8", same ? " (one routing)" : "", R, ug, ud, distinct.size(), bytes / ((ug + ud) * 1e3),
                    flop / ((ug + ud) * 1e6));
    }
    cudaStreamDestroy(st);
}

}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    if (do_bench) {
        bench(40, false);
        bench(40, true);
        bench(46, false);
        bench(46, true);
        return 0;
    }
    test_format(40);
    test_format(46);
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
