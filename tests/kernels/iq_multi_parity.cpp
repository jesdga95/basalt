// tests/kernels/iq_multi_parity.cpp - the IQ kernels that decode each weight part once for every
// column / entry, against the older kernels that decode it again per column.  Synthetic blocks, no model.
//
//     build/iq_multi_parity            the checks (a ctest; needs the GPU)
//     build/iq_multi_parity --bench    the checks, then old/new timings of both paths
//
// (1) iq_mmvq, every format it takes, ncols 1..8 and 11 (the > 8 split): the new output must be BITWISE equal to
//     the old one (iq_set_old_kernels), and within float rounding of a double reference built from the GPU's own
//     dequantizer (validated against gguf-py by iq_parity) and the dequantized q8_1 activations - a gross-error
//     guard, not the contract.
// (2) native_expert_grouped, every gate/up format with the down formats: groups of 0..11 entries (more than a pass
//     of GRP_NC), unused grid rows (cap_groups > groups), scattered destinations; bitwise, new vs old, the rows it
//     must not write included.
// (3) iq_mmvq's IQ3_S columns on the tensor-core kernel (MMVQ_MMA, turned on here where the device has it)
//     against the old kernel, bitwise; with MMVQ_MMA=1 in the environment (1) takes it as well.
//
// NVFP4 blocks come in three scale profiles (narrow; pack-like block values with 2e-5..2e-4 tails; pack-like with scale
// bytes 0x7F, 0xFF, bit 7 set, 0 and subnormals), all of which every path must decode alike.
//
// Random bytes are valid codes for every format here (all grid indices are in range); only the fp16 block scales
// are set, small enough that the grouped path's SwiGLU output keeps a finite fp16 q8_1 scale.
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace k = basalt::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template<typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}

const char* name_of(int t) {
    switch (t) {
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 29: return "IQ1_M";
        case 42: return "Q2_0";
        case 40: return "NVFP4";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        default: return "?";
    }
}
int block_values(int t) { return t == 20 || t == 7 || t == 8 ? 32 : t == 42 || t == 40 ? 64 : 256; }
// A UE4M3 sub-block scale byte.  Profile 0: exponent 1..4 (small blocks).  Profile 1: exponent 6..14, block values in
// the hundreds as the packs' are (their tails are 1e-5..1e-4).  Profile 2: profile 1 with the odd bytes mixed in: 0x7F
// (the NaN code, decoded as 0), the sign bit set (ignored), zero and subnormals.
uint8_t nvfp4_scale_byte(int prof, std::mt19937& rng) {
    if (prof == 0) return (uint8_t) (((1 + rng() % 4) << 3) | (rng() & 7));
    const uint8_t v = (uint8_t) (((6 + rng() % 9) << 3) | (rng() & 7));
    if (prof == 1) return v;
    switch (rng() % 8) {
        case 0: return 0x7F;
        case 1: return (uint8_t) (v | 0x80);
        case 2: return 0xFF;
        case 3: return 0x00;
        case 4: return (uint8_t) (rng() & 7);   // subnormal
        default: return v;
    }
}
// The FP32 tail scale: profile 0 in 0.5 .. 2, else log-uniform in 2e-5 .. 2e-4 (what the packs carry).
float nvfp4_tail_scale(int prof, std::mt19937& rng) {
    if (prof == 0) return std::uniform_real_distribution<float>(0.5f, 2.0f)(rng);
    return 2e-5f * std::exp(std::uniform_real_distribution<float>(0.0f, std::log(10.0f))(rng));
}
// The NVFP4 scale profile of the blob being made (-1: a random one per matrix)
int g_nv_profile = -1;
// NVFP4 experts end in {s_gate, s_up, s_down, 0}: FP32 scales
void append_nvfp4_tail(std::vector<uint8_t>& b, std::mt19937& rng) {
    const int prof = g_nv_profile >= 0 ? g_nv_profile : 0;
    const float t[4] = {nvfp4_tail_scale(prof, rng), nvfp4_tail_scale(prof, rng), nvfp4_tail_scale(prof, rng), 0.0f};
    const uint8_t* p = (const uint8_t*) t;
    b.insert(b.end(), p, p + sizeof t);
}


// `rows` rows of `n` values of format t: random bytes, then a finite fp16 scale in every block
std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 3);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        if (t == 40) {
            // NVFP4: four UE4M3 sub-block scales (profiles in nvfp4_scale_byte)
            const int prof = g_nv_profile >= 0 ? g_nv_profile : (int) (rng() % 3);
            for (int j = 0; j < 4; ++j) w[o + j] = nvfp4_scale_byte(prof, rng);
        } else if (t == 29) {
            // IQ1_M: the fp16 scale is the top nibbles of the four scale words (bytes 48..55); nibble 3 carries the
            // sign and the exponent's top bits: 0x1 / 0x2 (or 0x9 / 0xA) keeps it in 2^-11 .. 2^-3
            const uint8_t nib = (uint8_t) ((sgn(rng) == 0 ? 0x8 : 0x0) | (1 + (byte(rng) & 1)));
            w[o + 55] = (uint8_t) ((w[o + 55] & 0x0F) | (nib << 4));
        } else if (t == 12 || t == 13 || t == 7) {
            // K-quants and Q5_1: two fp16 scales {d, dmin} / {d, m}, small so SwiGLU keeps a finite fp16 q8_1 scale
            for (int j = 0; j < 2; ++j) {
                const uint16_t h = (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | ((2 + byte(rng) % 4) << 10) | man(rng));
                std::memcpy(&w[o + 2 * j], &h, 2);
            }
        } else {
            const uint16_t h = (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));   // 2^-13 .. 2^-6
            std::memcpy(&w[o], &h, 2);
        }
    }
    return w;
}

std::vector<float> random_x(size_t n, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n);
    for (auto& v : x) v = nd(rng);
    return x;
}

// q8_1 blocks (fp16 scale, fp16 sum, 32 int8) back to floats
std::vector<double> dequant_q8_1(const std::vector<uint8_t>& q, size_t n) {
    std::vector<double> v(n);
    for (size_t b = 0; b < n / 32; ++b) {
        uint16_t dh;
        std::memcpy(&dh, &q[b * 36], 2);
        const double d = k::f32_from_f16(dh);
        for (int i = 0; i < 32; ++i) v[b * 32 + i] = d * (int8_t) q[b * 36 + 4 + i];
    }
    return v;
}

// ------------------------------------------------------------------------------------------------ (1) iq_mmvq
void check_mmvq(int t, int n_in, int n_out, cudaStream_t s, std::mt19937& rng) {
    const auto w = random_rows(t, n_out, n_in, rng);
    uint8_t* dw = dalloc<uint8_t>(w.size());
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    // the double reference's weights: the GPU dequantizer (iq_parity checks it against gguf-py)
    float* dwf = dalloc<float>((size_t) n_out * n_in);
    k::iq_dequant_f32(t, dw, (int64_t) n_out * n_in, dwf, s);
    std::vector<float> wf((size_t) n_out * n_in);
    ck(cudaMemcpy(wf.data(), dwf, wf.size() * 4, cudaMemcpyDeviceToHost), "wf");
    const int max_cols = 11;
    const auto x = random_x((size_t) max_cols * n_in, rng);
    float* dx = dalloc<float>(x.size());
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    const size_t xq_bytes = (size_t) max_cols * (n_in / 32) * 36;
    uint8_t* dxq = dalloc<uint8_t>(xq_bytes);
    k::quantize_q8_1_rows(dx, max_cols, n_in, dxq, s);
    std::vector<uint8_t> xq(xq_bytes);
    ck(cudaMemcpy(xq.data(), dxq, xq_bytes, cudaMemcpyDeviceToHost), "xq");
    const auto xd = dequant_q8_1(xq, (size_t) max_cols * n_in);
    float* dy_old = dalloc<float>((size_t) max_cols * n_out);
    float* dy_new = dalloc<float>((size_t) max_cols * n_out);
    std::vector<float> y_old((size_t) max_cols * n_out), y_new(y_old.size());
    const int cols[] = {1, 2, 3, 4, 5, 6, 7, 8, 11};
    int bad = 0;
    double worst = 0.0;
    for (int nc : cols) {
        ck(cudaMemset(dy_old, 0xFF, y_old.size() * 4), "memset");
        ck(cudaMemset(dy_new, 0xFF, y_new.size() * 4), "memset");
        k::iq_set_old_kernels(true);
        k::iq_mmvq(t, dw, dxq, dy_old, n_in, n_out, nc, s);
        k::iq_set_old_kernels(false);
        k::iq_mmvq(t, dw, dxq, dy_new, n_in, n_out, nc, s);
        ck(cudaStreamSynchronize(s), "sync");
        ck(cudaMemcpy(y_old.data(), dy_old, y_old.size() * 4, cudaMemcpyDeviceToHost), "y_old");
        ck(cudaMemcpy(y_new.data(), dy_new, y_new.size() * 4, cudaMemcpyDeviceToHost), "y_new");
        size_t diff = 0;
        for (size_t i = 0; i < y_old.size(); ++i) diff += std::memcmp(&y_old[i], &y_new[i], 4) != 0;
        double num = 0, den = 0;
        bool finite = true;
        for (int c = 0; c < nc; ++c)
            for (int r = 0; r < n_out; ++r) {
                double acc = 0;
                for (int i = 0; i < n_in; ++i) acc += (double) wf[(size_t) r * n_in + i] * xd[(size_t) c * n_in + i];
                const float g = y_new[(size_t) c * n_out + r];
                finite = finite && std::isfinite(g);
                num += std::fabs(g - acc);
                den += std::fabs(acc);
            }
        const double rel = num / (den + 1e-30);
        worst = std::max(worst, rel);
        if (diff != 0 || !finite || !(rel < 1e-2)) {
            std::printf("  %-8s %5d x %5d  ncols %2d: %zu outputs differ from the old kernel, ref rel %.2e%s  FAIL\n",
                        name_of(t), n_out, n_in, nc, diff, rel, finite ? "" : ", non-finite");
            ++bad;
        }
    }
    std::printf("%-8s type %2d  %5d x %5d  iq_mmvq ncols 1..8, 11: %s (ref rel <= %.2e)\n", name_of(t), t, n_out, n_in,
                bad ? "FAIL" : "bitwise equal to the old kernel", worst);
    g_fail += bad;
    cudaFree(dw); cudaFree(dwf); cudaFree(dx); cudaFree(dxq); cudaFree(dy_old); cudaFree(dy_new);
}

// (3) MMVQ_MMA: iq_mmvq's IQ3_S columns on the tensor-core kernel, turned on here where the device has it,
// against the old per-column kernel (iq_set_old_kernels, which the tensor-core path never takes), bitwise; with row
// tails, both block sizes the launcher picks, and weights / activations off their 4- and 16-byte alignment.  Every
// call must take the tensor-core kernel (its launch counter), or the check checked nothing.
void check_mmvq_mma(cudaStream_t s, std::mt19937& rng) {
    if (!k::iq_mmvq_mma_available()) {
        std::printf("IQ3_S    tensor-core iq_mmvq (MMVQ_MMA): not available on this device or build "
                    "(needs int8 tensor cores): skipped\n");
        return;
    }
    const bool env = k::iq_mmvq_mma();
    struct Shape { int n_in, n_out, w_off, x_off; };
    const Shape shapes[] = {{2560, 6144, 0, 0}, {2560, 67, 0, 0}, {1024, 5, 0, 0}, {5120, 300, 0, 0},
                            {512, 6000, 0, 0},  {2560, 1000, 2, 4}};
    for (const Shape& sh : shapes) {
        const auto w = random_rows(21, sh.n_out, sh.n_in, rng);
        uint8_t* dw = dalloc<uint8_t>(w.size() + 16);
        ck(cudaMemcpy(dw + sh.w_off, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
        const auto x = random_x((size_t) 8 * sh.n_in, rng);
        float* dx = dalloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        uint8_t* dxq = dalloc<uint8_t>((size_t) 8 * (sh.n_in / 32) * 36 + 16);
        k::quantize_q8_1_rows(dx, 8, sh.n_in, dxq + sh.x_off, s);
        float* dy_old = dalloc<float>((size_t) 8 * sh.n_out);
        float* dy_new = dalloc<float>((size_t) 8 * sh.n_out);
        std::vector<uint32_t> y_old((size_t) 8 * sh.n_out), y_new(y_old.size());
        int bad = 0;
        for (int nc = 2; nc <= 8; ++nc) {
            ck(cudaMemset(dy_old, 0xFF, y_old.size() * 4), "memset");
            ck(cudaMemset(dy_new, 0xFE, y_new.size() * 4), "memset");
            k::iq_set_old_kernels(true);
            k::iq_mmvq(21, dw + sh.w_off, dxq + sh.x_off, dy_old, sh.n_in, sh.n_out, nc, s);
            k::iq_set_old_kernels(false);
            k::iq_set_mmvq_mma(true);
            const long before = k::iq_mmvq_mma_launches();
            k::iq_mmvq(21, dw + sh.w_off, dxq + sh.x_off, dy_new, sh.n_in, sh.n_out, nc, s);
            const long launches = k::iq_mmvq_mma_launches() - before;
            k::iq_set_mmvq_mma(env);
            ck(cudaStreamSynchronize(s), "sync");
            ck(cudaMemcpy(y_old.data(), dy_old, y_old.size() * 4, cudaMemcpyDeviceToHost), "y_old");
            ck(cudaMemcpy(y_new.data(), dy_new, y_new.size() * 4, cudaMemcpyDeviceToHost), "y_new");
            size_t diff = 0;
            bool finite = true;
            for (size_t i = 0; i < (size_t) nc * sh.n_out; ++i) {   // the columns written (the rest keep their fills)
                diff += y_old[i] != y_new[i];
                float f;
                std::memcpy(&f, &y_new[i], 4);
                finite = finite && std::isfinite(f);
            }
            for (size_t i = (size_t) nc * sh.n_out; i < y_new.size(); ++i)   // and nothing past them
                diff += y_new[i] != 0xFEFEFEFEu;
            if (diff != 0 || !finite || launches != 1) {
                std::printf("  IQ3_S    %5d x %5d  ncols %d%s: %zu outputs differ from the old kernel%s%s  FAIL\n",
                            sh.n_out, sh.n_in, nc, sh.w_off || sh.x_off ? " (off-aligned)" : "", diff,
                            finite ? "" : ", non-finite", launches == 1 ? "" : ", the tensor-core kernel did not run");
                ++bad;
            }
        }
        std::printf("IQ3_S    type 21  %5d x %5d  tensor-core iq_mmvq ncols 2..8%s: %s\n", sh.n_out, sh.n_in,
                    sh.w_off || sh.x_off ? " (off-aligned)" : "", bad ? "FAIL" : "bitwise equal to the old kernel");
        g_fail += bad;
        cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy_old); cudaFree(dy_new);
    }
}

// ---------------------------------------------------------------------------------- (2) native_expert_grouped
struct Grouped {
    k::NativeExpertLayout L;
    int n_groups = 0, n_ent = 0, cap_groups = 0, cap_ent = 0, n_tok = 0;
    std::vector<uint8_t*> blobs;
    unsigned long long* dptr = nullptr;
    int32_t *dstart = nullptr, *dn = nullptr, *ddst = nullptr, *dtok = nullptr;
    float* dx = nullptr;
    uint8_t *dxq = nullptr, *dscr = nullptr;
    float* dout = nullptr;
    size_t out_floats = 0;

    Grouped(int gu, int dt, int64_t H, int64_t FF, const std::vector<int>& counts, int tokens, std::mt19937& rng) {
        L = k::native_expert_layout(gu, dt, H, FF);
        n_groups = (int) counts.size();
        cap_groups = n_groups + 2;              // the extra grid rows must exit
        n_tok = tokens;
        std::vector<int32_t> start(1, 0), tok, dst;
        for (int c : counts) start.push_back(start.back() + c);
        n_ent = start.back();
        cap_ent = n_ent + 3;
        std::uniform_int_distribution<int> tk(0, tokens - 1);
        for (int e = 0; e < n_ent; ++e) tok.push_back(tk(rng));
        dst.resize(n_ent);
        std::iota(dst.begin(), dst.end(), 0);
        std::shuffle(dst.begin(), dst.end(), rng);
        std::vector<unsigned long long> ptr;
        for (int g = 0; g < n_groups; ++g) {
            // one blob per group: gate rows | up rows (format gu, H values) | down rows (format dt, FF values)
            g_nv_profile = g % 3;   // the narrow, the pack-like, the pack-like with odd scale bytes
            auto b = random_rows(gu, 2 * FF, H, rng);
            const auto d = random_rows(dt, H, FF, rng);
            b.insert(b.end(), d.begin(), d.end());
            if (gu == 40) append_nvfp4_tail(b, rng);
            g_nv_profile = -1;
            if (b.size() != L.bytes) { std::fprintf(stderr, "blob size %zu != %zu\n", b.size(), L.bytes); std::exit(2); }
            uint8_t* db = dalloc<uint8_t>(b.size());
            ck(cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice), "blob");
            blobs.push_back(db);
            ptr.push_back((unsigned long long) db);
        }
        dptr = dalloc<unsigned long long>(cap_groups);
        dstart = dalloc<int32_t>(cap_groups + 1);
        dn = dalloc<int32_t>(1);
        ddst = dalloc<int32_t>(cap_ent);
        dtok = dalloc<int32_t>(cap_ent);
        ck(cudaMemcpy(dptr, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice), "ptr");
        ck(cudaMemcpy(dstart, start.data(), start.size() * 4, cudaMemcpyHostToDevice), "start");
        ck(cudaMemcpy(dn, &n_groups, 4, cudaMemcpyHostToDevice), "n");
        if (n_ent) {
            ck(cudaMemcpy(ddst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "dst");
            ck(cudaMemcpy(dtok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice), "tok");
        }
        const auto x = random_x((size_t) tokens * H, rng);
        dx = dalloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        dxq = dalloc<uint8_t>((size_t) tokens * (H / 32) * 36);
        dscr = dalloc<uint8_t>(k::native_expert_scratch_bytes(cap_ent, FF));
        out_floats = (size_t) cap_ent * H;
        dout = dalloc<float>(out_floats);
    }
    ~Grouped() {
        for (auto* b : blobs) cudaFree(b);
        cudaFree(dptr); cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
        cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout);
    }
    void run(bool old, cudaStream_t s) {
        k::iq_set_old_kernels(old);
        k::quantize_q8_1_rows(dx, n_tok, L.n_embd, dxq, s);
        k::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, cap_groups, cap_ent, dxq, dscr, dout, s);
        k::iq_set_old_kernels(false);
    }
    std::vector<float> result(bool old, cudaStream_t s) {
        ck(cudaMemset(dout, 0xFF, out_floats * 4), "memset");      // NaN rows: must stay NaN where nothing writes
        ck(cudaMemset(dscr, 0, k::native_expert_scratch_bytes(cap_ent, L.n_ff)), "memset");
        run(old, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> o(out_floats);
        ck(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost), "out");
        return o;
    }
};

void check_grouped(int gu, int dt, int64_t H, int64_t FF, cudaStream_t s, std::mt19937& rng) {
    // 0..11 entries per group: one pass, a partial pass, two passes and three passes of GRP_NC = 4
    Grouped G(gu, dt, H, FF, {1, 3, 4, 5, 0, 8, 2, 11, 1}, 8, rng);
    const auto a = G.result(true, s), b = G.result(false, s);
    size_t diff = 0, written = 0;
    bool finite = true;
    for (size_t i = 0; i < a.size(); ++i) {
        diff += std::memcmp(&a[i], &b[i], 4) != 0;
        if (i < (size_t) G.n_ent * H) { ++written; finite = finite && std::isfinite(b[i]); }
    }
    const bool ok = diff == 0 && finite;
    std::printf("%-8s/%-7s %5lld x %4lld  native_expert_grouped, %d groups, %d entries: %s\n", name_of(gu),
                name_of(dt), (long long) H, (long long) FF, G.n_groups, G.n_ent,
                ok ? "bitwise equal to the old kernels" : "FAIL");
    if (!ok) {
        std::printf("  %zu of %zu floats differ%s\n", diff, a.size(), finite ? "" : ", non-finite outputs");
        ++g_fail;
    }
}

// ------------------------------------------------------------------------------------------------ --bench
float time_ms(cudaStream_t s, int it, const auto& fn) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    fn();
    cudaEventRecord(e0, s);
    for (int i = 0; i < it; ++i) fn();
    cudaEventRecord(e1, s);
    ck(cudaEventSynchronize(e1), "bench");
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / it;
}

void bench(cudaStream_t s, std::mt19937& rng) {
    std::printf("\n--bench: microseconds per call, old (per-column decode) / new (decode once), idle GPU assumed\n");
    const int n_in = 2560, n_out = 8192, it = 200;
    for (int t : {16, 17, 18, 21, 22, 29, 12, 13, 7, 8}) {
        const auto w = random_rows(t, n_out, n_in, rng);
        uint8_t* dw = dalloc<uint8_t>(w.size());
        ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
        const auto x = random_x((size_t) 8 * n_in, rng);
        float* dx = dalloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        uint8_t* dxq = dalloc<uint8_t>((size_t) 8 * (n_in / 32) * 36);
        k::quantize_q8_1_rows(dx, 8, n_in, dxq, s);
        float* dy = dalloc<float>((size_t) 8 * n_out);
        std::printf("%-8s iq_mmvq %d x %d:", name_of(t), n_out, n_in);
        for (int nc = 1; nc <= 8; ++nc) {
            float us[2];
            for (int old = 1; old >= 0; --old) {
                k::iq_set_old_kernels(old != 0);
                us[old] = 1e3f * time_ms(s, it, [&] { k::iq_mmvq(t, dw, dxq, dy, n_in, n_out, nc, s); });
            }
            std::printf("  %d: %.1f/%.1f", nc, us[1], us[0]);
        }
        std::printf("\n");
        k::iq_set_old_kernels(false);
        cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy);
    }
    if (k::iq_mmvq_mma_available()) {   // the dense gate's shape: decode-once dp4a / tensor-core
        const bool env = k::iq_mmvq_mma();
        const int gi = 2560, go = 6144;
        const auto w = random_rows(21, go, gi, rng);
        uint8_t* dw = dalloc<uint8_t>(w.size());
        ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
        const auto x = random_x((size_t) 8 * gi, rng);
        float* dx = dalloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        uint8_t* dxq = dalloc<uint8_t>((size_t) 8 * (gi / 32) * 36);
        k::quantize_q8_1_rows(dx, 8, gi, dxq, s);
        float* dy = dalloc<float>((size_t) 8 * go);
        std::printf("IQ3_S    iq_mmvq %d x %d, dp4a / tensor-core:", go, gi);
        for (int nc = 2; nc <= 8; ++nc) {
            float us[2];
            for (int mma = 0; mma < 2; ++mma) {
                k::iq_set_mmvq_mma(mma == 1);
                us[mma] = 1e3f * time_ms(s, it, [&] { k::iq_mmvq(21, dw, dxq, dy, gi, go, nc, s); });
            }
            std::printf("  %d: %.1f/%.1f", nc, us[0], us[1]);
        }
        std::printf("\n");
        k::iq_set_mmvq_mma(env);
        cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy);
    }
    // the verify window's shape: 2560 x 640 experts, 16 groups of m entries each
    // (GB/s: the 16 experts' bytes once per call, at the new kernel's time)
    for (auto [gu, dt] : {std::pair<int, int>{16, 42}, {17, 42}, {18, 42}, {21, 42}, {22, 42}, {29, 42}, {40, 40},
                          {12, 7}, {12, 8}, {13, 8}, {8, 8}}) {   // NVFP4 with its own down format; the Q4_K_XL and Q8_0 pairs
        std::printf("%-8s/%-5s grouped, 16 groups of m entries:", name_of(gu), name_of(dt));
        std::string gbs;
        for (int m : {1, 2, 3, 4, 8}) {
            Grouped G(gu, dt, 2560, 640, std::vector<int>(16, m), 8, rng);
            float us[2];
            for (int old = 1; old >= 0; --old)
                us[old] = 1e3f * time_ms(s, 50, [&] { G.run(old != 0, s); });
            std::printf("  m=%d: %.1f/%.1f", m, us[1], us[0]);
            char b[32];
            std::snprintf(b, sizeof b, " %.0f", 16.0 * (double) G.L.bytes / (us[0] * 1e3));
            gbs += b;
        }
        std::printf("   GB/s (new):%s\n", gbs.c_str());
    }
}

// both q8_1 activation quantizers (quantize_q8_1_rows and native_quantize_q8_1) against a host transcription
// of the formula: every finite block bit for bit; a block whose sum (or scale) overflows fp16 is stored
// finite (the largest half, its sign), with its values in the int8 range.
void check_q8_1_finite(cudaStream_t s, std::mt19937& rng) {
    const int n = 2560, rows = 3, nb = n / 32;
    std::vector<float> x = random_x((size_t) rows * n, rng);
    for (int i = 0; i < 32; ++i) x[(size_t) 5 * 32 + i] = 3000.0f;                  // sum 96,000: overflowed
    for (int i = 0; i < 32; ++i) x[(size_t) 9 * 32 + i] = i == 3 ? -70000.0f : 0.5f;   // sum -69,984.5
    for (int i = 0; i < 32; ++i) x[(size_t) 13 * 32 + i] = 2000.0f + i;            // sum 64,496: finite, kept
    x[(size_t) n + 40] = 2.0e7f;                                                    // amax / 127 overflowed too
    float* dx = dalloc<float>(x.size());
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    uint8_t* dq = dalloc<uint8_t>((size_t) rows * nb * 36);
    for (int path = 0; path < 2; ++path) {
        if (path == 0) k::quantize_q8_1_rows(dx, rows, n, dq, s);
        else k::native_quantize_q8_1(dx, dq, n, rows, s);
        ck(cudaStreamSynchronize(s), "quantize");
        std::vector<uint8_t> got((size_t) rows * nb * 36);
        ck(cudaMemcpy(got.data(), dq, got.size(), cudaMemcpyDeviceToHost), "q");
        int same = 0, clamped = 0, bad = 0;
        for (int b = 0; b < rows * nb; ++b) {
            const float* v = x.data() + (size_t) b * 32;
            float lane[32], amax = 0.0f;
            for (int i = 0; i < 32; ++i) { lane[i] = v[i]; amax = std::max(amax, std::fabs(v[i])); }
            for (int o = 16; o > 0; o >>= 1) {   // the warp's butterfly: every lane ends with the same sum
                float nx[32];
                for (int i = 0; i < 32; ++i) nx[i] = lane[i] + lane[i ^ o];
                std::memcpy(lane, nx, sizeof lane);
            }
            const float d = amax / 127.0f, sum = lane[0];
            const uint8_t* g = got.data() + (size_t) b * 36;
            uint16_t gd, gs;
            std::memcpy(&gd, g, 2);
            std::memcpy(&gs, g + 2, 2);
            const uint16_t od = k::f16_from_f32(d), os = k::f16_from_f32(sum);
            const bool finite_before = (od & 0x7c00) != 0x7c00 && (os & 0x7c00) != 0x7c00;
            if (finite_before) {
                uint8_t want[36];
                std::memcpy(want, &od, 2);
                std::memcpy(want + 2, &os, 2);
                for (int i = 0; i < 32; ++i)   // CUDA's roundf: half away from zero, as std::round
                    want[4 + i] = (uint8_t) (int8_t) (amax == 0.0f ? 0.0f : std::round(v[i] / d));
                // native_mmvq.cu is built with --use_fast_math (its divisions are approximate): there the sum must be
                // the same bits, the scale within one fp16 step and each value within one step
                bool near = path == 1 && gs == os && std::abs((int) gd - (int) od) <= 1;
                for (int i = 0; near && i < 32; ++i) near = std::abs((int) (int8_t) g[4 + i] - (int) (int8_t) want[4 + i]) <= 1;
                if (std::memcmp(want, g, 36) == 0 || near) ++same;
                else { ++bad; std::printf("  q8_1 path %d block %d: changed although it was finite\n", path, b); }
            } else {
                const float fd = k::f32_from_f16(gd), fs = k::f32_from_f16(gs);
                bool ok = std::isfinite(fd) && std::isfinite(fs) && fd > 0.0f &&
                          ((os & 0x7c00) != 0x7c00 ? gs == os : std::fabs(fs) == 65504.0f && (fs < 0) == (sum < 0));
                for (int i = 0; i < 32; ++i) ok = ok && (int8_t) g[4 + i] >= -127 && (int8_t) g[4 + i] <= 127;
                if (ok) ++clamped;
                else { ++bad; std::printf("  q8_1 path %d block %d: d %g sum %g not finite and clamped\n", path, b, fd, fs); }
            }
        }
        std::printf("q8_1 finite (%s): %d finite blocks as before, %d overflowed blocks clamped finite%s\n",
                    path == 0 ? "quantize_q8_1_rows" : "native_quantize_q8_1", same, clamped, bad ? "  FAIL" : "");
        g_fail += bad + (clamped != 3);
    }
    cudaFree(dx);
    cudaFree(dq);
}

}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    std::mt19937 rng(18);
    for (int t : {16, 17, 18, 20, 21, 22, 23, 29, 42, 40, 12, 13, 7, 8}) {
        check_mmvq(t, 2560, 67, s, rng);   // the model's n_embd; 67 rows: a partial block of 4 rows
        check_mmvq(t, 1024, 5, s, rng);
    }
    for (int gu : {16, 17, 18, 21, 22, 23, 29, 42}) {
        for (int dt : {20, 42}) check_grouped(gu, dt, 2560, 640, s, rng);   // the model's shape
        check_grouped(gu, 23, 1024, 512, s, rng);                           // IQ4_XS down needs n_ff % 256 == 0
    }
    for (auto [gu, dt] : {std::pair<int, int>{12, 7}, {12, 8}, {13, 8}, {13, 7}, {8, 8}}) {   // the Q4_K_XL and Q8_0 pairs
        check_grouped(gu, dt, 2560, 640, s, rng);
        check_grouped(gu, dt, 1024, 512, s, rng);
    }
    check_grouped(40, 40, 2560, 640, s, rng);                               // NVFP4: its own pair, with the scale tail
    check_grouped(40, 40, 1024, 512, s, rng);
    check_mmvq_mma(s, rng);
    check_q8_1_finite(s, rng);
    if (do_bench) bench(s, rng);
    std::printf("iq_multi_parity: %d failures\n", g_fail);
    cudaStreamDestroy(s);
    return g_fail ? 1 : 0;
}
