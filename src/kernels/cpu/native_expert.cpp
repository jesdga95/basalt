// src/kernels/cpu/native_expert.cpp - native (GGUF-form) experts on the CPU through the extracted ggml dot products.
// See the header.  The activation quantizers and the row dot products here are ggml's (src/kernels/cpu/ggml), so an
// IQ expert computes what llama.cpp's CPU backend computes for it.
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/iq_avx512.hpp"
#include "basalt/kernels/cpu/iq_avx2.hpp"
#include "basalt/kernels/cpu/kq_avx2.hpp"
#include "basalt/kernels/cpu/nvfp4_avx512.hpp"
#include "basalt/kernels/cpu/nvfp4_avx2.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/hadamard.hpp"
#include "basalt/kernels/cpu/tq_cpu.hpp"

#include "basalt/ggml/ggml.h"
#include "basalt/ggml/ggml_common_tq.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace basalt::kernels::cpu {
namespace {

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

// The trellis types are not in the vendored ggml: their geometry comes from ggml_common_tq.h, and their rows take
// Q8_0 activations (tq_cpu.hpp).
size_t tq_block_bytes(int type) {
    switch (type) {
        case 144: return sizeof(block_tq2_t);
        case 145: return sizeof(block_tqk6);
        case 146: return sizeof(block_tqk7);
        case 147: return sizeof(block_tqp5);
        case 148: return sizeof(block_tqp6);
        case 149: return sizeof(block_tqp7);
        default: return 0;
    }
}

// a weight type's row bytes, block size and activation type; false when the CPU has no dot product for it
bool weight_geometry(int type, int64_t n, size_t& row, int64_t& blck, int& act) {
    if (tq_supported(type)) {
        row = (size_t) (n / 128) * tq_block_bytes(type);
        blck = 128;
        act = (int) GGML_TYPE_Q8_0;
        return true;
    }
    if (type < 0 || type >= (int) GGML_TYPE_COUNT) return false;
    const ggml_type_traits_cpu* t = traits(type);
    if (t == nullptr || t->vec_dot == nullptr) return false;
    row = ggml_row_size((ggml_type) type, n);
    blck = ggml_blck_size((ggml_type) type);
    act = (int) t->vec_dot_type;
    return true;
}

}  // namespace

bool native_experts_available() noexcept { return true; }

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err) {
    size_t gu_row = 0, d_row = 0;
    int64_t gu_blck = 0, d_blck = 0;
    int gu_act = -1, d_act = -1;
    if (!weight_geometry(gu_type, n_embd, gu_row, gu_blck, gu_act) ||
        !weight_geometry(d_type, n_ff, d_row, d_blck, d_act)) {
        err = "native experts: no CPU dot product for type " +
              std::to_string(weight_geometry(gu_type, n_embd, gu_row, gu_blck, gu_act) ? d_type : gu_type);
        return false;
    }
    const ggml_type_traits_cpu* ag = traits(gu_act);
    const ggml_type_traits_cpu* ad = traits(d_act);
    if (ag == nullptr || ag->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
        err = "native experts: cannot quantize an activation for this layer";
        return false;
    }
    if (n_embd % gu_blck || n_ff % d_blck || n_embd % ggml_blck_size((ggml_type) gu_act) ||
        n_ff % ggml_blck_size((ggml_type) d_act)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = gu_act;
    f.d_act = d_act;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = gu_row;
    f.d_row = d_row;
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = 2 * f.up_off;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    // NVFP4 keeps its per-expert FP32 scales in a tail after down, so they move with the blob through every tier.
    // The scales cover all three projections: the two kinds cannot mix in one expert.
    if ((gu_type == GGML_TYPE_NVFP4) != (d_type == GGML_TYPE_NVFP4)) {
        err = "native experts: NVFP4 gate/up and down come together (the scale tail covers all three)";
        return false;
    }
    f.tail = gu_type == GGML_TYPE_NVFP4 ? kNvfp4Tail : 0;
    f.bytes += f.tail;
    f.act_bytes = ggml_row_size((ggml_type) gu_act, n_embd);
    f.h_bytes = ggml_row_size((ggml_type) d_act, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    if (f.had_gu) {   // Hadamard-folded gate/up: quantize H (s * x), never x itself
        thread_local std::vector<float> tmp;
        tmp.assign(x, x + f.n_embd);
        hadamard_rotate(tmp.data(), (int) f.n_embd, f.had_block, f.gu_signs);
        x = tmp.data();
    }
    static const bool avx512 = cpu_avx512_ok();
    if (f.gu_act == GGML_TYPE_Q8_K) {   // ggml's reference quantizer is scalar
        if (avx512) {
            quantize_q8_K_avx512(x, dst, f.n_embd);
            return;
        }
        quantize_q8_K_avx2(x, dst, f.n_embd);   // the engine runs on AVX2 CPUs at least
        return;
    }
    traits(f.gu_act)->from_float(x, dst, f.n_embd);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    if (f.had_d) {
        thread_local std::vector<float> tmp;
        tmp.assign(h, h + f.n_ff);
        hadamard_rotate(tmp.data(), (int) f.n_ff, f.had_block, f.d_signs);
        h = tmp.data();
    }
    traits(f.d_act)->from_float(h, dst, f.n_ff);
}

void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1) {
    // The multi-token kernels decode the weights once for all tokens; the single-token ggml dot below serves the
    // formats without one.  A token's rows round the same alone as in a group, so greedy output does not depend on
    // the number of drafts in a window.  A format with only an AVX-2 kernel (IQ4_XS) takes it on AVX-2 CPUs
    // only: an AVX-512 CPU keeps the ggml dot for it.
    if (tq_supported(f.gu_type)) {   // one block decode for all the tokens: a token's bits do not depend on the group
        tq_gu_rows(tq_isa(), f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    static const bool avx512 = cpu_avx512_ok();
    static const bool avx2 = cpu_avx2_ok();
    static const bool cpu512 = avx512;
    if ((iq512_supported(f.gu_type) || (!cpu512 && iq256_supported(f.gu_type)))) {
        if (avx512 && iq512_supported(f.gu_type)) {
            iq512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
        if (avx2 && iq256_supported(f.gu_type)) {
            iq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
    }
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    const int n = (int) f.n_embd;
    float sg = 1.f, su = 1.f;
    if (f.tail) {
        std::memcpy(&sg, blob + f.bytes - f.tail, 4);
        std::memcpy(&su, blob + f.bytes - f.tail + 4, 4);
    }
    // NVFP4 on AVX-512: the multi-token kernel, bitwise equal per token to the ggml dot it replaces.
    if (avx512 && f.gu_type == GGML_TYPE_NVFP4 && nvfp4_512_fits((int) f.n_embd, (int) f.n_ff)) {
        nvfp4_512_gu_rows(blob, f.gu_row, f.up_off, n, sg, su, act, nt, ff, r0, r1);
        return;
    }
    // The same on AVX2 CPUs, bitwise equal per token to the AVX2 ggml dot.
    if (!avx512 && avx2 && f.gu_type == GGML_TYPE_NVFP4 && nvfp4_256_fits((int) f.n_embd, (int) f.n_ff)) {
        nvfp4_256_gu_rows(blob, f.gu_row, f.up_off, n, sg, su, act, nt, ff, r0, r1);
        return;
    }
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = blob + f.up_off + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            if (f.tail) {
                g *= sg;
                u *= su;
            }
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    if (tq_supported(f.d_type)) {
        tq_rows(tq_isa(), f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles and absolutises the weights once per
    // block instead of once per token; the ggml dot is single-token.
    if (cpu_avx2_ok() && f.d_type == 20) {   // the same rule as the gate/up rows
        iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    const int n = (int) f.n_ff;
    float sd = 1.f;
    if (f.tail) std::memcpy(&sd, blob + f.bytes - f.tail + 8, 4);
    static const bool avx512 = cpu_avx512_ok();
    if (avx512 && f.d_type == GGML_TYPE_NVFP4 && nvfp4_512_fits((int) f.n_embd, (int) f.n_ff)) {
        nvfp4_512_down_rows(blob + f.down_off, f.d_row, n, sd, hq, nt, out, r0, r1);
        return;
    }
    if (!avx512 && cpu_avx2_ok() && f.d_type == GGML_TYPE_NVFP4 && nvfp4_256_fits((int) f.n_embd, (int) f.n_ff)) {
        nvfp4_256_down_rows(blob + f.down_off, f.d_row, n, sd, hq, nt, out, r0, r1);
        return;
    }
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = f.tail ? s * sd : s;
        }
    }
}

}  // namespace basalt::kernels::cpu
