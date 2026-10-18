// src/kernels/cpu/expert_layout.cpp - the per-layer expert table.  See the header.
#include "basalt/kernels/cpu/expert_layout.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(_MSC_VER)
// MSVC has no cpuid.h: CPUID is the __cpuidex intrinsic (registers in an array, subleaf in ECX) and XGETBV is an
// intrinsic of its own.
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <sstream>

namespace basalt::kernels::cpu {
namespace {
ExpertLayout g_layout;
}

const ExpertLayout& expert_layout() { return g_layout; }

int cpu_isa_cap() { return 3; }   // no cap

bool cpu_avx512_ok() {
    static const bool ok = [] {
        if (cpu_isa_cap() < 3) return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        if (!((r[2] >> 27) & 1u)) return false;             // OSXSAVE
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0xE6) != 0xE6) return false;          // the OS saves the AVX-512 state
        cpuid(7, 0);
        const unsigned ebx = r[1], ecx = r[2];
        return ((ebx >> 16) & 1u) && ((ebx >> 30) & 1u) && ((ebx >> 31) & 1u) && ((ecx >> 11) & 1u) && ((ecx >> 1) & 1u);
    }();
    return ok;
}

bool cpu_avx512bw_ok() {
    static const bool ok = [] {
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        if (!((r[2] >> 27) & 1u)) return false;             // OSXSAVE
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0xE6) != 0xE6) return false;          // the OS saves the AVX-512 state
        cpuid(7, 0);
        const unsigned ebx = r[1];
        return ((ebx >> 16) & 1u) && ((ebx >> 17) & 1u) && ((ebx >> 30) & 1u) && ((ebx >> 31) & 1u);
    }();
    return ok;
}

bool cpu_avx2_ok() {
    static const bool ok = [] {
        if (cpu_isa_cap() < 2) return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        const unsigned ecx1 = r[2];
        // FMA (12), OSXSAVE (27), AVX (28), F16C (29)
        if (!((ecx1 >> 12) & 1u) || !((ecx1 >> 27) & 1u) || !((ecx1 >> 28) & 1u) || !((ecx1 >> 29) & 1u)) return false;
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0x6) != 0x6) return false;            // the OS saves the SSE and AVX state
        cpuid(7, 0);
        return ((r[1] >> 5) & 1u) != 0;                    // AVX2
    }();
    return ok;
}

bool cpu_avx1_ok() {
    // AVX (Sandy Bridge, 2011): AVX + OSXSAVE with the OS saving the YMM state; FMA, F16C and AVX2 not needed.
    // From the upstream fork (rwkeyes).
    static const bool ok = [] {
        if (cpu_isa_cap() < 1) return false;
        unsigned r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
        { int x[4]; __cpuidex(x, 1, 0); for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i]; }
#else
        __cpuid_count(1, 0, r[0], r[1], r[2], r[3]);
#endif
        if (!((r[2] >> 27) & 1u) || !((r[2] >> 28) & 1u)) return false;   // OSXSAVE, AVX
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        return (xcr0 & 0x6) == 0x6;
    }();
    return ok;
}

bool cpu_sse42_ok() {
    static const bool ok = [] {
        unsigned r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
        { int x[4]; __cpuidex(x, 1, 0); for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i]; }
#else
        __cpuid_count(1, 0, r[0], r[1], r[2], r[3]);
#endif
        return ((r[2] >> 20) & 1u) && ((r[2] >> 23) & 1u);   // SSE4.2, POPCNT
    }();
    return ok;
}

std::string cpu_name() {
    unsigned r[12] = {};
    unsigned a = 0, b = 0, c = 0, d = 0;
#if defined(_MSC_VER)
    { int x[4]; __cpuidex(x, (int) 0x80000000u, 0); a = (unsigned) x[0]; b = (unsigned) x[1]; c = (unsigned) x[2]; d = (unsigned) x[3]; }
#else
    __cpuid(0x80000000u, a, b, c, d);
#endif
    if (a < 0x80000004u) return "unknown";
    for (unsigned i = 0; i < 3; ++i) {
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, (int) (0x80000002u + i), 0);
        for (int j = 0; j < 4; ++j) r[i * 4 + j] = (unsigned) x[j];
#else
        __cpuid(0x80000002u + i, r[i * 4], r[i * 4 + 1], r[i * 4 + 2], r[i * 4 + 3]);
#endif
    }
    char s[49] = {};
    std::memcpy(s, r, 48);
    std::string name(s);
    const size_t b0 = name.find_first_not_of(' '), b1 = name.find_last_not_of(' ');
    return b0 == std::string::npos ? std::string("unknown") : name.substr(b0, b1 - b0 + 1);
}

void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1) {
    if (cpu_avx512_ok()) q2_0_gguf_rows_multi(w, row_bytes, nblocks, a, nt, out, r0, r1);
    else q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void act_quant_any(const float* x, int n, ActQ& a) {
    if (cpu_avx512_ok()) act_quant_q8_1(x, n, a);
    else act_quant_q8_1_avx2(x, n, a);
}


bool expert_layout_load_text(const std::string& text, int64_t n_layers, int64_t n_expert, std::string& err) {
    ExpertLayout L;
    L.n_layers = n_layers;
    L.n_expert = n_expert;
    if (text.empty()) {
        err = "the expert layout is empty";
        return false;
    }
    std::istringstream in(text);
    L.native = true;
    L.fmt.resize((size_t) n_layers);
    L.offset.assign((size_t) n_layers, ~0ull);
    L.bytes.assign((size_t) n_layers, 0);
    L.max_blob = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            if (!line.empty() && line[0] == '#') {
                // "# experts vN: ...".  A version this engine does not know may mean columns it would misread,
                // so it is refused rather than parsed as far as the known columns go.
                static const char kHeader[] = "# experts v";
                if (line.compare(0, sizeof kHeader - 1, kHeader) == 0) {
                    L.version = std::atoi(line.c_str() + sizeof kHeader - 1);
                    if (L.version > kExpertLayoutVersion) {
                        err = "the expert layout is v" + std::to_string(L.version) + "; this engine reads up to v" +
                              std::to_string(kExpertLayoutVersion) + " (the model was converted by a newer "
                              "tools/convert/convert.py: update the engine)";
                        return false;
                    }
                }
                // the header records the expert count; a pruned model ships fewer experts than the geometry the
                // caller passes, which is a compile-time default, so the header wins.
                const size_t at = line.find("(n_expert ");
                if (at != std::string::npos) L.n_expert = std::atoll(line.c_str() + at + 10);
            }
            continue;
        }
        std::istringstream ss(line);
        long long l = -1, gt = -1, dt = -1;
        unsigned long long off = 0, blob = 0;
        if (!(ss >> l >> gt >> dt >> off >> blob) || l < 0 || l >= n_layers) {
            err = "expert layout: a malformed line: " + line;
            return false;
        }
        NativeFmt f;
        if (!native_fmt((int) gt, (int) dt, H, FF, f, err)) return false;
        if (f.bytes != blob) {
            err = "expert layout: layer " + std::to_string(l) + " blob is " + std::to_string(blob) +
                  " B but its formats make " + std::to_string(f.bytes);
            return false;
        }
        L.fmt[(size_t) l] = f;
        L.offset[(size_t) l] = off;
        L.bytes[(size_t) l] = blob;
        if (blob > L.max_blob) L.max_blob = blob;
    }
    uint64_t at = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        if (L.offset[(size_t) l] != at) {
            err = "expert layout: layer " + std::to_string(l) + " is missing or not contiguous";
            return false;
        }
        at += L.bytes[(size_t) l] * (uint64_t) L.n_expert;
    }
    L.total = at;
    g_layout = L;
    return true;
}

}  // namespace basalt::kernels::cpu
