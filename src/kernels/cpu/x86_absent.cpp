// src/kernels/cpu/x86_absent.cpp - the x86 kernels' names on a build that has none of them (Arm64).
//
// The shared dispatch (native_expert.cpp, expert_layout.cpp) names the AVX2 / AVX-512 kernels behind run-time probes.
// On Arm64 every probe says no (cpu_avx2_ok and cpu_avx512_ok read CPUID, which answers zero), so no kernel here is
// ever reached: the "supported" and "fits" questions answer false, and a kernel that is called anyway stops the engine
// by name rather than computing something. This keeps the dispatch files the x86 build's text.
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/iq_avx2.hpp"
#include "basalt/kernels/cpu/iq_avx512.hpp"
#include "basalt/kernels/cpu/kq_avx2.hpp"
#include "basalt/kernels/cpu/nvfp4_avx2.hpp"
#include "basalt/kernels/cpu/nvfp4_avx512.hpp"
#include "basalt/log.hpp"

#include <cstdlib>

namespace basalt::kernels::cpu {
namespace {

[[noreturn]] void absent(const char* name) {
    basalt::log::error("basalt", "the x86 CPU kernel %s was called on an Arm64 build", name);
    std::abort();
}

}  // namespace

// ---- the questions: nothing x86 is supported ----
bool iq512_supported(int) noexcept { return false; }
bool iq256_supported(int) noexcept { return false; }
bool iq_exact_512_supported(int) noexcept { return false; }
bool iq1s_512_fits(int) noexcept { return false; }
bool iq4nl256_fits(int) noexcept { return false; }
bool nvfp4_512_fits(int, int) noexcept { return false; }
bool nvfp4_256_fits(int, int) noexcept { return false; }
bool kq256_supported(int) noexcept { return false; }

// ---- the kernels: never reached ----
void quantize_q8_K_avx512(const float*, void*, int64_t) { absent("quantize_q8_K_avx512"); }
void quantize_q8_K_avx2(const float*, void*, int64_t) { absent("quantize_q8_K_avx2"); }

void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq512_gu_rows");
}
void iq512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { absent("iq512_rows"); }
void iq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq256_gu_rows");
}
void iq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { absent("iq256_rows"); }
void iq1s_512_gu_rows(const uint8_t*, size_t, size_t, int, float, float, bool, float, const void* const*, int, float* const*,
                      int, int) {
    absent("iq1s_512_gu_rows");
}
void iq1s_512_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { absent("iq1s_512_rows"); }
void iq_exact_512_gu_rows(int, const uint8_t*, size_t, size_t, int, float, float, bool, float, const void* const*, int,
                          float* const*, int, int) {
    absent("iq_exact_512_gu_rows");
}
void iq_exact_512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq_exact_512_rows");
}
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq4nl256_down_rows");
}
void nvfp4_512_gu_rows(const uint8_t*, size_t, size_t, int, float, float, const SwiGlu&, const void* const*, int, float* const*,
                       int, int) {
    absent("nvfp4_512_gu_rows");
}
void nvfp4_512_down_rows(const uint8_t*, size_t, int, float, const void* const*, int, float* const*, int, int) {
    absent("nvfp4_512_down_rows");
}
void nvfp4_256_gu_rows(const uint8_t*, size_t, size_t, int, float, float, const SwiGlu&, const void* const*, int, float* const*,
                       int, int) {
    absent("nvfp4_256_gu_rows");
}
void nvfp4_256_down_rows(const uint8_t*, size_t, int, float, const void* const*, int, float* const*, int, int) {
    absent("nvfp4_256_down_rows");
}

void kq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("kq256_gu_rows");
}
void kq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { absent("kq256_rows"); }

void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    absent("q2_0_gguf_rows_multi");
}
void q2_0_gguf_rows_multi_avx2(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    absent("q2_0_gguf_rows_multi_avx2");
}
void act_quant_q8_1(const float*, int, ActQ&) { absent("act_quant_q8_1"); }
void act_quant_q8_1_avx2(const float*, int, ActQ&) { absent("act_quant_q8_1_avx2"); }

}  // namespace basalt::kernels::cpu
