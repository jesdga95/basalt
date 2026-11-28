// include/basalt/kernels/mma_f16.cuh - the FP16 tensor-core inlines the models' attention MMAs share: `ldmatrix` and
// `mma.sync.m16n8k16` with f16 operands, and the BF16 / FP8 E4M3 -> packed FP16 conversions that feed them.  Device
// inlines plus one host helper (the dynamic shared-memory opt-in).  Include from a .cu.
//
// Everything here needs sm_80+; the device inlines are only defined for `__CUDA_ARCH__ >= 800` (or for the host pass
// of a `.cu` compile, so a file's own `#if` guards its calls).  `mma_smem_ready` is the run-time half of that: it
// refuses a device below sm_80, which a build for several architectures may well load a working cubin onto.
//
// FP16 holds far less range than BF16 or FP32, so every conversion from wider input saturates at +-65504, the largest
// finite FP16 value: an `Inf` operand turns a whole MMA tile's accumulator into NaN, while a saturated one only skews
// the few scores that huge value took part in.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace basalt::kernels::mma_f16 {

/// the largest finite FP16 value: wider input is saturated to it before conversion
constexpr float kFp16Max = 65504.0f;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

/// four 8x8 b16 matrices from shared memory into a warp's MMA A registers
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((uint32_t) __cvta_generic_to_shared(p)));
}

/// the same, transposed (the B operand of `row.col`)
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((uint32_t) __cvta_generic_to_shared(p)));
}

/// c += a * b, m16n8k16, f16 operands accumulated in f32
__device__ __forceinline__ void mma16816h(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/// an F32 as FP16, saturated to FP16's finite range so no MMA operand can be +-Inf (and so never NaN)
__device__ __forceinline__ __half f32_to_h_clamped(float v) {
    return __float2half_rn(fminf(fmaxf(v, -kFp16Max), kFp16Max));
}

/// two BF16 (one uint32_t) as packed FP16, saturated to FP16's finite range: BF16 reaches 3.39e38, FP16 only 65504
__device__ __forceinline__ uint32_t bf2_to_h2(uint32_t v) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
    const __half2 h = __float22half2_rn(
        make_float2(fminf(fmaxf(f.x, -kFp16Max), kFp16Max), fminf(fmaxf(f.y, -kFp16Max), kFp16Max)));
    return *reinterpret_cast<const uint32_t*>(&h);
}

/// two E4M3 codes (one uint32_t, the low byte the lower half) times a packed FP16 scale as packed FP16.  E4M3 peaks at
/// 448, so the product only needs the clamp when the scale itself is large.
__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t codes, __half2 scale) {
    const __half2 h = __hmul2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) codes, __NV_E4M3)), scale);
    return *reinterpret_cast<const uint32_t*>(&h);
}

/// the same with an F32 scale, the product taken in F32 and saturated to FP16: a power-of-two scale no smaller than
/// 2^-14 keeps every code in FP16's normal range, a smaller one rounds through FP16 subnormals (which `e4m3x2_to_h2`
/// with a scale already rounded to FP16 loses entirely), and 448 times a large scale saturates instead of overflowing.
__device__ __forceinline__ uint32_t e4m3x2_to_h2_f32scale(uint32_t codes, float scale) {
    const float2 f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) codes, __NV_E4M3)));
    const __half2 h = __float22half2_rn(make_float2(
        fminf(fmaxf(f.x * scale, -kFp16Max), kFp16Max), fminf(fmaxf(f.y * scale, -kFp16Max), kFp16Max)));
    return *reinterpret_cast<const uint32_t*>(&h);
}

#endif  // __CUDA_ARCH__ >= 800

/// Per-device cache for `mma_smem_ready`.  Each kernel instantiation owns its own instance (function pointers to
/// `kernel<true>` and `kernel<false>` share the same C++ type, so a function-local static inside a template on
/// `KernelFn` would alias them).
struct MmaSmemCache {
    static constexpr int kMaxDevs = 32;
    std::atomic<int8_t> state[kMaxDevs]{};   // 0 unqueried, 1 usable, -1 not
};

/// Raises `kernel`'s dynamic shared-memory limit to `smem_bytes` on the current device and reports whether the
/// tensor-core path may run there: false on a device below sm_80, and false on a card that rejects the allocation.
/// The answer is cached per device in `cache`, because a card that rejects the opt-in rejects it every time, and a
/// launch of the fallback kernel must not pay for the query.  A rejected `cudaFuncSetAttribute` leaves the error
/// latched: it is cleared here so the caller's next CUDA call is not blamed for it.
template <typename KernelFn>
inline bool mma_smem_ready(MmaSmemCache& cache, KernelFn kernel, size_t smem_bytes) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) return false;
    const auto cc_ok = [&dev] {
        int cc_major = 0;
        return cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
               cc_major >= 8;
    };
    if (dev >= MmaSmemCache::kMaxDevs) {
        const bool ok = cc_ok() &&
                        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes) ==
                            cudaSuccess;
        if (!ok) cudaGetLastError();
        return ok;
    }
    const int8_t s = cache.state[dev].load(std::memory_order_acquire);
    if (s != 0) return s > 0;
    const bool ok = cc_ok() &&
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes) ==
                        cudaSuccess;
    if (!ok) cudaGetLastError();
    cache.state[dev].store(ok ? 1 : -1, std::memory_order_release);
    return ok;
}

}  // namespace basalt::kernels::mma_f16
