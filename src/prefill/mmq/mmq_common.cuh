// SPDX-License-Identifier: MIT
// What the MMQ kernels need from ggml-cuda's common.cuh (llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d,
// https://github.com/ggml-org/llama.cpp): the error and shared-memory helpers, fast integer division, vector copy,
// the dp4a wrapper, programmatic dependent launch and the device and pool state.  Copyright (c) 2023-2026 The ggml
// authors.  MIT licence: see THIRD_PARTY_LICENSES.
#pragma once

#include "basalt/ggml/ggml.h"
#include "basalt/log.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "basalt/ggml/ggml_common.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
#error "the MMQ kernels are built for compute capability 12.0 only"
#endif

#define WARP_SIZE 32
#define MATRIX_ROW_PADDING 512  // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses
#define GGML_CUDA_MAX_DEVICES 16

#define GGML_UNUSED(x) (void) (x)
#define GGML_UNUSED_VARS(...) do { (void) sizeof((__VA_ARGS__, 0)); } while (0)
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
#define GGML_ABORT(...) do { basalt::log::error("basalt", __VA_ARGS__); std::abort(); } while (0)
#define GGML_ASSERT(x) if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg);

#define CUDA_CHECK(err)                                                                   \
    do {                                                                                  \
        auto err_ = (err);                                                                \
        if (err_ != cudaSuccess) {                                                        \
            ggml_cuda_error(#err, __func__, __FILE__, __LINE__, cudaGetErrorString(err_)); \
        }                                                                                 \
    } while (0)

int ggml_cuda_get_device();

#define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes)                                                       \
    do {                                                                                                   \
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = { false };                         \
        const int   id                                                = ggml_cuda_get_device();            \
        if (!shared_memory_limit_raised[id]) {                                                             \
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes)); \
            shared_memory_limit_raised[id] = true;                                                         \
        }                                                                                                  \
    } while (0)

static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
    return 32;
}

// Maximum number of bytes that can be copied in a single instruction.
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
    return 16;
}

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
    return __dp4a(a, b, c);
}

// Programmatic dependent launch: a kernel waits for its predecessor, or signals that the next may start.
static __device__ __forceinline__ void ggml_cuda_pdl_sync() {
    cudaGridDependencySynchronize();
}

static __device__ __forceinline__ void ggml_cuda_pdl_lc() {
    cudaTriggerProgrammaticLaunchCompletion();
}

// Kernels that use programmatic dependent launch take no __restrict__ pointers.
#define GGML_CUDA_RESTRICT

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
static const uint3 init_fastdiv_values(uint64_t d_64) {
    GGML_ASSERT(d_64 != 0);
    GGML_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());

    uint32_t d = (uint32_t)d_64;

    // compute L = ceil(log2(d));
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    // pack divisor as well to reduce error surface
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z>
    // fastdiv_values.z is unused and optimized away by the compiler.
    // Compute high 32 bits of n * mp
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    // add n, apply bit shift
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

// Calculate both division and modulo at once, returns <n/divisor, n%divisor>
static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z;
    return make_uint2(div_val, mod_val);
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
// Important: do not use this function if dst and src both point at registers.
//     Due to the strict aliasing rule the compiler can do incorrect optimizations if src and dst have different types.
//     The function is intended for copies between registers and SRAM/VRAM to make the compiler emit the right instructions.
//     If dst and src point at different address spaces then they are guaranteed to not be aliased.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(
        nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
        "You are misusing the alignment parameter for ggml_cuda_memcpy_1. "
        "The intent is for the parameter is only as a workaround if either one of the pointers is not properly aligned. "
        "If you use it to do more bytes per copy than ggml_cuda_max_cpy_bytes() the reads and writes may not be coalesced. "
        "Call ggml_cuda_memcpy_1 in a loop instead.");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

template <ggml_type type>
struct ggml_cuda_type_traits;

#define MMQ_TYPE_TRAITS(T, QK, QR, QI, BLOCK)       \
    template <>                                     \
    struct ggml_cuda_type_traits<T> {               \
        static constexpr int qk = QK;               \
        static constexpr int qr = QR;               \
        static constexpr int qi = QI;               \
        static constexpr int bs = sizeof(BLOCK);    \
    }

MMQ_TYPE_TRAITS(GGML_TYPE_Q2_0, QK2_0, QR2_0, QI2_0, block_q2_0);
MMQ_TYPE_TRAITS(GGML_TYPE_Q8_0, QK8_0, QR8_0, QI8_0, block_q8_0);
MMQ_TYPE_TRAITS(GGML_TYPE_Q5_1, QK5_1, QR5_1, QI5_1, block_q5_1);
MMQ_TYPE_TRAITS(GGML_TYPE_Q4_K, QK_K, QR4_K, QI4_K, block_q4_K);
MMQ_TYPE_TRAITS(GGML_TYPE_Q5_K, QK_K, QR5_K, QI5_K, block_q5_K);
MMQ_TYPE_TRAITS(GGML_TYPE_Q6_K, QK_K, QR6_K, QI6_K, block_q6_K);
MMQ_TYPE_TRAITS(GGML_TYPE_IQ2_XXS, QK_K, QR2_XXS, QI2_XXS, block_iq2_xxs);
MMQ_TYPE_TRAITS(GGML_TYPE_IQ2_XS, QK_K, QR2_XS, QI2_XS, block_iq2_xs);
MMQ_TYPE_TRAITS(GGML_TYPE_IQ2_S, QK_K, QR2_S, QI2_S, block_iq2_s);
MMQ_TYPE_TRAITS(GGML_TYPE_IQ3_XXS, QK_K, QR3_XXS, QI3_XXS, block_iq3_xxs);
MMQ_TYPE_TRAITS(GGML_TYPE_IQ3_S, QK_K, QR3_S, QI3_S, block_iq3_s);
MMQ_TYPE_TRAITS(GGML_TYPE_IQ4_NL, QK4_NL, QR4_NL, QI4_NL, block_iq4_nl);
MMQ_TYPE_TRAITS(GGML_TYPE_IQ4_XS, QK_K, QR4_XS, QI4_XS, block_iq4_xs);
MMQ_TYPE_TRAITS(GGML_TYPE_NVFP4, QK_NVFP4, QR_NVFP4, QI_NVFP4, block_nvfp4);

#undef MMQ_TYPE_TRAITS

// The device properties the MMQ launch code reads.
struct ggml_cuda_device_info {
    int device_count;

    struct cuda_device_info {
        int    nsm;        // number of streaming multiprocessors
        size_t smpbo;      // max. shared memory per block (with opt-in)
        int    warp_size;
    };

    cuda_device_info devices[GGML_CUDA_MAX_DEVICES] = {};
};

const ggml_cuda_device_info & ggml_cuda_info();

struct ggml_cuda_pool {
    virtual ~ggml_cuda_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
struct ggml_cuda_pool_alloc {
    ggml_cuda_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_cuda_pool_alloc() = default;

    explicit ggml_cuda_pool_alloc(ggml_cuda_pool & pool) : pool(&pool) {
    }

    ggml_cuda_pool_alloc(ggml_cuda_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_cuda_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_cuda_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() {
        return ptr;
    }

    ggml_cuda_pool_alloc(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc(ggml_cuda_pool_alloc &&) = delete;
    ggml_cuda_pool_alloc& operator=(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc& operator=(ggml_cuda_pool_alloc &&) = delete;
};

// Scratch memory for the launches of one device.
struct ggml_backend_cuda_context {
    int device;
    std::unique_ptr<ggml_cuda_pool> pools[GGML_CUDA_MAX_DEVICES];

    explicit ggml_backend_cuda_context(int device) : device(device) {}
    ~ggml_backend_cuda_context();

    static std::unique_ptr<ggml_cuda_pool> new_pool_for_device(int device);

    ggml_cuda_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(device);
        }
        return *pools[device];
    }
    ggml_cuda_pool & pool() { return pool(device); }
};
