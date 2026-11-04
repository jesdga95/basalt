// Arithmetic adapted from the MIT-licensed pinned ggml CUDA
// moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "basalt/kernels/native_moe.hpp"
#include "basalt/kernels/dec_moe.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/core/diag.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace basalt::kernels {
namespace {
__global__ void combine(const float* __restrict__ parts, const float* __restrict__ weights,
                        const float* __restrict__ shared, float* __restrict__ output,
                        int64_t n_embd, int k) {
    // blockIdx.y = the token of a multi-token launch (0 for the single one)
    const int64_t tk = blockIdx.y;
    parts += tk * k * n_embd; weights += tk * k; if (shared) shared += tk * n_embd; output += tk * n_embd;
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    float sum = parts[col] * weights[0];
    for (int expert = 1; expert < k; ++expert) {
        sum += parts[int64_t(expert) * n_embd + col] * weights[expert];
    }
    if (shared) sum += shared[col];
    output[col] = sum;
}
bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes)
            || (shared && !valid_span(shared, row_bytes))
            || overlap(output, row_bytes, parts, part_bytes)
            || overlap(output, row_bytes, weights, weight_bytes)
            || (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");
    combine<<<unsigned((n_embd + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        parts, weights, shared, output, n_embd, int(k));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (multi) requires a stream, width, 1..15 experts, tokens");
    combine<<<dim3(unsigned((n_embd + 255) / 256), (unsigned) n_tok), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        parts, weights, shared, output, n_embd, int(k));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// DEC_MOE: combine's sum with each entry's row read from its tier (dec_moe.hpp), after the tiers this layer
// uses have landed.  A VRAM row is +0.0 + row as the old path's hit add made it; the sum is combine's.
namespace {
// The shared expert's per-token gate as its own kernel applied it: a separate IEEE product, never fused into the sum.
__device__ __forceinline__ float sh_mul(float a, float b) {
    float d;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}
__global__ void dm_combine_kernel(const float* __restrict__ parts, const float* peer_rows, const float* cpu_rows,
                                  const DmPlan* __restrict__ plan, const float* __restrict__ weights,
                                  const float* __restrict__ shared, float* __restrict__ output, int64_t n_embd, int k,
                                  const volatile uint32_t* peer_done, const volatile uint32_t* host_flag, uint32_t ring,
                                  const float* __restrict__ sh_gate) {
    __shared__ int s_kind[16];
    const int64_t tk = blockIdx.y;
    if (threadIdx.x == 0 && peer_done != nullptr) {
        if (plan->counts[2] > 0) while (*peer_done < ring) __nanosleep(100);
        if (plan->counts[3] > 0) while (*host_flag < ring) __nanosleep(100);
        __threadfence_system();
    }
    if (threadIdx.x < k) s_kind[threadIdx.x] = plan->kind[tk * k + threadIdx.x];
    __syncthreads();
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    weights += tk * k;
    auto row = [&](int j) -> float {
        const int64_t off = (tk * k + j) * n_embd + col;
        const int kd = s_kind[j];
        if ((unsigned) kd <= 1u) return __fadd_rn(0.0f, parts[off]);   // CUDA0's own and its staged share
        return __ldcv(kd >= 2 ? peer_rows + off : cpu_rows + off);
    };
    float sum = row(0) * weights[0];
    for (int expert = 1; expert < k; ++expert) {
        sum += row(expert) * weights[expert];
    }
    if (shared) sum += sh_gate ? sh_mul(shared[tk * n_embd + col], sh_gate[tk]) : shared[tk * n_embd + col];
    output[tk * n_embd + col] = sum;
}
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release_gpu(unsigned* p, unsigned v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// DEC_GLUE & 4: the same sums, four columns a thread with every row loaded before the first add; DEP: a
// programmatic dependent of dec_moe_wait - CUDA0's rows, the weights and the shared row (all final once the wait
// kernel runs) are loaded before pdl_wait, only the mapped tiers' rows after it.  TAG: the peers' rows carry their
// own arrival (a dword still holding the tag word is not yet written).  WAITC (with TAG, not DEP): no wait kernel runs
// before this one; when the layer has CPU entries one thread of the launch waits for the CPU pool's word (cpu_done)
// and passes the ring on through cpu_seen (device), and the others read the tiers' rows once it has.
template<int KM, bool DEP, bool TAG = false, bool WAITC = false>
__global__ void dm_combine4_kernel(const float* __restrict__ parts, const float* peer_rows, const float* cpu_rows,
                                   const DmPlan* __restrict__ plan, const float* __restrict__ weights,
                                   const float* __restrict__ shared,
                                   float* __restrict__ output, int64_t n_embd, int k,
                                   const volatile uint32_t* tag_done = nullptr, uint32_t ring = 0,
                                   const volatile uint32_t* cpu_done = nullptr, unsigned* cpu_seen = nullptr,
                                   const float* __restrict__ sh_gate = nullptr) {
    static_assert(!WAITC || (TAG && !DEP), "the CPU pool's word is waited for inside a plain launch");
    __shared__ int s_kind[16];
    // the next kernel (the hc read) may start its blocks and stage weights while this one waits
    if (DEP || WAITC) pdl_launch_dependents();
    const int64_t tk = blockIdx.y;
    if (threadIdx.x < k) s_kind[threadIdx.x] = plan->kind[tk * k + threadIdx.x];
    const int64_t col = 4 * (int64_t(blockIdx.x) * blockDim.x + threadIdx.x);
    const bool live = col < n_embd;
    weights += tk * k;
    float4 v[KM];
    float w[KM];
    // every entry's CUDA0 row and weight, the shared row and its gate load in one round with the kinds (a tier entry's
    // CUDA0 row is read and not used), so no load waits for an earlier one to arrive
#pragma unroll
    for (int j = 0; j < KM; ++j) {
        if (live && j < k) {
            w[j] = weights[j];
            v[j] = *reinterpret_cast<const float4*>(parts + (tk * k + j) * n_embd + col);
        }
    }
    float4 h = shared && live ? *reinterpret_cast<const float4*>(shared + tk * n_embd + col)
                              : make_float4(0.f, 0.f, 0.f, 0.f);
    const float gt = shared && sh_gate ? sh_gate[tk] : 1.0f;
    __syncthreads();
    if (!live) return;   // (thread 0 of block (0, 0), WAITC's waiter, has column 0)
    // the mapped rows are read in place; a peer row's own words say it arrived, so its first read needs no wait
    auto mapped_row = [&](int j, int kd) {
        return reinterpret_cast<const float4*>((kd >= 2 ? peer_rows : cpu_rows) + (tk * k + j) * n_embd + col);
    };
#pragma unroll
    for (int j = 0; j < KM; ++j) {
        if (j >= k) break;
        if ((unsigned) s_kind[j] <= 1u)   // CUDA0's own and its staged share
            v[j] = make_float4(__fadd_rn(0.0f, v[j].x), __fadd_rn(0.0f, v[j].y), __fadd_rn(0.0f, v[j].z),
                               __fadd_rn(0.0f, v[j].w));
        else if (TAG && s_kind[j] >= 2)
            v[j] = __ldcv(mapped_row(j, 2));
    }
    if (shared && sh_gate)   // the shared expert's gate, as its own kernel applied it
        h = make_float4(sh_mul(h.x, gt), sh_mul(h.y, gt), sh_mul(h.z, gt), sh_mul(h.w, gt));
    if constexpr (DEP) pdl_wait();
    bool pmark = false;
    if constexpr (WAITC) {
        // as the wait kernel: every row after the pool's word when the layer has CPU entries (the peers' rows are not
        // polled while the pool reads its experts).  cpu_seen only ever steps through a window's rings in order, so
        // holding this ring means this ring's word was seen.
        const bool cpu = plan->counts[3] > 0;
        pmark = !cpu && plan->counts[2] > 0;   // only the peers' rows to wait for
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
            if (cpu) {
                st_release_gpu(cpu_seen, ring | 0x80000000u);   // waiting for the pool
                while (*cpu_done < ring) __nanosleep(64);
                __threadfence_system();
            }
            st_release_gpu(cpu_seen, pmark ? ring | 0x40000000u : ring);   // (pmark: waiting for the peers' rows)
        }
        if (cpu)
            while (ld_acquire_gpu(cpu_seen) != ring) __nanosleep(32);
    }
    // every other mapped row's load is issued before the first one is used
#pragma unroll
    for (int j = 0; j < KM; ++j) {
        if (j >= k) break;
        const int kd = s_kind[j];
        if ((unsigned) kd > 1u && !(TAG && kd >= 2)) v[j] = __ldcv(mapped_row(j, kd));
    }
    if constexpr (TAG) {
        // a peer row still holding the tag word is read again until none is; a released window (tag_done past ring)
        // ends the poll after one more read, with whatever the rows hold.  After a row that had to be waited for, the
        // later peer rows still waiting are read again together, so their landing costs one round trip, not one each.
        constexpr unsigned tag = 0x7fa5a5a5u;
        auto waiting = [&](const float4& x) {
            return __float_as_uint(x.x) == tag || __float_as_uint(x.y) == tag || __float_as_uint(x.z) == tag ||
                   __float_as_uint(x.w) == tag;
        };
#pragma unroll
        for (int j = 0; j < KM; ++j) {
            if (j >= k) break;
            if (s_kind[j] < 2) continue;
            float4* const src = const_cast<float4*>(mapped_row(j, 2));
            unsigned spins = 0;
            bool rel = false;
            while (waiting(v[j])) {
                if (rel) break;
                if ((++spins & 63u) == 0u && *tag_done >= ring) { __threadfence_system(); rel = true; }
                v[j] = __ldcv(src);
            }
            if (spins > 0) {
#pragma unroll
                for (int i = j + 1; i < KM; ++i) {
                    if (i >= k) break;
                    if (s_kind[i] >= 2 && waiting(v[i])) v[i] = __ldcv(mapped_row(i, 2));
                }
            }
            __stcs(src, make_float4(__uint_as_float(tag), __uint_as_float(tag), __uint_as_float(tag), __uint_as_float(tag)));
        }
        if (pmark && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) st_release_gpu(cpu_seen, ring);   // its rows are in
    }
    float sx = v[0].x * w[0], sy = v[0].y * w[0], sz = v[0].z * w[0], sw = v[0].w * w[0];
#pragma unroll
    for (int j = 1; j < KM; ++j) {
        if (j >= k) break;
        sx += v[j].x * w[j];
        sy += v[j].y * w[j];
        sz += v[j].z * w[j];
        sw += v[j].w * w[j];
    }
    if (shared) { sx += h.x; sy += h.y; sz += h.z; sw += h.w; }
    *reinterpret_cast<float4*>(output + tk * n_embd + col) = make_float4(sx, sy, sz, sw);
}
constexpr bool dm_combine4_on() { return true; }
}  // namespace

void dec_moe_combine(const float* parts, const float* peer_rows_m, const float* cpu_rows_m, const DmPlan* plan,
                     const float* weights, const float* shared, float* output, int64_t n_embd, int64_t k, int n_tok,
                     const volatile uint32_t* peer_done_m, const volatile uint32_t* host_flag_m, uint32_t ring,
                     void* stream, bool dep, const volatile uint32_t* tag_done, const volatile uint32_t* cpu_done,
                     unsigned* cpu_seen, const float* shared_gate) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1 ||
        (cpu_done != nullptr && (dep || tag_done == nullptr || cpu_seen == nullptr)))
        throw std::invalid_argument("dec_moe_combine requires a stream, width, 1..15 experts, tokens");
    if (dm_combine4_on() && peer_done_m == nullptr && n_embd % 4 == 0 && k <= 10) {
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(unsigned((n_embd / 4 + 127) / 128), (unsigned) n_tok);
        cfg.blockDim = dim3(128);
        cfg.stream = static_cast<cudaStream_t>(stream);
        cfg.attrs = at;
        cfg.numAttrs = dep ? 1 : 0;
        const int ki = int(k);
        const volatile uint32_t* none = nullptr;
        unsigned* nseen = nullptr;
        if (cpu_done != nullptr)
            (void) cudaLaunchKernelEx(&cfg, dm_combine4_kernel<10, false, true, true>, parts, peer_rows_m, cpu_rows_m, plan,
                                      weights, shared, output, n_embd, ki, tag_done, ring, cpu_done, cpu_seen, shared_gate);
        else if (dep && tag_done != nullptr)
            (void) cudaLaunchKernelEx(&cfg, dm_combine4_kernel<10, true, true>, parts, peer_rows_m, cpu_rows_m, plan, weights,
                                      shared, output, n_embd, ki, tag_done, ring, none, nseen, shared_gate);
        else if (dep)
            (void) cudaLaunchKernelEx(&cfg, dm_combine4_kernel<10, true>, parts, peer_rows_m, cpu_rows_m, plan, weights,
                                      shared, output, n_embd, ki, none, 0u, none, nseen, shared_gate);
        else
            (void) cudaLaunchKernelEx(&cfg, dm_combine4_kernel<10, false>, parts, peer_rows_m, cpu_rows_m, plan, weights,
                                      shared, output, n_embd, ki, none, 0u, none, nseen, shared_gate);
        const auto error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        return;
    }
    dm_combine_kernel<<<dim3(unsigned((n_embd + 255) / 256), (unsigned) n_tok), 256, 0,
                        static_cast<cudaStream_t>(stream)>>>(parts, peer_rows_m, cpu_rows_m, plan, weights, shared,
                                                             output, n_embd, int(k), peer_done_m, host_flag_m, ring,
                                                             shared_gate);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
