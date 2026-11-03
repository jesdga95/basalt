// src/kernels/cuda/philox.cuh - Philox 4x32-10, the counter-based generator behind every random draw on the GPU.
//
// Counter-based matters because it makes the stream a function of (seed, position) rather than of how many draws came
// before - so a batch can be sampled in any order and a run is reproducible.  `purpose` keys a stream of its own for
// the same (seed, counter): the sampler's pick is purpose 0, and the draws that speculative sampling adds (a drafter's
// drawn draft, the keep test of a draft) take the other purposes, so no two of a request's draws share a number.
#pragma once

#include <cstdint>

namespace basalt::kernels {

enum PhiloxPurpose : uint32_t {
    kPhiloxPick = 0,         // the sampler's draw of a row's token
    kPhiloxDraft = 1,        // a drafter's drawn draft (counter: the draft's position)
    kPhiloxKeep = 2,         // the keep test of a drawn draft (counter: the row's draw index)
};

__device__ __forceinline__ void philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3, uint32_t k0,
                                                 uint32_t k1) {
    const uint32_t hi0 = __umulhi(0x9E3779B9u, c0);
    const uint32_t hi1 = __umulhi(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
}

// A uniform in [0, 1) from (seed, counter) on the stream `purpose`.
__device__ __forceinline__ float philox_uniform(uint64_t seed, uint64_t counter, uint32_t purpose = kPhiloxPick) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        philox4x32_round(c0, c1, c2, c3, (uint32_t) i, purpose);
    }
    // 24 bits of mantissa, so the value is uniform in [0,1) with no rounding to 1.0
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

}  // namespace basalt::kernels
