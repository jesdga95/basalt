// include/basalt/kernels/expert_rows.hpp - the activation rows the tensor-core expert kernels (kernels/cuda/expert_tile.cuh)
// read: K values as f16 or bf16, each 16-value group stored as the pairs (v_j, v_{j+8}), j = 0..7, so that a pair of
// values a thread multiplies sits in one word beside the NVFP4 code byte holding the same two positions.
#pragma once

#if defined(__CUDACC__)
#define BASALT_XH_HD __host__ __device__
#else
#define BASALT_XH_HD
#endif

namespace basalt::kernels::xt {

/// Where value i of an activation row sits.
BASALT_XH_HD constexpr int xh_pos(int i) { return (i & ~15) | ((i & 7) << 1) | ((i >> 3) & 1); }

}  // namespace basalt::kernels::xt
