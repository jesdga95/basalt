// include/basalt/core/region_copy.hpp - a running state's device regions packed into one device buffer and back, in a
// few launches instead of a copy per region (a checkpoint of a model with hundreds of regions).
#pragma once

#include "basalt/core/checkpoint_regions.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace basalt::core {

/// The regions' bytes back to back into `dst` (device memory), queued on `stream`.
cudaError_t regions_gather(const std::vector<DeviceRegion>& regions, uint8_t* dst, cudaStream_t stream);
/// The bytes back to back in `src` (device memory) into the regions, queued on `stream`.
cudaError_t regions_scatter(const std::vector<DeviceRegion>& regions, const uint8_t* src, cudaStream_t stream);

}  // namespace basalt::core
