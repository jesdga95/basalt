// include/basalt/core/device_copy.hpp - a synchronous copy to the device that has landed when it returns.
//
// Header-only, so the vision libraries and the test programs that do not link `core` use it too.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace basalt::core {

/// Whether any CUDA device of this process is an integrated GPU (cudaDevAttrIntegrated), asked once.
inline bool any_integrated_device() {
    static const bool any = [] {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        for (int d = 0; d < n; ++d) {
            int integrated = 0;
            if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, d) == cudaSuccess && integrated != 0) return true;
            (void) cudaGetLastError();
        }
        return false;
    }();
    return any;
}

/// cudaMemcpy, whose host-to-device copy has landed when this returns.
///
/// cudaMemcpy from pageable host memory may return once the bytes are staged, before they reach the device (the runtime's
/// documentation says so). On a discrete card nothing in the engine has been seen to notice. On the integrated GPU it
/// does, on Windows and on Linux alike: a pageable copy of a table, then a kernel launched on a non-blocking stream,
/// then a copy of one word of that table on the same stream, then that stream synchronized, leaves the OLD word in
/// some 190 of 200 tries - the first copy lands after the last. Synchronizing the default stream after the pageable
/// copy makes it land (0 of 200); so does a copy through a stream of its own. A discrete card takes cudaMemcpy alone,
/// as before; with an integrated GPU in the machine every copy towards a device is followed by that synchronize, which
/// waits for no non-blocking stream.
inline cudaError_t device_copy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    const cudaError_t e = cudaMemcpy(dst, src, count, kind);
    if (e != cudaSuccess || kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice || !any_integrated_device())
        return e;
    return cudaStreamSynchronize(nullptr);   // the default stream: the staged bytes have reached the device
}

}  // namespace basalt::core
