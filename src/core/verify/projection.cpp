#include "common.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include <cmath>
#include <mutex>
#include <set>
#include <string>
#include <tuple>

namespace basalt::core::verify_detail {

void note_fallback(const char* what, int a, int b) {
    static std::mutex mu;
    static std::set<std::tuple<std::string, int, int>> seen;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert({what, a, b}).second)
        basalt::log::warn("basalt verify", "fallback: %s (types %d / %d)", what, a, b);
}
// a dense projection of the window: dense_tc where dense_tc_mask has the projection's bit and the type, else
// native_mmvq (either one for every window width, so a column's value never depends on T)
void dense_proj(int bit, int type, const void* w, const void* xq, float* y, int n_in, int n_out, int n, void* s) {
    if ((kernels::dense_tc_mask() & bit) && kernels::dense_tc_supported(type, n_in, n_out, w) &&
        kernels::dense_tc_type_on(type))
        kernels::dense_tc(type, w, xq, y, n_in, n_out, n, s);
    else {
        if (kernels::dense_tc_mask() & bit) note_fallback("dense_tc projection on native_mmvq", bit, type);
        kernels::native_mmvq(type, w, xq, y, n_in, n_out, n, s);
    }
}

// Memory bandwidth of a card from its attributes, bytes per second.
double card_bandwidth(int dev) {
    int khz = 0, bits = 0;
    if (cudaDeviceGetAttribute(&khz, cudaDevAttrMemoryClockRate, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&bits, cudaDevAttrGlobalMemoryBusWidth, dev) != cudaSuccess) {
        cudaGetLastError();
        return 0.0;
    }
    return 2.0 * khz * 1e3 * bits / 8.0;
}

}  // namespace basalt::core::verify_detail
