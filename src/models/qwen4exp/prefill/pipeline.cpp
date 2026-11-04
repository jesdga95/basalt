#include "state.hpp"
#include "layout.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/peer_experts.hpp"
#include <string>

namespace basalt::prefill {
using namespace detail;

float* Prefill::hand_buffer(int b) const { return b == 0 || b == 1 ? impl_->hand[b] : nullptr; }

int64_t Prefill::hand_rows() const { return impl_->hand[0] != nullptr && impl_->hand[1] != nullptr ? impl_->hand_rows : 0; }

bool Prefill::reserve_hand_off(int64_t rows, std::string& err) {
    Impl& m = *impl_;
    if (m.hand[0] != nullptr && m.hand[1] != nullptr && m.hand_rows >= rows) return true;
    for (int b = 0; b < 2; ++b) {
        if (m.hand[b] != nullptr) cudaFreeHost(m.hand[b]);
        m.hand[b] = nullptr;
    }
    m.hand_rows = 0;
    for (int b = 0; b < 2; ++b)
        if (cudaHostAlloc((void**) &m.hand[b], (size_t) rows * D * 4, cudaHostAllocPortable) != cudaSuccess) {
            (void) cudaGetLastError();
            err = "prefill: the layer split's hand-off buffers";
            return false;
        }
    m.hand_rows = rows;
    return true;
}

bool Prefill::drain_pipeline(std::string& err) {
    bool ok = true;

    if (next_run_.valid()) {
        if (!next_run_.get()) {
            err = next_err_;
            ok = false;
        }
    }

    if (next_ != nullptr) {
        std::string tail_err;
        if (!next_->drain_pipeline(tail_err)) {
            if (ok) err = tail_err;
            ok = false;
        }
    }

    return ok;
}

bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    const bool body_ok = run_impl(tokens, n, pos0, err);
    // the tokens the warm-up reads belong to this request: wait for it once the run that reads them is over
    impl_->warm.pending = false;
    if (impl_->warm.fut.valid() && (!body_ok || pos0 + n >= impl_->warm.to)) impl_->warm.fut.wait();

    std::string drain_err;
    const bool drain_ok = drain_pipeline(drain_err);

    if (!body_ok) return false;
    if (!drain_ok) {
        err = drain_err;
        return false;
    }
    return true;
}

}  // namespace basalt::prefill
