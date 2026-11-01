#include "run.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/core/peer_experts.hpp"
#include <string>
#include <future>

namespace basalt::prefill {
using namespace detail;

int Prefill::Run::Chunk::finish() {
    if (!ple_land()) return -1;   // a stage that ends before layer 1: the rows land anyway, the next gather starts
    if (issuer.joinable()) {
        issuer.join();
        owner.stats_.ms_experts_host += iss_ms;
        owner.stats_.experts_streamed += iss_streamed;
        owner.stats_.experts_dma += iss_dma;
    }
    owner.stats_.tokens += T;
    core::progress_at("reading the prompt (batched): finishing the chunk from token", p0);
    if (owner.next_ != nullptr && owner.keep_last_ && c0 + T >= n) return 0;   // kept: its later layers run on this card next
    if (owner.next_ != nullptr && owner.keep_last_ && owner.keep_here && owner.keep_here(p0 + T)) return 1;   // kept, and the run ends with it
    if (owner.next_ != nullptr) {
        if (T > m.hand_rows || m.hand[owner.hand_buf_] == nullptr) {
            err = "prefill: the layer split's hand-off buffers are too small for the chunk";
            return -1;
        }
        // The current hand-off slot was used two chunks ago.
        float* h = m.hand[owner.hand_buf_];
        if (cudaMemcpyAsync(h, m.R, (size_t) T * D * 4, cudaMemcpyDeviceToHost, m.cs) != cudaSuccess ||
            cudaStreamSynchronize(m.cs) != cudaSuccess) {
            err = std::string("prefill: the layer split's hand-off: ") + cudaGetErrorString(cudaGetLastError());
            return -1;
        }
        // Wait only for the DIRECT successor's previous chunk. That successor
        // may already have forwarded its older chunk to later GPUs.
        if (owner.on_stage_chunk && !owner.on_stage_chunk(p0 + T, err)) return -1;
        if (owner.next_run_.valid() && !owner.next_run_.get()) { err = owner.next_err_; return -1; }
        owner.next_err_.clear();
        owner.next_->hand_in_ = h;
        owner.next_->single_chunk_ = single_chunk;
        owner.next_run_ = std::async(std::launch::async, [&owner = owner, tokens = tokens, c0 = c0, T = T, p0 = p0] {
            return owner.next_->run_impl(tokens + c0, T, p0, owner.next_err_);
        });
        owner.hand_buf_ ^= 1;
        return 0;   // the last stage reports the chunk (on_chunk)
    }
    if (owner.on_chunk || owner.on_stage_chunk) {
        if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
            err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
            return -1;
        }
        if (owner.on_stage_chunk && !owner.on_stage_chunk(p0 + T, err)) return -1;
        if (owner.on_chunk && !owner.on_chunk(m.R, T, p0, err)) return -1;
    }
    return 0;
}

}  // namespace basalt::prefill
