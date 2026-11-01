#include "run.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/core/peer_experts.hpp"
#include <algorithm>
#include <string>

namespace basalt::prefill {
using namespace detail;

bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    err.clear();
    Run run(*this, tokens, n, pos0, err);
    return run.execute();
}

bool Prefill::Run::execute() {
    // The direct successor's future lives on the Prefill object. Intermediate
    // stages therefore do not drain the complete remaining GPU chain here.
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    owner.rows_in_place_ = false;
    if (m.hist_on && !rows_in_place) m.hist_tokens += n;   // a kept chunk's tokens counted in the run that kept it
    // the MMQ row table lives in the borrowed cache slots, which the refill after a prompt overwrites with experts:
    // write it again for every prompt (a layout is reused as long as the chunk and the slots are the same)
    if (m.ids_identity != nullptr) mmq::iota(m.ids_identity, m.T * K, m.cs);
    // The PLE rows of a chunk are read from the model file on the host (an SSD read per missed row): the chunk
    // after this one is read on a thread while the GPU runs this one, into the other of two buffers.  The rows
    // depend only on the tokens (the two before a position name its n-grams), so this is the same data.
    for (int64_t c0 = 0; c0 < n; c0 += chunk_len(c0)) {
        if (owner.should_stop && owner.should_stop()) { err = "cancelled"; return false; }
        Chunk chunk(*this, c0);
        if (!chunk.execute()) return false;
        const int flow = chunk.finish();
        if (flow < 0) return false;
        if (flow > 0) break;
    }
    return finish();
}

bool Prefill::Run::finish() {
    // Do not drain the successor here. This is the overlap: an intermediate
    // stage can return while later GPUs are still processing the previous chunk.
    ss.ple_prev[0] = prev[0];
    ss.ple_prev[1] = prev[1];
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    // an expert copy that failed on the copy stream surfaces here, not in the next request
    if (const cudaError_t cst = cudaStreamSynchronize(m.copy); cst != cudaSuccess) {
        err = std::string("prefill: expert copy stream: ") + cudaGetErrorString(cst);
        return false;
    }
    owner.stats_.ms_total += ms_since(t_start);
    return true;
}

bool Prefill::Run::Chunk::execute() {
    core::progress_at("reading the prompt (batched): preparing the chunk from token", p0);
    ++owner.stats_.chunks;
    if (!inputs()) return false;
    if (!start_stream()) return false;
    m.ahead_layer[0] = m.ahead_layer[1] = m.guess_layer = -1;
    for (int64_t l = LB; l < LE; ++l) {
        if (!layer(l)) return false;
    }
    return true;
}

}  // namespace basalt::prefill
