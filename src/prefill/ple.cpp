#include "run.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/ple.hpp"
#include "basalt/kernels/native_ple_postops.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/prefill/kernels.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <exception>
#include <string>
#include <vector>
#include <future>

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::ple_gather(int64_t c0, int buf, std::string& e) {
    const int64_t T = chunk_len(c0);
    auto at = [&](int64_t i) { return i < 2 ? prev0[i] : (int32_t) tokens[i - 2]; };   // prev0, then the tokens
    int32_t pv[2] = {at(c0), at(c0 + 1)};
    for (int64_t t = 0; t < T; ++t) {
        const int32_t tok = (int32_t) tokens[c0 + t];
        basalt::kernels::ngram_rows(&tok, pv, 1, ss.ple.consts,
                                    m.ple_rows[buf].data() + t * basalt::kernels::PLE_N_HEADS);
        pv[0] = pv[1];
        pv[1] = tok;
    }
    return ss.ple.table->gather_rows(m.ple_rows[buf].data(), (size_t) T, m.ple_emb_host[buf], e);
}

// the chunk's rows onto the device just before layer 1 reads them, and the next chunk's gather started
bool Prefill::Run::Chunk::ple_land() {
    if (!ple_pending) return true;
    ple_pending = false;
    const auto tp = Clock::now();
    if (!ple_next.get()) {
        err = ple_next_err;
        return false;
    }
    if (cudaMemcpyAsync(m.ple_emb, m.ple_emb_host[ple_buf], (size_t) T * N * 4, cudaMemcpyHostToDevice, m.cs) !=
            cudaSuccess ||
        cudaEventRecord(m.ple_copied[ple_buf], m.cs) != cudaSuccess) {
        err = std::string("prefill: the PLE rows' upload failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    if (c0 + T < n) {
        // the other buffer's upload (a chunk ago) is done before the SSD thread refills it
        if (cudaEventSynchronize(m.ple_copied[ple_buf ^ 1]) != cudaSuccess) {
            err = std::string("prefill: the PLE rows' upload failed: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        ple_next = std::async(std::launch::async, [&run = run, c1 = c0 + T, b = ple_buf ^ 1] {
            return run.ple_gather(c1, b, run.ple_next_err);
        });
    }
    ple_buf ^= 1;
    if (m.warm.pending && m.warm.run_pos0 == pos0) {
        m.warm.pending = false;
        m.warm.fut = std::async(std::launch::async, [&m = m, sess = m.ss] {
            try {
                constexpr size_t kSlice = 2048;   // tokens per request, so the reader is shared with the chunk reads
                const std::vector<int64_t>& w = m.warm.tok;
                std::vector<uint32_t> rows;
                std::string e;
                for (size_t a = 2; a < w.size(); a += kSlice) {
                    const size_t b = std::min(a + kSlice, w.size());
                    rows.resize((b - a) * basalt::kernels::PLE_N_HEADS);
                    for (size_t i = a; i < b; ++i) {
                        const int32_t tok = (int32_t) w[i];
                        const int32_t pv[2] = {(int32_t) w[i - 2], (int32_t) w[i - 1]};
                        basalt::kernels::ngram_rows(&tok, pv, 1, sess->ple.consts,
                                                    rows.data() + (i - a) * basalt::kernels::PLE_N_HEADS);
                    }
                    if (!sess->ple.table->warm_rows(rows.data(), b - a, e)) return;
                }
            } catch (const std::exception&) {}   // only a warm-up: the later read fetches what is missing
        });
    }
    owner.stats_.ms_ple += ms_since(tp);
    return true;
}

bool Prefill::Run::Chunk::ple_layer(int64_t l) {
    if (l == 1 && ple_on && ple_batch) {
        // the whole chunk at once, in sub-batches carved from the idle scratch region: the key and value
        // projections as GEMMs (a token at a time they re-read the whole BF16 key per token), the rest with the per-token kernels' arithmetic (native_ple_postops_batch)
        const auto tp = Clock::now();
        const basalt::kernels::PleWeights& pw = ss.ple.w;
        constexpr int64_t HD = basalt::kernels::NG_HC_DIM;
        const uint64_t per_token = (uint64_t) (3 * HD + N + 4) * 4 + (uint64_t) N * 2 + 4096;
        const int64_t SB = std::min<int64_t>(T, (int64_t) (m.region_bytes / per_token));
        for (int64_t s0 = 0; s0 < T; s0 += SB) {
            const int64_t nb = std::min(SB, T - s0);
            uint8_t* q = m.region;
            auto carve_f = [&](size_t n) { float* p = (float*) q; q += (n * 4 + 255) & ~(size_t) 255; return p; };
            float* key = carve_f((size_t) nb * HD);
            float* qn = carve_f((size_t) nb * HD);
            float* gated = carve_f((size_t) nb * HD);
            float* val = carve_f((size_t) nb * N);
            float* gate = carve_f((size_t) nb * 4);
            uint16_t* e16 = (uint16_t*) carve_f((size_t) nb * N / 2);
            const float* emb = m.ple_emb + s0 * N;
            if (pw.key_bf16 != nullptr) {
                to_bf16(emb, e16, nb * N, m.cs);
                m.gemm.bf16(e16, pw.key_bf16, key, nb, HD, N);
            } else {
                to_f16(emb, e16, nb * N, m.cs);
                m.gemm.native(e16, pw.key_native_type, pw.key_native_data, key, nb, HD, N);
                to_bf16(emb, e16, nb * N, m.cs);
            }
            m.gemm.bf16(e16, pw.value_bf16, val, nb, N, N);
            try {
                basalt::kernels::native_ple_postops_batch(key, m.R + s0 * D, val, ss.ple.hist, pw, qn, gated,
                                                          gate, (int) nb, m.cs);
            } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
        }
        owner.stats_.ms_ple += ms_since(tp);
    } else if (l == 1 && ple_on) {
        const auto tp = Clock::now();
        for (int64_t t = 0; t < T; ++t) {
            basalt::kernels::PleOut po;
            po.normalized = m.ple_norm;
            po.result = m.R + t * D;
            try {
                basalt::kernels::ple_block(m.ple_emb + t * N, m.R + t * D, ss.ple.hist, ss.ple.w, po,
                                           ss.ple.scratch, m.cs);
            } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
            basalt::kernels::ple_history_advance(ss.ple.hist, m.ple_norm, m.cs);
        }
        owner.stats_.ms_ple += ms_since(tp);
    }
    return true;
}

}  // namespace basalt::prefill
