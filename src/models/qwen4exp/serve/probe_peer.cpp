// src/models/qwen4exp/serve/probe_peer.cpp - the peer path `PROBE rates` measures: dec_moe_parity's host-rung rings on a
// small synthetic expert set on the peer card, so no tier's residency or cache is read or written.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "basalt/program/probe_rates.hpp"

#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"

#include <algorithm>
#include <memory>

namespace basalt::models::qwen4exp {

using namespace basalt::models::qwen4exp;
namespace K = basalt::kernels;
namespace C = basalt::kernels::cpu;

namespace {

class DecMoeRings final : public ProbePeerPath {
public:
    static constexpr int kRings = 24, kExperts = 8, kTok = 3;   // entries spread over kTok tokens of a window
    int dev = 0, k = 10, n_expert = 0;
    K::NativeExpertLayout L;
    int blocks_ = 0;
    bool q2 = false;
    DevBuf blob, ptab, lay, scr, arr;
    HostBuf ring_buf, xq, words, rows, diag;
    Stream st;
    size_t xq_stride = 0;

    bool setup(const ProbeEnv& env, const C::NativeFmt& f, std::string& why) override {
        dev = env.peer_device;
        k = std::max(env.top_k, kExperts);
        n_expert = (int) env.n_expert;
        L = K::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff, 0.0f, f.up_type);
        if (!K::dec_moe_formats_ok(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_type)) { why = "format_not_covered"; return false; }
        q2 = f.gu_type == 42;
        core::OnDevice od(dev);
        blocks_ = K::dec_moe_peer_blocks(q2);
        if (blocks_ <= 0) { why = "no_peer_kernel"; return false; }
        const size_t nexp = (size_t) env.n_expert;
        xq_stride = (size_t) f.n_embd / 32 * 36 * kTok + 64;
        std::vector<unsigned long long> pt(nexp, 0ull);
        bool ok = blob.alloc(dev, L.bytes * kExperts) && ptab.alloc(dev, nexp * 8) && lay.alloc(dev, sizeof L) &&
                  scr.alloc(dev, K::dec_moe_peer_scratch_bytes((int) f.n_embd, (int) f.n_ff)) && arr.alloc(dev, 64) &&
                  st.make(dev);
        if (ok) {
            for (int e = 0; e < kExperts && (size_t) e < nexp; ++e)
                pt[(size_t) e] = (unsigned long long) ((uint8_t*) blob.p + (size_t) e * L.bytes);
            ok = cudaMemset(blob.p, 0, L.bytes * kExperts) == cudaSuccess &&
                 cudaMemset(scr.p, 0, K::dec_moe_peer_scratch_bytes((int) f.n_embd, (int) f.n_ff)) == cudaSuccess &&
                 cudaMemcpy(ptab.p, pt.data(), nexp * 8, cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy(lay.p, &L, sizeof L, cudaMemcpyHostToDevice) == cudaSuccess;
        }
        const unsigned fl = cudaHostAllocMapped | cudaHostAllocPortable;
        ok = ok && ring_buf.alloc(sizeof(K::DmRing) * kRings + 256, fl) && xq.alloc(xq_stride * kRings + 256, fl) &&
             words.alloc(256, fl) && rows.alloc((size_t) K::kDmCap * (size_t) f.n_embd * 4 + 256, fl) &&
             diag.alloc(K::kDmDiagWords * 4 + 64, fl);
        if (!ok) { (void) cudaGetLastError(); why = "alloc_failed"; return false; }
        return true;
    }

    int rings() const override { return kRings; }
    int blocks() const override { return blocks_; }

    void fill(int n) override {
        K::DmRing* R = (K::DmRing*) ring_buf.p;
        for (int r = 0; r < kRings; ++r) {
            R[r] = K::DmRing{};
            R[r].n_tok = kTok;
            R[r].n_peer = n;
            for (int j = 0; j < n; ++j) {
                const int i = (j % kTok) * k + j / kTok;
                R[r].ids[i] = j % kExperts;
                R[r].kind[i] = 2;
            }
        }
    }

    // One window of kRings doorbells rung from the host; `before(r)` runs ahead of ring r.  Returns each ring's round
    // trip in us (empty on a hang, after the release).
    bool window(const C::NativeFmt& f, const std::function<void(int)>& before, std::vector<double>& rt,
                std::string& err) override {
        volatile uint32_t* w = (volatile uint32_t*) words.p;
        std::memset(words.p, 0, 256);
        std::memset(diag.p, 0, K::kDmDiagWords * 4);
        {
            core::OnDevice od(dev);
            cudaMemsetAsync(arr.p, 0, 64, st.s);
            K::DmPeerArgs pa;
            pa.rings_m = (const K::DmRing*) ring_buf.p; pa.xq_m = (const uint8_t*) xq.p; pa.xq_stride = xq_stride;
            pa.seq_m = (const volatile uint32_t*) ((uint32_t*) words.p + 16);
            pa.done_m = (uint32_t*) words.p + 32; pa.rows_m = (float*) rows.p;
            pa.ptr = (const unsigned long long*) ptab.p; pa.lay = (const K::NativeExpertLayout*) lay.p;
            pa.n_expert = n_expert;
            pa.k = k; pa.n_embd = (int) f.n_embd; pa.n_ff = (int) f.n_ff; pa.steps = kRings;
            pa.scratch = scr.p; pa.arrive = (unsigned int*) arr.p; pa.diag_m = (uint32_t*) diag.p;
            if (!K::dec_moe_peer_launch(pa, blocks_, st.s, q2)) { err = "peer launch failed"; return false; }
        }
        rt.clear();
        bool ok = true;
        for (int r = 0; r < kRings && ok; ++r) {
            before(r);
            const auto t0 = ProbeClock::now();
            w[16] = ((uint32_t) r + 1) | (uint32_t) kTok << 16 | 1u << 28;
            while (w[32] < (uint32_t) r + 1) {
                if (us_since(t0) > 500000.0) { ok = false; break; }
            }
            rt.push_back(us_since(t0));
        }
        if (!ok) {
            w[16] = UINT32_MAX;
            w[32] = UINT32_MAX;
            err = "peer ring hang: " + K::dec_moe_peer_diag((const uint32_t*) diag.p);
        }
        core::OnDevice od(dev);
        cudaStreamSynchronize(st.s);
        return ok;
    }
};

}  // namespace

std::shared_ptr<ProbePeerPath> make_probe_peer_path() { return std::make_shared<DecMoeRings>(); }

}  // namespace basalt::models::qwen4exp
