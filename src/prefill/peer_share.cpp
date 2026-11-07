// src/prefill/peer_share.cpp - multi-GPU: a second card's share of a prompt chunk's routed experts.
//
// Per layer the main card quantizes the activations into mapped host memory. A layer's experts are staged in groups of
// at most launch_experts, a new one also where the caller starts one (group i in staging buffer i % stage_bufs), each group's
// rows run in launches of at most launch_rows rows (launch j in buffer set j % bufs), and four streams carry them:
//   stage    (second card): a group's blobs into its staging buffer (copies device-to-device or over its own link, or
//                           the caller's staging); the first groups of a layer whose experts are known early start before
//                           its routing (`prepare`);
//   products (second card): gather the launch's rows out of mapped host memory, gate/up, the caller's step, down (or the
//                           caller's products);
//   out      (second card): copy the result rows into mapped host memory;
//   rows     (main card):   a kernel on a high-priority stream reads them into the caller's block. Neither the activations
//                           nor the rows queue behind the expert copies on a copy engine.
// Events order the reuse of each buffer, so no host thread waits; the host's per-layer tables alternate between two sets,
// which the caller's next-but-one layer may rewrite only after `join` (each layer starts after the main card's stream
// reached it).
#include "basalt/prefill/peer_share.hpp"

#include "basalt/core/on_device.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace basalt::prefill {
namespace {

// The main card quantizes a layer's activations this many rows at a time into its own memory and its copy engine moves
// them to the host: a kernel writing to mapped host memory would hold the main card's SMs for the whole transfer.
constexpr int64_t kQuantRows = 2048;

int64_t max_launches(const PeerShare::Dims& d) {
    // a launch closes at its rows, at its experts, or splits an expert: at most one launch more per expert
    return (d.max_rows + d.launch_rows - 1) / d.launch_rows + d.n_expert + 1;
}

// The groups of `experts`: [first, end) runs of at most `per` experts, a new one also at an expert that starts one.
std::vector<std::pair<size_t, size_t>> cut_groups(const std::vector<PeerShare::Expert>& experts, int per) {
    std::vector<std::pair<size_t, size_t>> g;
    for (size_t i = 0; i < experts.size(); ++i)
        if (g.empty() || experts[i].group || i - g.back().first == (size_t) per) g.push_back({i, i + 1});
        else g.back().second = i + 1;
    return g;
}

}  // namespace

struct PeerShare::Impl {
    int main = -1, peer = -1;
    Dims d;
    std::vector<void*> dev_peer, host;
    // main card
    cudaStream_t s_main = nullptr, s_rows = nullptr;
    cudaEvent_t ev_x = nullptr, ev_xq = nullptr, ev_end = nullptr, ev_rows[kMaxBufs] = {};
    // second card
    cudaStream_t s_stage = nullptr, s_prod = nullptr, s_out = nullptr;
    cudaEvent_t ev_xin = nullptr, ev_staged[kMaxBufs] = {}, ev_free[kMaxBufs] = {}, ev_done[kMaxBufs] = {}, ev_out[kMaxBufs] = {};
    uint8_t* stage[kMaxBufs] = {};
    void* xq[kMaxBufs] = {};
    void* hq = nullptr;
    float *gu = nullptr, *h = nullptr, *out[kMaxBufs] = {};
    int32_t *ident = nullptr, *tok_dev = nullptr, *bounds_dev = nullptr;
    std::unique_ptr<mmq::Context> ctx;
    // pinned, portable; the activations mapped too
    void* hxq = nullptr;
    void* dxq = nullptr;                    // main card: kQuantRows rows of q8_1 on their way to hxq
    float* hrows[kMaxBufs] = {};
    int32_t *tok_h[2] = {}, *bounds_h[2] = {};
    int set = 0;
    uint64_t launch = 0, group = 0;
    cudaError_t err = cudaSuccess;          // the worker's last error, read after `drain`
    // groups of the next run staged by `prepare`: how many, the first one's number, its blobs
    int prepared = 0;
    uint64_t prepared_at = 0;
    std::vector<const uint8_t*> prepared_blobs;

    // The second card's calls are issued by a thread of its own: when its queues are full a call waits, and the main card's
    // thread must not.
    std::thread worker;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::function<void()>> jobs;
    bool busy = false, quit = false;

    void post(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(mu);
            jobs.push_back(std::move(f));
        }
        cv.notify_all();
    }
    void drain() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return jobs.empty() && !busy; });
    }
    void work() {
        cudaSetDevice(peer);
        for (;;) {
            std::function<void()> f;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || !jobs.empty(); });
                if (jobs.empty()) return;
                f = std::move(jobs.front());
                jobs.pop_front();
                busy = true;
            }
            f();
            {
                std::lock_guard<std::mutex> lk(mu);
                busy = false;
            }
            cv.notify_all();
        }
    }

    // Queues the copies of `blobs` into the staging buffer of group number `gi`, once the last group there is done.
    void stage_group(const Layer& L, const std::vector<std::pair<const uint8_t*, bool>>& blobs, uint64_t gi) {
        const int b = (int) (gi % (uint64_t) d.stage_bufs);
        const core::OnDevice od(peer);
        cudaStreamWaitEvent(s_stage, ev_free[b], 0);
        if (L.stage) L.stage(blobs, stage[b], s_stage);
        else
            for (size_t i = 0; i < blobs.size(); ++i)
                cudaMemcpyAsync(stage[b] + L.stride * i, blobs[i].first, L.blob_bytes,
                                blobs[i].second ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, s_stage);
        cudaEventRecord(ev_staged[b], s_stage);
    }

    // The MMQ products of one launch, its rows gathered out of the layer's q8_1 activations.
    void products(const Layer& L, const Launch& l, int b, cudaStream_t ps) {
        mmq::gather_q8(l.act, l.n_tokens, l.tok, xq[b], l.rows, d.n_embd, ps);
        mmq::Product p;
        p.w = l.stage + L.gu_stride * (size_t) l.q0;
        p.type = L.gu_type;
        p.w_rows = 2 * d.n_ff;
        p.w_cols = d.n_embd;
        p.expert_bytes = L.gu_stride;
        p.n = l.n;
        p.xq = xq[b];
        p.bounds = l.bounds;
        p.ids = ident;
        p.total_rows = l.rows;
        p.max_rows = l.max_rows;
        p.dst = gu;
        p.ld_dst = p.w_rows;
        if (L.up_type >= 0 && L.up_type != L.gu_type) {   // gate rows, then up rows of their own type
            p.w_rows = d.n_ff;
            ctx->run(p, ps);
            p.w = (const uint8_t*) p.w + L.up_at;
            p.type = L.up_type;
            p.dst = gu + d.n_ff;
        }
        ctx->run(p, ps);
        L.act(gu, h, l, ps);
        mmq::quantize(h, nullptr, hq, L.d_type, d.n_ff, d.n_ff, l.rows, ps);
        cudaStreamWaitEvent(ps, ev_out[b], 0);   // the last rows out of this buffer have left
        mmq::Product q = p;
        q.w = l.stage + L.down_off + L.d_stride * (size_t) l.q0;
        q.type = L.d_type;
        q.w_rows = d.n_embd;
        q.w_cols = d.n_ff;
        q.expert_bytes = L.d_stride;
        q.xq = hq;
        q.dst = l.out;
        q.ld_dst = d.n_embd;
        ctx->run(q, ps);
    }

    ~Impl() {
        if (worker.joinable()) {
            {
                std::lock_guard<std::mutex> lk(mu);
                quit = true;
            }
            cv.notify_all();
            worker.join();
        }
        if (peer >= 0) {
            const core::OnDevice od(peer);
            for (cudaStream_t s : {s_stage, s_prod, s_out})
                if (s) {
                    cudaStreamSynchronize(s);
                    cudaStreamDestroy(s);
                }
            if (ev_xin) cudaEventDestroy(ev_xin);
            for (int b = 0; b < kMaxBufs; ++b)
                for (cudaEvent_t e : {ev_staged[b], ev_free[b], ev_done[b], ev_out[b]})
                    if (e) cudaEventDestroy(e);
            ctx.reset();
            for (void* p : dev_peer) cudaFree(p);
        }
        if (main >= 0) {
            const core::OnDevice od(main);
            for (cudaStream_t s : {s_main, s_rows})
                if (s) {
                    cudaStreamSynchronize(s);
                    cudaStreamDestroy(s);
                }
            for (cudaEvent_t e : {ev_x, ev_xq, ev_end})
                if (e) cudaEventDestroy(e);
            for (cudaEvent_t e : ev_rows)
                if (e) cudaEventDestroy(e);
            if (dxq) cudaFree(dxq);
        }
        for (void* p : host) cudaFreeHost(p);
    }
};

PeerShare::PeerShare() = default;
PeerShare::~PeerShare() = default;

// The buffers in one place, so that the sizes `open` takes and the ones the plans count are the same.
template <class Peer, class Host>
static void layout(const PeerShare::Dims& d, Peer&& peer, Host&& host) {
    const int64_t R = d.launch_rows, G = max_launches(d);
    host(d.act_bytes);                                               // hxq
    for (int b = 0; b < d.stage_bufs; ++b) peer(d.group_bytes);     // stage
    for (int b = 0; b < d.bufs; ++b) {
        peer(mmq::q8_bytes(R, d.n_embd));                            // xq
        peer((size_t) R * (size_t) d.n_embd * 4);                    // out
        host((size_t) R * (size_t) d.n_embd * 4);                    // hrows
    }
    peer((size_t) R * (size_t) d.n_ff * 2 * 4);                      // gu
    peer((size_t) R * (size_t) d.n_ff * 4);                          // h
    peer(mmq::q8_bytes(R, d.n_ff));                                  // hq
    peer((size_t) R * 4);                                            // ident
    peer((size_t) d.max_rows * 4);                                   // tok
    peer((size_t) G * (size_t) (d.launch_experts + 1) * 4);          // bounds
    for (int s = 0; s < 2; ++s) {
        host((size_t) d.max_rows * 4);                               // tok
        host((size_t) G * (size_t) (d.launch_experts + 1) * 4);      // bounds
    }
}

uint64_t PeerShare::peer_bytes(const Dims& d) {
    uint64_t sum = 0;
    layout(d, [&](size_t b) { sum += (b + 255) / 256 * 256; }, [](size_t) {});
    return sum;
}

uint64_t PeerShare::host_bytes(const Dims& d) {
    uint64_t sum = 0;
    layout(d, [](size_t) {}, [&](size_t b) { sum += b; });
    return sum;
}

bool PeerShare::open(int main_device, int peer_device, const Dims& d, std::string& err) {
    if (d.bufs < 2 || d.bufs > kMaxBufs || d.stage_bufs < 1 || d.stage_bufs > kMaxBufs) {
        err = "peer share: 2 or 3 launch buffers, 1 to 3 staging buffers";
        return false;
    }
    auto m = std::make_unique<Impl>();
    m->d = d;
    bool ok = true;
    std::vector<void*> pv, hv;
    {
        m->peer = peer_device;
        const core::OnDevice od(peer_device);
        layout(d, [&](size_t b) {
            void* p = nullptr;
            if (ok && cudaMalloc(&p, b) == cudaSuccess) pv.push_back(p), m->dev_peer.push_back(p);
            else ok = false;
        }, [](size_t) {});
        ok = ok && cudaStreamCreateWithFlags(&m->s_stage, cudaStreamNonBlocking) == cudaSuccess &&
             cudaStreamCreateWithFlags(&m->s_prod, cudaStreamNonBlocking) == cudaSuccess &&
             cudaStreamCreateWithFlags(&m->s_out, cudaStreamNonBlocking) == cudaSuccess &&
             cudaEventCreateWithFlags(&m->ev_xin, cudaEventDisableTiming) == cudaSuccess;
        for (int b = 0; ok && b < kMaxBufs; ++b)
            for (cudaEvent_t* e : {&m->ev_staged[b], &m->ev_free[b], &m->ev_done[b], &m->ev_out[b]})
                ok = ok && cudaEventCreateWithFlags(e, cudaEventDisableTiming) == cudaSuccess;
        if (ok) {
            size_t i = 0;
            for (int b = 0; b < d.stage_bufs; ++b) m->stage[b] = (uint8_t*) pv[i++];
            for (int b = 0; b < d.bufs; ++b) {
                m->xq[b] = pv[i++];
                m->out[b] = (float*) pv[i++];
            }
            m->gu = (float*) pv[i++];
            m->h = (float*) pv[i++];
            m->hq = pv[i++];
            m->ident = (int32_t*) pv[i++];
            m->tok_dev = (int32_t*) pv[i++];
            m->bounds_dev = (int32_t*) pv[i++];
            m->ctx = std::make_unique<mmq::Context>();
            mmq::iota(m->ident, d.launch_rows, m->s_prod);
            ok = cudaStreamSynchronize(m->s_prod) == cudaSuccess;
        }
    }
    {
        m->main = main_device;
        const core::OnDevice od(main_device);
        int lo = 0, hi = 0;
        cudaDeviceGetStreamPriorityRange(&lo, &hi);
        ok = ok && cudaStreamCreateWithFlags(&m->s_main, cudaStreamNonBlocking) == cudaSuccess &&
             cudaStreamCreateWithPriority(&m->s_rows, cudaStreamNonBlocking, hi) == cudaSuccess &&
             cudaEventCreateWithFlags(&m->ev_x, cudaEventDisableTiming) == cudaSuccess &&
             cudaEventCreateWithFlags(&m->ev_xq, cudaEventDisableTiming) == cudaSuccess &&
             cudaEventCreateWithFlags(&m->ev_end, cudaEventDisableTiming) == cudaSuccess;
        for (int b = 0; ok && b < kMaxBufs; ++b) ok = cudaEventCreateWithFlags(&m->ev_rows[b], cudaEventDisableTiming) == cudaSuccess;
        ok = ok && cudaMalloc(&m->dxq, mmq::q8_bytes(kQuantRows, d.n_embd)) == cudaSuccess;
        layout(d, [](size_t) {}, [&](size_t b) {
            void* p = nullptr;
            if (ok && cudaHostAlloc(&p, b, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess)
                hv.push_back(p), m->host.push_back(p);
            else ok = false;
        });
        if (ok) {
            size_t i = 0;
            m->hxq = hv[i++];
            for (int b = 0; b < d.bufs; ++b) m->hrows[b] = (float*) hv[i++];
            for (int s = 0; s < 2; ++s) {
                m->tok_h[s] = (int32_t*) hv[i++];
                m->bounds_h[s] = (int32_t*) hv[i++];
            }
        }
    }
    if (!ok) {
        (void) cudaGetLastError();
        err = "out of memory for the second card's prompt share";
        return false;
    }
    Impl* w = m.get();
    m->worker = std::thread([w] { w->work(); });
    m_ = std::move(m);
    return true;
}

void PeerShare::prepare(const Layer& L, const std::vector<Expert>& experts) {
    Impl& m = *m_;
    m.prepared = 0;
    m.prepared_at = m.group;
    m.prepared_blobs.clear();
    const std::vector<std::pair<size_t, size_t>> cut = cut_groups(experts, m.d.launch_experts);
    for (size_t k = 0; k < cut.size() && m.prepared < m.d.stage_bufs; ++k, ++m.prepared) {
        const auto [i0, i1] = cut[k];
        std::vector<std::pair<const uint8_t*, bool>> blobs;
        for (size_t i = i0; i < i1; ++i) {
            blobs.emplace_back(experts[i].blob, experts[i].on_peer);
            m.prepared_blobs.push_back(experts[i].blob);
        }
        const uint64_t gi = m.group++;
        Impl* w = &m;
        m.post([w, L, blobs = std::move(blobs), gi] { w->stage_group(L, blobs, gi); });
    }
}

bool PeerShare::run(const Layer& L, const float* x, int n_tokens, const std::vector<Expert>& experts, float* dst,
                    void* main_stream, std::string& err) {
    Impl& m = *m_;
    const Dims& d = m.d;
    const int prepared = m.prepared;
    m.prepared = 0;
    if ((!L.stage && (L.blob_bytes > L.stride || L.stride * (size_t) d.launch_experts > d.group_bytes)) ||
        n_tokens > d.max_tokens) {
        err = "peer share: a layer does not fit the share's buffers";
        return false;
    }
    for (size_t i = 0; i < m.prepared_blobs.size() && prepared > 0; ++i)
        if (i >= experts.size() || experts[i].blob != m.prepared_blobs[i]) {
            err = "peer share: the experts differ from the prepared ones";
            return false;
        }
    // the groups and their launches: this layer's table set, rows launch by launch in expert order
    const int s = m.set;
    m.set ^= 1;
    int32_t* tok = m.tok_h[s];
    int32_t* bounds = m.bounds_h[s];
    struct Cut {
        int64_t r0 = 0, rows = 0, max_rows = 0;
        int b0 = 0, q0 = 0, n = 0;                    // its bounds in the table; its first expert in the group and how many
    };
    struct Group {
        size_t i0 = 0, i1 = 0;
        std::vector<Cut> launches;
    };
    std::vector<Group> groups;
    int64_t r = 0, nb = 0;
    const int64_t max_b = max_launches(d) * (d.launch_experts + 1);
    for (const auto& [i0, i1] : cut_groups(experts, d.launch_experts)) {
        Group g;
        g.i0 = i0;
        g.i1 = i1;
        // the group's rows, cut into launches of at most launch_rows rows
        Cut l;
        int k0 = 0;
        for (size_t i = g.i0; i < g.i1;) {
            const Expert& e = experts[i];
            if (l.n == 0) {
                if (nb + d.launch_experts + 1 > max_b) {
                    err = "peer share: too many launches";
                    return false;
                }
                l.r0 = r;
                l.b0 = (int) nb;
                l.q0 = (int) (i - g.i0);
                bounds[nb++] = 0;
            }
            const int take = (int) std::min<int64_t>(e.n - k0, d.launch_rows - l.rows);
            if (r + take > d.max_rows) {
                err = "peer share: more rows than the chunk has";
                return false;
            }
            std::copy_n(e.tok + k0, take, tok + r);
            r += take;
            l.rows += take;
            l.max_rows = std::max<int64_t>(l.max_rows, take);
            ++l.n;
            bounds[nb++] = (int32_t) l.rows;
            k0 += take;
            if (k0 == e.n) {
                ++i;
                k0 = 0;
            }
            if (l.rows == d.launch_rows || i == g.i1) {   // the launch is full, or the group ends
                if (l.rows > 0) g.launches.push_back(l);
                else nb = l.b0;
                l = Cut{};
            }
        }
        groups.push_back(std::move(g));
    }
    if (r > 0) {   // the layer's activations into mapped host memory, once the second card has read the last layer's
        const core::OnDevice od(m.main);
        cudaEventRecord(m.ev_x, (cudaStream_t) main_stream);
        cudaStreamWaitEvent(m.s_main, m.ev_x, 0);
        cudaStreamWaitEvent(m.s_rows, m.ev_x, 0);
        cudaStreamWaitEvent(m.s_main, m.ev_xin, 0);
        if (L.quantize) {
            L.quantize(x, n_tokens, m.hxq, m.s_main);
        } else {   // block column kb of row r lies at (kb n_tokens + r) blocks: a piece's rows are a strided copy
            const size_t kb = (size_t) mmq::q8_blocks(d.n_embd);
            const size_t bb = (mmq::q8_bytes(1, d.n_embd) - mmq::q8_bytes(0, d.n_embd)) / kb;   // bytes of one block
            for (int64_t r0 = 0; r0 < n_tokens; r0 += kQuantRows) {
                const int64_t nr = std::min<int64_t>(kQuantRows, n_tokens - r0);
                mmq::quantize(x + (size_t) r0 * (size_t) d.n_embd, nullptr, m.dxq, L.gu_type, d.n_embd, d.n_embd, nr,
                              m.s_main);
                cudaMemcpy2DAsync((uint8_t*) m.hxq + (size_t) r0 * bb, (size_t) n_tokens * bb, m.dxq, (size_t) nr * bb,
                                  (size_t) nr * bb, kb, cudaMemcpyDeviceToHost, m.s_main);
            }
        }
        cudaEventRecord(m.ev_xq, m.s_main);
    }
    // the group numbers: the prepared ones, then new ones
    std::vector<uint64_t> nums;
    std::vector<std::vector<std::pair<const uint8_t*, bool>>> blobs(groups.size());
    std::vector<const uint8_t*> all_blobs;
    for (const Expert& e : experts) all_blobs.push_back(e.blob);
    for (size_t gi = 0; gi < groups.size(); ++gi) {
        nums.push_back((int) gi < prepared ? m.prepared_at + gi : m.group++);
        if ((int) gi >= prepared)
            for (size_t i = groups[gi].i0; i < groups[gi].i1; ++i) blobs[gi].emplace_back(experts[i].blob, experts[i].on_peer);
    }
    Impl* w = &m;
    m.post([w, L, n_tokens, dst, r, nb, tok, bounds, prepared, groups = std::move(groups), nums = std::move(nums),
            blobs = std::move(blobs), all_blobs = std::move(all_blobs)] {
        Impl& m = *w;
        const Dims& d = m.d;
        const cudaStream_t mr = m.s_rows, ps = m.s_prod, so = m.s_out;
        if (r > 0) {
            const core::OnDevice od(m.peer);
            cudaMemcpyAsync(m.tok_dev, tok, (size_t) r * 4, cudaMemcpyHostToDevice, ps);
            cudaMemcpyAsync(m.bounds_dev, bounds, (size_t) nb * 4, cudaMemcpyHostToDevice, ps);
            cudaStreamWaitEvent(ps, m.ev_xq, 0);
        }
        bool first = true;
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            const Group& g = groups[gi];
            if ((int) gi >= prepared) m.stage_group(L, blobs[gi], nums[gi]);
            const int sb = (int) (nums[gi] % (uint64_t) d.stage_bufs);
            const core::OnDevice od(m.peer);
            if (!g.launches.empty()) cudaStreamWaitEvent(ps, m.ev_staged[sb], 0);
            for (const Cut& l : g.launches) {
                const int b = (int) (m.launch++ % (uint64_t) d.bufs);
                Launch lv;
                lv.stage = m.stage[sb];
                lv.q0 = l.q0;
                lv.n = l.n;
                lv.blob = all_blobs.data() + g.i0 + l.q0;
                lv.tok = m.tok_dev + l.r0;
                lv.bounds = m.bounds_dev + l.b0;
                lv.bounds_host = bounds + l.b0;
                lv.rows = l.rows;
                lv.max_rows = l.max_rows;
                lv.act = m.hxq;
                lv.n_tokens = n_tokens;
                lv.first = first;
                lv.out = m.out[b];
                first = false;
                if (L.products) {
                    cudaStreamWaitEvent(ps, m.ev_out[b], 0);   // the last rows out of this buffer have left
                    L.products(lv, ps);
                } else {
                    m.products(L, lv, b, ps);
                }
                cudaEventRecord(m.ev_done[b], ps);
                // the rows out, once the main card has copied the last ones from there
                const size_t bytes = (size_t) l.rows * (size_t) d.n_embd * 4;
                cudaStreamWaitEvent(so, m.ev_done[b], 0);
                cudaStreamWaitEvent(so, m.ev_rows[b], 0);
                cudaMemcpyAsync(m.hrows[b], m.out[b], bytes, cudaMemcpyDeviceToHost, so);
                cudaEventRecord(m.ev_out[b], so);
                const core::OnDevice om(m.main);
                cudaStreamWaitEvent(mr, m.ev_out[b], 0);
                kernels::copy_from_mapped(dst + (size_t) l.r0 * (size_t) d.n_embd, m.hrows[b], (int64_t) (bytes / 4), mr);
                cudaEventRecord(m.ev_rows[b], mr);
            }
            // the staging buffer is free once the group's products are done (or at once, without rows)
            if (g.launches.empty()) cudaEventRecord(m.ev_free[sb], m.s_stage);
            else cudaEventRecord(m.ev_free[sb], ps);
        }
        {   // the activations' host buffer is free once the layer's gathers are done
            const core::OnDevice op(m.peer);
            cudaEventRecord(m.ev_xin, ps);
        }
        const core::OnDevice od(m.main);
        cudaEventRecord(m.ev_end, mr);
        m.err = cudaGetLastError();
    });
    return true;
}

bool PeerShare::join(void* stream, std::string& err) {
    m_->drain();
    if (m_->err != cudaSuccess) {
        err = std::string("peer share: ") + cudaGetErrorString(m_->err);
        return false;
    }
    const core::OnDevice od(m_->main);
    cudaStreamWaitEvent((cudaStream_t) stream, m_->ev_end, 0);
    return true;
}

}  // namespace basalt::prefill
