// src/prefill/prefill.cpp - see include/basalt/prefill/prefill.hpp.
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/on_device.hpp"

#include "basalt/core/layout.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/native_qsa_indexer.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/ple.hpp"
#include "basalt/kernels/native_ple_postops.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/kv_stream.hpp"
#include "basalt/kernels/cvec.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"
#include "basalt/kernels/qsa_prompt_attn.hpp"
#include "basalt/kernels/qsa_select.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/moe_fused.hpp"
#include "basalt/prefill/moe_fused_iq.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/core/peer_experts.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>


namespace basalt::prefill {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float EPS = 1e-6f;
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10, NE = 512;
constexpr int64_t C = 10240, ZV = 6144, HV = 48;
// staging holds the largest blob of the pack (a native pack's blobs differ per layer)
inline int64_t MAXBLOB() { return (int64_t) basalt::kernels::cpu::expert_layout().max_blob; }
constexpr int STAGE = 8;           // host->device expert staging ring (chunks below stream_all_min())
// From the chunk size stream_all_min() on, every non-resident expert of every layer streams in a fixed order through a
// ring_slots()-slot ring (nearly every expert is routed at such a chunk), so the copy engine keeps working through the
// attention halves instead of waiting for each layer's routing.
constexpr int RING_MAX = 256;           // the arrays; the ring itself is ring_slots()
// The chunk size from which every expert streams: below it only the routed experts are staged.  Staging every
// non-resident expert costs the same whatever the chunk, so it pays only once most of them are routed anyway.
int64_t g_stream_all_min = 6144;
inline int64_t stream_all_min() { return g_stream_all_min; }
// The streamed ring's slots (one whole expert blob each): the most a layout holds (the same for every pack), and the
// depth the next layout is laid out with (a prompt read beside decoding slots lends fewer cache slots with a shallower
// ring).
constexpr int kPrefillRing = 256;
int g_ring_depth = kPrefillRing;
// Whether the fused experts' buffers apply: a streamed chunk of a pack the fused kernels cover.
inline bool fused_ring() {
    if (!fused::enabled()) return false;
    if (core::prompt_peer()) return false;   // multi-GPU: a prompt peer keeps the MMQ path and its buffer sizes
    const basalt::kernels::cpu::ExpertLayout& lay = basalt::kernels::cpu::expert_layout();
    if (!lay.native) return true;
    // EVERY layer: fused_layout() shrinks the MoE buffers to the fused path's needs, so a layer the native kernels do
    // not cover would run MMQ in them at the full chunk and overflow them.  A pack with such a layer keeps MMQ's
    // buffers; its covered layers still take the fused kernels.
    static const bool all = [&lay] {
        for (const auto& f : lay.fmt)
            if (!fused::native_supported(f.gu_type, f.d_type)) return false;
        return !lay.fmt.empty();
    }();
    return all;
}
inline int ring_slots(size_t T) { return (int64_t) T >= stream_all_min() ? g_ring_depth : STAGE; }
constexpr int DQ = 2;              // dequantized-expert ring (FP16 gate/up + down)
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Either cudaMalloc (owned, freed with the object) or a bump allocation from a borrowed region; with no base and
// no region it only counts, which is how `bytes_needed` sizes the region.
struct Alloc {
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
    bool count_only = false;
    std::vector<void*>* owned = nullptr;
    template <typename T> T* take(size_t n, bool& ok) {
        const uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        if (count_only) { used += bytes; return nullptr; }
        if (base != nullptr) {
            if (used + bytes > cap) { ok = false; return nullptr; }
            T* p = (T*) (base + used);
            used += bytes;
            return p;
        }
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; return nullptr; }
        owned->push_back(p);
        used += bytes;
        return (T*) p;
    }
};

}  // namespace

// The experts the arena could not pin are copied into pinned buffers by these threads, ahead of the launches.  Copied
// in line by the launching thread they would leave the GPU without queued work while each memcpy runs.  Job j - a layer's
// j-th unpinned expert, in launch order - lands in host buffer j % kRing, which is free again once the DMA of job
// j - kRing (recorded by the launching thread, `issued`) is done.
struct Stager {
    // the pinned ring's depth: how far the host copies can run ahead of the DMAs of the unpinned experts' blobs
    int kRing = 32;
    // how many jobs ahead of the one being copied the OS is asked to read from the file
    static constexpr size_t kAhead = 64;
    // a job whose bytes are only in the model file (fd >= 0) is read into the buffer with pread at `off`; the others
    // are copied from `src`
    struct Job { const uint8_t* src; size_t bytes; int fd = -1; uint64_t off = 0; };
    // reader threads for the file-backed jobs: each blocks in its read, so the threads are the reads in flight
    static constexpr int kIoThreads = 16;
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    std::vector<cudaEvent_t> dma_done;
    std::vector<Job> jobs;
    std::unique_ptr<std::atomic<int>[]> ready;
    size_t ready_cap = 0;
    // gen << 32 | n << 16 | next index: a claim is a CAS on the generation it woke for (a thread late from the
    // previous layer can never take a job of this one)
    std::atomic<uint64_t> head{0};
    std::atomic<int> issued{0}, active{0};
    uint32_t gen = 0;
    bool quit = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int device = 0;

    bool init(size_t blob_bytes, int nthreads) {
        buf.assign((size_t) kRing, nullptr);
        pinned.assign((size_t) kRing, 0);
        dma_done.assign((size_t) kRing, nullptr);
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            pinned[i] = cudaHostAlloc((void**) &buf[i], blob_bytes, cudaHostAllocDefault) == cudaSuccess;
            if (!pinned[i]) {
                cudaGetLastError();
                pageable[(size_t) i].resize(blob_bytes);
                buf[i] = pageable[(size_t) i].data();
            }
            if (cudaEventCreateWithFlags(&dma_done[i], cudaEventDisableTiming) != cudaSuccess) return false;
        }
        cudaGetDevice(&device);
        for (int t = 0; t < nthreads; ++t) threads.emplace_back([this] { work(); });
        return true;
    }
    ~Stager() {
        finish();
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto& t : threads) t.join();
        for (int i = 0; i < kRing; ++i) {
            if (dma_done[i]) cudaEventDestroy(dma_done[i]);
            if (buf[i] && pinned[i]) cudaFreeHost(buf[i]);
        }
    }
    void work() {
        cudaSetDevice(device);
        uint32_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            for (;;) {
                active.fetch_add(1, std::memory_order_acq_rel);
                const int j = claim(seen);
                if (j < 0) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                const int b = j % kRing;
                if (j >= kRing)   // job j - kRing's DMA from this buffer is queued
                    while (issued.load(std::memory_order_acquire) <= j - kRing) std::this_thread::yield();
                // and done - for a generation's first kRing jobs that is the previous generation's last DMA from
                // the buffer, which nothing else waits for when a chunk ends without a sync (no MTP) or the DMA
                // was a ring entry the routing skipped (an event never recorded returns at once)
                cudaEventSynchronize(dma_done[b]);
                const Job& jb = jobs[(size_t) j];
                if (jb.fd >= 0) {
                    if (!read_file(jb, buf[b])) std::memcpy(buf[b], jb.src, jb.bytes);   // the mapping as the fallback
                } else {
                    const size_t a = (size_t) j + kAhead;
                    if (a < jobs.size() && jobs[a].fd < 0) basalt::platform::advise_willneed(jobs[a].src, jobs[a].bytes);
                    std::memcpy(buf[b], jb.src, jb.bytes);
                }
                ready[(size_t) j].store(1, std::memory_order_release);
                active.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }
    static bool read_file(const Job& jb, uint8_t* dst) {
        size_t done = 0;
        while (done < jb.bytes) {
            // One chunk per call, so a count short of the rest is normal and this loop picks up where it stopped.
            const int64_t n = basalt::platform::pread(jb.fd, dst + done, jb.bytes - done, jb.off + done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            done += (size_t) n;
        }
        return true;
    }
    int claim(uint32_t g) {
        uint64_t cur = head.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (cur >> 32) != g) return -1;
            const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
            if (j >= n) return -1;
            if (head.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_acquire)) return j;
        }
    }
    /// A layer's jobs; the previous layer's are finished (finish()).
    void start(std::vector<Job>&& js) {
        if (js.empty()) return;
        std::lock_guard<std::mutex> lk(mu);
        jobs = std::move(js);
        if (ready_cap < jobs.size()) {
            ready_cap = jobs.size() * 2;
            ready.reset(new std::atomic<int>[ready_cap]);
        }
        for (size_t i = 0; i < jobs.size(); ++i) ready[i].store(0, std::memory_order_relaxed);
        bool any_file = false;
        for (size_t i = 0; i < jobs.size(); ++i) {
            if (jobs[i].fd >= 0) any_file = true;
            else if (i < kAhead) basalt::platform::advise_willneed(jobs[i].src, jobs[i].bytes);
        }
        while (any_file && (int) threads.size() < kIoThreads) threads.emplace_back([this] { work(); });
        issued.store(0);
        ++gen;
        head.store((uint64_t) gen << 32 | (uint64_t) jobs.size() << 16, std::memory_order_release);
        cv.notify_all();
    }
    /// Job j's bytes, in a pinned buffer (waits for the copy).
    const uint8_t* wait(int j) {
        while (!ready[(size_t) j].load(std::memory_order_acquire)) std::this_thread::yield();
        return buf[j % kRing];
    }
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.
    void issued_one(int j, cudaStream_t copy) {
        cudaEventRecord(dma_done[j % kRing], copy);
        issued.store(j + 1, std::memory_order_release);
    }
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish() {
        head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
        issued.store(1 << 30, std::memory_order_release);
        while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
};

// A stager job for the expert's blob `b`: read from the model file when that is where it lives.
static Stager::Job make_job(core::ExpertSource& src, int64_t l, int32_t e, const uint8_t* b, size_t bytes) {
    Stager::Job j{b, bytes};
    int fd = -1;
    uint64_t off = 0;
    if (src.file_extent(l, e, fd, off)) { j.fd = fd; j.off = off; }
    return j;
}

// multi-GPU: the peer GPU's share of a prompt chunk's experts.  Per MoE layer the primary copies its normed
// activations over P2P, the peer quantizes the rows routed to the experts it holds, runs the same MMQ products the
// primary would (gathered from its own slots), and copies the result rows back into the primary's Dm rows - the rows
// are laid out local-first, so the peer's are one contiguous block at the end.
struct PeerPrefill {
    core::PeerExperts* peer = nullptr;
    int dev = -1;
    int64_t cap_rows = 0, T_max = 0;
    cudaStream_t s = nullptr;
    cudaEvent_t ev_in = nullptr;    // on the primary: the activations and the row tables are ready
    cudaEvent_t ev_done = nullptr;  // on the peer: its rows have landed in the primary's Dm
    // multi-GPU: the result rows go back group by group on a second stream while the next group computes
    cudaStream_t s_out = nullptr;
    static constexpr int kGrpEv = NE / 16 + 2;
    cudaEvent_t ev_grp[kGrpEv] = {};
    bool out_pending = false;       // s_out may still read Dm (the next layer's products wait for it)
    // the peer's per-row buffers hold one GROUP's rows (a group = up to 16 experts and at most G rows), the result
    // rows double-buffered, and there is no row cap (every peer-held expert's rows go to the peer)
    int64_t G = 0;
    void *Xq_g = nullptr, *Hq_g = nullptr;
    float *GU_g = nullptr, *H_g = nullptr, *Dm_b[2] = {};
    cudaEvent_t ev_dm[2] = {};
    bool dm_live[2] = {};
    std::vector<std::pair<size_t, size_t>> groups;
    // PEER STREAMING: a share of the experts the primary would stream over ITS PCIe link in a
    // big chunk is streamed by the peer over its own link instead (into its own ring) and computed there - it halves
    // the primary's copy load and moves expert work to the card that idles through most of the MoE half.
    double ps_frac = 0.0;
    int RP = 0;
    std::vector<uint8_t*> pstage;
    std::vector<cudaEvent_t> pcopied, pused;
    std::vector<char> plive;
    cudaStream_t s_cp = nullptr;
    struct PsEntry { int32_t l, e; const uint8_t* blob; };
    std::vector<PsEntry> pseq;            // this chunk's peer-streamed experts, layer by layer in id order
    std::vector<size_t> pseq_start;
    std::vector<char> ps_flag;            // [layer * NE + e]: streamed by the peer in this chunk
    size_t p_issued = 0, pk = 0;
    float* mixed = nullptr;
    int32_t *src = nullptr, *bounds = nullptr, *ident = nullptr;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    std::unique_ptr<mmq::Context> ctx;
    mmq::Context* run_ctx = nullptr;      // the context its products run in: ctx, or a layer split helper stage's
    std::vector<int32_t> bounds_host;
    std::vector<void*> owned;
    ~PeerPrefill() {
        if (dev < 0) return;
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(dev);
        if (s) cudaStreamSynchronize(s);
        ctx.reset();
        for (void* p : owned) cudaFree(p);
        if (ev_done) cudaEventDestroy(ev_done);
        if (s_out) cudaStreamSynchronize(s_out);
        for (cudaEvent_t e : ev_grp) if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : ev_dm) if (e) cudaEventDestroy(e);
        if (s_cp) cudaStreamSynchronize(s_cp);
        for (cudaEvent_t e : pcopied) if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : pused) if (e) cudaEventDestroy(e);
        if (s_cp) cudaStreamDestroy(s_cp);
        if (s_out) cudaStreamDestroy(s_out);
        if (s) cudaStreamDestroy(s);
        cudaSetDevice(prev);
        if (ev_in) cudaEventDestroy(ev_in);
    }
};

// Copies a blob from the peer card's memory to the primary through pinned host buffers with only asynchronous calls:
// the peer's device-to-host copies run ahead of the primary's host-to-device ones, and no host thread blocks in a
// synchronous staged peer copy.
struct PeerStage {
    // a failed call here would let a buffer be reused early and corrupt weights silently: stop instead
    static void must(cudaError_t e, const char* what) {
        if (e != cudaSuccess) {
            std::fprintf(stderr, "prefill peer stage: %s: %s\n", what, cudaGetErrorString(e));
            std::exit(1);
        }
    }
    static constexpr int kBufs = 16;
    uint8_t* buf[kBufs] = {};
    cudaEvent_t d2h[kBufs] = {}, h2d[kBufs] = {};
    cudaStream_t stream = nullptr;
    int dev = -1, n = 0;
    bool ok = false;
    size_t bytes = 0;
    bool setup(int peer_dev, size_t blob_bytes) {
        if (ok && dev == peer_dev && bytes >= blob_bytes) return true;
        release();
        dev = peer_dev;
        bytes = blob_bytes;
        for (int i = 0; i < kBufs; ++i)
            if (cudaHostAlloc((void**) &buf[i], bytes, cudaHostAllocDefault) != cudaSuccess) { cudaGetLastError(); release(); return false; }
        // h2d is recorded on the primary's copy stream, so it belongs to the current (primary) device; the rest to the peer
        for (int i = 0; i < kBufs; ++i)
            if (cudaEventCreateWithFlags(&h2d[i], cudaEventDisableTiming) != cudaSuccess) { release(); return false; }
        const core::OnDevice od(dev);
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) { release(); return false; }
        for (int i = 0; i < kBufs; ++i)
            if (cudaEventCreateWithFlags(&d2h[i], cudaEventDisableTiming) != cudaSuccess) { release(); return false; }
        ok = true;
        return true;
    }
    // queue `src` (on the peer) into `dst` (on the current device) behind `copy_stream`'s earlier work
    void copy(void* dst, const void* src, size_t nbytes, cudaStream_t copy_stream) {
        const int b = n++ % kBufs;
        {
            const core::OnDevice od(dev);
            must(cudaStreamWaitEvent(stream, h2d[b], 0), "wait for the buffer's previous host-to-device copy");
            must(cudaMemcpyAsync(buf[b], src, nbytes, cudaMemcpyDeviceToHost, stream), "device-to-host copy");
            must(cudaEventRecord(d2h[b], stream), "record the device-to-host copy");
        }
        must(cudaStreamWaitEvent(copy_stream, d2h[b], 0), "wait for the device-to-host copy");
        must(cudaMemcpyAsync(dst, buf[b], nbytes, cudaMemcpyHostToDevice, copy_stream), "host-to-device copy");
        must(cudaEventRecord(h2d[b], copy_stream), "record the host-to-device copy");
    }
    void release() {
        ok = false;
        for (int i = 0; i < kBufs; ++i) {
            if (buf[i]) cudaFreeHost(buf[i]);
            buf[i] = nullptr;
            if (d2h[i]) cudaEventDestroy(d2h[i]);
            if (h2d[i]) cudaEventDestroy(h2d[i]);
            d2h[i] = h2d[i] = nullptr;
        }
        if (stream) { const core::OnDevice od(dev); cudaStreamDestroy(stream); }
        stream = nullptr;
    }
    ~PeerStage() { release(); }
};

struct Prefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    core::ExpertSource* src = nullptr;
    const core::ExpertCache* cache = nullptr;
    const int32_t* host_res = nullptr;
    int64_t T = 0, T_max = 0;
    bool borrowed = false;
    uint32_t* hist = nullptr;                // routed-entry counts per (layer, expert); see route_hist_reset
    int64_t hist_n = 0, hist_tokens = 0;
    bool hist_on = false;
    cudaStream_t cs = nullptr, copy = nullptr;
    Gemm gemm;
    std::vector<void*> owned;
    // chunk buffers
    float *emb = nullptr, *R = nullptr, *xn = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
    float* grs = nullptr;                    // the hyper-connection read's row scales (T x 4)
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
    float* mixed = nullptr;
    uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
    float* bo = nullptr;
    // GDN
    float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *hbuf = nullptr, *y = nullptr;
    uint16_t* y_h = nullptr;
    // QSA
    float *Kc = nullptr, *Vc = nullptr, *Qf = nullptr, *q = nullptr, *idx_raw = nullptr, *q_idx = nullptr, *attn = nullptr;
    uint16_t* attn_h = nullptr;
    int32_t* steps_dev = nullptr;
    std::vector<int32_t> steps_host;
    int32_t* sel_ids = nullptr;
    float* sel_scores = nullptr;          // [sel_batch, max_blocks]
    int64_t sel_batch = 256, max_blocks = 0;
    float* attn_scratch = nullptr;
    int64_t attn_batch = 32, cap = 0;
    // MoE
    float *logits = nullptr, *w = nullptr, *GU = nullptr, *Dm = nullptr, *sgate = nullptr, *sup = nullptr,
          *shared = nullptr, *sg = nullptr;
    int32_t *ids = nullptr, *slot_dev = nullptr, *src_dev = nullptr;
    uint16_t *Xs = nullptr, *Hh = nullptr, *sh_h = nullptr;
    // MMQ: the activations quantized per layer, H in FP32 and its group's quantized rows, the identity
    // row map, the group bounds, the group buffers of gathered experts
    void *Xq = nullptr, *Hq = nullptr;
    float* H = nullptr;
    int32_t *ids_identity = nullptr, *bounds_dev = nullptr;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    std::vector<int32_t> bounds_host;
    std::unique_ptr<mmq::Context> mmq_ctx;
    std::vector<int32_t> ids_host, slot_host, src_host, cnt, off;
    // The CPU share of a small chunk's experts: the layer's normalized activations and the pool's output rows (pinned),
    // the pool's per-token activations and jobs, the stream and events that bring the activations to the host, and the
    // running per-expert costs the share is balanced from.
    float *cpu_x = nullptr, *cpu_rows = nullptr;
    size_t cpu_x_n = 0, cpu_rows_n = 0;
    std::vector<uint8_t> cpu_nact;
    std::vector<kernels::cpu::ActQ> cpu_actq;
    std::vector<kernels::cpu::ExpertJobMulti> cpu_jobs;
    std::vector<char> cpu_need;
    cudaStream_t cpu_s = nullptr;
    cudaEvent_t cpu_ev_in = nullptr, cpu_ev_x = nullptr, cpu_gev[2] = {};
    // The share of the streamed experts the CPU computes. Fixed, not timing-driven: the CPU and the GPU round
    // differently, so a share that moved with timing would change the output from run to run.
    double cpu_share_now = 0.6;
    // The grouping tables in mapped pinned memory, [ids | slot | src] of T_max * K each, then the MMQ bounds: kernels
    // read and write them in place.  A cudaMemcpyAsync of them queues behind the expert blobs the copy stream already
    // holds (up to `ring` of them), and the GPU idles meanwhile.
    int32_t* grp_host = nullptr;
    int32_t* grp_dev = nullptr;          // its device alias
    size_t grp_n = 0, grp_tk = 0;        // int32s allocated; T_max * K (the offset of slot, and of src past it)
    uint16_t* dq_gu[DQ] = {};
    uint16_t* dq_d[DQ] = {};
    uint8_t* stage_dev[RING_MAX] = {};
    int ring = STAGE;                        // the slots of this layout's ring (ring_slots)
    std::unique_ptr<Stager> stager;          // the unpinned experts' host copies
    cudaEvent_t copied[RING_MAX] = {}, used[RING_MAX] = {};
    bool stage_live[RING_MAX] = {};
    // the event that releases each ring slot: its own `used`, or - when an MMQ group is gathered in one launch - the
    // `used` of the last slot gathered with it, recorded once for all of them (a later record only waits longer)
    int used_of[RING_MAX] = {};
    // PLE
    float* ple_emb = nullptr;
    std::vector<float> ple_pageable[2];      // the fallback when no more RAM can be pinned
    float* ple_emb_host[2] = {};             // pinned, double-buffered: the next chunk's rows are read while this
    cudaEvent_t ple_copied[2] = {};          // one runs; the event marks that buffer's upload done
    std::vector<uint32_t> ple_rows[2];
    float* ple_norm = nullptr;
    uint8_t* region = nullptr;               // the attention/MoE scratch region (idle while the PLE block runs)
    uint64_t region_bytes = 0;
    PrefillStats* stats = nullptr;
    // KV streaming: one layer's whole K/V, staged from the host copy per layer and chunk (identity layout)
    basalt::kernels::KvHostPools stage;
    int32_t* ident_table = nullptr;
    // layer split: the device, and the hand-off to the next stage (two pinned chunk buffers, used in turn)
    int device = -1;
    float* hand[2] = {};
    // the chunk's token ids on the device, for one batched embedding gather
    int32_t* tok_dev = nullptr;
    std::vector<int32_t> tok_host;
    std::unique_ptr<PeerPrefill> pp;         // multi-GPU: the peer GPU's expert share (set_peer)
    core::PeerExperts* ring_peer = nullptr;  // --peer-exclusive: experts held only there are copied from its memory
    PeerStage peer_stage;                    // the asynchronous host-staged copy of those experts
};

namespace {
// the staging pool of a streamed session: every page of one layer (same sequence in init and bytes_needed)
void take_stage(Alloc& o, const core::SessionState& ss, const basalt::kernels::QsaShapes& s,
                basalt::kernels::KvHostPools& st, bool& ok) {
    const core::QsaState& q0 = ss.qsa_states[ss.qsa_primary()];
    if (q0.kv_mode != 1) return;
    const size_t rows = (size_t) q0.n_pages * s.n_head_kv * s.page_size;
    if (q0.kv_q4) {
        st.k_q4 = o.take<uint8_t>(rows * basalt::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
        st.v_q4 = o.take<uint8_t>(rows * basalt::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
    } else if (q0.kv_int8) {
        st.k_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.v_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.k_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
        st.v_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
    } else {
        st.k_pool = o.take<uint16_t>(rows * s.head_dim, ok);
        st.v_pool = o.take<uint16_t>(rows * s.head_dim, ok);
    }
}
basalt::kernels::QsaAttnPools pools_of(const basalt::kernels::KvHostPools& h, const int32_t* table) {
    basalt::kernels::QsaAttnPools p;
    p.k_pool = h.k_pool; p.v_pool = h.v_pool; p.k_q = h.k_q; p.v_q = h.v_q; p.k_scale = h.k_scale; p.v_scale = h.v_scale;
    p.k_q4 = h.k_q4; p.v_q4 = h.v_q4;
    p.page_table = table;
    return p;
}
}  // namespace

Prefill::Prefill() : impl_(new Impl) {}
Prefill::~Prefill() { release(); }

void Prefill::reset() {
    release();
    impl_.reset(new Impl);
    stats_ = PrefillStats{};
}

void Prefill::release() {
    if (!impl_) return;
    if (impl_->cs) cudaStreamSynchronize(impl_->cs);
    if (impl_->copy) cudaStreamSynchronize(impl_->copy);
    for (int i = 0; i < RING_MAX; ++i) {
        if (impl_->copied[i]) cudaEventDestroy(impl_->copied[i]);
        if (impl_->used[i]) cudaEventDestroy(impl_->used[i]);
    }
    for (int b = 0; b < 2; ++b) {
        if (impl_->hand[b]) cudaFreeHost(impl_->hand[b]);
        if (impl_->ple_copied[b]) cudaEventDestroy(impl_->ple_copied[b]);
        if (impl_->ple_emb_host[b] && impl_->ple_pageable[b].empty()) cudaFreeHost(impl_->ple_emb_host[b]);
    }
    if (impl_->copy) cudaStreamDestroy(impl_->copy);
    if (impl_->grp_host) cudaFreeHost(impl_->grp_host);
    for (float* p : {impl_->cpu_x, impl_->cpu_rows})
        if (p) cudaFreeHost(p);
    if (impl_->cpu_s) cudaStreamDestroy(impl_->cpu_s);
    for (cudaEvent_t e : {impl_->cpu_ev_in, impl_->cpu_ev_x, impl_->cpu_gev[0], impl_->cpu_gev[1]})
        if (e) cudaEventDestroy(e);
    for (void* p : impl_->owned) cudaFree(p);
}

namespace {
constexpr int64_t GEMM_SCRATCH = 32ll << 20;        // FP16 elements for the largest dequantized dense weight
constexpr size_t GEMM_WS = 32u << 20;               // cuBLAS workspace

// THE ATTENTION HALF AND THE MoE HALF SHARE THEIR BUFFERS.  A layer runs its attention (GDN or QSA), writes it back
// into the residual, and only then its MoE, so the three sets of scratch are never live at once: one region the size
// of the largest holds them all.  That is a large part of what a prompt token costs, which is what lets a chunk
// grow (every expert is streamed once per chunk, so a bigger chunk streams fewer bytes per token).  The sizes are
// counted with the same `take` sequence `init` uses; a mismatch makes `init` fail with "do not fit", never overlap.
// the conv input, which the recurrence reuses as its scratch
uint64_t gdn_qkv_bytes(size_t T) { return std::max<uint64_t>((uint64_t) T * C * 4, gdn_work_bytes((int64_t) T)); }
uint64_t gdn_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<uint8_t>(gdn_qkv_bytes(T), ok); a.take<float>(T * ZV, ok); a.take<float>(T * 2 * HV, ok); a.take<float>(T * HV, ok);
    a.take<float>(T * HV, ok); a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    return a.used;
}
// the hyper-connection's rows kept between halves: the mix weights, the bottleneck, the normed residual and its scales
uint64_t hc_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * D, ok); a.take<float>(T * LR, ok); a.take<uint16_t>(T * LR, ok);
    a.take<uint16_t>(T * D, ok); a.take<float>(T * HC, ok);
    return a.used;
}
uint64_t qsa_set_bytes(size_t T, int64_t cap, int64_t max_blocks, int64_t sel_batch, int64_t attn_batch,
                       const basalt::kernels::QsaShapes& s) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * 512, ok); a.take<float>(T * 512, ok); a.take<float>(T * 12288, ok); a.take<float>(T * ZV, ok);
    a.take<float>(T * 128, ok); a.take<float>(T * 512, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    a.take<int32_t>(T * (size_t) cap, ok);
    a.take<float>((size_t) sel_batch * (size_t) max_blocks, ok);
    a.take<float>((size_t) attn_batch * basalt::kernels::qsa_decode_attn_scratch_floats(cap, s), ok);
    return a.used;
}
// Which layers' experts go through MMQ (both weight types covered; the Q2_0 pack always - its blob
// is converted to GGUF Q2_0 blocks on the gather), whether any layer keeps the FP16 path (IQ1_M), and the largest
// gate/up and down matrices a group buffer slot holds.
constexpr int MMQ_GROUP = 16;                  // experts per MMQ launch (the gather is per expert, as blobs arrive)
// MMQ reads up to one 256-value tile past a matrix's last row when the row length is not a multiple of it (the down
// product: 640 values).  Those bytes meet zero activations, which is harmless only if they decode to finite numbers -
// llama.cpp zero-pads after every tensor, and so does a group buffer: this many zeroed bytes follow its last expert.
constexpr size_t MMQ_TAIL = 4096;
// A group buffer's down part ends in MMQ_GROUP 16-byte slots after its zeroed tail: each NVFP4 expert's FP32 scale
// tail {s_gate, s_up, s_down, 0}, gathered with it and applied in mmq::swiglu.
constexpr size_t MMQ_SCALES = (size_t) MMQ_GROUP * 16;
struct MmqPlan {
    bool any = false, fallback = true;
    std::vector<char> layer;                   // per layer: MMQ
    size_t gu_max = 0, d_max = 0;
};
const MmqPlan& mmq_plan() {
    static const MmqPlan plan = [] {
        MmqPlan p;
        const auto& lay = basalt::kernels::cpu::expert_layout();
        const bool on = mmq::built();
        const int64_t layers = lay.native ? (int64_t) lay.fmt.size() : lay.n_layers;
        p.layer.assign((size_t) std::max<int64_t>(layers, 0), 0);
        p.fallback = !on || layers <= 0;
        for (int64_t l = 0; on && l < layers; ++l) {
            const int gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42, dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
            // a tile on every GPU for these shapes (gate+up: 1280 rows, down: N rows), else the FP16 path
            if (!mmq::fits(gt, 1280) || !mmq::fits(dt, N)) {
                // Q6_K experts never take the FP16 path: a build without their MMQ instance refuses the pack
                if (gt == 14 || dt == 14) {
                    std::fprintf(stderr, "prefill: layer %lld has Q6_K experts and no MMQ kernel takes them\n", (long long) l);
                    std::exit(1);
                }
                p.fallback = true;
                continue;
            }
            p.layer[(size_t) l] = 1;
            p.any = true;
            p.gu_max = std::max(p.gu_max, mmq::matrix_bytes(gt, 1280, N));
            p.d_max = std::max(p.d_max, mmq::matrix_bytes(dt, N, 640));
        }
        return p;
    }();
    return plan;
}
// The scale slots of a group buffer (see MMQ_SCALES)
float* grp_scales(uint8_t* grp_d) { return (float*) (grp_d + MMQ_GROUP * mmq_plan().d_max + MMQ_TAIL); }
// A layout whose chunks run the fused experts (a streamed chunk of stream_all_min() tokens or more).  Its GU, H and Xq hold only the fused path's grouping tables, int8 H and per-token
// int8 activations, and Hq nothing: far less than MMQ's FP32 GU / H and per-slot q8_1 rows, which is what
// lets a bigger chunk or ring fit in the slots the prompt path borrows.  A last chunk below stream_all_min() still runs
// MMQ in the same buffers, so each keeps MMQ's size for stream_all_min() - 1 tokens.  `src`: the layout streams experts
// (Prefill::init got an ExpertSource; without one no chunk takes the streamed walk, so no chunk is fused).
bool fused_layout(size_t T, bool src) {
    return src && fused_ring() && mmq_plan().any && ring_slots(T) > STAGE && (int64_t) T >= stream_all_min();
}
// The MoE buffers MMQ and the fused path share: GU and H in floats, Xq and Hq in bytes.  Without `fused` (the
// default): MMQ's, for T tokens.
struct MoeBufs { size_t gu, h, xq, hq; };
// An NVFP4 pack's prompt experts run on the FP4 tensor cores: the activation planes, token scales and tile tables sit
// in Xq (the layer's rows) and Hq (one group's H, a group of up to MMQ_GROUP experts).
bool fp4_pack() {
    static const bool any = [] {
        const auto& lay = basalt::kernels::cpu::expert_layout();
        if (!lay.native) return false;
        for (const auto& f : lay.fmt)
            if (f.gu_type == GGML_TYPE_NVFP4 && f.d_type == GGML_TYPE_NVFP4 && f.tail != 0) return true;
        return false;
    }();
    return any;
}
// A chunk shorter than this leaves most expert tiles nearly empty and the per-expert launches dominate: MMQ is faster.
constexpr int64_t FP4_MIN_TOKENS = 1024;
constexpr size_t al256(size_t v) { return (v + 255) & ~(size_t) 255; }
size_t fp4_x_bytes(size_t rows, int64_t ne) {
    return al256(fp4::a4_bytes((int64_t) rows, N, (int) ne)) + al256(fp4::scale_count((int64_t) rows, (int) ne) * 4) +
           al256((size_t) (ne + 1) * 4);
}
size_t fp4_h_bytes(size_t rows) {
    return al256(fp4::a4_bytes((int64_t) rows, 640, MMQ_GROUP)) + al256(fp4::scale_count((int64_t) rows, MMQ_GROUP) * 4) +
           al256((size_t) 2 * (MMQ_GROUP + 1) * 4);
}
// Every layer's experts are NVFP4 on the FP4 path, so a chunk of FP4_MIN_TOKENS or more never runs MMQ.
bool fp4_every_layer() {
    static const bool all = [] {
        const auto& lay = basalt::kernels::cpu::expert_layout();
        const MmqPlan& mp = mmq_plan();
        if (!lay.native || lay.fmt.empty() || !mp.any || mp.layer.size() < lay.fmt.size()) return false;
        for (size_t l = 0; l < lay.fmt.size(); ++l)
            if (!mp.layer[l] || lay.fmt[l].gu_type != GGML_TYPE_NVFP4 || lay.fmt[l].d_type != GGML_TYPE_NVFP4 ||
                lay.fmt[l].tail == 0)
                return false;
        return true;
    }();
    return all;
}
// Rows of GU and H a group's products hold at once.  The FP4 path indexes them by the group's own rows and splits a
// group above this cap (an expert has at most T rows); a chunk below FP4_MIN_TOKENS runs MMQ, which indexes them by
// the layer's absolute row, so the cap also covers that chunk's rows.
size_t group_rows_cap(size_t T) {
    if (!(fp4_every_layer() && (int64_t) T >= FP4_MIN_TOKENS)) return T * K;
    return std::min(T * K, std::max(T, (size_t) (FP4_MIN_TOKENS - 1) * K));
}
MoeBufs moe_bufs(size_t T, int64_t n_expert, bool fused) {
    if (!fused) {
        // MMQ's int8 planes are only needed for rows a chunk below FP4_MIN_TOKENS runs; the FP4 planes cover the rest
        const size_t qt = fp4_every_layer() && (int64_t) T >= FP4_MIN_TOKENS ? (size_t) (FP4_MIN_TOKENS - 1) : T;
        const size_t gr = group_rows_cap(T);
        size_t xq = mmq::q8_bytes((int64_t) (qt * K), N), hq = mmq::q8_bytes((int64_t) (qt * K), 640);
        if (fp4_pack()) { xq = std::max(xq, fp4_x_bytes(T * K, n_expert)); hq = std::max(hq, fp4_h_bytes(gr)); }
        return {gr * 1280, gr * 640, xq, hq};
    }
    const size_t ts = (size_t) std::min<int64_t>((int64_t) T, stream_all_min() - 1);   // MMQ's last small chunk
    return {std::max(ts * K * 1280, (fused::group_bytes((int64_t) (T * K), (int) n_expert) + 3) / 4),
            std::max(ts * K * 640, (fused::act_bytes((int64_t) (T * K), 640) + 3) / 4),
            std::max(mmq::q8_bytes((int64_t) (ts * K), N), fused::act_bytes((int64_t) T, N)),
            mmq::q8_bytes((int64_t) (ts * K), 640)};
}
struct Fp4Ptrs {
    uint8_t *a4x, *a4h;
    float *gsx, *gsh;
    int32_t *tox, *toh, *rel;
};
// Where the FP4 path's tables sit in Xq (the layer's activations) and Hq (a group's H), as moe_bufs sized them.
Fp4Ptrs fp4_ptrs(void* xq, void* hq, int64_t T, int64_t n_expert) {
    const size_t rows = (size_t) (T * K), ne = (size_t) n_expert;
    Fp4Ptrs p;
    uint8_t* x = (uint8_t*) xq;
    p.a4x = x;
    p.gsx = (float*) (x + al256(fp4::a4_bytes((int64_t) rows, N, (int) ne)));
    p.tox = (int32_t*) ((uint8_t*) p.gsx + al256(fp4::scale_count((int64_t) rows, (int) ne) * 4));
    uint8_t* h = (uint8_t*) hq;
    p.a4h = h;
    const size_t hrows = group_rows_cap((size_t) T);
    p.gsh = (float*) (h + al256(fp4::a4_bytes((int64_t) hrows, 640, MMQ_GROUP)));
    p.toh = (int32_t*) ((uint8_t*) p.gsh + al256(fp4::scale_count((int64_t) hrows, MMQ_GROUP) * 4));
    p.rel = p.toh + (MMQ_GROUP + 1);
    return p;
}
uint64_t moe_set_bytes(size_t T, int64_t n_expert, bool fused) {
    const MmqPlan& mp = mmq_plan();
    const MoeBufs mb = moe_bufs(T, n_expert, fused);
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * n_expert, ok); a.take<float>(T * K, ok); a.take<int32_t>(T * K, ok); a.take<int32_t>(T * K, ok);
    a.take<int32_t>(T * K, ok);
    if (mp.fallback) a.take<uint16_t>(T * K * N, ok);
    a.take<float>(mb.gu, ok);
    if (mp.fallback) a.take<uint16_t>(T * K * 640, ok);
    a.take<float>(T * K * N, ok); a.take<float>(T * 640, ok);
    a.take<float>(T * 640, ok); a.take<uint16_t>(T * 640, ok); a.take<float>(T * N, ok); a.take<float>(T, ok);
    if (mp.any) {
        a.take<uint8_t>(mb.xq, ok);
        a.take<float>(mb.h, ok);
        a.take<uint8_t>(mb.hq, ok);
    }
    return a.used;
}
}

void Prefill::set_session(core::SessionState& ss) { impl_->ss = &ss; }

bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                   core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes) {
    Impl& m = *impl_;
    m.wt = &wt; m.g = &g; m.ss = &ss; m.src = src; m.cache = cache; m.host_res = host_res;
    m.T = chunk; m.cs = (cudaStream_t) stream; m.stats = &stats_;
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
        err = "prefill: geometry differs from the artifact's"; return false;
    }
    cudaGetDevice(&m.device);
    if (stage_le_ < 0) stage_le_ = g.n_layers;
    if (stage_lb_ < 0 || stage_lb_ >= stage_le_ || stage_le_ > g.n_layers || (stage_le_ < g.n_layers) != (next_ != nullptr)) {
        err = "prefill: the stage's layer range is wrong";
        return false;
    }
    for (int b = 0; next_ != nullptr && b < 2; ++b)
        if (!m.hand[b] && cudaHostAlloc((void**) &m.hand[b], (size_t) chunk * D * 4, cudaHostAllocPortable) != cudaSuccess) {
            err = "prefill: the layer split's hand-off buffers";
            return false;
        }
    if (m.tok_dev == nullptr) {
        if (const cudaError_t e = cudaMalloc((void**) &m.tok_dev, (size_t) chunk * sizeof(int32_t)); e != cudaSuccess) {
            err = std::string("prefill: the token id buffer (") + cudaGetErrorString(e) + ")";
            return false;
        }
        m.owned.push_back(m.tok_dev);
        m.tok_host.resize((size_t) chunk);
    }
    if (cudaStreamCreateWithFlags(&m.copy, cudaStreamNonBlocking) != cudaSuccess) { err = "prefill: copy stream"; return false; }
    const size_t T = (size_t) chunk;
    m.T_max = chunk;
    m.borrowed = borrow != nullptr;
    bool ok = true;
    // one-time: events, the stager, the host buffers (for the largest chunk), the identity page table
    for (int i = 0; i < RING_MAX; ++i) {
        if (cudaEventCreateWithFlags(&m.copied[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
        if (cudaEventCreateWithFlags(&m.used[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
    }
    if (!m.stager) {
        m.stager = std::make_unique<Stager>();
        const int hw = (int) std::thread::hardware_concurrency();
        const int threads = std::max(2, std::min(4, hw / 4));
        if (!m.stager->init((size_t) MAXBLOB(), threads)) ok = false;
    }
    m.steps_host.resize(T * basalt::kernels::kStepCount);
    m.ids_host.resize(T * K); m.slot_host.resize(T * K); m.src_host.resize(T * K); m.cnt.resize(m.g->n_expert); m.off.resize(m.g->n_expert + 1);
    {
        const size_t need = 3 * T * K + (size_t) (2 * (m.g->n_expert + m.g->n_expert / MMQ_GROUP + 2));
        if (m.grp_n < need) {
            if (m.grp_host) cudaFreeHost(m.grp_host);
            m.grp_host = m.grp_dev = nullptr;
            m.grp_n = m.grp_tk = 0;
            void *h = nullptr, *d = nullptr;
            if (cudaHostAlloc(&h, need * 4, cudaHostAllocMapped | (core::peer_portable() ? cudaHostAllocPortable : 0)) == cudaSuccess &&
                cudaHostGetDevicePointer(&d, h, 0) == cudaSuccess) {
                m.grp_host = (int32_t*) h;
                m.grp_dev = (int32_t*) d;
                m.grp_n = need;
                m.grp_tk = T * K;
            } else {                              // the copies, as before
                if (h) cudaFreeHost(h);
                cudaGetLastError();
            }
        }
    }
    for (int b = 0; b < 2; ++b) {
        if (!m.ple_emb_host[b] &&
            cudaHostAlloc((void**) &m.ple_emb_host[b], (size_t) T * N * 4, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            m.ple_pageable[b].resize(T * N);          // pageable: the upload is staged before it returns
            m.ple_emb_host[b] = m.ple_pageable[b].data();
        }
        if (!m.ple_copied[b] && cudaEventCreateWithFlags(&m.ple_copied[b], cudaEventDisableTiming) != cudaSuccess)
            ok = false;
        m.ple_rows[b].resize(T * basalt::kernels::PLE_N_HEADS);
    }
    if (ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {   // KV streaming: the staging pool's identity page table
        const int64_t pages = ss.qsa_states[ss.qsa_primary()].n_pages;
        std::vector<int32_t> ident((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) ident[(size_t) i] = (int32_t) i;
        if (cudaMalloc((void**) &m.ident_table, ident.size() * 4) != cudaSuccess ||
            cudaMemcpy(m.ident_table, ident.data(), ident.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
            ok = false;
        else
            m.owned.push_back(m.ident_table);
    }
    if (!ok) { err = "prefill: host buffers or events for a chunk of " + std::to_string(chunk) + " tokens"; return false; }
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    {
        uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
        void* ws = o.take<uint8_t>(GEMM_WS, ok);
        if (!ok) { err = "prefill: GEMM scratch does not fit"; return false; }
        if (!m.gemm.init_external(stream, gs, GEMM_SCRATCH, ws, GEMM_WS, err)) return false;
    }
    if (!carve(T, &o)) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit";
        return false;
    }
    return true;
}

// Every device buffer of a chunk of T tokens, from the Alloc `alloc` (after the GEMM scratch and workspace): `init`
// once, and `relayout` for a request's own chunk.  The order is `bytes_needed`'s.
bool Prefill::carve(size_t T, void* alloc) {
    Impl& m = *impl_;
    Alloc& o = *static_cast<Alloc*>(alloc);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    bool ok = true;
    m.R = o.take<float>(T * D, ok);
    m.inj = o.take<float>(T * HC, ok);
    m.mixed = o.take<float>(T * N, ok); m.mixed_bf = o.take<uint16_t>(T * N, ok);
    m.mixed_h = o.take<uint16_t>(T * N, ok); m.bo = o.take<float>(T * N, ok);
    m.steps_dev = o.take<int32_t>(T * basalt::kernels::kStepCount, ok);
    basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    m.cap = basalt::kernels::qsa_selection_width(basalt::kernels::kTopkMaxCells, s);
    m.max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    {
        // one region for the attention half's and the MoE half's scratch (see gdn_set_bytes)
        const bool fz = fused_layout(T, m.src != nullptr);
        const MoeBufs mb = moe_bufs(T, m.g->n_expert, fz);
        const uint64_t region = std::max({gdn_set_bytes(T), qsa_set_bytes(T, m.cap, m.max_blocks, m.sel_batch,
                                                                           m.attn_batch, s), moe_set_bytes(T, m.g->n_expert, fz),
                                          hc_set_bytes(T)});
        uint8_t* base = o.take<uint8_t>((size_t) region, ok);
        m.region = base;
        m.region_bytes = region;
        {
            // the hyper-connection's rows are only live between one half's end and the next half's mix, when the
            // region holds no scratch, and the embedding rows before the first layer's: they sit at its start
            Alloc h;
            h.base = base; h.cap = region; h.owned = &m.owned;
            m.gated = h.take<float>(T * D, ok); m.lo = h.take<float>(T * LR, ok); m.lo16 = h.take<uint16_t>(T * LR, ok);
            m.xn16 = h.take<uint16_t>(T * D, ok); m.grs = h.take<float>(T * HC, ok);
            m.emb = m.gated;
        }
        Alloc a;
        a.base = base; a.cap = region; a.owned = &m.owned;
        m.qkv = (float*) a.take<uint8_t>(gdn_qkv_bytes(T), ok); m.z = a.take<float>(T * ZV, ok); m.ab = a.take<float>(T * 2 * HV, ok);
        m.gate = a.take<float>(T * HV, ok); m.beta = a.take<float>(T * HV, ok); m.hbuf = a.take<float>(T * C, ok);
        m.y = a.take<float>(T * ZV, ok); m.y_h = a.take<uint16_t>(T * ZV, ok);
        Alloc b;
        b.base = base; b.cap = region; b.owned = &m.owned;
        m.Kc = b.take<float>(T * 512, ok); m.Vc = b.take<float>(T * 512, ok); m.Qf = b.take<float>(T * 12288, ok);
        m.q = b.take<float>(T * ZV, ok); m.idx_raw = b.take<float>(T * 128, ok); m.q_idx = b.take<float>(T * 512, ok);
        m.attn = b.take<float>(T * ZV, ok); m.attn_h = b.take<uint16_t>(T * ZV, ok);
        m.sel_ids = b.take<int32_t>(T * (size_t) m.cap, ok);
        m.sel_scores = b.take<float>((size_t) m.sel_batch * (size_t) m.max_blocks, ok);
        m.attn_scratch = b.take<float>((size_t) m.attn_batch * basalt::kernels::qsa_decode_attn_scratch_floats(m.cap, s), ok);
        Alloc c;
        c.base = base; c.cap = region; c.owned = &m.owned;
        m.logits = c.take<float>(T * m.g->n_expert, ok); m.w = c.take<float>(T * K, ok); m.ids = c.take<int32_t>(T * K, ok);
        m.slot_dev = c.take<int32_t>(T * K, ok); m.src_dev = c.take<int32_t>(T * K, ok);
        const MmqPlan& mp = mmq_plan();
        m.Xs = mp.fallback ? c.take<uint16_t>(T * K * N, ok) : nullptr;
        m.GU = c.take<float>(mb.gu, ok);
        m.Hh = mp.fallback ? c.take<uint16_t>(T * K * 640, ok) : nullptr;
        m.Dm = c.take<float>(T * K * N, ok);
        m.sgate = c.take<float>(T * 640, ok); m.sup = c.take<float>(T * 640, ok); m.sh_h = c.take<uint16_t>(T * 640, ok);
        m.shared = c.take<float>(T * N, ok); m.sg = c.take<float>(T, ok);
        if (mp.any) {
            m.Xq = c.take<uint8_t>(mb.xq, ok);
            m.H = c.take<float>(mb.h, ok);
            m.Hq = c.take<uint8_t>(mb.hq, ok);
        }
        if (base == nullptr) ok = false;
    }
    for (int i = 0; i < DQ; ++i) { m.dq_gu[i] = o.take<uint16_t>(1280 * 2560, ok); m.dq_d[i] = o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        m.ids_identity = o.take<int32_t>(T * K, ok);
        m.bounds_dev = o.take<int32_t>((size_t) (2 * (m.g->n_expert + m.g->n_expert / MMQ_GROUP + 2)), ok);
        m.grp_gu = o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        m.grp_d = o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL + MMQ_SCALES, ok);
        // (written at every run's start, not here: when serving, these are live expert-cache slots until a request
        // lends them - a write now would corrupt a resident expert)
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
    }
    m.ring = ring_slots(T);
    for (int i = 0; i < m.ring; ++i) {
        m.stage_dev[i] = o.take<uint8_t>((size_t) MAXBLOB(), ok);
        m.stage_live[i] = false;                        // a new buffer: nothing of an earlier layout to wait for
        m.used_of[i] = i;
    }
    m.ple_emb = o.take<float>(T * N, ok);
    m.ple_norm = o.take<float>((size_t) basalt::kernels::NG_HC_DIM, ok);
    take_stage(o, ss, s, m.stage, ok);
    m.T = (int64_t) T;
    return ok;
}

bool Prefill::relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err) {
    Impl& m = *impl_;
    if (!m.borrowed || borrow == nullptr || chunk <= 0 || chunk > m.T_max) {
        err = "prefill: relayout needs borrowed buffers and a chunk of at most " + std::to_string(m.T_max);
        return false;
    }
    if (cudaStreamSynchronize(m.cs) != cudaSuccess || cudaStreamSynchronize(m.copy) != cudaSuccess) {
        err = "prefill: relayout: the stream failed";
        return false;
    }
    bool ok = true;
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    void* ws = o.take<uint8_t>(GEMM_WS, ok);
    if (ok) m.gemm.rebind(gs, GEMM_SCRATCH, ws, GEMM_WS);
    if (!ok || !carve((size_t) chunk, &o)) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit";
        return false;
    }
    return true;
}

int64_t Prefill::chunk() const { return impl_->T; }

bool Prefill::draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                       std::string& err) {
    Impl& m = *impl_;
    // A ring (KV streaming: the drafter's window, page p in slot p % n_slots over a host copy) takes the same appends
    // with its own page table and host copy, as a streamed main layer does; the cells written are those the window can
    // still reach (r0 below), which the ring holds, so no two of them share a slot..
    core::QsaState& st = mtp.kv_state_rw();
    if (n <= 0 || m.g == nullptr || m.region == nullptr || (st.kv_mode != 0 && st.kv_mode != 2) ||
        st.kv_hybrid || mtp.device() != m.device)
        return false;
    const auto t0 = Clock::now();
    const core::ModelGeometry& g = *m.g;
    const int64_t Nn = g.n_embd, HCN = g.hc * g.n_embd, KV = g.n_head_kv * g.head_dim;
    constexpr int kQ8_0 = 8;   // GGML_TYPE_Q8_0
    const float* w_ne = mtp.tensor_f32("pre_fc_norm_embedding.weight");
    const void* w_fe = mtp.tensor_q8("fc_embedding.weight");
    const float* w_nh = mtp.tensor_f32("pre_fc_norm_hidden.weight");
    const void* w_fh = mtp.tensor_q8("fc_hidden.weight");
    const float* w_hn = mtp.tensor_f32("attn_hyper_connection.hc_norm.weight");
    const uint16_t* w_dn = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_down.weight");
    const uint16_t* w_up = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_up.weight");
    const void* w_k = mtp.tensor_q8("self_attn.k_proj.weight");
    const void* w_v = mtp.tensor_q8("self_attn.v_proj.weight");
    const float* w_kn = mtp.tensor_f32("self_attn.k_norm.weight");
    if (!w_ne || !w_fe || !w_nh || !w_fh || !w_hn || !w_dn || !w_up || !w_k || !w_v || !w_kn) return false;
    const core::NativeEmbed* nemb = core::native_embed();
    const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
    if (!nemb && (wemb == nullptr || wemb->codebook_iq4nl || wemb->ne0 != g.n_embd || wemb->group_elems <= 0 ||
                  (wemb->code_bits != 2 && wemb->code_bits != 4 && wemb->code_bits != 8)))
        return false;
    // the cells the drafter's window can still reach
    const int64_t r0 = std::max<int64_t>(0, mtp.first_needed() - cell0);
    if (r0 >= n) return true;
    // per row: emb/e2 (N), en16 (N half), hn/h2/Rm/gated (HCN), hn16/xn16 (HCN half), lo (LR) + lo16, grs, mixed (N) +
    // mixed_h, K and V (KV each), the token id
    // the drafter's Q8_0 matrices through Q8_1 x Q8_0 MMQ - its own pass's integer dot products (mmvq), so its K/V
    // stay close to what the drafter computes itself
    const bool q8 = mmq::built() && mmq::fits(kQ8_0, Nn) && mmq::fits(kQ8_0, KV);
    const uint64_t per_row = 4 * (2 * Nn + 4 * HCN + LR + HC + Nn + 2 * KV + 1) + 2 * (Nn + 2 * HCN + LR + Nn) + 64 +
                             (q8 ? (uint64_t) mmq::q8_bytes(g.hc, Nn) + 4 * g.hc : 0);
    int64_t B = std::min<int64_t>(n - r0, (int64_t) (m.region_bytes / per_row) & ~(int64_t) 63);
    if (st.kv_mode == 2)   // a ring: one batch's cells must not share a slot (a batch can straddle one page more)
        B = std::min<int64_t>(B, ((st.n_slots - 1) * basalt::kernels::qsa_real_shapes().page_size) & ~(int64_t) 63);
    if (B < 64) return false;
    uint8_t* q = m.region;
    auto carve = [&](size_t bytes) { void* p = q; q += (bytes + 255) & ~(size_t) 255; return p; };
    float* emb = (float*) carve((size_t) B * Nn * 4);
    float* e2 = (float*) carve((size_t) B * Nn * 4);
    uint16_t* en16 = (uint16_t*) carve((size_t) B * Nn * 2);
    float* hn = (float*) carve((size_t) B * HCN * 4);
    float* h2 = (float*) carve((size_t) B * HCN * 4);
    float* Rm = (float*) carve((size_t) B * HCN * 4);
    float* gated = (float*) carve((size_t) B * HCN * 4);
    uint16_t* hn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    uint16_t* xn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    float* lo = (float*) carve((size_t) B * LR * 4);
    uint16_t* lo16 = (uint16_t*) carve((size_t) B * LR * 2);
    float* grs = (float*) carve((size_t) B * HC * 4);
    float* mixed = (float*) carve((size_t) B * Nn * 4);
    uint16_t* mixed_h = (uint16_t*) carve((size_t) B * Nn * 2);
    float* Kc = (float*) carve((size_t) B * KV * 4);
    float* Vc = (float*) carve((size_t) B * KV * 4);
    int32_t* tok = (int32_t*) carve((size_t) B * 4);
    void* xq = q8 ? carve(mmq::q8_bytes(B * g.hc, Nn)) : nullptr;
    int32_t* ident = q8 ? (int32_t*) carve((size_t) B * g.hc * 4) : nullptr;
    int32_t* bnd = q8 ? (int32_t*) carve(16) : nullptr;
    if ((uint64_t) (q - m.region) > m.region_bytes) return false;
    if (!mtp.idle(err)) return false;   // the drafter's own stream (its graph uploads) before this writes its K/V
    basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    std::vector<int32_t> tk((size_t) B);
    if (q8) {
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
        mmq::iota(ident, B * g.hc, m.cs);
    }
    // y[rows, n_out] = x[rows, k] . w^T for a Q8_0 matrix: MMQ from the FP32 rows (bounds slot 0: rows, 1: rows*hc)
    auto proj = [&](const float* x, const uint16_t* x16, const void* w, float* y, int64_t rows, int64_t n_out,
                    int64_t k, int slot) {
        if (!q8) { m.gemm.native(x16, kQ8_0, w, y, rows, n_out, k); return; }
        mmq::quantize(x, nullptr, xq, kQ8_0, k, k, rows, m.cs);
        mmq::Product p;
        p.w = w; p.type = kQ8_0; p.w_rows = n_out; p.w_cols = k; p.expert_bytes = mmq::matrix_bytes(kQ8_0, n_out, k);
        p.n = 1; p.xq = xq; p.bounds = bnd + 2 * slot; p.ids = ident; p.total_rows = rows; p.max_rows = rows;
        p.dst = y; p.ld_dst = n_out;
        m.mmq_ctx->run(p, m.cs);
    };
    for (int64_t b0 = r0; b0 < n; b0 += B) {
        const int64_t nb = std::min(B, n - b0), c0 = cell0 + b0;
        for (int64_t i = 0; i < nb; ++i) tk[(size_t) i] = next_tokens[b0 + i];
        const int32_t bh[4] = {0, (int32_t) nb, 0, (int32_t) (nb * g.hc)};
        if (q8 && cudaMemcpyAsync(bnd, bh, sizeof bh, cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
            err = "prefill: the draft bounds' upload failed";
            return false;
        }
        if (cudaMemcpyAsync(tok, tk.data(), (size_t) nb * 4, cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
            err = "prefill: the draft tokens' upload failed";
            return false;
        }
        // the input branches: the next token's embedding, and this cell's final residual rows
        if (nemb) {
            nemb->gather_dev(tok, nb, emb, m.cs);
        } else {
            const auto* codes = (const uint8_t*) wemb->data;
            const auto* scales = (const float*) (codes + wemb->codes_bytes);
            const auto* offsets = wemb->has_offset ? (const float*) (codes + wemb->codes_bytes + wemb->scales_bytes)
                                                   : nullptr;
            basalt::kernels::embedding_gather_dev(codes, scales, offsets, tok, (int) nb, wemb->ne0, wemb->code_bits,
                                                  wemb->code_bias, wemb->group_elems,
                                                  (uint64_t) (wemb->ne0 / (8 / wemb->code_bits)),
                                                  (uint64_t) (wemb->ne0 / wemb->group_elems), emb, m.cs);
        }
        rms_rows(emb, w_ne, nb, Nn, Nn, EPS, m.cs);
        if (!q8) to_f16(emb, en16, nb * Nn, m.cs);
        proj(emb, en16, w_fe, e2, nb, Nn, Nn, 0);
        cudaMemcpyAsync(hn, R_rows + (size_t) b0 * HCN, (size_t) nb * HCN * 4, cudaMemcpyDeviceToDevice, m.cs);
        rms_rows(hn, w_nh, nb, HCN, HCN, EPS, m.cs);
        if (!q8) to_f16(hn, hn16, nb * HCN, m.cs);
        proj(hn, hn16, w_fh, h2, nb * g.hc, Nn, Nn, 1);   // every stream through fc_hidden
        basalt::kernels::add_streams_broadcast(h2, e2, Rm, Nn, (int) g.hc, (int) nb, m.cs);
        // the attention hyper-connection's read (its mixed input only: this pass writes nothing back)
        gr_norm_rs(Rm, w_hn, EPS, grs, xn16, nb, m.cs);
        m.gemm.bf16_w(xn16, w_dn, lo, nb, LR, HCN);
        gr_silu(lo, lo16, nb, m.cs);
        m.gemm.bf16_w(lo16, w_up, gated, nb, HCN, LR);
        gr_mix_r(Rm, grs, w_hn, gated, mixed, nullptr, nb, m.cs, mixed_h);
        // K and V into the drafter's cache, as the prompt path's QSA layers append theirs
        proj(mixed, mixed_h, w_k, Kc, nb, KV, Nn, 0);
        proj(mixed, mixed_h, w_v, Vc, nb, KV, Nn, 0);
        rms_rows(Kc, w_kn, nb * g.n_head_kv, g.head_dim, g.head_dim, EPS, m.cs);
        rope(Kc, nb, g.n_head_kv, g.head_dim, KV, c0, basalt::kernels::rope_scaling(), m.cs);
        if (st.kv_rot) {   // rotated as the drafter's own decode stores them (mtp.cpp)
            basalt::kernels::fwht256_inplace_cuda(Kc, nb * g.n_head_kv, m.cs);
            basalt::kernels::fwht256_inplace_cuda(Vc, nb * g.n_head_kv, m.cs);
        }
        if (st.kv_q4) {
            basalt::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, c0, nb, Kc, Vc, s, m.cs, &st.host);
        } else {
            kv_append(Kc, Vc, nb, c0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs, &st.host);
        }
    }
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = std::string("prefill: the draft layer's K/V: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    mtp.ms_prefill += ms_since(t0);
    return true;
}

void Prefill::set_ring_peer(core::PeerExperts* peer) {
    impl_->ring_peer = peer != nullptr && peer->valid() ? peer : nullptr;
}

bool Prefill::set_peer(core::PeerExperts* peer, int64_t cap_rows, std::string& err) {
    Impl& m = *impl_;
    if (peer == nullptr || !peer->valid()) { m.pp.reset(); return true; }
    // Without P2P the peer still takes the rows of the routed-only chunks (below stream_all_min()): the activations and
    // result rows cross through host staging (cudaMemcpyPeerAsync) and the streamed ring stays off.
    if (!mmq_plan().any) { err = "prefill peer: needs the MMQ prompt path"; return false; }
    auto pp = std::make_unique<PeerPrefill>();
    pp->peer = peer;
    pp->T_max = peer->p2p() ? m.T_max : std::min<int64_t>(m.T_max, stream_all_min() - 1);
    pp->cap_rows = std::max<int64_t>(1, std::min<int64_t>(cap_rows, pp->T_max * K));
    const MmqPlan& mp = mmq_plan();
    if (cudaEventCreateWithFlags(&pp->ev_in, cudaEventDisableTiming) != cudaSuccess) { err = "prefill peer: event"; return false; }
    int prev = 0;
    cudaGetDevice(&prev);
    pp->dev = peer->device();
    cudaSetDevice(pp->dev);
    bool ok = cudaStreamCreateWithFlags(&pp->s, cudaStreamNonBlocking) == cudaSuccess &&
              cudaEventCreateWithFlags(&pp->ev_done, cudaEventDisableTiming) == cudaSuccess;
    if (ok) {
        ok = cudaStreamCreateWithFlags(&pp->s_out, cudaStreamNonBlocking) == cudaSuccess;
        for (int i = 0; ok && i < PeerPrefill::kGrpEv; ++i)
            ok = cudaEventCreateWithFlags(&pp->ev_grp[i], cudaEventDisableTiming) == cudaSuccess;
    }
    auto take = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (!ok || cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; return nullptr; }
        pp->owned.push_back(p);
        return p;
    };
    pp->mixed = (float*) take((size_t) pp->T_max * N * 4);
    {
        pp->cap_rows = pp->T_max * K;   // no row cap: only the row tables grow with it
        pp->G = pp->T_max;              // one expert never has more rows than the chunk has tokens
        const int64_t Gr = pp->G;
        pp->Xq_g = take(mmq::q8_bytes(Gr, N));
        pp->Hq_g = take(mmq::q8_bytes(Gr, 640));
        pp->GU_g = (float*) take((size_t) Gr * 1280 * 4);
        pp->H_g = (float*) take((size_t) Gr * 640 * 4);
        pp->Dm_b[0] = (float*) take((size_t) Gr * N * 4);
        pp->Dm_b[1] = (float*) take((size_t) Gr * N * 4);
        for (int b = 0; b < 2 && ok; ++b) ok = cudaEventCreateWithFlags(&pp->ev_dm[b], cudaEventDisableTiming) == cudaSuccess;
        pp->ps_frac = 0.35;   // the share of the primary's streamed experts the peer streams
        if (!mp.fallback && pp->T_max >= stream_all_min()) {
            pp->RP = 48;
            pp->pstage.assign((size_t) pp->RP, nullptr);
            pp->pcopied.assign((size_t) pp->RP, nullptr);
            pp->pused.assign((size_t) pp->RP, nullptr);
            pp->plive.assign((size_t) pp->RP, 0);
            ok = ok && cudaStreamCreateWithFlags(&pp->s_cp, cudaStreamNonBlocking) == cudaSuccess;
            for (int i = 0; ok && i < pp->RP; ++i) {
                pp->pstage[(size_t) i] = (uint8_t*) take((size_t) MAXBLOB());
                ok = ok && cudaEventCreateWithFlags(&pp->pcopied[(size_t) i], cudaEventDisableTiming) == cudaSuccess &&
                     cudaEventCreateWithFlags(&pp->pused[(size_t) i], cudaEventDisableTiming) == cudaSuccess;
            }
        } else {
            pp->ps_frac = 0.0;
        }
    }
    const int64_t Rt = pp->cap_rows;
    pp->src = (int32_t*) take((size_t) Rt * 4);
    pp->ident = (int32_t*) take((size_t) Rt * 4);
    pp->bounds = (int32_t*) take((size_t) (2 * (NE + NE / MMQ_GROUP + 2)) * 4);
    pp->grp_gu = (uint8_t*) take(MMQ_GROUP * mp.gu_max + MMQ_TAIL);
    pp->grp_d = (uint8_t*) take(MMQ_GROUP * mp.d_max + MMQ_TAIL + MMQ_SCALES);
    if (ok) {
        pp->ctx = std::make_unique<mmq::Context>();
        pp->run_ctx = pp->ctx.get();
        mmq::iota(pp->ident, pp->cap_rows, pp->s);
        ok = cudaStreamSynchronize(pp->s) == cudaSuccess;
    }
    size_t fb = 0, tb = 0;
    cudaMemGetInfo(&fb, &tb);
    cudaSetDevice(prev);
    if (!ok) { err = "prefill peer: the peer's buffers do not fit (raise --peer-reserve-mib or lower --peer-prefill-rows)"; return false; }
    std::fprintf(stderr, "basalt prefill: peer GPU %d computes its experts' rows of each prompt chunk, with compact group "
                         "buffers%s; %zu MiB left free on it\n", pp->dev,
                 pp->ps_frac > 0.0 ? (", streaming " + std::to_string((int) (pp->ps_frac * 100 + 0.5)) +
                                      "% of the primary's streamed experts through a " + std::to_string(pp->RP) + "-slot ring").c_str()
                                   : "", fb >> 20);
    m.pp = std::move(pp);
    return true;
}

namespace {
double g_pinned_share = 1.0;
}
void Prefill::set_pinned_share(double share) { g_pinned_share = share; }
void Prefill::set_stream_all_min(int64_t tokens) { g_stream_all_min = tokens; }
void Prefill::set_cpu_pool(kernels::cpu::ExpertPool* pool) { cpu_pool_ = pool; }
void Prefill::set_cpu_share(double share) { impl_->cpu_share_now = std::max(0.0, std::min(share, 1.0)); }
void Prefill::set_ring_depth(int slots) { g_ring_depth = std::max(STAGE + 1, std::min(slots, kPrefillRing)); }
int Prefill::ring_depth() const { return impl_->ring; }
double Prefill::pinned_share() { return g_pinned_share; }

namespace { constexpr uint64_t kSizingSlack = 8u << 20; }   // alignment slack of a borrowed region

uint64_t Prefill::bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk) {
    // the same allocation sequence as `init`, counted
    const size_t T = (size_t) chunk;
    bool ok = true;
    Alloc o;
    o.count_only = true;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    o.take<uint8_t>(GEMM_WS, ok);
    auto f = [&](size_t n) { o.take<float>(n, ok); };
    // `carve`'s order, buffer for buffer: R, inj, mixed, mixed_bf, mixed_h, bo (the hyper-connection rows live in the region).
    f(T * D);
    f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok); f(T * N);
    o.take<int32_t>(T * basalt::kernels::kStepCount, ok);
    basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const int64_t cap = basalt::kernels::qsa_selection_width(basalt::kernels::kTopkMaxCells, s);
    const int64_t max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    o.take<uint8_t>((size_t) std::max({gdn_set_bytes(T), qsa_set_bytes(T, cap, max_blocks, 256, 32, s),
                                       moe_set_bytes(T, g.n_expert, fused_layout(T, true)),
                                       hc_set_bytes(T)}), ok);
    for (int i = 0; i < DQ; ++i) { o.take<uint16_t>(1280 * 2560, ok); o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        o.take<int32_t>(T * K, ok);
        o.take<int32_t>((size_t) (2 * (g.n_expert + g.n_expert / MMQ_GROUP + 2)), ok);
        o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL + MMQ_SCALES, ok);
    }
    for (int i = 0; i < ring_slots(T); ++i) o.take<uint8_t>((size_t) MAXBLOB(), ok);
    f(T * N);
    f((size_t) basalt::kernels::NG_HC_DIM);
    basalt::kernels::KvHostPools stage;
    take_stage(o, ss, s, stage, ok);
    return o.used + kSizingSlack;
}

bool Prefill::layout_fits(const core::ModelGeometry& g, core::SessionState& ss, int64_t chunk, bool with_source,
                          uint64_t* used, uint64_t* capacity) {
    Prefill p;
    Impl& m = *p.impl_;
    m.g = &g;
    m.ss = &ss;
    m.src = with_source ? reinterpret_cast<core::ExpertSource*>(alignof(std::max_align_t)) : nullptr;   // only tested for null
    const uint64_t cap = bytes_needed(g, ss, chunk) - kSizingSlack;   // the layout must fit the count itself
    bool ok = true;
    Alloc o;
    o.base = reinterpret_cast<uint8_t*>(uint64_t{1} << 20);   // never dereferenced
    o.cap = cap;
    o.owned = &m.owned;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    o.take<uint8_t>(GEMM_WS, ok);
    ok = p.carve((size_t) chunk, &o) && ok;
    if (used) *used = o.used;
    if (capacity) *capacity = cap;
    return ok && o.used <= cap;
}

int64_t Prefill::ring_slots_for(int64_t chunk) { return ring_slots((size_t) chunk); }

namespace {

const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err) {
    const core::WeightRef* r = v.get(suffix);
    if (!r) err = v.name(suffix) + " is missing";
    return r;
}
bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy = 0) {
    if (!w->native_data) { err = "prefill: " + name + " has no native GGUF blocks (run with --native)"; return false; }
    gm.native(X, w->native_type, w->native_data, Y, T, w->ne1, w->ne0, ldy);
    return true;
}
bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
               std::string& err, int64_t ldy = 0) {
    if (w->kind != core::WeightKind::Bf16InF32 || !w->data) { err = "prefill: " + name + " is not a resident BF16 tensor"; return false; }
    gm.bf16_w(X, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy);
    return true;
}

}  // namespace

bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    err.clear();
    Impl& m = *impl_;
    const core::OnDevice on_device(m.device);
    if (m.hist_on) m.hist_tokens += n;
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    const auto t_start = Clock::now();
    const int64_t LB = stage_lb_, LE = stage_le_;
    // The direct successor's future lives on the Prefill object. Intermediate
    // stages therefore do not drain the complete remaining GPU chain here.
    basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
    const bool single_chunk = hand_in_ == nullptr ? n <= m.T : single_chunk_;
    const cudaStream_t cs = (cudaStream_t) m.cs;
    // the MMQ row table lives in the borrowed cache slots, which the refill after a prompt overwrites with experts:
    // write it again for every prompt (a layout is reused as long as the chunk and the slots are the same)
    if (m.ids_identity != nullptr) mmq::iota(m.ids_identity, m.T * K, m.cs);
    // The PLE rows of a chunk are read from the model file on the host (an SSD read per missed row): the chunk
    // after this one is read on a thread while the GPU runs this one, into the other of two buffers.  The rows
    // depend only on the tokens (the two before a position name its n-grams), so this is the same data.
    const bool ple_on = ss.ple.ready() && LB <= 1 && 1 < LE;
    // the PLE block batched over the chunk: the pinned postops and a BF16 or GGUF-native key (else token by token)
    const bool ple_batch = ple_on && basalt::kernels::ple_native_postops_enabled() &&
                           (ss.ple.w.key_bf16 != nullptr || ss.ple.w.key_native_data != nullptr) &&
                           m.region_bytes / ((uint64_t) (3 * basalt::kernels::NG_HC_DIM + N + 4) * 4 + (uint64_t) N * 2 + 4096) >= 64;
    const int32_t prev0[2] = {prev[0], prev[1]};
    auto ple_gather = [&m, &ss, tokens, n, prev0](int64_t c0, int buf, std::string& e) -> bool {
        const int64_t T = std::min(m.T, n - c0);
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
    };
    std::string ple_next_err;
    std::future<bool> ple_next;             // declared after everything it reads: an early return waits for it
    int ple_buf = 0;

    for (int64_t c0 = 0; c0 < n; c0 += m.T) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        const int64_t T = std::min(m.T, n - c0), p0 = pos0 + c0;
        // the peer takes the rows of a chunk only up to its buffers' size
        PeerPrefill* const ppc = (m.pp && T <= m.pp->T_max) ? m.pp.get() : nullptr;
        core::progress_at("reading the prompt (batched): preparing the chunk from token", p0);
        ++stats_.chunks;
        // ---- embeddings, broadcast to the four streams - or, in a later stage of a layer split, the rows the
        // previous stage handed on
        if (hand_in_ != nullptr) {
            if (cudaMemcpyAsync(m.R, hand_in_ + (size_t) c0 * D, (size_t) T * D * 4, cudaMemcpyHostToDevice, m.cs) !=
                cudaSuccess) {
                err = "prefill: the layer split's hand-off upload failed";
                return false;
            }
        }
        // the whole chunk's rows in one gather (the same per-element arithmetic as the per-token path, so the
        // same bits); a chunk with picture rows, or a token outside the table, takes the per-token path
        bool batched = hand_in_ == nullptr && m.tok_dev != nullptr;
        const core::NativeEmbed* nemb = core::native_embed();
        const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
        if (batched && nemb == nullptr &&
            (wemb == nullptr || wemb->codebook_iq4nl || wemb->ne0 != g.n_embd || wemb->group_elems <= 0 ||
             (wemb->code_bits != 2 && wemb->code_bits != 4 && wemb->code_bits != 8)))
            batched = false;
        for (int64_t t = 0; batched && t < T; ++t) {
            const int64_t tok = tokens[c0 + t];
            if ((embd_rows && embd_rows[p0 + t]) || tok < 0 || (wemb && tok >= wemb->ne1)) batched = false;
            else m.tok_host[(size_t) t] = (int32_t) tok;
        }
        if (batched) {
            if (cudaMemcpyAsync(m.tok_dev, m.tok_host.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice,
                                m.cs) != cudaSuccess) {
                err = "prefill: the token id upload failed";
                return false;
            }
            if (nemb) {
                nemb->gather_dev(m.tok_dev, T, m.emb, m.cs);
            } else {
                const auto* codes = (const uint8_t*) wemb->data;
                const auto* scales = (const float*) (codes + wemb->codes_bytes);
                const auto* offsets = wemb->has_offset ? (const float*) (codes + wemb->codes_bytes + wemb->scales_bytes)
                                                       : nullptr;
                basalt::kernels::embedding_gather_dev(codes, scales, offsets, m.tok_dev, (int) T, wemb->ne0,
                                                      wemb->code_bits, wemb->code_bias, wemb->group_elems,
                                                      (uint64_t) (wemb->ne0 / (8 / wemb->code_bits)),
                                                      (uint64_t) (wemb->ne0 / wemb->group_elems), m.emb, m.cs);
            }
        }
        for (int64_t t = 0; hand_in_ == nullptr && !batched && t < T; ++t) {
            const float* row = embd_rows ? embd_rows[p0 + t] : nullptr;
            if (row) {
                if (cudaMemcpyAsync(m.emb + t * N, row, (size_t) N * 4, cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
                    err = "prefill: the image embedding upload failed";
                    return false;
                }
            } else if (!core::embed_row(*m.wt, g, tokens[c0 + t], m.emb + t * N, m.cs, err)) {
                return false;
            }
        }
        if (hand_in_ == nullptr) gr_broadcast(m.emb, m.R, T, m.cs);
        // ---- the PLE rows of the whole chunk, one batched SSD request on a thread (see ple_gather).  A later chunk's
        // were read ahead during the previous chunk; the first chunk's are read beside layer 0 - they are needed from
        // layer 1 on, and gathering them here first would leave the GPU idle for the whole read
        if (ple_on && !ple_next.valid())
            ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c0, b = ple_buf] {
                return ple_gather(c0, b, ple_next_err);
            });
        bool ple_pending = ple_on;
        // the chunk's rows onto the device just before layer 1 reads them, and the next chunk's gather started
        auto ple_land = [&]() -> bool {
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
            if (c0 + m.T < n) {
                // the other buffer's upload (a chunk ago) is done before the SSD thread refills it
                if (cudaEventSynchronize(m.ple_copied[ple_buf ^ 1]) != cudaSuccess) {
                    err = std::string("prefill: the PLE rows' upload failed: ") + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c1 = c0 + m.T, b = ple_buf ^ 1] {
                    return ple_gather(c1, b, ple_next_err);
                });
            }
            ple_buf ^= 1;
            stats_.ms_ple += ms_since(tp);
            return true;
        };
        for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
        // ---- the QSA step records of every position in the chunk
        for (int64_t t = 0; t < T; ++t) basalt::kernels::qsa_step_fill(m.steps_host.data() + t * basalt::kernels::kStepCount, p0 + t, s);
        cudaMemcpyAsync(m.steps_dev, m.steps_host.data(), (size_t) T * basalt::kernels::kStepCount * 4,
                        cudaMemcpyHostToDevice, m.cs);

        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < LB; ++l) (core::is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        // this chunk's stream - every non-resident expert of every layer, layer by layer in id order (entry
        // k lands in ring slot k % ring); a copy is issued once the entry `ring` before it is consumed (its slot's
        // `used` event recorded), so the copy stream never waits on an event that is not queued yet
        const basalt::kernels::cpu::ExpertLayout& lay0 = basalt::kernels::cpu::expert_layout();
        const bool stream_all = m.ring > STAGE && T >= stream_all_min() && m.src != nullptr;
        const bool ps_on = stream_all && ppc && ppc->ps_frac > 0.0;
        if (ppc && !ps_on) ppc->ps_flag.clear();
        // multi-GPU: the peer's ring - issue its copies up to `limit` / give entries back (the peer device is current)
        auto p_issue_until = [&](size_t limit) {
            PeerPrefill& P = *ppc;
            limit = std::min(limit, P.pseq.size());
            while (P.p_issued < limit) {
                const PeerPrefill::PsEntry& en = P.pseq[P.p_issued];
                const size_t sl = P.p_issued % (size_t) P.RP;
                if (P.plive[sl]) cudaStreamWaitEvent(P.s_cp, P.pused[sl], 0);
                cudaMemcpyAsync(P.pstage[sl], en.blob, (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(en.l),
                                cudaMemcpyHostToDevice, P.s_cp);
                cudaEventRecord(P.pcopied[sl], P.s_cp);
                P.plive[sl] = 1;
                ++P.p_issued;
            }
        };
        auto p_release_to = [&](int64_t l, int32_t e_stop, cudaStream_t ps) {
            PeerPrefill& P = *ppc;
            while (P.pk < P.pseq_start[(size_t) l + 1] && P.pseq[P.pk].e < e_stop) {
                cudaEventRecord(P.pused[P.pk % (size_t) P.RP], ps);
                ++P.pk;
                p_issue_until(P.pk + (size_t) P.RP);
            }
        };
        struct StreamEntry { int32_t l, e; const uint8_t* blob; int job; const uint8_t* pdev; };
        std::vector<StreamEntry> seq;
        std::vector<size_t> seq_start;
        size_t consumed = 0;
        if (stream_all) {
            seq_start.assign((size_t) g.n_layers + 1, 0);
            std::vector<Stager::Job> js;
            if (ps_on) {
                ppc->pseq.clear();
                ppc->pseq_start.assign((size_t) g.n_layers + 1, 0);
                ppc->ps_flag.assign((size_t) g.n_layers * m.g->n_expert, 0);
                ppc->p_issued = ppc->pk = 0;
            }
            double ps_acc = 0.0;
            for (int64_t l = LB; l < LE; ++l) {
                seq_start[(size_t) l] = seq.size();
                if (ps_on) ppc->pseq_start[(size_t) l] = ppc->pseq.size();
                for (int32_t e = 0; e < m.g->n_expert; ++e) {
                    if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                    if (ppc && ppc->peer && ppc->peer->has(l, e)) continue;   // multi-GPU: computed on (or read from) the peer
                    int job = -1;
                    const uint8_t* b = nullptr;
                    if (m.ring_peer && m.ring_peer->exclusive() && m.ring_peer->has(l, e)) {   // no host copy: from the peer
                        seq.push_back({(int32_t) l, e, nullptr, -1, m.ring_peer->slot_ptr(l, e)});
                        continue;
                    }
                    {
                        b = m.src->blob(l, e);
                        if (!b) { err = "prefill: expert source has no blob"; return false; }
                        if (ps_on && m.src->pinned(l, e)) {   // multi-GPU: every ps_frac-th one goes to the peer's ring
                            ps_acc += ppc->ps_frac;
                            if (ps_acc >= 1.0) {
                                ps_acc -= 1.0;
                                ppc->pseq.push_back({(int32_t) l, e, b});
                                ppc->ps_flag[(size_t) l * m.g->n_expert + e] = 1;
                                continue;
                            }
                        }
                        if (!m.src->pinned(l, e)) {
                            job = (int) js.size();
                            js.push_back(make_job(*m.src, l, e, b, (size_t) lay0.blob_bytes(l)));
                        }
                    }
                    seq.push_back({(int32_t) l, e, b, job, nullptr});
                }
            }
            for (int64_t l = LE; l <= g.n_layers; ++l) {
                seq_start[(size_t) l] = seq.size();
                if (ps_on) ppc->pseq_start[(size_t) l] = ppc->pseq.size();
            }
            m.stager->start(std::move(js));
        }
        struct StagerDone {
            Stager* st;
            ~StagerDone() { if (st) st->finish(); }
        } chunk_stager_done{stream_all ? m.stager.get() : nullptr};
        // the stream is issued by its own host thread, so the thread launching the layers' kernels never waits
        // behind a host copy of an unpinned blob (that wait would leave the GPU idle).  The same copies in the same order into the same slots, and a slot is refilled only once
        // the compute stream has recorded that it is done with it: the same results.
        std::atomic<size_t> a_issued{0}, a_consumed{0};
        std::atomic<bool> a_stop{false};
        double iss_ms = 0;
        int64_t iss_streamed = 0, iss_dma = 0;
        std::thread issuer;
        struct IssuerJoin {
            std::atomic<bool>* stop;
            std::thread* t;
            ~IssuerJoin() { if (t->joinable()) { stop->store(true); t->join(); } }
        } issuer_join{&a_stop, &issuer};
        const bool threaded_issue = stream_all;
        const bool peer_async = m.ring_peer && !m.ring_peer->p2p() &&
                                m.peer_stage.setup(m.ring_peer->device(), (size_t) MAXBLOB());
        if (threaded_issue) {
            issuer = std::thread([&] {
                const core::OnDevice od(m.device);
                for (size_t idx = 0; idx < seq.size(); ++idx) {
                    while (idx >= a_consumed.load(std::memory_order_acquire) + (size_t) m.ring) {
                        if (a_stop.load(std::memory_order_acquire)) return;
                        std::this_thread::yield();
                    }
                    const StreamEntry& en = seq[idx];
                    const int sl = (int) (idx % (size_t) m.ring);
                    const auto th = Clock::now();
                    const size_t bytes = (size_t) lay0.blob_bytes(en.l);
                    if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
                    if (en.pdev != nullptr) {
                        if (peer_async) m.peer_stage.copy(m.stage_dev[sl], en.pdev, bytes, m.copy);
                        else cudaMemcpyPeerAsync(m.stage_dev[sl], m.device, en.pdev, m.ring_peer->device(), bytes, m.copy);
                        ++iss_dma;
                    } else if (en.job < 0) {
                        cudaMemcpyAsync(m.stage_dev[sl], en.blob, bytes, cudaMemcpyHostToDevice, m.copy);
                        ++iss_dma;
                    } else {
                        const uint8_t* hb = m.stager->wait(en.job);
                        cudaMemcpyAsync(m.stage_dev[sl], hb, bytes, cudaMemcpyHostToDevice, m.copy);
                        m.stager->issued_one(en.job, m.copy);
                    }
                    cudaEventRecord(m.copied[sl], m.copy);
                    m.stage_live[sl] = true;
                    iss_ms += ms_since(th);
                    ++iss_streamed;
                    a_issued.store(idx + 1, std::memory_order_release);
                }
            });
        }
        // the consumer's side: entry k's copy is on the copy stream (the thread issued it), then k is given back
        auto wait_issued = [&](size_t k) {
            while (a_issued.load(std::memory_order_acquire) <= k) std::this_thread::yield();
        };
        auto give_back = [&](size_t upto) { a_consumed.store(upto, std::memory_order_release); };
        if (ps_on) {
            int pd = 0;
            cudaGetDevice(&pd);
            cudaSetDevice(ppc->dev);
            p_issue_until((size_t) ppc->RP);
            cudaSetDevice(pd);
        }
        bool normed = false;   // the previous half's write already normed R for this half (grs, xn16)
        for (int64_t l = LB; l < LE; ++l) {
            core::progress_beat();   // the serve watchdog: a prompt chunk of 8192 tokens is still moving
            core::progress_at("reading the prompt (batched): layer", l, p0);   // a stall names layer and chunk
            const core::LayerView v(*m.wt, l);
            if (l == std::max<int64_t>(LB, 1) && !ple_land()) return false;   // the PLE rows, read from layer 1 on
            // ---- the PLE block at layer 1, token by token (its conv reads the previous tokens' rows)
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
                stats_.ms_ple += ms_since(tp);
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
                stats_.ms_ple += ms_since(tp);
            }
            for (int half = 0; half < 2; ++half) {
                // ---- the hyper-connection read of this half
                const char* pre = half == 0 ? "hc_attn_" : "hc_ffn_";
                const std::string sn = std::string(pre) + "norm.weight", sd = std::string(pre) + "down.weight",
                                  su = std::string(pre) + "up.weight", si = std::string(pre) + "inject.weight";
                const core::WeightRef *wn = need(v, sn.c_str(), err), *wd = need(v, sd.c_str(), err),
                                      *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                if (!wn || !wd || !wu || !wi) return false;
                if (!normed) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.grs, m.xn16, T, m.cs);
                normed = false;
                if (!bf16_proj(m.gemm, wd, m.xn16, m.lo, T, sd, err)) return false;
                gr_silu(m.lo, m.lo16, T, m.cs);
                if (!bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err)) return false;
                if (!bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err)) return false;
                gr_mix_r(m.R, m.grs, (const float*) wn->data, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h);

                if (half == 0 && !core::is_qsa_layer(g, l)) {
                    // ======================= GDN =======================
                    const core::WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                          *wo = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                          *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                          *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                          *wsa = need(v, "ssm_a", err);
                    if (!wqkv || !wg || !wo || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                    float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV)) return false;
                    if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV)) return false;
                    gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, m.cs);
                    gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, m.cs);
                    // the conv input is dead after the conv: the recurrence's scratch
                    gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.qkv, m.y, m.y_h, T, m.cs);
                    if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err)) return false;
                    ++gdn_index;
                } else if (half == 0) {
                    // ======================= QSA =======================
                    const core::QsaState& st = ss.qsa_states[qsa_index];
                    const core::WeightRef *wq = need(v, "attn_q.weight", err), *wk = need(v, "attn_k.weight", err),
                                          *wv = need(v, "attn_v.weight", err), *wo = need(v, "attn_output.weight", err),
                                          *wik = need(v, "indexer.k_proj.weight", err),
                                          *wiq = need(v, "indexer.q_proj.weight", err),
                                          *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                          *wiqn = need(v, "indexer.q_norm.weight", err),
                                          *wikn = need(v, "indexer.k_norm.weight", err);
                    if (!wq || !wk || !wv || !wo || !wik || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                    if (!native_proj(m.gemm, wk, m.mixed_h, m.Kc, T, v.name("attn_k.weight"), err)) return false;
                    if (!native_proj(m.gemm, wv, m.mixed_h, m.Vc, T, v.name("attn_v.weight"), err)) return false;
                    if (!native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wik, m.mixed_bf, m.idx_raw, T, v.name("indexer.k_proj.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wiq, m.mixed_bf, m.q_idx, T, v.name("indexer.q_proj.weight"), err)) return false;
                    rms_rows(m.Kc, (const float*) wkn->data, T * 2, 256, 256, EPS, m.cs);
                    rope(m.Kc, T, 2, 256, 512, p0, basalt::kernels::rope_scaling(), m.cs);
                    // KV streaming: this layer's cells [0, p0) come in from the host copy to the staging pool, and the
                    // chunk's cells go to the host copy, the staging pool, and the VRAM slots of resident blocks
                    const bool staged = st.kv_mode == 1;
                    if (staged) {
                        basalt::kernels::kv_stage_from_host(pools_of(m.stage, m.ident_table), st.host,
                                                            core::qsa_kv_format(st),
                                                            (p0 + s.page_size - 1) / s.page_size, s, m.cs);
                    }
                    const basalt::kernels::KvHostPools* host_w = &st.host;
                    if (st.kv_hybrid) {   // K8V4: K INT8 unrotated, V rotated Q4_0 (only V and the output rotate)
                        basalt::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        kv_append(m.Kc, m.Kc, T, p0, st.page_table, s.page_size, nullptr, nullptr,
                                  st.k_q, st.k_q, st.k_scale, st.k_scale, m.cs, nullptr,   // mode 0: no host mirror
                                  staged ? &m.stage : nullptr);
                        basalt::kernels::kv_append_q4(st.v_q4, st.v_q4, st.page_table, p0, T, m.Vc, m.Vc, s, m.cs,
                                                      nullptr, staged ? &m.stage : nullptr);
                    } else {
                        if (st.kv_rot) {   // rotated K and V (kv_q4.hpp), the queries below too, the output back
                            basalt::kernels::fwht256_inplace_cuda(m.Kc, T * 2, m.cs);
                            basalt::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        }
                        if (st.kv_q4)
                            basalt::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, p0, T, m.Kc, m.Vc, s, m.cs,
                                                          host_w, staged ? &m.stage : nullptr);
                        else
                            kv_append(m.Kc, m.Vc, T, p0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs,
                                      host_w, staged ? &m.stage : nullptr);
                    }
                    split_q(m.Qf, m.q, T, m.cs);
                    rms_rows(m.q, (const float*) wqn->data, T * 24, 256, 256, EPS, m.cs);
                    rope(m.q, T, 24, 256, 6144, p0, basalt::kernels::rope_scaling(), m.cs);
                    if (st.kv_rot) basalt::kernels::fwht256_inplace_cuda(m.q, T * 24, m.cs);
                    rms_rows(m.q_idx, (const float*) wiqn->data, T * 4, 128, 128, EPS, m.cs);
                    rope(m.q_idx, T, 4, 128, 512, p0, basalt::kernels::rope_scaling(), m.cs);
                    // the indexer appends, token by token; then scores + selection for many queries at once:
                    // a query reads completed blocks (final once completed) and `dead` for its own tail block
                    const basalt::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    // the chunk's appends in three launches instead of one per token (the same end state:
                    // the queries below read it only after the whole chunk is appended).
                    try {
                        basalt::kernels::native_qsa_indexer_append_batch(m.idx_raw, T, p0, 0, (const float*) wikn->data,
                                                                         EPS, ib, s, st.max_cells,
                                                                         basalt::kernels::rope_scaling(), m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill indexer: ") + e.what(); return false; }
                    for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                        const int64_t nb = std::min(m.sel_batch, T - t0);
                        const int32_t* steps0 = m.steps_dev + t0 * basalt::kernels::kStepCount;
                        // the grid reaches the batch's last query's n_bid (they rise with the position)
                        const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * basalt::kernels::kStepCount +
                                                                                basalt::kernels::kStepNBid)] + 1;
                        // the scores on tensor cores (3xTF32: FP32-level, not bitwise)
                        if (!basalt::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512,
                                                                             steps0, nb, m.max_blocks, s, m.sel_scores,
                                                                             m.cs, active))
                            basalt::kernels::qsa_block_scores(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs, active);
                        basalt::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                        m.sel_ids + t0 * m.cap, m.cs, active);
                    }
                    const basalt::kernels::QsaAttnPools pools = staged ? pools_of(m.stage, m.ident_table)
                                                                       : core::qsa_attn_pools(st);
                    // the whole chunk on tensor cores, one block per (query, KV head), FP32-level
                    // accuracy but not bitwise (qsa_prompt_attn.hpp). Q4_0 KV (or a shape it does not take): the
                    // decode kernel, 32 queries at a time (K8V4 runs the tensor kernel's mode 3: INT8 K,
                    // V dequantized from its q4_0 blocks to fp16 at gather)
                    if (!basalt::kernels::qsa_prompt_attn_batch(m.q, pools, m.sel_ids, m.steps_dev, m.cap, s,
                                                                            m.attn, T, m.cs))
                        for (int64_t t0 = 0; t0 < T; t0 += m.attn_batch) {
                            const int64_t nb = std::min(m.attn_batch, T - t0);
                            basalt::kernels::qsa_decode_attn_batch(m.q + t0 * ZV, pools, m.sel_ids + t0 * m.cap,
                                                                   m.steps_dev + t0 * basalt::kernels::kStepCount, m.cap,
                                                                   s, m.attn_scratch, m.attn + t0 * ZV, nb, m.cs);
                        }
                    if (st.kv_rot || st.kv_hybrid) basalt::kernels::fwht256_inplace_cuda(m.attn, T * 24, m.cs);
                    gate_attn(m.attn, m.Qf, m.attn_h, T, m.cs);
                    if (!native_proj(m.gemm, wo, m.attn_h, m.bo, T, v.name("attn_output.weight"), err)) return false;
                    ++qsa_index;
                } else {
                    // ======================= MoE =======================
                    const core::WeightRef *wr = need(v, "ffn_gate_inp.weight", err),
                                          *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                                          *wsg = need(v, "ffn_gate_shexp.weight", err),
                                          *wsu = need(v, "ffn_up_shexp.weight", err),
                                          *wsd = need(v, "ffn_down_shexp.weight", err);
                    if (!wr || !wgi || !wsg || !wsu || !wsd) return false;
                    if (!bf16_proj(m.gemm, wr, m.mixed_bf, m.logits, T, v.name("ffn_gate_inp.weight"), err)) return false;
                    route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);
                    if (m.hist_on) route_count(m.ids, T * K, m.hist + l * m.g->n_expert, (int) m.g->n_expert, m.cs);
                    // the shared expert and its scalar gate
                    if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
                    if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;
                    swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs);
                    if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;
                    if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
                    m.gemm.bf16(m.mixed_bf, (const uint16_t*) wgi->data, m.sg, T, 1, N);
                    // The Q2_0 pack's experts on the fused int8 kernels (moe_fused.hpp),
                    // grouped on the GPU: no host sync.  Only where every expert's place is known before the routing -
                    // the streamed walk, in which every non-resident expert of the layer comes through the ring in id
                    // order - and where the MMQ buffers exist: they hold the fused path's own (the per-token int8
                    // activations in Xq, the int8 H in H, the grouping tables in GU).  Chunks below stream_all_min()
                    // keep MMQ.  A native pack's layer takes the native kernels
                    // (moe_fused_iq.hpp) where they cover its two formats, else MMQ (or the FP16 path: IQ1_M).
                    const basalt::kernels::cpu::ExpertLayout& lay = basalt::kernels::cpu::expert_layout();
                    const bool use_mmq = mmq_plan().any && mmq_plan().layer[(size_t) l];
                    const int mmq_gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42;
                    const int mmq_dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
                    // NVFP4 experts: FP4 block-scaled tensor cores, read in place from the blobs
                    const bool fp4_l = use_mmq && lay.native && mmq_gt == GGML_TYPE_NVFP4 && mmq_dt == GGML_TYPE_NVFP4 &&
                                       lay.fmt[(size_t) l].tail != 0 && T >= FP4_MIN_TOKENS;
                    // a prompt peer: MMQ only, whether or not set_peer took it (it can decline), as fused_ring() sized
                    // the ring and the buffers for it
                    const bool no_peer = !core::prompt_peer();
                    const bool fused_nat = use_mmq && stream_all && no_peer && lay.native && fused::native_supported(mmq_gt, mmq_dt);
                    const bool fused_l = (use_mmq && stream_all && no_peer && !lay.native && fused::enabled()) || fused_nat;
                    bool peer_now = false;                // multi-GPU: the peer computed rows of this layer (MMQ path only)
                    // CPU share: this layer's CPU experts (rows [T * K - rows_cpu, T * K) of Dm) run on a thread that
                    // reads on_cpu, so the future (which waits for the thread) is declared after it
                    std::vector<char> on_cpu;
                    double cpu_ms = 0;
                    std::future<bool> cpu_fut;
                    int64_t rows_cpu = 0, n_cpu = 0, n_stream = 0;
                    if (fused_l) {
                        if (static bool said = false; !said) {
                            said = true;
                            std::fprintf(stderr, "basalt: prompt experts on the fused int8 kernels\n");
                        }
                        // the layer's input to int8 once per token; the rows of each expert from the router's ids
                        if (fused_nat) fused::quantize_act_native(m.mixed, T, N, m.Xq, m.cs);
                        else fused::quantize_act(m.mixed, T, N, m.Xq, m.cs);
                        fused::group(m.ids, T * K, (int) K, (int) m.g->n_expert, m.GU, m.slot_dev, m.src_dev, m.cs);
                        // launches over the experts in id order, each at most kMaxBatch experts of which at most a
                        // third of the ring streamed: the next batch's blobs arrive while one computes.  An expert
                        // the routing did not pick has no tiles; its ring slot is given back with its batch.
                        const size_t per = (size_t) std::max(1, m.ring / 3);
                        size_t k = seq_start[(size_t) l];
                        const size_t kend = seq_start[(size_t) l + 1];
                        for (int32_t e = 0; e < m.g->n_expert;) {
                            fused::Batch b;
                            b.e0 = e;
                            const size_t k0 = k;
                            for (; e < m.g->n_expert && e - b.e0 < fused::kMaxBatch; ++e) {
                                if (k < kend && seq[k].e == e) {
                                    if (k - k0 == per) break;
                                    b.blob[e - b.e0] = m.stage_dev[k % (size_t) m.ring];
                                    ++k;
                                } else {                  // not streamed: resident (the walk streams all others)
                                    b.blob[e - b.e0] = m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]);
                                    // counted whether routed or not (the routing stays on the GPU): at a streamed
                                    // chunk's size (>= 1024 tokens x 10 of 512) nearly every expert is routed
                                    ++stats_.experts_resident;
                                }
                            }
                            b.e1 = e;
                            if (k > k0) {
                                // one copy stream, in order: the batch's last blob covers the others
                                wait_issued(k - 1);
                                cudaStreamWaitEvent(m.cs, m.copied[(k - 1) % (size_t) m.ring], 0);
                            }
                            if (fused_nat) {
                                const auto& f = lay.fmt[(size_t) l];
                                const fused::NativeGeom ng{f.gu_type, f.d_type, f.gu_row, f.d_row, f.up_off, f.down_off};
                                fused::experts_native(b, ng, (int) m.g->n_expert, T * K, m.GU, m.Xq, m.src_dev, m.H,
                                                      m.Dm, m.cs);
                            } else {
                                fused::experts(b, (int) m.g->n_expert, T * K, m.GU, m.Xq, m.src_dev, m.H, m.Dm, m.cs);
                            }
                            for (size_t kk = k0; kk < k; ++kk) {
                                cudaEventRecord(m.used[kk % (size_t) m.ring], m.cs);
                                m.used_of[kk % (size_t) m.ring] = (int) (kk % (size_t) m.ring);
                            }
                            if (k > k0) {
                                consumed = k;
                                give_back(consumed);
                            }
                        }
                    } else {
                        // group the (token, k) pairs by expert on the host
                        // (the sync below also orders this layer's writes of slot/src/bounds after the previous
                        // layer's kernels that read them)
                        const bool grp_mapped = m.grp_host != nullptr;
                        int32_t* ids_h = grp_mapped ? m.grp_host : m.ids_host.data();
                        int32_t* slot_h = grp_mapped ? m.grp_host + m.grp_tk : m.slot_host.data();
                        int32_t* src_h = grp_mapped ? m.grp_host + 2 * m.grp_tk : m.src_host.data();
                        if (grp_mapped) copy_i32(m.grp_dev, m.ids, T * K, m.cs);
                        else cudaMemcpyAsync(m.ids_host.data(), m.ids, (size_t) T * K * 4, cudaMemcpyDeviceToHost, m.cs);
                        // CPU share (small chunks, experts with a host copy): the layer's activations go to the host on
                        // their own stream while the routing is read
                        const bool cpu_maybe = cpu_pool_ != nullptr && !stream_all && m.src != nullptr && lay.native &&
                                               next_ == nullptr && stage_lb_ == 0;
                        if (cpu_maybe) {
                            const size_t want = (size_t) T * N;
                            if (m.cpu_x_n < want) {
                                if (m.cpu_x) cudaFreeHost(m.cpu_x);
                                m.cpu_x = nullptr;
                                m.cpu_x_n = 0;
                                if (cudaHostAlloc((void**) &m.cpu_x, want * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
                                    err = "prefill: cannot allocate the CPU experts' activations";
                                    return false;
                                }
                                m.cpu_x_n = want;
                            }
                            if (m.cpu_s == nullptr) {
                                cudaStreamCreateWithFlags(&m.cpu_s, cudaStreamNonBlocking);
                                cudaEventCreateWithFlags(&m.cpu_ev_in, cudaEventDisableTiming);
                                cudaEventCreateWithFlags(&m.cpu_ev_x, cudaEventDisableTiming);
                                cudaEventCreate(&m.cpu_gev[0]);
                                cudaEventCreate(&m.cpu_gev[1]);
                            }
                            cudaEventRecord(m.cpu_ev_in, m.cs);
                            cudaStreamWaitEvent(m.cpu_s, m.cpu_ev_in, 0);
                            cudaMemcpyAsync(m.cpu_x, m.mixed, want * sizeof(float), cudaMemcpyDeviceToHost, m.cpu_s);
                            cudaEventRecord(m.cpu_ev_x, m.cpu_s);
                        }
                        // a stall here is the GPU (this layer's attention and router, or the previous layer's
                        // work), not the host: the watchdog's report says so (only its text changes)
                        core::progress_at("reading the prompt (batched): waiting for the GPU (attention, router) at layer",
                                          l, p0);
                        cudaStreamSynchronize(m.cs);
                        core::progress_at("reading the prompt (batched): layer", l, p0);
                        std::fill(m.cnt.begin(), m.cnt.end(), 0);
                        for (int64_t i = 0; i < T * K; ++i) {
                            const int32_t e = ids_h[(size_t) i];
                            if (e < 0 || e >= m.g->n_expert) { err = "prefill: routed id out of range"; return false; }
                            ++m.cnt[(size_t) e];
                        }
                        // multi-GPU: the rows of the experts the peer computes go last, as one block [rows_local, T*K)
                        const bool pre_mmq = mmq_plan().any && mmq_plan().layer[(size_t) l];
                        std::vector<char> on_peer;
                        int64_t rows_local = T * K, rows_peer = 0;
                        if (ppc && pre_mmq) {
                            on_peer.assign((size_t) m.g->n_expert, 0);
                            for (int32_t e = 0; e < m.g->n_expert; ++e) {
                                const int32_t c = m.cnt[(size_t) e];
                                if (c > 0 && !ppc->ps_flag.empty() && ppc->ps_flag[(size_t) l * m.g->n_expert + e]) {   // peer-streamed
                                    on_peer[(size_t) e] = 2;
                                    rows_peer += c;
                                    continue;
                                }
                                if (c == 0 || (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) ||
                                    !ppc->peer || !ppc->peer->has(l, e))
                                    continue;
                                if (rows_peer + c > ppc->cap_rows) { continue; }
                                on_peer[(size_t) e] = 1;
                                rows_peer += c;
                            }
                            rows_local = T * K - rows_peer;
                        }
                        if (cpu_maybe) {
                            // Non-resident experts with a host copy and few rows go to the CPU, fewest rows first, up to
                            // the share at which the CPU and the GPU's streaming end together.
                            std::vector<std::pair<int32_t, int32_t>> cand;   // (rows, expert)
                            for (int32_t e = 0; e < m.g->n_expert; ++e) {
                                const int32_t c = m.cnt[(size_t) e];
                                if (c == 0 || (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) ||
                                    (!on_peer.empty() && on_peer[(size_t) e]) ||
                                    (m.ring_peer && m.ring_peer->exclusive() && m.ring_peer->has(l, e)))
                                    continue;
                                ++n_stream;
                                int fd = -1;
                                uint64_t fo = 0;
                                if (c <= basalt::kernels::cpu::MAXT && (m.src->pinned(l, e) || !m.src->file_extent(l, e, fd, fo)))
                                    cand.emplace_back(c, e);
                            }
                            std::sort(cand.begin(), cand.end());
                            const size_t take = std::min(cand.size(), (size_t) std::llround(m.cpu_share_now * (double) n_stream));
                            if (take > 0) {
                                on_cpu.assign((size_t) m.g->n_expert, 0);
                                for (size_t i = 0; i < take; ++i) {
                                    on_cpu[(size_t) cand[i].second] = 1;
                                    rows_cpu += cand[i].first;
                                }
                                n_cpu = (int64_t) take;
                                rows_local -= rows_cpu;
                            }
                        }
                        {
                            int32_t r = 0;
                            for (int32_t e = 0; e < m.g->n_expert; ++e)
                                if ((on_peer.empty() || !on_peer[(size_t) e]) && (on_cpu.empty() || !on_cpu[(size_t) e])) {
                                    m.off[(size_t) e] = r;
                                    r += m.cnt[(size_t) e];
                                }
                            // the peer's rows in order_peer's order (peer-held, then peer-streamed): its groups need them contiguous
                            for (int kind = 1; kind <= 2 && !on_peer.empty(); ++kind)
                                for (int32_t e = 0; e < m.g->n_expert; ++e)
                                    if (on_peer[(size_t) e] == kind) { m.off[(size_t) e] = r; r += m.cnt[(size_t) e]; }
                            for (int32_t e = 0; e < m.g->n_expert && !on_cpu.empty(); ++e)   // the CPU's block, last
                                if (on_cpu[(size_t) e]) { m.off[(size_t) e] = r; r += m.cnt[(size_t) e]; }
                            m.off[(size_t) m.g->n_expert] = r;
                        }
                        std::vector<int32_t> fill(m.off.begin(), m.off.end() - 1);
                        for (int64_t i = 0; i < T * K; ++i) {
                            const int32_t e = ids_h[(size_t) i];
                            const int32_t p = fill[(size_t) e]++;
                            slot_h[(size_t) i] = p;
                            src_h[(size_t) p] = (int32_t) (i / K);
                        }
                        if (grp_mapped) {
                            copy_i32(m.slot_dev, m.grp_dev + m.grp_tk, T * K, m.cs);
                            copy_i32(m.src_dev, m.grp_dev + 2 * m.grp_tk, T * K, m.cs);
                        } else {
                            cudaMemcpyAsync(m.slot_dev, m.slot_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, m.cs);
                            cudaMemcpyAsync(m.src_dev, m.src_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, m.cs);
                        }
                        // the experts, in id order: resident ones from VRAM, the others through the staging ring
                        std::vector<int32_t> order, order_peer;
                        for (int32_t e = 0; e < m.g->n_expert; ++e)
                            if (m.cnt[(size_t) e] > 0 && (on_peer.empty() || on_peer[(size_t) e] != 2) &&
                                (on_cpu.empty() || !on_cpu[(size_t) e]))
                                (!on_peer.empty() && on_peer[(size_t) e] ? order_peer : order).push_back(e);
                        for (int32_t e = 0; e < m.g->n_expert && !on_peer.empty(); ++e)   // the peer-streamed ones last:
                            if (on_peer[(size_t) e] == 2) order_peer.push_back(e);        // their copies get the most time
                        // CPU share: the pool runs the CPU's experts while this thread stages and runs the GPU's; their
                        // unweighted rows land in Dm's row order, so the combine weights them like any other row
                        if (rows_cpu > 0) {
                            const size_t want = (size_t) rows_cpu * N;
                            if (m.cpu_rows_n < want) {
                                if (m.cpu_rows) cudaFreeHost(m.cpu_rows);
                                m.cpu_rows = nullptr;
                                m.cpu_rows_n = 0;
                                const size_t cap = want * 2;   // headroom: few reallocations
                                if (cudaHostAlloc((void**) &m.cpu_rows, cap * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
                                    err = "prefill: cannot allocate the CPU experts' rows";
                                    return false;
                                }
                                m.cpu_rows_n = cap;
                            }
                            const int64_t r0c = T * K - rows_cpu;
                            const basalt::kernels::cpu::NativeFmt& cf = lay.fmt[(size_t) l];
                            namespace kc = basalt::kernels::cpu;
                            const bool q2 = kc::q2_native_kernels(cf.gu_type);
                            if (q2 && m.cpu_actq.size() < (size_t) T) m.cpu_actq.resize((size_t) T);
                            if (!q2 && m.cpu_nact.size() < (size_t) T * kc::kNativeActBytes) m.cpu_nact.resize((size_t) T * kc::kNativeActBytes);
                            // the jobs are built here: the source's blob() is not for two threads at once
                            m.cpu_need.assign((size_t) T, 0);   // the tokens the CPU's experts read
                            m.cpu_jobs.clear();
                            for (int32_t e = 0; e < m.g->n_expert; ++e) {
                                if (!on_cpu[(size_t) e]) continue;
                                kc::ExpertJobMulti j;
                                j.blob = m.src->blob(l, e);
                                if (j.blob == nullptr) { err = "prefill: expert source has no blob"; return false; }
                                j.nt = m.cnt[(size_t) e];
                                for (int r = 0; r < j.nt; ++r) {
                                    const int64_t pr = (int64_t) m.off[(size_t) e] + r;
                                    const int32_t tk = src_h[(size_t) pr];
                                    m.cpu_need[(size_t) tk] = 1;
                                    if (q2) j.act[r] = &m.cpu_actq[(size_t) tk];
                                    else j.nact[r] = m.cpu_nact.data() + (size_t) tk * kc::kNativeActBytes;
                                    j.out[r] = m.cpu_rows + (size_t) (pr - r0c) * N;
                                }
                                m.cpu_jobs.push_back(j);
                            }
                            cpu_fut = std::async(std::launch::async, [&m, &cf, &cpu_ms, q2, T, this]() -> bool {
                                const auto t0 = Clock::now();
                                cudaEventSynchronize(m.cpu_ev_x);   // the activations are on the host
                                for (int64_t t = 0; t < T; ++t) {
                                    if (!m.cpu_need[(size_t) t]) continue;
                                    if (q2) kc::act_quant_any(m.cpu_x + (size_t) t * N, (int) N, m.cpu_actq[(size_t) t]);
                                    else kc::native_quant_act(cf, m.cpu_x + (size_t) t * N, m.cpu_nact.data() + (size_t) t * kc::kNativeActBytes);
                                }
                                cpu_pool_->run_split_multi_native(cf, m.cpu_jobs.data(), (int) m.cpu_jobs.size());
                                cpu_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                                return true;
                            });
                        }
                        const size_t mmq_gub = use_mmq ? mmq::matrix_bytes(mmq_gt, 1280, N) : 0;
                        const size_t mmq_db = use_mmq ? mmq::matrix_bytes(mmq_dt, N, 640) : 0;
                        if (use_mmq) {
                            // the layer's activations as q8_1 rows in expert order, straight from `mixed`
                            if (!fp4_l) mmq::quantize(m.mixed, m.src_dev, m.Xq, mmq_gt, N, N, T * K, m.cs);
                            // each group's rows: absolute bounds (gate/up reads the layer's rows), relative ones (down
                            // reads the group's own quantized H)
                            const size_t n = order.size(), ng = (n + MMQ_GROUP - 1) / MMQ_GROUP;
                            m.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
                            for (size_t j = 0; j < n; ++j) m.bounds_host[j] = m.off[(size_t) order[j]];
                            m.bounds_host[n] = (int32_t) rows_local;
                            for (size_t g = 0; g < ng; ++g)
                                for (size_t i = 0; i <= MMQ_GROUP; ++i)
                                    m.bounds_host[n + 1 + g * (MMQ_GROUP + 1) + i] =
                                        m.bounds_host[std::min(n, g * MMQ_GROUP + i)] - m.bounds_host[g * MMQ_GROUP];
                            if (grp_mapped) {
                                int32_t* bh = m.grp_host + 3 * m.grp_tk;
                                std::memcpy(bh, m.bounds_host.data(), m.bounds_host.size() * 4);
                                copy_i32(m.bounds_dev, m.grp_dev + 3 * m.grp_tk, (int64_t) m.bounds_host.size(), m.cs);
                            } else {
                                cudaMemcpyAsync(m.bounds_dev, m.bounds_host.data(), m.bounds_host.size() * 4,
                                                cudaMemcpyHostToDevice, m.cs);
                            }
                            if (fp4_l) {   // the activations as FP4 tiles in expert order
                                const Fp4Ptrs fp = fp4_ptrs(m.Xq, m.Hq, m.T, m.g->n_expert);
                                fp4::plan(m.bounds_dev, (int) n, fp.tox, m.cs);
                                fp4::quantize(m.mixed, m.src_dev, m.bounds_dev, fp.tox, (int) n, fp.a4x, fp.gsx, N, N,
                                              rows_local, m.cs);
                            }
                        } else {
                            gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs);
                        }
                        // multi-GPU: the peer's share, enqueued before the primary's own experts so both cards work at once
                        peer_now = use_mmq && !order_peer.empty();
                        if (peer_now) {
                            PeerPrefill& P = *ppc;
                            cudaEventRecord(P.ev_in, m.cs);
                            int prevd = 0;
                            cudaGetDevice(&prevd);
                            cudaSetDevice(P.dev);
                            const cudaStream_t ps = P.s;
                            cudaStreamWaitEvent(ps, P.ev_in, 0);
                            if (P.out_pending) { cudaStreamWaitEvent(ps, P.ev_done, 0); P.out_pending = false; }
                            cudaMemcpyPeerAsync(P.mixed, P.dev, m.mixed, prevd, (size_t) T * N * 4, ps);
                            cudaMemcpyAsync(P.src, src_h + rows_local, (size_t) rows_peer * 4, cudaMemcpyHostToDevice, ps);
                            const size_t n = order_peer.size();
                            {
                                // groups of up to MMQ_GROUP experts and at most G rows, each computed from its own rows
                                P.groups.clear();
                                {
                                    size_t g0 = 0;
                                    int64_t gr = 0;
                                    for (size_t j = 0; j < n; ++j) {
                                        const int64_t c = m.cnt[(size_t) order_peer[j]];
                                        if (j > g0 && (j - g0 == (size_t) MMQ_GROUP || gr + c > P.G)) { P.groups.push_back({g0, j}); g0 = j; gr = 0; }
                                        gr += c;
                                    }
                                    if (n > g0) P.groups.push_back({g0, n});
                                }
                                const size_t ng = P.groups.size();
                                P.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
                                for (size_t j = 0; j < n; ++j) P.bounds_host[j] = m.off[(size_t) order_peer[j]] - (int32_t) rows_local;
                                P.bounds_host[n] = (int32_t) rows_peer;
                                for (size_t g2 = 0; g2 < ng; ++g2)
                                    for (size_t i = 0; i <= MMQ_GROUP; ++i)
                                        P.bounds_host[n + 1 + g2 * (MMQ_GROUP + 1) + i] =
                                            P.bounds_host[std::min(P.groups[g2].second, P.groups[g2].first + i)] -
                                            P.bounds_host[P.groups[g2].first];
                                cudaMemcpyAsync(P.bounds, P.bounds_host.data(), P.bounds_host.size() * 4, cudaMemcpyHostToDevice, ps);
                                const auto& f = lay.fmt[(size_t) l];
                                for (size_t g2 = 0; g2 < ng; ++g2) {
                                    const size_t j0 = P.groups[g2].first, j1 = P.groups[g2].second;
                                    const int ngx = (int) (j1 - j0);
                                    int64_t maxr = 0;
                                    for (size_t j = j0; j < j1; ++j) {
                                        const int32_t e = order_peer[j];
                                        maxr = std::max<int64_t>(maxr, m.cnt[(size_t) e]);
                                        const uint8_t* bd = nullptr;
                                        int psl = -1;
                                        if (on_peer[(size_t) e] == 2) {   // from the peer's ring
                                            p_release_to(l, e, ps);
                                            if (P.pk >= P.pseq_start[(size_t) l + 1] || P.pseq[P.pk].e != e) {
                                                err = "prefill: a peer-streamed expert is not next in the peer's ring";
                                                cudaSetDevice(prevd);
                                                return false;
                                            }
                                            psl = (int) (P.pk % (size_t) P.RP);
                                            cudaStreamWaitEvent(ps, P.pcopied[(size_t) psl], 0);
                                            bd = P.pstage[(size_t) psl];
                                        } else {
                                            bd = P.peer->slot_ptr(l, e);
                                        }
                                        const size_t q = j - j0;
                                        if (lay.native)
                                            mmq::gather_native(bd, bd + f.up_off, mmq_gub / 2, bd + f.down_off, mmq_db,
                                                               P.grp_gu + q * mmq_gub, P.grp_d + q * mmq_db, ps,
                                                               f.tail ? bd + f.bytes - f.tail : nullptr,
                                                               grp_scales(P.grp_d) + 4 * q);
                                        else
                                            mmq::gather_q2(bd, P.grp_gu + q * mmq_gub, P.grp_d + q * mmq_db, ps);
                                        if (psl >= 0) {   // the slot is free once gathered
                                            cudaEventRecord(P.pused[(size_t) psl], ps);
                                            ++P.pk;
                                            p_issue_until(P.pk + (size_t) P.RP);
                                        }
                                    }
                                    const int64_t r0 = P.bounds_host[j0], nr = P.bounds_host[j1] - r0;
                                    if (nr <= 0) continue;
                                    const int32_t* rel = P.bounds + n + 1 + g2 * (MMQ_GROUP + 1);
                                    cudaMemsetAsync(P.grp_gu + (size_t) ngx * mmq_gub, 0, MMQ_TAIL, ps);
                                    cudaMemsetAsync(P.grp_d + (size_t) ngx * mmq_db, 0, MMQ_TAIL, ps);
                                    mmq::quantize(P.mixed, P.src + r0, P.Xq_g, mmq_gt, N, N, nr, ps);
                                    mmq::Product gu;
                                    gu.w = P.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                                    gu.n = ngx; gu.xq = P.Xq_g; gu.bounds = rel; gu.ids = P.ident;
                                    gu.total_rows = nr; gu.max_rows = maxr; gu.dst = P.GU_g; gu.ld_dst = 1280;
                                    P.run_ctx->run(gu, ps);
                                    mmq::swiglu(P.GU_g, P.H_g, nr, 640, !lay.native, ps,
                                                lay.native && f.tail ? grp_scales(P.grp_d) : nullptr, rel, ngx);
                                    mmq::quantize(P.H_g, nullptr, P.Hq_g, mmq_dt, 640, 640, nr, ps);
                                    const int b = (int) (g2 & 1);
                                    if (P.dm_live[b]) cudaStreamWaitEvent(ps, P.ev_dm[b], 0);   // its last rows have left
                                    mmq::Product dn;
                                    dn.w = P.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                                    dn.n = ngx; dn.xq = P.Hq_g; dn.bounds = rel;
                                    dn.ids = P.ident; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = P.Dm_b[b];
                                    dn.ld_dst = N;
                                    P.run_ctx->run(dn, ps);
                                    cudaEventRecord(P.ev_grp[g2 % PeerPrefill::kGrpEv], ps);
                                    cudaStreamWaitEvent(P.s_out, P.ev_grp[g2 % PeerPrefill::kGrpEv], 0);
                                    cudaMemcpyPeerAsync(m.Dm + (size_t) (rows_local + r0) * N, prevd, P.Dm_b[b], P.dev,
                                                        (size_t) nr * N * 4, P.s_out);
                                    cudaEventRecord(P.ev_dm[b], P.s_out);
                                    P.dm_live[b] = true;
                                }
                            }
                            cudaEventRecord(P.ev_done, P.s_out);
                            P.out_pending = true;
                            cudaSetDevice(prevd);
                        }
                        if (ps_on) {   // multi-GPU: this layer's peer-ring entries the routing did not pick give their slots back
                            int pd = 0;
                            cudaGetDevice(&pd);
                            cudaSetDevice(ppc->dev);
                            p_release_to(l, (int32_t) m.g->n_expert, ppc->s);
                            cudaSetDevice(pd);
                        }
                        // --peer-exclusive: an expert only the peer holds has no host copy.  It is copied from the peer's
                        // memory into a staging slot, or (with direct access between the cards) read in place.
                        auto peer_ring = [&](int32_t e) {
                            return m.ring_peer && m.ring_peer->exclusive() && m.ring_peer->has(l, e) &&
                                   !(m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0);
                        };
                        auto peer_direct = [&](int32_t e) { return peer_ring(e) && m.ring_peer->p2p(); };
                        // Stage ahead: the copy stream moves blobs host -> device while the compute stream works.
                        int stage_next = 0;
                        std::vector<int> stage_of(order.size(), -1);
                        // the unpinned ones are copied to pinned buffers by the stager's threads, in this order
                        std::vector<int> job_of(order.size(), -1);
                        if (!stream_all) {
                            std::vector<Stager::Job> js;
                            for (size_t j = 0; j < order.size(); ++j) {
                                const int32_t e = order[j];
                                if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                                if (peer_ring(e)) continue;
                                if (m.src->pinned(l, e)) continue;
                                job_of[j] = (int) js.size();
                                const uint8_t* b = m.src->blob(l, e);
                                if (!b) { err = "prefill: expert source has no blob"; return false; }
                                js.push_back(make_job(*m.src, l, e, b, (size_t) lay.blob_bytes(l)));
                            }
                            m.stager->start(std::move(js));
                        }
                        StagerDone stager_done{stream_all ? nullptr : m.stager.get()};
                        auto stage_one = [&](size_t j) -> bool {
                            const int32_t e = order[j];
                            const bool resident = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                            if (resident || peer_direct(e)) return true;
                            const int sl = stage_next;
                            stage_next = (stage_next + 1) % STAGE;
                            const auto th = Clock::now();
                            const bool via_peer = peer_ring(e);
                            const bool pinned = !via_peer && m.src->pinned(l, e);
                            const uint8_t* b = pinned ? m.src->blob(l, e) : nullptr;
                            if (pinned && !b) { err = "prefill: expert source has no blob"; return false; }
                            if (via_peer) {
                                if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
                                if (peer_async) m.peer_stage.copy(m.stage_dev[sl], m.ring_peer->slot_ptr(l, e),
                                                                  (size_t) lay.blob_bytes(l), m.copy);
                                else cudaMemcpyPeerAsync(m.stage_dev[sl], m.device, m.ring_peer->slot_ptr(l, e),
                                                         m.ring_peer->device(), (size_t) lay.blob_bytes(l), m.copy);
                                ++stats_.experts_dma;
                            } else if (pinned) {
                                // DMA straight from the page-locked arena: the copy stream only waits for the slot
                                if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
                                cudaMemcpyAsync(m.stage_dev[sl], b, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
                                ++stats_.experts_dma;
                            } else {
                                // copied to a pinned buffer by the stager (waits only if it is behind), then DMA
                                const uint8_t* hb = m.stager->wait(job_of[j]);
                                if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
                                cudaMemcpyAsync(m.stage_dev[sl], hb, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
                                m.stager->issued_one(job_of[j], m.copy);
                            }
                            cudaEventRecord(m.copied[sl], m.copy);
                            m.stage_live[sl] = true;
                            stage_of[j] = sl;
                            stats_.ms_experts_host += ms_since(th);
                            ++stats_.experts_streamed;
                            return true;
                        };
                        // In the streamed walk an MMQ group is gathered in ONE launch, after ONE wait on its last
                        // streamed copy (the copy stream is in order), and its ring slots are released by ONE event.  A
                        // wait and a record per expert would put a GPU bubble each on the compute stream.  The same bytes into the same group slots.  A ring
                        // entry the routing skipped inside an open group first gathers what the group holds so far
                        // (`flush`), so no more than a group's entries are ever held back from the issuer.
                        const bool group_gather = stream_all && use_mmq && lay.native &&
                                                  MMQ_GROUP <= mmq::kGatherGroupMax;
                        mmq::GatherGroup gg;
                        int gg_slots[MMQ_GROUP];
                        int gg_nslots = 0;   // ring slots gathered by the next flush
                        // the FP4 path's open group: its experts' blobs (read in place until the down product is done)
                        const uint8_t* fp4_blob[MMQ_GROUP];
                        size_t fp4_j0 = 0;
                        int fp4_n = 0;
                        auto fp4_flush = [&]() {
                            if (fp4_n == 0) return;
                            if (gg_nslots > 0) cudaStreamWaitEvent(m.cs, m.copied[gg_slots[gg_nslots - 1]], 0);
                            const Fp4Ptrs fp = fp4_ptrs(m.Xq, m.Hq, m.T, m.g->n_expert);
                            // runs of consecutive experts of at most the rows GU and H hold: GU and H index a run's own
                            // rows, the layer's Dm and activation planes the absolute ones
                            fp4::Group fg;
                            fg.n = fp4_n;
                            for (int i = 0; i < fp4_n; ++i) fg.blob[i] = fp4_blob[i];
                            fg.bounds_host = m.bounds_host.data() + fp4_j0;
                            fg.bounds_dev = m.bounds_dev + fp4_j0;
                            fg.a4x = fp.a4x; fg.gsx = fp.gsx; fg.tox = fp.tox + fp4_j0;
                            fg.down_off = lay.fmt[(size_t) l].down_off;
                            fg.tail_off = lay.fmt[(size_t) l].bytes - lay.fmt[(size_t) l].tail;
                            fg.gu = m.GU; fg.h = m.H; fg.dm = m.Dm;
                            fg.a4h = fp.a4h; fg.gsh = fp.gsh; fg.toh = fp.toh; fg.rel = fp.rel;
                            fg.tails = grp_scales(m.grp_d);
                            fp4::run_group(fg, (int64_t) group_rows_cap((size_t) m.T), m.cs);
                            if (gg_nslots > 0) {   // the blobs are read: give the ring slots back
                                const int rel_slot = gg_slots[gg_nslots - 1];
                                cudaEventRecord(m.used[rel_slot], m.cs);
                                for (int i = 0; i < gg_nslots; ++i) m.used_of[gg_slots[i]] = rel_slot;
                                gg_nslots = 0;
                            }
                            fp4_n = 0;
                        };
                        auto flush = [&]() {
                            if (fp4_l) { fp4_flush(); return; }
                            if (gg.n <= gg.first) return;
                            const auto& f = lay.fmt[(size_t) l];
                            if (gg_nslots > 0) {   // the copies land in order: the last one covers the others
                                cudaStreamWaitEvent(m.cs, m.copied[gg_slots[gg_nslots - 1]], 0);
                            }
                            float* sc = f.tail ? grp_scales(m.grp_d) : nullptr;
                            if (!mmq::gather_native_group(gg, f.up_off, mmq_gub / 2, f.down_off, mmq_db, m.grp_gu, mmq_gub,
                                                          m.grp_d, mmq_db, m.cs, sc ? f.bytes - f.tail : 0, sc)) {
                                for (int i = gg.first; i < gg.n; ++i) {   // not 16-byte aligned: one at a time
                                    const uint8_t* b = gg.blob[i];
                                    mmq::gather_native(b, b + f.up_off, mmq_gub / 2, b + f.down_off, mmq_db,
                                                       m.grp_gu + i * mmq_gub, m.grp_d + i * mmq_db, m.cs,
                                                       sc ? b + f.bytes - f.tail : nullptr, sc ? sc + 4 * i : nullptr);
                                }
                            }
                            if (gg_nslots > 0) {
                                const int rel = gg_slots[gg_nslots - 1];
                                cudaEventRecord(m.used[rel], m.cs);
                                for (int i = 0; i < gg_nslots; ++i) m.used_of[gg_slots[i]] = rel;
                            }
                            gg_nslots = 0;
                            gg.first = gg.n;
                        };
                        // one expert's products from its blob on the device; `slot` (a ring slot, or -1 for a resident
                        // expert) is released once the blob is read
                        auto compute = [&](size_t j, const uint8_t* blob_dev, int slot) -> bool {
                            const int32_t e = order[j];
                            if (fp4_l) {
                                if (fp4_n == 0) fp4_j0 = j;
                                fp4_blob[fp4_n++] = blob_dev;
                                if (slot >= 0) gg_slots[gg_nslots++] = slot;
                                // a group at the walk's end of MMQ_GROUP experts; outside the streamed walk the staging
                                // ring reuses a slot as soon as the next expert starts, so one expert at a time
                                if (group_gather && j % MMQ_GROUP != MMQ_GROUP - 1 && j + 1 < order.size()) return true;
                                fp4_flush();
                                return true;
                            }
                            if (use_mmq) {
                                // gather the expert into its group slot (GGUF blocks, unchanged or converted)
                                const size_t q = j % MMQ_GROUP;
                                if (group_gather) {
                                    gg.blob[q] = blob_dev;
                                    gg.n = (int) q + 1;
                                    if (slot >= 0) gg_slots[gg_nslots++] = slot;
                                    if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
                                    flush();
                                    gg = mmq::GatherGroup{};
                                } else if (lay.native) {
                                    const auto& f = lay.fmt[(size_t) l];
                                    mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gub / 2, blob_dev + f.down_off,
                                                       mmq_db, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs,
                                                       f.tail ? blob_dev + f.bytes - f.tail : nullptr,
                                                       grp_scales(m.grp_d) + 4 * q);
                                } else {
                                    mmq::gather_q2(blob_dev, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
                                }
                                if (slot >= 0 && !group_gather) { cudaEventRecord(m.used[slot], m.cs); m.used_of[slot] = slot; }
                                if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
                                // the group's products: gate/up, swiglu, the group's H to q8_1, down
                                const size_t j0 = j - q, g = j0 / MMQ_GROUP, n = order.size();
                                const int ngx = (int) (q + 1);
                                const int64_t r0 = m.bounds_host[j0], nr = m.bounds_host[j + 1] - r0;
                                int64_t maxr = 0;
                                for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
                                // the zeroed tail after the group's last expert (see MMQ_TAIL)
                                cudaMemsetAsync(m.grp_gu + (size_t) ngx * mmq_gub, 0, MMQ_TAIL, m.cs);
                                cudaMemsetAsync(m.grp_d + (size_t) ngx * mmq_db, 0, MMQ_TAIL, m.cs);
                                mmq::Product gu;
                                gu.w = m.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                                gu.n = ngx; gu.xq = m.Xq; gu.bounds = m.bounds_dev + j0; gu.ids = m.ids_identity;
                                gu.total_rows = T * K; gu.max_rows = maxr; gu.dst = m.GU; gu.ld_dst = 1280;
                                m.mmq_ctx->run(gu, m.cs);
                                const bool scaled = lay.native && lay.fmt[(size_t) l].tail != 0;
                                mmq::swiglu(m.GU + r0 * 1280, m.H + r0 * 640, nr, 640, !lay.native, m.cs,
                                            scaled ? grp_scales(m.grp_d) : nullptr,
                                            m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1), ngx);
                                mmq::quantize(m.H + r0 * 640, nullptr, m.Hq, mmq_dt, 640, 640, nr, m.cs);
                                mmq::Product dn;
                                dn.w = m.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                                dn.n = ngx; dn.xq = m.Hq; dn.bounds = m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1);
                                dn.ids = m.ids_identity; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = m.Dm + r0 * N;
                                dn.ld_dst = N;
                                m.mmq_ctx->run(dn, m.cs);
                                return true;
                            }
                            const int q = (int) (j % DQ);
                            if (lay.native) {
                                // a native pack's layer, dequantized by llama.cpp's own formulas
                                const auto& f = lay.fmt[(size_t) l];
                                // NVFP4: the tail's FP32 scales {s_gate, s_up, s_down} go into the fp16 weights
                                const float* tail = f.tail ? (const float*) (blob_dev + f.bytes - f.tail) : nullptr;
                                basalt::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff, f.n_embd,
                                                                   m.dq_gu[q], m.cs, tail);
                                basalt::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, m.dq_d[q], m.cs,
                                                                tail ? tail + 2 : nullptr);
                            } else {
                                blob_dequant_f16(blob_dev, m.dq_gu[q], m.dq_d[q], m.cs);
                            }
                            if (slot >= 0) { cudaEventRecord(m.used[slot], m.cs); m.used_of[slot] = slot; }
                            const int64_t o0 = m.off[(size_t) e], ne = m.cnt[(size_t) e];
                            m.gemm.f16(m.Xs + o0 * N, m.dq_gu[q], m.GU + o0 * 1280, ne, 1280, N);
                            swiglu_interleaved(m.GU + o0 * 1280, m.Hh + o0 * 640, ne, m.cs);
                            m.gemm.f16(m.Hh + o0 * 640, m.dq_d[q], m.Dm + o0 * N, ne, N, 640);
                            return true;
                        };
                        if (cpu_maybe) cudaEventRecord(m.cpu_gev[0], m.cs);
                        if (!stream_all) {
                            size_t staged = 0;
                            const size_t lookahead = STAGE - 1;
                            for (size_t j = 0; j < order.size(); ++j) {
                                while (staged < order.size() && staged <= j + lookahead) {
                                    if (!stage_one(staged)) return false;
                                    ++staged;
                                }
                                const int32_t e = order[j];
                                if (stage_of[j] < 0) {
                                    ++stats_.experts_resident;
                                    const int32_t rs = m.host_res ? m.host_res[(size_t) l * m.g->n_expert + e] : -1;
                                    const uint8_t* bd = rs >= 0 && m.cache ? m.cache->device_slot(rs) : m.ring_peer->slot_ptr(l, e);
                                    if (!compute(j, bd, -1)) return false;
                                } else {
                                    cudaStreamWaitEvent(m.cs, m.copied[stage_of[j]], 0);
                                    if (!compute(j, m.stage_dev[stage_of[j]], stage_of[j])) return false;
                                }
                            }
                        } else {
                            // the streamed walk: this layer's entries [k, kend) in id order; an entry the routing did not
                            // pick only gives its slot back
                            size_t k = seq_start[(size_t) l];
                            const size_t kend = seq_start[(size_t) l + 1];
                            auto release_to = [&](int32_t e_stop) {
                                while (k < kend && seq[k].e < e_stop) {
                                    if (gg_nslots > 0) flush();   // the open group's slots get their event first
                                    const int sl = (int) (k % (size_t) m.ring);
                                    cudaEventRecord(m.used[sl], m.cs);
                                    m.used_of[sl] = sl;
                                    consumed = ++k;
                                    give_back(consumed);
                                }
                            };
                            for (size_t j = 0; j < order.size(); ++j) {
                                const int32_t e = order[j];
                                release_to(e);
                                if (k < kend && seq[k].e == e) {
                                    const int sl = (int) (k % (size_t) m.ring);
                                    wait_issued(k);
                                    if (!group_gather) {
                                        cudaStreamWaitEvent(m.cs, m.copied[sl], 0);
                                    }
                                    if (!compute(j, m.stage_dev[sl], sl)) return false;
                                    consumed = ++k;
                                    if (!group_gather || gg_nslots == 0) give_back(consumed);   // its group was gathered
                                } else {
                                    const bool r0 = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                                    const uint8_t* bp = r0 ? m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e])
                                                           : (ppc && ppc->peer ? ppc->peer->slot_ptr(l, e) : nullptr);   // over the cap: P2P
                                    if (bp == nullptr) { err = "prefill: an expert is neither resident, streamed nor on the peer"; return false; }
                                    ++stats_.experts_resident;
                                    if (!compute(j, bp, -1)) return false;
                                    if (group_gather && gg_nslots == 0) give_back(consumed);
                                }
                            }
                            release_to(m.g->n_expert);
                        }
                    }
                    if (cpu_fut.valid()) {   // CPU share: its rows into Dm's tail
                        cudaEventRecord(m.cpu_gev[1], m.cs);   // after the GPU's expert work
                        if (!cpu_fut.get()) { err = "prefill: a CPU expert has no blob"; return false; }
                        cudaMemcpyAsync(m.Dm + (size_t) (T * K - rows_cpu) * N, m.cpu_rows, (size_t) rows_cpu * N * sizeof(float),
                                        cudaMemcpyHostToDevice, m.cs);
                        stats_.experts_cpu += n_cpu;
                        stats_.cpu_share = m.cpu_share_now;
                    }
                    if (peer_now) {   // multi-GPU: the peer's rows are in Dm (or, without P2P, in host memory)
                        cudaStreamWaitEvent(m.cs, ppc->ev_done, 0);
                    }
                    moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs);
                }
                // ---- the hyper-connection write of this half; fused with the next half's norm when nothing else
                // touches R in between (not the stage's last half, not before the PLE block of layer 1, not under a
                // control vector)
                const int64_t nl = half == 0 ? l : l + 1;
                const bool fuse = nl < LE && !(half == 1 && nl == 1 && ple_on) &&
                                  !(half == 1 && basalt::kernels::cvec().covers(l));
                const core::WeightRef* wnn = nullptr;
                if (fuse) {
                    const core::LayerView vn(*m.wt, nl);
                    wnn = need(vn, half == 0 ? "hc_ffn_norm.weight" : "hc_attn_norm.weight", err);
                    if (!wnn) return false;
                }
                if (wnn) {
                    gr_write_norm_rs(m.R, m.bo, m.inj, HC, (const float*) wnn->data, EPS, m.grs, m.xn16, T, m.cs);
                    normed = true;
                } else {
                    gr_write(m.R, m.bo, m.inj, HC, T, m.cs);
                }
                if (half == 1 && basalt::kernels::cvec().covers(l))   // --control-vector-scaled
                    basalt::kernels::cvec_apply(m.R, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
            }
        }
        if (!ple_land()) return false;   // a stage that ends before layer 1: the rows land anyway, the next gather starts
        if (issuer.joinable()) {
            issuer.join();
            stats_.ms_experts_host += iss_ms;
            stats_.experts_streamed += iss_streamed;
            stats_.experts_dma += iss_dma;
        }
        stats_.tokens += T;
        core::progress_at("reading the prompt (batched): finishing the chunk from token", p0);
        if (next_ != nullptr) {
            // The current hand-off slot was used two chunks ago.
            float* h = m.hand[hand_buf_];
            if (cudaMemcpyAsync(h, m.R, (size_t) T * D * 4, cudaMemcpyDeviceToHost, m.cs) != cudaSuccess ||
                cudaStreamSynchronize(m.cs) != cudaSuccess) {
                err = std::string("prefill: the layer split's hand-off: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            // Wait only for the DIRECT successor's previous chunk. That successor
            // may already have forwarded its older chunk to later GPUs.
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            if (next_run_.valid() && !next_run_.get()) { err = next_err_; return false; }
            next_err_.clear();
            next_->hand_in_ = h;
            next_->single_chunk_ = single_chunk;
            next_run_ = std::async(std::launch::async, [this, tokens, c0, T, p0] {
                return next_->run_impl(tokens + c0, T, p0, next_err_);
            });
            hand_buf_ ^= 1;
            continue;   // the last stage reports the chunk (on_chunk)
        }
        if (on_chunk || on_stage_chunk) {
            if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
                err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            if (on_chunk && !on_chunk(m.R, T, p0, err)) return false;
        }
    }
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
    stats_.ms_total += ms_since(t_start);
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

void Prefill::route_hist_reset() {
    Impl& m = *impl_;
    if (m.g == nullptr) return;
    {
        const core::OnDevice od(m.device);
        if (m.hist == nullptr) {
            m.hist_n = m.g->n_layers * m.g->n_expert;
            if (cudaMalloc((void**) &m.hist, (size_t) m.hist_n * 4) != cudaSuccess) { m.hist = nullptr; m.hist_on = false; }
            else m.owned.push_back(m.hist);
        }
        if (m.hist != nullptr) {
            cudaStreamSynchronize(m.cs);
            cudaMemset(m.hist, 0, (size_t) m.hist_n * 4);
        }
        m.hist_tokens = 0;
        m.hist_on = m.hist != nullptr;
    }
    if (next_ != nullptr) next_->route_hist_reset();
}

bool Prefill::route_hist_take(std::vector<uint32_t>& out, int64_t& tokens) {
    Impl& m = *impl_;
    if (!m.hist_on) return false;
    m.hist_on = false;
    {
        const core::OnDevice od(m.device);
        cudaStreamSynchronize(m.cs);
        std::vector<uint32_t> h((size_t) m.hist_n);
        if (cudaMemcpy(h.data(), m.hist, (size_t) m.hist_n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        if (out.size() != h.size()) out.assign(h.size(), 0u);
        for (size_t i = 0; i < h.size(); ++i) out[i] += h[i];
        tokens = std::max(tokens, m.hist_tokens);
    }
    if (next_ != nullptr) next_->route_hist_take(out, tokens);
    return true;
}

bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    const bool body_ok = run_impl(tokens, n, pos0, err);

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
