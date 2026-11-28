// include/basalt/model/batch_driver.hpp - decode_batch for any model (docs/architecture/batching.md). A model's session derives
// BatchSession and writes the hooks below: its rows' inputs to the card, the forward of a window of segments, each session's
// logits back and its draws. The driver does the rest: the checks, the segments, each session's head rows, the graph of each
// composition (its first window direct, then captured behind it; the most recently used kept), the order of the streams
// and one sync for the whole batch. A model whose host steers its forward between launches (experts placed by the host
// between per-layer graphs) runs its forward directly every window and keeps its own graphs (batch_captured false); a
// session that cannot batch (batch_ready false) has the batch's windows run in turn.
#pragma once

#include "basalt/core/graph.hpp"
#include "basalt/log.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/model/batch_window.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace basalt::model {

class BatchSession;

/// A batch window in flight: its segments, its sessions (the leader, whose scratch and stream it runs on, first), their
/// requests and each one's head rows (the last `head[i]` rows of segment i take the head: every row for row_tokens, the
/// last for next_token or logits, none otherwise).
struct BatchView {
    const BatchWindow& w;
    BatchSession* const* ss;
    const DecodeRequest* reqs;
    const int* head;
    int n;
    cudaStream_t stream;
};

/// What every session of one model shares for its batch windows: the graphs by composition. A model keeps one beside its
/// shared forward scratch; sessions that return the same one batch together.
struct BatchShared {
    static constexpr size_t kGraphs = 128;
    static constexpr size_t kSeen = 256;
    GraphLru<core::CapturedGraph> graphs{kGraphs};
    /// `seen` values: a composition's windows so far ran direct once; or its graph found no device memory.
    static constexpr uint8_t kSeenOnce = 1, kGraphless = 2;
    GraphLru<uint8_t> seen{kSeen};
    bool graph_short_noted = false;
};

class BatchSession : public ModelSession {
public:
    Result decode_batch(ModelSession* const* peers, const DecodeRequest* reqs, int n) final {
        if (n == 1) return peers[0]->decode_window(reqs[0]);
        const std::string who = batch_name();
        std::vector<BatchSession*> ss;
        for (int i = 0; i < n; ++i) {
            BatchSession* s = dynamic_cast<BatchSession*>(peers[i]);
            if (s == nullptr || &s->batch_shared() != &batch_shared() || std::find(ss.begin(), ss.end(), s) != ss.end())
                return {Status::Error, who + ": a batch of sessions that are not distinct sessions of this model"};
            ss.push_back(s);
        }
        if (ss[0] != this) return {Status::Error, who + ": a batch runs on its first session"};
        if (std::any_of(ss.begin(), ss.end(), [](const BatchSession* s) { return !s->batch_ready(); })) {
            for (int i = 0; i < n; ++i)
                if (Result r = peers[i]->decode_window(reqs[i]); !r.ok()) return r;
            return {};
        }
        BatchWindow w;
        std::string err;
        if (!w.build(reqs, n, batch_rows(), batch_context(), err)) return {Status::Error, who + ": " + err};
        const cudaStream_t st = batch_stream();
        // every session readies first (what each queues on its own stream runs side by side), then the shared scratch and
        // the caches are the leader's stream's: whatever another session queued has run
        for (int i = 0; i < n; ++i)
            if (!ss[(size_t) i]->batch_reserve(reqs[i], err)) return {Status::Refused, who + ": " + err};
        for (int i = 1; i < n; ++i)
            if (cudaStreamSynchronize(ss[(size_t) i]->batch_stream()) != cudaSuccess) return {Status::Error, who + ": a session's stream failed"};
        std::vector<int> head((size_t) n, 0);
        for (int i = 0; i < n; ++i) {
            const DecodeRequest& r = reqs[i];
            head[(size_t) i] = r.row_tokens ? r.n_tokens : (r.next_token || r.logits) ? 1 : 0;
        }
        const BatchView v{w, ss.data(), reqs, head.data(), n, st};
        GraphLru<core::CapturedGraph>::Key key;
        if (!batch_stage(v, key, err)) return {Status::Error, who + ": " + err};
        struct End {   // every exit after the stage ends the window (batch_end): an early one here, the last after its sync
            BatchSession& s;
            bool open = true;
            ~End() {
                if (open) s.batch_end(std::nullopt);
            }
        } end{*this};
        if (batch_key_per_seg()) {
            for (int i = 0; i < n; ++i)
                key.insert(key.end(), {(uint64_t) (uintptr_t) ss[(size_t) i], (uint64_t) w.segs[(size_t) i].rows, (uint64_t) head[(size_t) i]});
        } else {
            for (int i = 0; i < n; ++i) key.push_back((uint64_t) (uintptr_t) ss[(size_t) i]);
            key.push_back((uint64_t) w.rows);
        }
        BatchShared& sh = batch_shared();
        if (!batch_captured()) {
            if (!batch_forward(v, err)) {
                cudaStreamSynchronize(st);
                return {Status::Error, who + ": " + err};
            }
        } else if (core::CapturedGraph* g = sh.graphs.find(key)) {
            if (!g->launch(st, err)) return {Status::Error, who + ": " + err};
        } else if (uint8_t* seen = sh.seen.find(key); seen == nullptr) {
            sh.seen.insert(std::move(key), uint8_t(BatchShared::kSeenOnce));
            if (!batch_forward(v, err)) return {Status::Error, who + ": " + err};
        } else if (*seen == BatchShared::kGraphless) {
            if (!batch_forward(v, err)) return {Status::Error, who + ": " + err};
        } else {
            // a composition's first window runs as launched, then is captured for the next ones: the graph is built while
            // the card runs the window
            if (!batch_forward(v, err)) return {Status::Error, who + ": " + err};
            core::CapturedGraph fresh;
            if (!fresh.begin(st, err)) return {Status::Error, who + ": " + err};
            const auto drop = [&] {   // the stream leaves the capture before the window's end syncs it
                cudaGraph_t dropped = nullptr;
                cudaStreamEndCapture(st, &dropped);
                if (dropped) cudaGraphDestroy(dropped);
                cudaGetLastError();
            };
            bool ok = false;
            try {
                ok = batch_forward(v, err);
            } catch (...) {
                drop();
                throw;
            }
            if (!ok) {
                drop();
                return {Status::Error, who + ": " + err};
            }
            if (!fresh.end(st, err)) {
                // the graph's own device memory is what the card is short of: this window ran direct above and the
                // composition keeps running without a graph; any other failure fails the window
                if (fresh.status() != cudaErrorMemoryAllocation) return {Status::Error, who + ": " + err};
                (void) cudaGetLastError();
                err.clear();
                if (!sh.graph_short_noted) {
                    sh.graph_short_noted = true;
                    basalt::log::warn("basalt batch", "%s: a window's graph found no device memory: windows of its composition run without one",
                                      who.c_str());
                }
                *seen = BatchShared::kGraphless;
            } else {
                sh.graphs.insert(std::move(key), std::move(fresh));
            }
        }
        if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return {Status::Error, who + ": CUDA: " + cudaGetErrorString(e)};
        // each session its rows back, then its draws, all queued; one sync, then the picks
        std::vector<const int32_t*> picks((size_t) n, nullptr);
        for (int i = 0; i < n; ++i) {
            ss[(size_t) i]->batch_take(v, i);
            if (head[(size_t) i] > 0 && (picks[(size_t) i] = ss[(size_t) i]->batch_draw(v, i, err)) == nullptr)
                return {Status::Error, who + ": " + err};
        }
        const cudaError_t synced = cudaStreamSynchronize(st);
        end.open = false;
        if (!batch_end(synced) || synced != cudaSuccess)
            return {Status::Error, who + (synced == cudaSuccess ? ": the window was stopped" : ": the batch window failed")};
        for (int i = 0; i < n; ++i) {
            const DecodeRequest& r = reqs[i];
            const int h = head[(size_t) i];
            if (h == 0) continue;
            if (r.row_tokens) std::copy(picks[(size_t) i], picks[(size_t) i] + h, r.row_tokens);
            if (r.next_token) *r.next_token = picks[(size_t) i][h - 1];
        }
        return {};
    }

protected:
    // ---- the hooks a model writes (docs/architecture/batching.md, "Adding batching to a model")

    /// The error messages' prefix: the model's name.
    virtual const char* batch_name() const = 0;
    /// The positions a session holds.
    virtual int64_t batch_context() const = 0;
    /// The model's shared batch state (the same object for every session of one model).
    virtual BatchShared& batch_shared() = 0;
    /// The stream this session's work runs on; a batch runs on its leader's.
    virtual cudaStream_t batch_stream() const = 0;
    /// Room for the request's positions in this session's caches, and whatever the session queues on its own stream before
    /// a window (the last window's commit); every session's runs before the driver waits on any (false: the batch is
    /// refused).
    virtual bool batch_reserve(const DecodeRequest& r, std::string& err) = 0;
    /// On the leader: every row's token and position to the card, outside the graph (v.w.tokens, v.w.positions). `key`
    /// takes whatever else the forward's launches depend on (a launch bound); the driver adds the composition.
    virtual bool batch_stage(const BatchView& v, GraphLru<core::CapturedGraph>::Key& key, std::string& err) = 0;
    /// On the leader: every row through the layers on the shared scratch, each segment's stateful kernels reading and
    /// writing v.ss[i]'s state, then the head over at least each segment's head rows, and whatever per-segment state the
    /// window leaves (a drafter's context). Launches only while batch_captured(): a composition's first window runs it
    /// twice, as launched and then captured. False with `err`: the window failed (the driver drains the stream and reports
    /// it).
    virtual bool batch_forward(const BatchView& v, std::string& err) = 0;
    /// True: batch_forward only launches and the driver captures a graph per composition. False: the host steers the
    /// forward (it reads results between launches), so the driver runs it directly every window and the model keeps its
    /// own graphs of the stretches between.
    virtual bool batch_captured() const { return true; }
    /// False when batch_forward's launches depend only on the total row count (segment boundaries read from device memory).
    virtual bool batch_key_per_seg() const { return true; }
    /// False while this session holds state its batch path does not cover: the batch's windows run in turn.
    virtual bool batch_ready() const { return true; }
    /// On session i, queued on v.stream: segment i's head rows into its own logits (what row_logprobs, copy_logits and
    /// resample_row read next), its r.logits, and its host bookkeeping (history).
    virtual void batch_take(const BatchView& v, int i) = 0;
    /// On session i, queued on v.stream: its draws of its head rows with its own sampler; the host array of the picks it
    /// returns is read after the batch's sync (null with `err`: the draw failed).
    virtual const int32_t* batch_draw(const BatchView& v, int i, std::string& err) = 0;
    /// On the leader, once on every exit after a successful batch_stage: `synced` holds the status of the batch's stream
    /// sync after every take and draw, and is empty on an earlier exit. False: the window was stopped and nothing of it
    /// is taken.
    virtual bool batch_end(std::optional<cudaError_t> synced) { return synced == cudaSuccess; }
};

}  // namespace basalt::model
