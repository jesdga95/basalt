// src/prefill/timing.cpp - see timing.hpp.
#include "timing.hpp"

#include "basalt/core/diag.hpp"
#include "basalt/log.hpp"

#include <algorithm>
#include <cstdio>
#include <string>

namespace basalt::prefill::detail {

namespace {

constexpr const char* kSpanName[(size_t) Span::Count] = {
    "inputs", "ple", "hc", "gdn", "qsa", "kv", "router", "shared", "routing", "experts", "ring", "cpu", "combine",
    "callback"};
constexpr const char* kSrcName[(size_t) Src::Count] = {"pinned", "ram", "file", "card"};
constexpr size_t kEventBatch = 256;   // events made at a time as the pool grows

void append(std::string& s, const char* fmt, double v) {
    char b[64];
    std::snprintf(b, sizeof b, fmt, v);
    s += b;
}

std::FILE* chunk_log() {
    static std::FILE* const f = [] {
        const std::string& path = basalt::core::diagnostics().prefill_timing_log;
        if (path.empty()) return (std::FILE*) nullptr;
        std::FILE* p = std::fopen(path.c_str(), "ab");
        if (p == nullptr) basalt::log::warn("basalt prefill", "--prefill-timing-log %s: cannot open it", path.c_str());
        return p;
    }();
    return f;
}

}  // namespace

Timeline::~Timeline() {
    for (cudaEvent_t e : ev_) cudaEventDestroy(e);
}

void Timeline::open(bool on, cudaStream_t cs, int64_t pos0, int64_t n) {
    on_ = on;
    in_chunk_ = false;
    pending_ = 0;
    chunks_.clear();
    if (!on_) return;
    pos0_ = pos0;
    n_ = n;
    record(Span::Count, cs);   // the timeline's start: it closes no interval
}

void Timeline::record(Span s, cudaStream_t cs) {
    if (pending_ == ev_.size()) {
        for (size_t i = 0; i < kEventBatch; ++i) {
            cudaEvent_t e = nullptr;
            if (cudaEventCreate(&e) != cudaSuccess) {
                (void) cudaGetLastError();
                basalt::log::warn("basalt prefill", "--prefill-timing: no more CUDA events; this read is not timed");
                on_ = false;
                return;
            }
            ev_.push_back(e);
        }
        mark_.resize(ev_.size());
    }
    if (cudaEventRecord(ev_[pending_], cs) != cudaSuccess) {
        (void) cudaGetLastError();
        basalt::log::warn("basalt prefill", "--prefill-timing: an event record failed; this read is not timed");
        on_ = false;
        return;
    }
    mark_[pending_++] = {s, s == Span::Count || !in_chunk_ ? -1 : (int) chunks_.size() - 1};
}

void Timeline::fold_pending() {
    const Clock::time_point t0 = Clock::now();
    // an error already pending is left to its own check (the reads below clear only their own)
    if (cudaPeekAtLastError() != cudaSuccess) {
        on_ = false;
        return;
    }
    for (size_t i = 1; i < pending_; ++i) {
        float ms = 0;
        if (cudaEventElapsedTime(&ms, ev_[i - 1], ev_[i]) != cudaSuccess) {
            (void) cudaGetLastError();
            continue;
        }
        if (mark_[i].chunk >= 0) chunks_[(size_t) mark_[i].chunk].gpu[(size_t) mark_[i].span] += ms;
    }
    // the last mark starts the next interval
    std::swap(ev_[0], ev_[pending_ - 1]);
    mark_[0] = mark_[pending_ - 1];
    pending_ = 1;
    if (!chunks_.empty())
        chunks_.back().host[(size_t) Host::Events] +=
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void Timeline::chunk_begin(int64_t p0, int64_t T, bool stream_all) {
    if (!on_) return;
    Chunk c;
    c.p0 = p0;
    c.T = T;
    c.stream_all = stream_all;
    c.t0 = Clock::now();
    c.unix_s = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    chunks_.push_back(c);
    in_chunk_ = true;
}

void Timeline::chunk_end() {
    if (!on_ || chunks_.empty()) return;
    Chunk& c = chunks_.back();
    in_chunk_ = false;
    c.wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - c.t0).count();
}

bool Timeline::wait_begins(cudaEvent_t copied, Span before, cudaStream_t cs) {
    // a pending error stays for its own check; a running copy's NotReady is cleared, as it is no error
    if (!on_ || cudaPeekAtLastError() != cudaSuccess) return false;
    if (cudaEventQuery(copied) != cudaErrorNotReady) return false;
    (void) cudaGetLastError();
    mark(before, cs);
    return true;
}

void Timeline::close(double wall_ms) {
    fold();
    if (!on_) return;
    on_ = false;
    if (chunks_.empty()) return;
    static int64_t reads = 0;
    const int64_t id = reads++;
    using Spans = std::array<double, (size_t) Span::Count>;
    Spans g{};
    std::array<double, (size_t) Host::Count> h{};
    std::array<uint64_t, (size_t) Src::Count> b{};
    int64_t copies = 0, resident = 0, cpu_n = 0, waits = 0;
    uint64_t resident_b = 0;
    std::string sizes;
    for (size_t c = 0; c < chunks_.size(); ++c) {
        const Chunk& k = chunks_[c];
        for (size_t s = 0; s < g.size(); ++s) g[s] += k.gpu[s];
        for (size_t s = 0; s < h.size(); ++s) h[s] += k.host[s];
        for (size_t s = 0; s < b.size(); ++s) b[s] += k.bytes[s];
        copies += k.copies;
        resident += k.resident;
        resident_b += k.resident_bytes;
        cpu_n += k.cpu_experts;
        waits += k.waits;
        sizes += (c ? ", " : "") + std::to_string((long long) k.T);
    }
    double gpu_total = 0;
    for (double v : g) gpu_total += v;
    std::string line = "read " + std::to_string((long long) id) + ": " + std::to_string((long long) n_) +
                       " tokens at " + std::to_string((long long) pos0_) + " in " +
                       std::to_string(chunks_.size()) + " chunks (" + sizes + "), ";
    append(line, "%.1f ms; GPU ", wall_ms);
    append(line, "%.1f ms =", gpu_total);
    for (size_t s = 0; s < g.size(); ++s) {
        line += s ? ", " : " ";
        line += kSpanName[s];
        append(line, " %.1f", g[s]);
        if ((Span) s == Span::Ring) line += " (" + std::to_string((long long) waits) + " waits)";
    }
    constexpr double GiB = 1073741824.0;
    append(line, "; host: sync %.1f", h[(size_t) Host::Sync]);
    append(line, ", callback %.1f", h[(size_t) Host::Callback]);
    append(line, " (draft K/V %.1f)", h[(size_t) Host::DraftKv]);
    append(line, ", stager %.1f", h[(size_t) Host::Stager]);
    line += ", CPU share " + std::to_string((long long) cpu_n);
    append(line, " experts in %.1f ms", h[(size_t) Host::CpuPool]);
    append(line, " (waited %.1f)", h[(size_t) Host::CpuWait]);
    append(line, ", event reads %.1f", h[(size_t) Host::Events]);
    double streamed = 0;
    for (uint64_t v : b) streamed += (double) v;
    append(line, "; streamed %.2f GiB in ", streamed / GiB);
    line += std::to_string((long long) copies) + " copies:";
    for (size_t s = 0; s < b.size(); ++s) {
        line += s ? ", " : " ";
        line += kSrcName[s];
        append(line, " %.2f", (double) b[s] / GiB);
    }
    line += "; resident " + std::to_string((long long) resident);
    append(line, " expert-layers (%.2f GiB)", (double) resident_b / GiB);
    basalt::log::info("basalt prefill", "%s", line.c_str());

    std::FILE* f = chunk_log();
    if (f == nullptr) return;
    for (size_t c = 0; c < chunks_.size(); ++c) {
        const Chunk& k = chunks_[c];
        std::string j = "{\"read\":" + std::to_string((long long) id) + ",\"chunk\":" + std::to_string(c) +
                        ",\"p0\":" + std::to_string((long long) k.p0) + ",\"tokens\":" + std::to_string((long long) k.T) +
                        ",\"stream_all\":" + (k.stream_all ? "true" : "false");
        append(j, ",\"unix_s\":%.3f", k.unix_s);
        append(j, ",\"wall_ms\":%.3f", k.wall_ms);
        j += ",\"gpu_ms\":{";
        for (size_t s = 0; s < g.size(); ++s) {
            j += (s ? ",\"" : "\"") + std::string(kSpanName[s]) + "\":";
            append(j, "%.3f", k.gpu[s]);
        }
        j += "},\"ring_waits\":" + std::to_string((long long) k.waits);
        append(j, ",\"sync_ms\":%.3f", k.host[(size_t) Host::Sync]);
        append(j, ",\"callback_ms\":%.3f", k.host[(size_t) Host::Callback]);
        append(j, ",\"draft_kv_ms\":%.3f", k.host[(size_t) Host::DraftKv]);
        append(j, ",\"stager_ms\":%.3f", k.host[(size_t) Host::Stager]);
        j += ",\"cpu_experts\":" + std::to_string((long long) k.cpu_experts);
        append(j, ",\"cpu_pool_ms\":%.3f", k.host[(size_t) Host::CpuPool]);
        append(j, ",\"cpu_wait_ms\":%.3f", k.host[(size_t) Host::CpuWait]);
        append(j, ",\"event_reads_ms\":%.3f", k.host[(size_t) Host::Events]);
        j += ",\"copies\":" + std::to_string((long long) k.copies) + ",\"bytes\":{";
        for (size_t s = 0; s < b.size(); ++s)
            j += (s ? ",\"" : "\"") + std::string(kSrcName[s]) + "\":" + std::to_string((unsigned long long) k.bytes[s]);
        j += "},\"resident\":" + std::to_string((long long) k.resident) +
             ",\"resident_bytes\":" + std::to_string((unsigned long long) k.resident_bytes) + "}\n";
        std::fputs(j.c_str(), f);
    }
    std::fflush(f);
}

}  // namespace basalt::prefill::detail
