#include "common.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/remote_expert_opt.hpp"
#include "basalt/core/gdn_head_split.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace basalt::core {
using namespace verify_detail;

std::string Verifier::profile_report() {
    if (!prof_on_ || prof_windows_ == 0) return std::string();
    static const char* names[kProfPer] = {"-", "hc-read0", "q8+qkv/q-idx gemv", "conv", "ab", "z", "rec", "q8+kv-idx",
                                          "k/v+norm-rope", "kv+idx append", "q+q-idx", "scores+topk", "kv-resolve",
                                          "attention", "gate", "", "out-proj", "hc-read1+router", "shared+quant",
                                          "waitA", "VRAM hits", "waitB", "PCIe grp", "waitCPU", "copy+combine",
                                          "(gap)", "head", "  hc0 norm", "  hc0 down", "  hc0 up", "", "", ""};
    std::string out;
    char b[80];
    double total = 0;
    for (int k = 0; k < 2; ++k) {
        out += k == 0 ? " GDN layers:" : " | QSA layers:";
        for (int i = 0; i < kProfPer; ++i) {
            if (prof_sum_[k][i] <= 0) continue;
            total += prof_sum_[k][i];
            std::snprintf(b, sizeof b, " %s %.2f", names[i], prof_sum_[k][i] / 1e6 / (double) prof_windows_);
            out += b;
        }
    }
    std::snprintf(b, sizeof b, " | total %.2f ms/window over %lld windows", total / 1e6 / (double) prof_windows_, (long long) prof_windows_);
    out += b;
    std::snprintf(b, sizeof b, " | graph launch %.3f ms/window", ms_launch / (double) prof_windows_);
    out += b;
    ms_launch = 0;
    DecodeGaps& dg = decode_gaps();
    if (dg.n_v2d > 0 && dg.n_d2v > 0) {
        std::snprintf(b, sizeof b, " | GPU idle verify->draft %.3f, draft GPU %.3f, idle draft->verify %.3f ms",
                      dg.v2d / 1e6 / (double) dg.n_v2d, dg.draft_gpu / 1e6 / (double) dg.n_v2d,
                      dg.d2v / 1e6 / (double) dg.n_d2v);
        out += b;
    }
    if (dg.host_on) {
        static const char* hn[32] = {"peer/copy sync", "profile read", "host sample", "run tail", "commit", "emit",
                                     "draft stage", "draft launch", "draft wait", "draft sync", "draft prof read",
                                     "run entry", "stage+ple", "dm_sync", "peer launch", "graph launch",
                                     "adapt join", "observe", "apply_pending", "window/penalty", "adapt thread", "", "", "", "", "", "", "", "commit GPU",
                                     "commit>draft", "verify>commit", ""};
        out += " | host ms/window:";
        for (int i = 0; i < 31; ++i) {
            if (!hn[i][0]) continue;
            std::snprintf(b, sizeof b, " %s %.3f", hn[i], dg.hms[i] / (double) prof_windows_);
            out += b;
            dg.hms[i] = 0;
        }
        const double gs = prof_graph0_ / 1e6 / (double) prof_windows_;
        std::snprintf(b, sizeof b, " | graph start->layer0 %.3f ms", gs);
        out += b;
        prof_graph0_ = 0;
    }
    dg.v2d = dg.d2v = dg.draft_gpu = 0;
    dg.n_v2d = dg.n_d2v = 0;
    for (auto& r : prof_sum_) for (double& d : r) d = 0;
    prof_windows_ = 0;
    return out;
}

DecodeGaps& decode_gaps() {
    static DecodeGaps g;
    return g;
}

DecodeEvents& decode_events() {
    static DecodeEvents e;
    return e;
}

void DecodeEvents::enable(size_t max_windows) {
    if (on) return;
    ev.assign((max_windows + 2) * kKinds, nullptr);
    set.assign(ev.size(), 0);
    tags.assign(max_windows + 2, 0);
    hs.assign(max_windows + 2, std::array<double, 10>{});
    for (auto& e : ev)
        if (cudaEventCreate(&e) != cudaSuccess) { cudaGetLastError(); e = nullptr; }
    on = true;
    // align the host clock to the event clock: poll an event's completion tightly, keep the smallest offset
    auto host_ms = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    cudaEventCreate(&cal);
    cudaEventRecord(cal, nullptr);
    cudaEventSynchronize(cal);
    cal_host = 1e30;
    cudaEvent_t e;
    cudaEventCreate(&e);
    for (int i = 0; i < 200; ++i) {
        cudaEventRecord(e, nullptr);
        while (cudaEventQuery(e) == cudaErrorNotReady) {}
        const double h = host_ms();
        float t = 0;
        cudaEventElapsedTime(&t, cal, e);
        cal_host = std::min(cal_host, h - (double) t);
    }
    cudaEventDestroy(e);
}

void DecodeEvents::record(Kind k, cudaStream_t s, size_t ahead) {
    if (!on) return;
    const size_t i = (window + ahead) * kKinds + k;
    if (i >= ev.size() || ev[i] == nullptr) return;
    cudaEventRecord(ev[i], s);
    set[i] = 1;
}

std::string DecodeEvents::report() {
    if (!on) return {};
    cudaDeviceSynchronize();
    std::vector<double> v[8];
    std::vector<std::pair<double, unsigned>> idle;
    std::vector<size_t> idle_w;
    auto at = [&](size_t w, int k) { return ev[w * kKinds + k]; };
    auto ok = [&](size_t w, int k) { return w * kKinds + k < set.size() && set[w * kKinds + k]; };
    auto el = [&](cudaEvent_t a, cudaEvent_t b) { float ms = 0; cudaEventElapsedTime(&ms, a, b); return (double) ms; };
    for (size_t w = 1; w + 1 <= window; ++w) {
        bool all = ok(w + 1, kVerifyPre) && ok(w + 1, kRunEntry);
        for (int k = 0; k < kRunEntry; ++k) all = all && ok(w, k);
        if (!all) continue;
        v[0].push_back(el(at(w, kVerifyPre), at(w, kVerifyPost)));
        v[1].push_back(el(at(w, kVerifyPost), at(w, kCommitPre)));
        v[2].push_back(el(at(w, kCommitPre), at(w, kCommitPost)));
        v[3].push_back(el(at(w, kVerifyPost), at(w, kDraftPre)));
        v[4].push_back(el(at(w, kDraftPre), at(w, kDraftPost)));
        const double c = el(at(w, kVerifyPre), at(w, kCommitPost)), d = el(at(w, kVerifyPre), at(w, kDraftPost));
        v[5].push_back(el(at(w, kVerifyPre), at(w + 1, kVerifyPre)) - std::max(c, d));
        idle.emplace_back(v[5].back(), tags[w]);
        idle_w.push_back(w);
        v[6].push_back(el(at(w, kVerifyPre), at(w + 1, kVerifyPre)));
        v[7].push_back(el(at(w + 1, kRunEntry), at(w + 1, kVerifyPre)));
    }
    static const char* nm[8] = {"verify", "verify>commit", "commit", "verify>draft", "draft", "idle before next verify",
                                "window", "run entry>launch"};
    std::string out;
    char b[400];
    std::snprintf(b, sizeof b, "basalt decode events (GPU ms, %zu windows; median / mean):", v[0].size());
    out = b;
    for (int i = 0; i < 8; ++i) {
        if (v[i].empty()) continue;
        std::vector<double> x = v[i];
        std::sort(x.begin(), x.end());
        double sum = 0;
        for (double y : x) sum += y;
        std::snprintf(b, sizeof b, " %s %.3f/%.3f;", nm[i], x[x.size() / 2], sum / (double) x.size());
        out += b;
    }
    // the idle gap's tail: the mean over the windows by tag, and how much of the total the gaps over 0.1 ms hold
    double tot = 0, big = 0;
    for (auto& g : idle) { tot += g.first; if (g.first > 0.1) big += g.first; }
    std::snprintf(b, sizeof b, " | idle total %.3f ms/window, in gaps over 0.1 ms %.3f", idle.empty() ? 0.0 : tot / (double) idle.size(),
                  idle.empty() ? 0.0 : big / (double) idle.size());
    out += b;
    std::vector<double> sorted;
    for (auto& g : idle) sorted.push_back(g.first);
    std::sort(sorted.begin(), sorted.end());
    if (!sorted.empty()) {
        std::snprintf(b, sizeof b, " p90 %.3f p99 %.3f max %.3f", sorted[sorted.size() * 9 / 10], sorted[sorted.size() * 99 / 100],
                      sorted.back());
        out += b;
    }
    for (unsigned m = 0; m < 12; ++m) {
        double sum = 0;
        long n = 0;
        for (auto& g : idle) if (m == 0 ? g.second == 0 : (g.second & (1u << (m - 1)))) { sum += g.first; ++n; }
        if (n == 0) continue;
        std::snprintf(b, sizeof b, " | tag%u: n %ld mean idle %.3f", m, n, sum / (double) n);
        out += b;
    }
    // the host's stations between a draft's return and the next launch: mean interval for gaps over 0.1 ms and the rest
    for (int big_set = 1; big_set >= 0; --big_set) {
        double sum[8] = {}, th[4] = {};
        long n = 0, thn = 0;
        for (size_t i = 0; i < idle.size(); ++i) {
            if ((idle[i].first > 0.1) != (big_set == 1)) continue;
            const auto& h = hs[idle_w[i]];
            bool ok2 = true;
            for (int k = 0; k < 6; ++k) ok2 = ok2 && h[(size_t) k] > 0;
            if (!ok2) continue;
            for (int k = 0; k < 5; ++k) sum[k] += h[(size_t) k + 1] - h[(size_t) k];
            sum[5] += idle[i].first;
            sum[6] += v[7][i];
            if (h[8] > 0 && h[7] > 0 && h[6] > 0) { th[3] += h[9] > 0 ? h[8] - h[9] : 0; th[0] += h[8] - h[0]; th[1] += h[6] - h[8]; th[2] += h[7] - h[6]; ++thn; }
            ++n;
        }
        if (n == 0) continue;
        if (thn > 0) {
            std::snprintf(b, sizeof b, " | adapt thread (windows with one, %s gaps): start after stamp0 %.3f, work %.3f, presync %.3f, start after spawn %.3f",
                          big_set ? "big" : "small", th[0] / (double) thn, th[1] / (double) thn, th[2] / (double) thn, th[3] / (double) thn);
            out += b;
        }
        std::snprintf(b, sizeof b, " | host %s gaps (n %ld) ms: draft>adapt-join %.3f join>loop-end %.3f loop-end>top %.3f top>run-dm_sync %.3f dm_sync>launch %.3f (GPU idle %.3f, of it run-entry>launch %.3f)",
                      big_set ? "big" : "small", n, sum[0] / (double) n, sum[1] / (double) n, sum[2] / (double) n,
                      sum[3] / (double) n, sum[4] / (double) n, sum[5] / (double) n, sum[6] / (double) n);
        out += b;
    }
    // the stamps on the GPU timeline: ms after the later of the commit's and the draft's end
    for (int big_set = 1; big_set >= 0; --big_set) {
        double sum[10] = {};
        long n = 0;
        for (size_t i = 0; i < idle.size(); ++i) {
            if ((idle[i].first > 0.1) != (big_set == 1)) continue;
            const size_t w = idle_w[i];
            const auto& h = hs[w];
            bool ok2 = true;
            for (int k = 0; k < 6; ++k) ok2 = ok2 && h[(size_t) k] > 0;
            if (!ok2) continue;
            const double c = el(cal, at(w, kCommitPost)), d = el(cal, at(w, kDraftPost));
            const double done = std::max(c, d) + cal_host;
            for (int k = 0; k < 6; ++k) sum[k] += h[(size_t) k] - done;
            sum[6] += el(cal, at(w + 1, kRunEntry)) + cal_host - done;
            sum[7] += el(cal, at(w + 1, kVerifyPre)) + cal_host - done;
            sum[8] += c - d;
            ++n;
        }
        if (n == 0) continue;
        std::snprintf(b, sizeof b, " | %s gaps, host stamps after the GPU's draft+commit end (ms):", big_set ? "big" : "small");
        out += b;
        static const char* sn[6] = {"draft returned", "adapt joined", "loop end", "loop top", "after dm_sync", "pre-launch"};
        for (int k = 0; k < 6; ++k) {
            const int kk = k == 5 ? 5 : k;
            std::snprintf(b, sizeof b, " %s %.3f;", sn[k], sum[kk] / (double) n);
            out += b;
        }
        std::snprintf(b, sizeof b, " run-entry event %.3f; verify-pre event %.3f; commit end - draft end %.3f", sum[6] / (double) n,
                      sum[7] / (double) n, sum[8] / (double) n);
        out += b;
    }
    std::fill(set.begin(), set.end(), 0);
    std::fill(tags.begin(), tags.end(), 0);
    for (auto& h : hs) h.fill(0.0);
    window = 0;
    return out;
}

void Verifier::collect_profile() {
    const ModelGeometry& g = *g_;
    cudaMemcpy(prof_h_.data(), prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost);
    const int64_t L = g.n_layers;
    auto at = [&](int64_t l, int i) { return prof_h_[(size_t) (l * kProfPer + i)]; };
    // the derived columns read stamps the hc-read kernels write themselves or the next layer's first; a slot no kernel
    // stamped is 0, so a missing or out-of-order stamp contributes nothing instead of an unsigned wrap
    const auto gap = [](unsigned long long to, unsigned long long from) {
        return (from != 0 && to != 0 && to >= from) ? (double) (to - from) : 0.0;
    };
    for (int64_t l = 0; l < L; ++l) {
        const int kind = is_qsa_layer(g, l) ? 1 : 0;
        unsigned long long prev = at(l, 0);
        for (int i = 1; i <= 24; ++i) {
            const unsigned long long x = at(l, i);
            if (x == 0 || x < prev) continue;
            prof_sum_[kind][i] += (double) (x - prev);
            prev = x;
        }
        if (l + 1 < L) prof_sum_[kind][25] += gap(at(l + 1, 0), at(l, 24));
        const double dn = gap(at(l, 27), at(l, 0)), dd = gap(at(l, 28), at(l, 27)), du = gap(at(l, 1), at(l, 28));
        if (dn > 0 && dd > 0 && du > 0) {   // the split exists: show it split, not twice
            prof_sum_[kind][27] += dn;      // hc-read0: norm
            prof_sum_[kind][28] += dd;      //           down
            prof_sum_[kind][29] += du;      //           up (through both halves, as before)
            prof_sum_[kind][1] -= gap(at(l, 1), at(l, 0));   // (hc-read0 shown split)
        }
    }
    prof_sum_[0][26] += gap(at(L, 1), at(L, 0));
    prof_graph0_ += gap(at(0, 0), at(L, 2));
    DecodeGaps& dg = decode_gaps();
    if (dg.draft_end != 0 && at(0, 0) > dg.draft_end) {
        dg.d2v += (double) (at(0, 0) - dg.draft_end);
        ++dg.n_d2v;
        dg.hms[28] += gap(at(L, 4), at(L, 3)) / 1e6;          // the commit's GPU span
        dg.hms[29] += gap(at(L, 4), dg.draft_end) / 1e6;      // the commit still running after the draft
        dg.hms[30] += gap(at(L, 3), dg.verify_prev_end) / 1e6;   // the window's end to the commit's start
    }
    dg.verify_prev_end = at(L, 1);
    dg.draft_end = 0;
    dg.verify_end = at(L, 1);
    ++prof_windows_;
}

}  // namespace basalt::core
