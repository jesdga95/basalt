// src/model/gpu_check.cpp - the several-GPUs conformance check (gpu_check.hpp): a run's logits and greedy tokens, recorded and compared.
#include "basalt/model/gpu_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace basalt::model {

namespace {

constexpr char kMagic[8] = {'B', 'G', 'P', 'U', 'T', 'R', '1', 0};

int argmax(const float* p, int64_t n) {
    int best = 0;
    for (int64_t i = 1; i < n; ++i)
        if (p[i] > p[best]) best = (int) i;
    return best;
}

}  // namespace

Result record_gpu_trace(ModelSession& s, const std::vector<int32_t>& prompt, int steps, int64_t n_vocab, GpuTrace& out,
                        const std::vector<int32_t>* forced) {
    if (prompt.size() < 2) return {Status::Error, "the GPU check needs a prompt of two tokens or more (--tokens-file)"};
    if (Result r = s.prepare(); !r.ok()) return r;
    out = GpuTrace{};
    out.n_vocab = n_vocab;
    PrefillRequest p;
    p.tokens = prompt.data();
    p.n_tokens = (int) prompt.size() - 1;
    p.total_tokens = p.n_tokens;
    if (Result r = s.prefill_chunk(p); !r.ok()) return r;
    s.end_prompt();
    int32_t tok = prompt.back();
    int64_t pos = (int64_t) prompt.size() - 1;
    if (forced) steps = std::min(steps, (int) forced->size());
    out.logits.assign((size_t) steps * (size_t) n_vocab, 0.f);
    for (int i = 0; i < steps; ++i) {
        int32_t next = -1;
        DecodeRequest d;
        d.tokens = &tok;
        d.n_tokens = 1;
        d.pos = pos;
        d.next_token = &next;
        d.logits = out.logits.data() + (size_t) i * (size_t) n_vocab;
        if (Result r = s.decode_window(d); !r.ok()) return r;
        out.tokens.push_back(next);
        tok = forced ? (*forced)[(size_t) i] : next;
        ++pos;
    }
    return {};
}

bool write_gpu_trace(const std::string& path, const GpuTrace& t, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        err = path + ": cannot write";
        return false;
    }
    const int64_t head[2] = {t.n_vocab, t.steps()};
    bool ok = std::fwrite(kMagic, 1, 8, f) == 8 && std::fwrite(head, sizeof head[0], 2, f) == 2 &&
              std::fwrite(t.tokens.data(), sizeof(int32_t), t.tokens.size(), f) == t.tokens.size() &&
              std::fwrite(t.logits.data(), sizeof(float), t.logits.size(), f) == t.logits.size();
    ok = std::fclose(f) == 0 && ok;
    if (!ok) err = path + ": write failed";
    return ok;
}

bool read_gpu_trace(const std::string& path, GpuTrace& t, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = path + ": cannot read";
        return false;
    }
    char magic[8];
    int64_t head[2];
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, kMagic, 8) == 0 && std::fread(head, sizeof head[0], 2, f) == 2 &&
              head[0] > 0 && head[1] >= 0 && head[1] < (1 << 20);
    if (ok) {
        t.n_vocab = head[0];
        t.tokens.resize((size_t) head[1]);
        t.logits.resize((size_t) head[1] * (size_t) head[0]);
        ok = std::fread(t.tokens.data(), sizeof(int32_t), t.tokens.size(), f) == t.tokens.size() &&
             std::fread(t.logits.data(), sizeof(float), t.logits.size(), f) == t.logits.size();
    }
    std::fclose(f);
    if (!ok) err = path + ": not a GPU check trace";
    return ok;
}

GpuCompare compare_gpu_traces(const GpuTrace& ref, const GpuTrace& run) {
    GpuCompare c;
    if (ref.n_vocab != run.n_vocab || ref.n_vocab <= 0) return c;
    const int64_t V = ref.n_vocab;
    c.steps = std::min(ref.steps(), run.steps());
    int agree = 0;
    for (int i = 0; i < c.steps; ++i) {
        const float* a = ref.logits.data() + (size_t) i * (size_t) V;
        const float* b = run.logits.data() + (size_t) i * (size_t) V;
        if (std::memcmp(a, b, (size_t) V * sizeof(float)) == 0) ++c.bitwise;
        else if (c.first_diff < 0) c.first_diff = i;
        if (ref.tokens[(size_t) i] != run.tokens[(size_t) i] && c.first_token_diff < 0) c.first_token_diff = i;
        double ma = -1e300, mb = -1e300;
        for (int64_t v = 0; v < V; ++v) {
            c.max_abs = std::max(c.max_abs, (double) std::fabs(a[v] - b[v]));
            ma = std::max(ma, (double) a[v]);
            mb = std::max(mb, (double) b[v]);
        }
        double za = 0, zb = 0;
        for (int64_t v = 0; v < V; ++v) {
            za += std::exp((double) a[v] - ma);
            zb += std::exp((double) b[v] - mb);
        }
        const double la = std::log(za), lb = std::log(zb);
        double kl = 0;
        for (int64_t v = 0; v < V; ++v) {
            const double lpa = (double) a[v] - ma - la;
            kl += std::exp(lpa) * (lpa - ((double) b[v] - mb - lb));
        }
        c.kld += kl;
        c.max_kld = std::max(c.max_kld, kl);
        if (argmax(a, V) == argmax(b, V)) ++agree;
    }
    if (c.steps > 0) {
        c.kld /= c.steps;
        c.top1 = (double) agree / c.steps;
    }
    return c;
}

std::string gpu_compare_line(const GpuCompare& c) {
    char b[320];
    const std::string greedy = c.first_token_diff < 0 ? "identical" : "differs from step " + std::to_string(c.first_token_diff);
    std::snprintf(b, sizeof b, "gpu check: %d steps, %d bitwise equal, greedy %s, max |dlogit| %.3g, KLD %.3g (max step %.3g), top-1 %.2f%%: %s", c.steps,
                  c.bitwise, greedy.c_str(), c.max_abs, c.kld, c.max_kld, 100.0 * c.top1, c.same() ? "BITWISE" : "DIFFERS");
    return b;
}

}  // namespace basalt::model
