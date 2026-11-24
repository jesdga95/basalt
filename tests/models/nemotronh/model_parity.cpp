// tests/models/nemotronh/model_parity.cpp - the whole Nemotron-H model against reference.py's float64 logits, and its
// paths against each other.
//
//     nemotronh_model_parity PACK REF.bin [--vram N] [--ram N] [--frozen]
//
// PACK is a converted checkpoint (synth_checkpoint.py's small one, or a real one), REF.bin reference.py's logits of its
// tokens.  --vram / --ram set the expert tier's homes (experts per layer on the card / in RAM; the rest stay in the file),
// so the card's, the CPU's and the file's experts all run; --frozen keeps every expert where it starts.  Exits 77
// without both files or a GPU.
//
// An expert the CPU computes rounds its activations as Q8_0, one on the card as q8_1: the paths below agree bitwise only
// while every expert stays where it was (a small pack whose experts all fit on the card, or --frozen); a tier that moves
// experts between them is held to the reference's KL instead.
//
//   (a) decode: every token alone (one-row windows), each row's logits against the reference: the relative error of the
//       logits, the KL divergence of the reference's softmax from the engine's, top-1 agreement
//   (b) prompt: the first half of the tokens as one prompt, the rest one-row windows: against the reference
//   (c) windows: windows of up to 8 rows that keep a random number of their rows (the rest run again in the next): the
//       kept rows' logits bitwise those of (a) when no expert moves, else within the reference's KL
//   (d) checkpoint: the running state saved mid-way, the session decoding on, the state restored: the same tokens again
//   (e) with REF.bin.mtp (reference.py's MTP drafts; the session then drafts, so (a)-(d) keep its cache too): after each
//       token, the MTP head's first two greedy drafts against the reference's: the ids, and the first's probability
#include "basalt/core/checkpoint_transfer.hpp"
#include "basalt/model/adapter.hpp"
#include "models/nemotronh/adapter.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace basalt;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

struct Ref {
    std::vector<int32_t> tokens;
    int vocab = 0;
    std::vector<float> logits;   // [n][vocab]
};

bool read_ref(const std::string& path, Ref& r) {
    std::ifstream f(path, std::ios::binary);
    int32_t n = 0, v = 0;
    if (!f.read((char*) &n, 4) || !f.read((char*) &v, 4) || n <= 0 || v <= 0) return false;
    r.vocab = v;
    r.tokens.resize((size_t) n);
    r.logits.resize((size_t) n * v);
    return (bool) f.read((char*) r.tokens.data(), (std::streamsize) n * 4) &&
           (bool) f.read((char*) r.logits.data(), (std::streamsize) r.logits.size() * 4);
}

struct Score {
    double rel = 0, kl = 0, kl_max = 0;
    int top1 = 0, n = 0;
    void add(const float* got, const float* want, int V) {
        double num = 0, den = 0, mg = -1e30, mw = -1e30;
        int ag = 0, aw = 0;
        for (int i = 0; i < V; ++i) {
            num += std::fabs((double) got[i] - want[i]);
            den += std::fabs((double) want[i]);
            if (got[i] > mg) { mg = got[i]; ag = i; }
            if (want[i] > mw) { mw = want[i]; aw = i; }
        }
        double zg = 0, zw = 0;
        for (int i = 0; i < V; ++i) { zg += std::exp(got[i] - mg); zw += std::exp(want[i] - mw); }
        double k = 0;
        for (int i = 0; i < V; ++i) {
            const double lw = want[i] - mw - std::log(zw), lg = got[i] - mg - std::log(zg);
            k += std::exp(lw) * (lw - lg);
        }
        rel = std::max(rel, num / (den + 1e-30));
        kl += k;
        kl_max = std::max(kl_max, k);
        top1 += ag == aw;
        ++n;
    }
    std::string str() const {
        char b[200];
        std::snprintf(b, sizeof b, "rel %.2e, KL mean %.2e max %.2e, top-1 %d/%d", rel, n ? kl / n : 0.0, kl_max, top1, n);
        return b;
    }
};

// one window of `n` rows at `pos`: every row's logits into `out` ([n][V])
bool window(model::ModelSession& s, const int32_t* tok, int n, int64_t pos, int V, std::vector<float>& out) {
    std::vector<int32_t> rows((size_t) n);
    model::DecodeRequest r;
    r.tokens = tok;
    r.n_tokens = n;
    r.pos = pos;
    r.row_tokens = rows.data();
    if (const model::Result e = s.decode_window(r); !e.ok()) {
        std::fprintf(stderr, "decode: %s\n", e.message.c_str());
        return false;
    }
    out.resize((size_t) n * V);
    for (int i = 0; i < n; ++i)
        if (!s.copy_logits(i, out.data() + (size_t) i * V).ok()) return false;
    return true;
}

// reference.py's MTP drafts: per position, the first draft's id and probability, the second's
bool read_mtp(const std::string& path, std::vector<float>& d) {
    std::ifstream f(path, std::ios::binary);
    int32_t n = 0;
    if (!f.read((char*) &n, 4) || n <= 0) return false;
    d.resize((size_t) n * 4);
    return (bool) f.read((char*) d.data(), (std::streamsize) d.size() * 4);
}

bool prompt(model::ModelSession& s, const int32_t* tok, int n, int64_t pos) {
    model::PrefillRequest r;
    r.tokens = tok;
    r.n_tokens = n;
    r.pos = pos;
    r.total_tokens = n;
    if (const model::Result e = s.prefill_chunk(r); !e.ok()) {
        std::fprintf(stderr, "prefill: %s\n", e.message.c_str());
        return false;
    }
    return true;
}

std::vector<uint8_t> save(model::ModelSession& s, int64_t pos) {
    core::RunningState rs;
    if (!s.running_state(pos, true, rs).ok()) return {};
    std::vector<uint8_t> bytes(rs.bytes());
    std::memcpy(bytes.data(), rs.head.data(), rs.head.size());
    std::string err;
    if (!core::checkpoint_copy_out({{bytes.data() + rs.head.size(), rs.regions}}, (cudaStream_t) rs.stream, err)) return {};
    return bytes;
}

bool restore(model::ModelSession& s, const std::vector<uint8_t>& bytes, int64_t pos) {
    core::RunningState rs;
    if (!s.running_state(pos, false, rs).ok() || rs.bytes() != bytes.size()) return false;
    std::string err;
    if (!core::checkpoint_copy_in({{bytes.data() + rs.head.size(), rs.regions}}, (cudaStream_t) rs.stream, err)) return false;
    return s.running_restored(pos, bytes.data()).ok();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: nemotronh_model_parity PACK REF.bin [--vram N] [--ram N]: skipped\n");
        return 77;
    }
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    Ref ref;
    if (!read_ref(argv[2], ref)) {
        std::printf("cannot read %s: skipped\n", argv[2]);
        return 77;
    }
    model::LoadRequest req;
    req.paths = {argv[1]};
    req.max_context = 4096;
    bool frozen = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--frozen") frozen = true;
        else if (a == "--vram" && i + 1 < argc) req.diag_tier_vram = std::atoi(argv[++i]);
        else if (a == "--ram" && i + 1 < argc) req.diag_tier_ram = std::atoi(argv[++i]);
    }
    req.diag_tier_frozen = frozen;
    std::vector<float> mtp;
    const bool drafts = read_mtp(std::string(argv[2]) + ".mtp", mtp);
    if (drafts) {   // the session drafts: every check below keeps the MTP head's cache
        req.draft = true;
        req.spec = 4;
    }
    auto adapter = models::nemotronh::make_adapter();
    model::Started st;
    if (const model::Result e = adapter->start(req, st); !e.ok()) {
        std::fprintf(stderr, "start: %s\n", e.message.c_str());
        return 1;
    }
    model::ModelSession& s = *st.sessions.front();
    const int N = (int) ref.tokens.size(), V = ref.vocab;
    if (st.model->desc().n_vocab != V) {
        std::fprintf(stderr, "the pack's vocabulary %lld is not the reference's %d\n", (long long) st.model->desc().n_vocab, V);
        return 1;
    }
    const int32_t* tok = ref.tokens.data();
    const model::ModelSession::ExpertCounts c0 = s.expert_counts();
    (void) c0;

    // (a) one-row windows
    std::vector<float> dec((size_t) N * V), w;
    Score sa;
    for (int t = 0; t < N; ++t) {
        if (!window(s, tok + t, 1, t, V, w)) return 1;
        std::memcpy(dec.data() + (size_t) t * V, w.data(), (size_t) V * 4);
        sa.add(w.data(), ref.logits.data() + (size_t) t * V, V);
    }
    check(sa.top1 >= N - 2 && sa.kl / N < 5e-3, "decode, one row at a time: " + sa.str());

    // (b) a prompt of the first half, then one-row windows
    const int P = N / 2;
    Score sb;
    bool same_as_decode = true;
    if (!prompt(s, tok, P - 1, 0)) return 1;
    for (int t = P - 1; t < N; ++t) {
        if (!window(s, tok + t, 1, t, V, w)) return 1;
        sb.add(w.data(), ref.logits.data() + (size_t) t * V, V);
        (void) same_as_decode;
    }
    check(sb.top1 >= (N - P + 1) - 2 && sb.kl / sb.n < 5e-3, "prompt of " + std::to_string(P - 1) + " then decode: " + sb.str());

    // (c) windows of up to 8 rows keeping a random number of them
    std::mt19937 rng(7);
    int t = 0, windows = 0;
    double worst = 0;
    bool bitwise = true;
    Score sc;
    while (t < N) {
        const int n = std::min(N - t, 1 + (int) (rng() % 8));
        if (!window(s, tok + t, n, t, V, w)) return 1;
        const int keep = 1 + (int) (rng() % (unsigned) n);
        for (int i = 0; i < keep; ++i) {
            const float* a = w.data() + (size_t) i * V;
            const float* b = dec.data() + (size_t) (t + i) * V;
            if (std::memcmp(a, b, (size_t) V * 4)) bitwise = false;
            for (int j = 0; j < V; ++j) worst = std::max(worst, (double) std::fabs(a[j] - b[j]));
            sc.add(a, b, V);
        }
        t += keep;
        ++windows;
    }
    char buf[200];
    // the placement moved when the tier promoted or swapped an expert since the start
    const bool fixed = frozen || s.expert_counts().lookups == s.expert_counts().hits;
    std::snprintf(buf, sizeof buf, "%d windows of 1-8 rows keeping some: the kept rows %s one-row decode's (max abs %.2e; %s)",
                  windows, bitwise ? "bitwise" : "close to", worst, sc.str().c_str());
    check(fixed ? bitwise : sc.kl / sc.n < 5e-3, buf);

    // (d) the running state saved at P, decoding on to the end, restored: the same logits again
    {
        if (!window(s, tok, 1, 0, V, w)) return 1;   // a new conversation from position 0
        for (int i = 1; i < P; ++i)
            if (!window(s, tok + i, 1, i, V, w)) return 1;
        const std::vector<uint8_t> state = save(s, P);
        std::vector<float> first, again;
        bool ok = !state.empty();
        for (int i = P; ok && i < N; ++i) {
            ok = window(s, tok + i, 1, i, V, w);
            first.insert(first.end(), w.begin(), w.end());
        }
        ok = ok && restore(s, state, P);
        for (int i = P; ok && i < N; ++i) {
            ok = window(s, tok + i, 1, i, V, w);
            again.insert(again.end(), w.begin(), w.end());
        }
        double d = 0;
        Score sd;
        for (size_t i = 0; ok && i < first.size(); ++i) d = std::max(d, (double) std::fabs(first[i] - again[i]));
        for (size_t r = 0; ok && r < first.size() / (size_t) V; ++r) sd.add(again.data() + r * V, first.data() + r * V, V);
        const bool fixed = frozen || s.expert_counts().lookups == s.expert_counts().hits;
        std::snprintf(buf, sizeof buf, "the state saved at %d and restored: decoding on gives the same logits (max abs %.2e; %s)", P,
                      d, sd.str().c_str());
        check(ok && (fixed ? d == 0 : sd.kl / sd.n < 5e-3), buf);
    }
    // (e) the MTP head's drafts after each token, against the reference's: token by token from the start, then after a
    // prompt of the first half (the head's cache filled by the prompt's chunk)
    if (drafts) {
        const int M = std::min((int) mtp.size() / 4, N - 1);
        for (int from : {0, P - 1}) {
            if (from > 0 && !prompt(s, tok, from, 0)) return 1;
            int agree0 = 0, agree1 = 0, both = 0;
            double dp = 0;
            for (int t = from; t < M; ++t) {
                if (!window(s, tok + t, 1, t, V, w)) return 1;   // from 0: a new conversation
                if (!s.draft_begin(tok[t + 1], t + 1, 2).ok()) return 1;
                int32_t ids[2] = {-1, -1};
                float conf[2] = {0, 0};
                int n = 0;
                if (!s.draft_end(ids, conf, 2, n, nullptr).ok()) return 1;
                const float* r = mtp.data() + (size_t) t * 4;
                if (n >= 1 && ids[0] == (int32_t) r[0]) {
                    ++agree0;
                    dp += std::fabs(conf[0] - r[1]);
                    if (n >= 2) {
                        ++both;
                        agree1 += ids[1] == (int32_t) r[2];
                    }
                }
            }
            char eb[300];
            std::snprintf(eb, sizeof eb, "MTP drafts after tokens %d-%d%s: the first agrees at %d of %d (probability off by %.2e on average), the second at %d of %d",
                          from + 1, M, from ? " (a prompt first)" : "", agree0, M - from, agree0 ? dp / agree0 : 0.0, agree1, both);
            // after a prompt the model's hidden rows come from the prompt path (8-bit activations: on the synthetic pack its
            // KL is ~100x one-row decode's), and the synthetic head's near-flat rows (top probabilities 0.02-0.1) flip at
            // near ties: three in four there, all but two token by token
            const int need0 = from ? (3 * (M - from) + 3) / 4 : M - from - 2;
            check(agree0 >= need0 && agree1 >= both - 3 && (agree0 == 0 || dp / agree0 < 0.05), eb);
        }
    }
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
