// tests/models/nemotronh/model_parity.cpp - the whole Nemotron-H model against reference.py's float64 logits, and its
// paths against each other.
//
//     nemotronh_model_parity PACK REF.bin [--vram N] [--ram N]
//
// PACK is a converted checkpoint (synth_checkpoint.py's small one, or a real one), REF.bin reference.py's logits of its
// tokens.  --vram / --ram pin the expert tier (experts per layer on the card / in RAM; the rest stay in the file), so
// the card's, the CPU's and the file's experts all run.  Exits 77 without both files or a GPU.
//
//   (a) decode: every token alone (one-row windows), each row's logits against the reference: the relative error of the
//       logits, the KL divergence of the reference's softmax from the engine's, top-1 agreement
//   (b) prompt: the first half of the tokens as one prompt, the rest one-row windows: against the reference
//   (c) windows: windows of up to 8 rows that keep a random number of their rows (the rest run again in the next): the
//       kept rows' logits bitwise those of (a) when every expert is on the card, else within the CPU's rounding
//   (d) checkpoint: the running state saved mid-way, the session decoding on, the state restored: the same tokens again
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
    for (int i = 3; i + 1 < argc; i += 2) {
        if (std::string(argv[i]) == "--vram") req.diag_tier_vram = std::atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--ram") req.diag_tier_ram = std::atoi(argv[i + 1]);
    }
    const bool all_on_card = req.diag_tier_vram < 0 && req.diag_tier_ram < 0;
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
    while (t < N) {
        const int n = std::min(N - t, 1 + (int) (rng() % 8));
        if (!window(s, tok + t, n, t, V, w)) return 1;
        const int keep = 1 + (int) (rng() % (unsigned) n);
        for (int i = 0; i < keep; ++i) {
            const float* a = w.data() + (size_t) i * V;
            const float* b = dec.data() + (size_t) (t + i) * V;
            if (std::memcmp(a, b, (size_t) V * 4)) bitwise = false;
            for (int j = 0; j < V; ++j) worst = std::max(worst, (double) std::fabs(a[j] - b[j]));
        }
        t += keep;
        ++windows;
    }
    char buf[200];
    std::snprintf(buf, sizeof buf, "%d windows of 1-8 rows keeping some: the kept rows %s one-row decode's (max abs %.2e)", windows,
                  bitwise ? "bitwise" : "close to", worst);
    check(all_on_card ? bitwise : worst < 0.05, buf);

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
        for (size_t i = 0; ok && i < first.size(); ++i) d = std::max(d, (double) std::fabs(first[i] - again[i]));
        std::snprintf(buf, sizeof buf, "the state saved at %d and restored: decoding on gives the same logits (max abs %.2e)", P, d);
        check(ok && (all_on_card ? d == 0 : d < 0.05), buf);
    }
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
