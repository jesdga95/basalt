// Shared sparse MQA against a double CPU reference, including raw BF16 rows and the BF16 output boundary.
#include "basalt/kernels/sparse_mqa.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { if (n) check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc"); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void upload(const std::vector<T>& v) { check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
    std::vector<T> download(size_t n) const {
        std::vector<T> v(n);
        check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return v;
    }
};

float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

std::vector<uint8_t> encode_rows(std::vector<float>& values, basalt::kernels::MqaRowType type) {
    std::vector<uint8_t> bytes;
    if (type == basalt::kernels::MqaRowType::F32) {
        bytes.resize(values.size() * sizeof(float));
        std::memcpy(bytes.data(), values.data(), bytes.size());
    } else {
        bytes.resize(values.size() * sizeof(uint16_t));
        for (size_t i = 0; i < values.size(); ++i) {
            uint16_t bits;
            if (type == basalt::kernels::MqaRowType::BF16) {
                const __nv_bfloat16 h = __float2bfloat16_rn(values[i]);
                std::memcpy(&bits, &h, sizeof(bits));
                values[i] = __bfloat162float(h);
            } else {
                const __half h = __float2half_rn(values[i]);
                std::memcpy(&bits, &h, sizeof(bits));
                values[i] = __half2float(h);
            }
            std::memcpy(bytes.data() + i * sizeof(bits), &bits, sizeof(bits));
        }
    }
    return bytes;
}

struct Error { double max_abs = 0, rms_rel = 0; };

Error compare(const std::vector<float>& got, const std::vector<double>& ref) {
    double se = 0, sr = 0;
    Error e;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double) got[i] - ref[i];
        e.max_abs = std::max(e.max_abs, std::fabs(d));
        se += d * d;
        sr += ref[i] * ref[i];
    }
    e.rms_rel = std::sqrt(se / std::max(sr, 1e-30));
    return e;
}

std::vector<double> reference(int T, int n_sel, int win_cap, const std::vector<int32_t>& sel,
                              const std::vector<int32_t>& limit, const std::vector<int32_t>& start,
                              const std::vector<int32_t>& win_n, const std::vector<float>& q,
                              const std::vector<float>& win, const std::vector<float>& comp,
                              const std::vector<float>& sinks, const std::vector<int32_t>& pos,
                              const basalt::kernels::MqaRopeTable& rope, float scale, bool round_bf16) {
    using namespace basalt::kernels;
    std::vector<double> out((size_t) T * kSparseMqaHeads * kSparseMqaDim);
    for (int t = 0; t < T; ++t) {
        std::vector<const float*> keys;
        for (int j = 0; j < win_n[t]; ++j) {
            const int row = (start[t] + j) % win_cap;
            keys.push_back(&win[(size_t) row * kSparseMqaDim]);
        }
        for (int i = 0; i < n_sel; ++i) {
            const int row = sel[(size_t) t * n_sel + i];
            if (row >= 0 && row < limit[t]) keys.push_back(&comp[(size_t) row * kSparseMqaDim]);
        }
        for (int h = 0; h < kSparseMqaHeads; ++h) {
            const float* qh = &q[((size_t) t * kSparseMqaHeads + h) * kSparseMqaDim];
            std::vector<double> score(keys.size());
            double m = sinks[h];
            for (size_t k = 0; k < keys.size(); ++k) {
                double dot = 0;
                for (int d = 0; d < kSparseMqaDim; ++d) dot += (double) qh[d] * keys[k][d];
                score[k] = dot * scale;
                m = std::max(m, score[k]);
            }
            double den = std::exp((double) sinks[h] - m);
            double o[kSparseMqaDim]{};
            for (size_t k = 0; k < keys.size(); ++k) {
                const double w = std::exp(score[k] - m);
                den += w;
                for (int d = 0; d < kSparseMqaDim; ++d) o[d] += w * keys[k][d];
            }
            for (double& x : o) x /= den;
            const float p = (float) pos[t];
            for (int i = 0; i < kSparseMqaRopeDim / 2; ++i) {
                const float angle = p * rope.step[i];
                const float c = std::cos(angle) * rope.mscale;
                const float s = -std::sin(angle) * rope.mscale;
                const double x0 = o[kSparseMqaDim - kSparseMqaRopeDim + 2 * i];
                const double x1 = o[kSparseMqaDim - kSparseMqaRopeDim + 2 * i + 1];
                o[kSparseMqaDim - kSparseMqaRopeDim + 2 * i] = x0 * c - x1 * s;
                o[kSparseMqaDim - kSparseMqaRopeDim + 2 * i + 1] = x0 * s + x1 * c;
            }
            for (int d = 0; d < kSparseMqaDim; ++d) {
                float x = (float) o[d];
                if (round_bf16) x = bf16_round(x);
                out[((size_t) t * kSparseMqaHeads + h) * kSparseMqaDim + d] = x;
            }
        }
    }
    return out;
}

Error run(basalt::kernels::MqaRowType type, bool round_bf16, uint32_t seed, const char* label) {
    using namespace basalt::kernels;
    constexpr int T = 4, win_cap = 136, n_sel = kSparseMqaTopK, n_comp = 1100;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.5f);
    std::vector<float> q((size_t) T * kSparseMqaHeads * kSparseMqaDim), win((size_t) win_cap * kSparseMqaDim),
        comp((size_t) n_comp * kSparseMqaDim), sinks(kSparseMqaHeads);
    for (float& x : q) x = nd(rng);
    for (float& x : win) x = nd(rng);
    for (float& x : comp) x = nd(rng);
    for (float& x : sinks) x = nd(rng);
    if (type == MqaRowType::BF16) for (float& x : q) x = bf16_round(x);

    const std::vector<uint8_t> win_bytes = encode_rows(win, type), comp_bytes = encode_rows(comp, type);
    std::vector<int32_t> sel((size_t) T * n_sel), limit = {n_comp, 917, 512, 1024}, start = {172, 43, 290, 135},
                         win_n = {128, 100, 1, 0}, pos = {31, 4000, 70000, 190000};
    for (int& id : sel) {
        id = (int) (rng() % n_comp);
        if (rng() % 11 == 0) id = -1;
    }
    MqaRopeTable rope{};
    rope.mscale = 1.0f;
    for (int i = 0; i < kSparseMqaRopeDim / 2; ++i)
        rope.step[i] = std::pow(10000.f, -2.0f * i / kSparseMqaRopeDim) / 16.0f;
    constexpr float scale = 1.0f / 22.627416997969522f;
    const auto ref = reference(T, n_sel, win_cap, sel, limit, start, win_n, q, win, comp, sinks, pos, rope, scale, round_bf16);

    Device<float> dq(q.size()), dsinks(sinks.size()), dout(q.size()), dworkspace(sparse_mqa_workspace_floats(T, n_sel));
    Device<uint8_t> dwin(win_bytes.size()), dcomp(comp_bytes.size());
    Device<int32_t> dsel(sel.size()), dlimit(limit.size()), dstart(start.size()), dwin_n(win_n.size()), dpos(pos.size());
    dq.upload(q); dsinks.upload(sinks); dwin.upload(win_bytes); dcomp.upload(comp_bytes);
    dsel.upload(sel); dlimit.upload(limit); dstart.upload(start); dwin_n.upload(win_n); dpos.upload(pos);

    SparseMqaArgs a;
    a.n_tokens = T; a.q = dq.p; a.win = dwin.p; a.win_type = type; a.win_cap = win_cap; a.win_start = dstart.p; a.win_n = dwin_n.p;
    a.comp = dcomp.p; a.comp_type = type; a.sel = dsel.p; a.n_sel = n_sel; a.comp_limit = dlimit.p; a.sinks = dsinks.p;
    a.scale = scale; a.pos = dpos.p; a.rope = rope; a.round_output_bf16 = round_bf16; a.out = dout.p;
    cudaStream_t stream{};
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    sparse_mqa(a, dworkspace.p, stream);
    check(cudaStreamSynchronize(stream), "sparse_mqa batch");
    const auto got = dout.download(q.size());
    const Error e = compare(got, ref);

    bool independent = true;
    Device<float> one_out(kSparseMqaHeads * kSparseMqaDim);
    Device<float> one_ws(sparse_mqa_workspace_floats(1, n_sel));
    for (int t = 0; t < T && independent; ++t) {
        SparseMqaArgs one = a;
        one.n_tokens = 1; one.q = dq.p + (size_t) t * kSparseMqaHeads * kSparseMqaDim;
        one.win_start = dstart.p + t; one.win_n = dwin_n.p + t; one.sel = dsel.p + (size_t) t * n_sel;
        one.comp_limit = dlimit.p + t; one.pos = dpos.p + t; one.out = one_out.p;
        sparse_mqa(one, one_ws.p, stream);
        check(cudaStreamSynchronize(stream), "sparse_mqa single");
        const auto single = one_out.download(kSparseMqaHeads * kSparseMqaDim);
        independent = std::memcmp(single.data(), got.data() + (size_t) t * single.size(), single.size() * sizeof(float)) == 0;
    }
    if (type == MqaRowType::BF16) {
        for (int i = 0; i < 8; ++i) sparse_mqa(a, dworkspace.p, stream);
        cudaEvent_t begin{}, end{};
        check(cudaEventCreate(&begin), "cudaEventCreate");
        check(cudaEventCreate(&end), "cudaEventCreate");
        check(cudaEventRecord(begin, stream), "cudaEventRecord begin");
        constexpr int repeats = 100;
        for (int i = 0; i < repeats; ++i) sparse_mqa(a, dworkspace.p, stream);
        check(cudaEventRecord(end, stream), "cudaEventRecord end");
        check(cudaEventSynchronize(end), "cudaEventSynchronize");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, begin, end), "cudaEventElapsedTime");
        std::printf("bench sparse_mqa BF16 T=%d topk=%d window=%d %.2f us/call\n", T, n_sel, kSparseMqaWindow,
                    ms * 1000.0f / repeats);
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }
    cudaStreamDestroy(stream);
    std::printf("%-24s max_abs %.3e rel_RMS %.3e token-independent %s\n", label, e.max_abs, e.rms_rel, independent ? "yes" : "NO");
    if (e.rms_rel > (type == MqaRowType::BF16 ? 2e-3 : 2e-5) || !independent) throw std::runtime_error("sparse MQA parity failed");
    return e;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("no GPU\n"); return 77; }
    try {
        run(basalt::kernels::MqaRowType::BF16, true, 11, "BF16 rows/output");
        run(basalt::kernels::MqaRowType::F16, false, 12, "FP16 rows");
        run(basalt::kernels::MqaRowType::F32, false, 13, "FP32 rows");
        std::puts("PASSED");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }
}
