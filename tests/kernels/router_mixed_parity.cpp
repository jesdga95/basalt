#include "basalt/kernels/router.hpp"
#include "basalt/models/deepseek4/router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kExperts = 256;
constexpr int kRoutes = 6;
constexpr int kTokens = 8;
constexpr float kScale = 1.5f;

void check(cudaError_t e, const char* op) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(e));
}

template <typename T>
struct Device {
    T* p = nullptr;
    explicit Device(size_t n) {
        check(cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(1, n) * sizeof(T)), "cudaMalloc");
    }
    ~Device() {
        if (p) cudaFree(p);
    }
    void upload(const std::vector<T>& h) {
        if (!h.empty()) check(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
    std::vector<T> download(size_t n) const {
        std::vector<T> h(n);
        check(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        return h;
    }
};

struct Output {
    std::vector<int32_t> ids;
    std::vector<float> weights;
};

Output run(const std::vector<float>& logits, const std::vector<int32_t>& fixed_ids,
           const std::vector<uint8_t>& fixed_mask, const std::vector<uint8_t>& alternate_mask,
           const std::vector<float>& bias, const std::vector<float>& alternate_bias) {
    const int n = (int) fixed_mask.size();
    Device<float> dl(logits.size()), db(bias.size()), da(alternate_bias.size());
    Device<int32_t> df(fixed_ids.size()), di((size_t) n * kRoutes);
    Device<uint8_t> dm(fixed_mask.size()), dx(alternate_mask.size());
    dl.upload(logits);
    db.upload(bias);
    da.upload(alternate_bias);
    df.upload(fixed_ids);
    dm.upload(fixed_mask);
    dx.upload(alternate_mask);

    basalt::kernels::RouterMixedArgs args;
    args.n_expert = kExperts;
    args.k = kRoutes;
    args.weight_scale = kScale;
    args.gate = basalt::kernels::RouterGate::SqrtSoftplus;
    args.fixed_ids = df.p;
    args.fixed_mask = dm.p;
    args.alternate_mask = dx.p;
    args.bias = db.p;
    args.alternate_bias = da.p;

    Output out;
    out.ids.resize((size_t) n * kRoutes);
    out.weights.resize((size_t) n * kRoutes);
    Device<float> dw(out.weights.size());
    basalt::kernels::router_select_mixed(dl.p, n, args, di.p, dw.p, nullptr);
    check(cudaDeviceSynchronize(), "router_select_mixed");
    out.ids = di.download(out.ids.size());
    out.weights = dw.download(out.weights.size());
    return out;
}

Output run_table(const std::vector<float>& logits, const std::vector<int32_t>& fixed_table,
                 const std::vector<int32_t>& fixed_rows, const std::vector<uint8_t>& fixed_mask,
                 const std::vector<uint8_t>& alternate_mask, const std::vector<float>& bias,
                 const std::vector<float>& alternate_bias) {
    const int n = (int) fixed_mask.size();
    Device<float> dl(logits.size()), db(bias.size()), da(alternate_bias.size());
    Device<int32_t> dt(fixed_table.size()), dr(fixed_rows.size()), di((size_t) n * kRoutes);
    Device<uint8_t> dm(fixed_mask.size()), dx(alternate_mask.size());
    dl.upload(logits); db.upload(bias); da.upload(alternate_bias); dt.upload(fixed_table); dr.upload(fixed_rows);
    dm.upload(fixed_mask); dx.upload(alternate_mask);

    basalt::kernels::RouterMixedArgs args;
    args.n_expert = kExperts; args.k = kRoutes; args.weight_scale = kScale;
    args.gate = basalt::kernels::RouterGate::SqrtSoftplus;
    args.fixed_table = dt.p; args.fixed_rows = dr.p; args.fixed_table_rows = (int) (fixed_table.size() / kRoutes);
    args.fixed_mask = dm.p; args.alternate_mask = dx.p; args.bias = bias.empty() ? nullptr : db.p;
    args.alternate_bias = alternate_bias.empty() ? nullptr : da.p;

    Output out;
    out.ids.resize((size_t) n * kRoutes); out.weights.resize((size_t) n * kRoutes);
    Device<float> dw(out.weights.size());
    basalt::kernels::router_select_mixed(dl.p, n, args, di.p, dw.p, nullptr);
    check(cudaDeviceSynchronize(), "router_select_mixed table");
    out.ids = di.download(out.ids.size()); out.weights = dw.download(out.weights.size());
    return out;
}

Output run_v4_router(const std::vector<float>& x, const std::vector<float>& gate, int n_embd,
                     const std::vector<int32_t>& fixed_table, const std::vector<int32_t>& fixed_rows,
                     const std::vector<uint8_t>& fixed_mask, const std::vector<uint8_t>& alternate_mask,
                     const std::vector<float>& bias, const std::vector<float>& alternate_bias, bool use_model_step) {
    const int n = (int) fixed_mask.size();
    Device<float> dx(x.size()), dg(gate.size()), db(bias.size()), da(alternate_bias.size());
    Device<int32_t> dt(fixed_table.size()), dr(fixed_rows.size()), di((size_t) n * kRoutes);
    Device<uint8_t> dm(fixed_mask.size()), da_mask(alternate_mask.size());
    Device<float> dlogits((size_t) n * kExperts), dw((size_t) n * kRoutes);
    dx.upload(x); dg.upload(gate); db.upload(bias); da.upload(alternate_bias);
    dt.upload(fixed_table); dr.upload(fixed_rows); dm.upload(fixed_mask); da_mask.upload(alternate_mask);

    const auto args = basalt::models::deepseek4::router_mixed_args(
        kExperts, (int) (fixed_table.size() / kRoutes), dt.p, dr.p, dm.p, da_mask.p, db.p, da.p);
    if (use_model_step) {
        basalt::models::deepseek4::router_step(dx.p, dg.p, n, n_embd, args, dlogits.p, di.p, dw.p, nullptr);
    } else {
        basalt::kernels::RouterArgs scoring;
        scoring.n_expert = kExperts; scoring.n_embd = n_embd; scoring.k = kRoutes;
        scoring.weight_scale = kScale; scoring.gate = basalt::kernels::RouterGate::SqrtSoftplus;
        basalt::kernels::router_logits(dx.p, dg.p, n, scoring, dlogits.p, nullptr);
        basalt::kernels::router_select_mixed(dlogits.p, n, args, di.p, dw.p, nullptr);
    }
    check(cudaDeviceSynchronize(), "V4 router step");
    Output out;
    out.ids = di.download((size_t) n * kRoutes);
    out.weights = dw.download((size_t) n * kRoutes);
    return out;
}

double score(double x) {
    return std::sqrt(x > 20.0 ? x : std::log1p(std::exp(x)));
}

Output reference(const std::vector<float>& logits, const std::vector<int32_t>& fixed_ids,
                 const std::vector<uint8_t>& fixed_mask, const std::vector<uint8_t>& alternate_mask,
                 const std::vector<float>& bias, const std::vector<float>& alternate_bias) {
    const int n = (int) fixed_mask.size();
    Output out;
    out.ids.resize((size_t) n * kRoutes);
    out.weights.resize((size_t) n * kRoutes);
    for (int t = 0; t < n; ++t) {
        std::vector<int32_t> chosen;
        if (fixed_mask[(size_t) t]) {
            for (int r = 0; r < kRoutes; ++r) chosen.push_back(fixed_ids[(size_t) t * kRoutes + r]);
        } else {
            const auto& b = alternate_mask[(size_t) t] ? alternate_bias : bias;
            std::vector<int32_t> order(kExperts);
            for (int e = 0; e < kExperts; ++e) order[(size_t) e] = e;
            std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t c) {
                const double sa = score(logits[(size_t) t * kExperts + a]) + b[(size_t) a];
                const double sc = score(logits[(size_t) t * kExperts + c]) + b[(size_t) c];
                return sa > sc;
            });
            chosen.assign(order.begin(), order.begin() + kRoutes);
        }

        double sum = 0.0;
        for (int r = 0; r < kRoutes; ++r) {
            const int32_t e = chosen[(size_t) r];
            out.ids[(size_t) t * kRoutes + r] = e;
            sum += score(logits[(size_t) t * kExperts + e]);
        }
        for (int r = 0; r < kRoutes; ++r) {
            const int32_t e = chosen[(size_t) r];
            out.weights[(size_t) t * kRoutes + r] =
                (float) (score(logits[(size_t) t * kExperts + e]) / (sum + 1e-20) * kScale);
        }
    }
    return out;
}

bool same_row(const Output& a, int at, const Output& b, int bt) {
    return !std::memcmp(a.ids.data() + (size_t) at * kRoutes, b.ids.data() + (size_t) bt * kRoutes,
                        kRoutes * sizeof(int32_t)) &&
           !std::memcmp(a.weights.data() + (size_t) at * kRoutes, b.weights.data() + (size_t) bt * kRoutes,
                        kRoutes * sizeof(float));
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;

    try {
        std::mt19937 rng(19);
        std::uniform_real_distribution<float> logit(-3.f, 3.f), bias_dist(-0.25f, 0.25f);
        std::vector<float> logits((size_t) kTokens * kExperts), bias(kExperts), alternate_bias(kExperts);
        std::vector<int32_t> fixed_ids((size_t) kTokens * kRoutes);
        std::vector<uint8_t> fixed_mask(kTokens), alternate_mask(kTokens);
        for (float& x : logits) x = logit(rng);
        for (float& x : bias) x = bias_dist(rng);
        for (float& x : alternate_bias) x = bias_dist(rng);
        for (int t = 0; t < kTokens; ++t) {
            fixed_mask[(size_t) t] = (t % 2 == 0);
            alternate_mask[(size_t) t] = (t == 3 || t == 5);
            for (int r = 0; r < kRoutes; ++r) fixed_ids[(size_t) t * kRoutes + r] = (t * 17 + r * 43) % kExperts;
        }
        // Equal dynamic scores verify that the lower expert id wins every tie.
        for (int e = 0; e < kExperts; ++e) logits[(size_t) 7 * kExperts + e] = 0.f;
        std::fill(bias.begin(), bias.end(), -10.f);
        bias[17] = bias[3] = 5.f;

        const Output got = run(logits, fixed_ids, fixed_mask, alternate_mask, bias, alternate_bias);
        const Output ref = reference(logits, fixed_ids, fixed_mask, alternate_mask, bias, alternate_bias);
        double max_abs = 0.0, max_rel = 0.0;
        int id_errors = 0;
        for (size_t i = 0; i < got.ids.size(); ++i) {
            id_errors += got.ids[i] != ref.ids[i];
            const double d = std::fabs((double) got.weights[i] - ref.weights[i]);
            max_abs = std::max(max_abs, d);
            max_rel = std::max(max_rel, d / std::max(std::fabs((double) ref.weights[i]), 1e-20));
        }
        std::printf("router mixed T=%d experts=%d topk=%d: ids %d differ, weights max abs %.3g rel %.3g\n",
                    kTokens, kExperts, kRoutes, id_errors, max_abs, max_rel);
        if (id_errors || max_abs > 2e-6 || max_rel > 2e-5 || got.ids[(size_t) 7 * kRoutes] != 3 ||
            got.ids[(size_t) 7 * kRoutes + 1] != 17) return 1;

        const int token_vocab = kTokens / 2;
        std::vector<int32_t> token_ids(kTokens);
        for (int t = 0; t < kTokens; ++t) token_ids[(size_t) t] = t;
        basalt::models::deepseek4::RouteRows route_rows;
        std::string route_error;
        if (!basalt::models::deepseek4::build_route_rows(token_ids.data(), kTokens, token_vocab, 2, 43, 3,
                                                         route_rows, route_error)) {
            std::fprintf(stderr, "V4 hash route plan: %s\n", route_error.c_str());
            return 1;
        }
        basalt::models::deepseek4::RouteRows score_rows;
        if (!basalt::models::deepseek4::build_route_rows(token_ids.data(), kTokens, token_vocab, 3, 43, 3,
                                                         score_rows, route_error) ||
            std::any_of(score_rows.fixed_mask.begin(), score_rows.fixed_mask.end(), [](uint8_t x) { return x != 0; }) ||
            score_rows.alternate_mask != route_rows.alternate_mask) {
            std::fprintf(stderr, "V4 score route plan: %s\n", route_error.c_str());
            return 1;
        }
        const int32_t invalid_token = -1;
        if (basalt::models::deepseek4::build_route_rows(&invalid_token, 1, token_vocab, 2, 43, 3,
                                                        score_rows, route_error)) {
            std::fprintf(stderr, "V4 route plan accepted a negative token id\n");
            return 1;
        }
        for (int t = 0; t < kTokens; ++t) {
            const bool text = t < token_vocab;
            if (route_rows.fixed_rows[(size_t) t] != (text ? t : 0) ||
                route_rows.fixed_mask[(size_t) t] != (uint8_t) text ||
                route_rows.alternate_mask[(size_t) t] != (uint8_t) !text) return 1;
        }
        const std::vector<int32_t>& table_rows = route_rows.fixed_rows;
        const std::vector<uint8_t>& hash_mask = route_rows.fixed_mask;
        const std::vector<uint8_t>& image_mask = route_rows.alternate_mask;
        std::vector<int32_t> fixed_table((size_t) token_vocab * kRoutes);
        for (int t = 0; t < token_vocab; ++t)
            std::copy_n(fixed_ids.data() + (size_t) t * kRoutes, kRoutes,
                        fixed_table.data() + (size_t) t * kRoutes);
        constexpr int router_embd = 4096;
        std::uniform_real_distribution<float> activation(-0.3f, 0.3f), gate_weight(-0.001f, 0.001f);
        std::vector<float> router_x((size_t) kTokens * router_embd);
        std::vector<float> router_gate((size_t) kExperts * router_embd);
        for (float& x : router_x) x = activation(rng);
        for (float& x : router_gate) x = gate_weight(rng);
        const Output v4_step = run_v4_router(router_x, router_gate, router_embd, fixed_table, table_rows,
                                              hash_mask, image_mask, bias, alternate_bias, true);
        const Output v4_direct = run_v4_router(router_x, router_gate, router_embd, fixed_table, table_rows,
                                                hash_mask, image_mask, bias, alternate_bias, false);
        const bool v4_step_equal = v4_step.ids == v4_direct.ids &&
                                   !std::memcmp(v4_step.weights.data(), v4_direct.weights.data(),
                                                v4_step.weights.size() * sizeof(float));
        std::printf("DeepSeek4 routed expert step composition: ids and weights %s\n",
                    v4_step_equal ? "bitwise" : "DIFFERS");
        if (!v4_step_equal) return 1;
        const Output table_got = run_table(logits, fixed_table, table_rows, hash_mask, image_mask, {}, alternate_bias);
        const Output table_ref = reference(logits, fixed_ids, hash_mask, image_mask,
                                           std::vector<float>(kExperts, 0.f), alternate_bias);
        int table_id_errors = 0;
        double table_max_abs = 0.0, table_max_rel = 0.0;
        for (size_t i = 0; i < table_got.ids.size(); ++i) {
            table_id_errors += table_got.ids[i] != table_ref.ids[i];
            const double d = std::fabs((double) table_got.weights[i] - table_ref.weights[i]);
            table_max_abs = std::max(table_max_abs, d);
            table_max_rel = std::max(table_max_rel, d / std::max(std::fabs((double) table_ref.weights[i]), 1e-20));
        }
        std::printf("router fixed-table/null-bias parity: ids %d differ, weights max abs %.3g rel %.3g\n",
                    table_id_errors, table_max_abs, table_max_rel);
        if (table_id_errors || table_max_abs > 2e-6 || table_max_rel > 2e-5) return 1;

        bool independent = true;
        for (const auto [start, count] : {std::pair{0, 1}, {1, 3}, {4, 2}, {6, 2}}) {
            const size_t l0 = (size_t) start * kExperts, l1 = (size_t) (start + count) * kExperts;
            const size_t r0 = (size_t) start * kRoutes, r1 = (size_t) (start + count) * kRoutes;
            std::vector<float> ls(logits.begin() + (ptrdiff_t) l0, logits.begin() + (ptrdiff_t) l1);
            std::vector<int32_t> fs(fixed_ids.begin() + (ptrdiff_t) r0, fixed_ids.begin() + (ptrdiff_t) r1);
            std::vector<uint8_t> ms(fixed_mask.begin() + start, fixed_mask.begin() + start + count);
            std::vector<uint8_t> as(alternate_mask.begin() + start, alternate_mask.begin() + start + count);
            const Output slice = run(ls, fs, ms, as, bias, alternate_bias);
            for (int t = 0; t < count; ++t) independent = independent && same_row(slice, t, got, start + t);
        }
        std::printf("router mixed window/position independence: %s\n", independent ? "bitwise identical" : "DIFFERS");
        if (!independent) return 1;

        basalt::kernels::RouterMixedArgs invalid;
        bool refused = false;
        try {
            basalt::kernels::router_select_mixed(nullptr, 1, invalid, nullptr, nullptr, nullptr);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        if (!refused) return 1;

        const size_t fixed_tokens = (size_t) std::count(fixed_mask.begin(), fixed_mask.end(), uint8_t{1});
        const size_t dynamic_tokens = kTokens - fixed_tokens;
        const size_t alternate_tokens = (size_t) std::count(image_mask.begin(), image_mask.end(), uint8_t{1});
        const size_t logical_bytes =
            dynamic_tokens * kExperts * sizeof(float) + alternate_tokens * kExperts * sizeof(float) +
            fixed_tokens * kRoutes * sizeof(float) + fixed_tokens * kRoutes * sizeof(int32_t) +
            fixed_tokens * sizeof(int32_t) + (size_t) kTokens * 2 +
            (size_t) kTokens * kRoutes * (sizeof(int32_t) + sizeof(float));
        Device<float> dl(logits.size()), dw(got.weights.size()), da(alternate_bias.size());
        Device<int32_t> dt(fixed_table.size()), dr(table_rows.size()), di(got.ids.size());
        Device<uint8_t> dm(fixed_mask.size()), dx(image_mask.size());
        dl.upload(logits); dt.upload(fixed_table); dr.upload(table_rows); dm.upload(fixed_mask); dx.upload(image_mask);
        da.upload(alternate_bias);
        basalt::kernels::RouterMixedArgs args;
        args.n_expert = kExperts; args.k = kRoutes; args.weight_scale = kScale;
        args.fixed_table = dt.p; args.fixed_rows = dr.p; args.fixed_table_rows = (int) (fixed_table.size() / kRoutes);
        args.fixed_mask = dm.p; args.alternate_mask = dx.p;
        args.bias = nullptr; args.alternate_bias = da.p;
        cudaEvent_t begin{}, end{};
        check(cudaEventCreate(&begin), "cudaEventCreate");
        check(cudaEventCreate(&end), "cudaEventCreate");
        for (int i = 0; i < 50; ++i) basalt::kernels::router_select_mixed(dl.p, kTokens, args, di.p, dw.p, nullptr);
        check(cudaEventRecord(begin), "cudaEventRecord");
        constexpr int iterations = 1000;
        for (int i = 0; i < iterations; ++i) basalt::kernels::router_select_mixed(dl.p, kTokens, args, di.p, dw.p, nullptr);
        check(cudaEventRecord(end), "cudaEventRecord");
        check(cudaEventSynchronize(end), "cudaEventSynchronize");
        float ms = 0.f;
        check(cudaEventElapsedTime(&ms, begin, end), "cudaEventElapsedTime");
        const double us = (double) ms * 1000.0 / iterations;
        const double gbps = logical_bytes / (us * 1000.0);
        std::printf("router fixed-table microbench: %.3f us/call, %.2f GB/s logical traffic (%zu bytes/call)\n",
                    us, gbps, logical_bytes);
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "router_mixed_parity: %s\n", e.what());
        return 1;
    }
}
