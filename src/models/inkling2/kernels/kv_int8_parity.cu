// src/models/inkling2/kernels/kv_int8_parity.cu - INT8 KV writer and scalar/MMA reader parity against CPU references.
// With --bench, also time decode window attention over every cache format at the model's depths. With --reduced, check only
// a short list of MMA cases and one replayed decode graph (for a race checker).
#include "basalt/core/device_copy.hpp"
#include "prep.cuh"
#include "basalt/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <random>
#include <vector>

namespace ink = basalt::models::inkling2;
constexpr int kKeys100Ki = 100 * 1024;
constexpr int kKeysPartTail = 128 * 1024 - 1000;   // a deep row whose last slice is partly filled

static const char* mma_ring_name(const ink::Int8MmaPlanInfo& info) {
    return info.stages == 0 ? "direct" : info.ping_pong ? "ping-pong-ring" : "ring";
}

static const char* mma_plan_name(ink::Int8MmaPlan plan) {
    switch (plan) {
        case ink::Int8MmaPlan::Auto: return "auto";
        case ink::Int8MmaPlan::TwoStageRing: return "two-stage-ring";
        case ink::Int8MmaPlan::ThreeStageRing: return "three-stage-ring";
        case ink::Int8MmaPlan::OneStagePingPong: return "one-stage-ping-pong";
        case ink::Int8MmaPlan::NarrowRing: return "narrow-ring";
        case ink::Int8MmaPlan::Direct: return "direct";
    }
    return "unknown";
}

__global__ void encode_rows(const float* input, uint8_t* output, int rows) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= rows) return;

    float v[4], scale[2];
#pragma unroll
    for (int u = 0; u < 4; ++u) v[u] = input[(size_t) row * ink::kHeadDim + lane * 4 + u];
    ink::prep::int8_scales(v, 1.f, lane, scale);
#pragma unroll
    for (int u = 0; u < 4; ++u)
        ink::prep::store_kv(output + (size_t) row * ink::kKvInt8RowBytes, false, true,
                            (size_t) lane * 4 + u, v[u], scale, lane);
}

static bool check_cuda(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "kv_int8_parity: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// A decode window of `rows` rows (positions depth - rows ..) over one visit's cache in each format: a sliding-window ring
// (the fused kernel) and a full cache at each depth (prepare + split attention). us a call, and the K and V bytes one row
// reads over that time (the cache's own bytes: every row of a window reads them, from L2 after the first).
static bool bench_attention() {
    struct Fmt { const char* name; bool fp8, int8; };
    const Fmt fmts[] = {{"fp16", false, false}, {"e4m3", true, false}, {"int8", false, true}};
    const int64_t depths[] = {0, 8192, 32768, 65536, 98304, kKeys100Ki, 120000, kKeysPartTail, 131072, 196608, 250000, 262144};
    const int widths[] = {1, 2, 3, 5, 6, 7, 8};
    int dev = 0;
    cudaDeviceProp prop{};
    if (!check_cuda(cudaGetDevice(&dev), "bench device") ||
        !check_cuda(cudaGetDeviceProperties(&prop, dev), "bench device properties")) return false;
    std::printf("kv bench: device=%s SMs=%d L2=%d shared_per_SM=%zu\n",
                prop.name, prop.multiProcessorCount, prop.l2CacheSize, prop.sharedMemPerMultiprocessor);
    float *y = nullptr, *taps = nullptr, *qs = nullptr, *rs = nullptr, *kr = nullptr, *vr = nullptr, *table = nullptr;
    float *q = nullptr, *r = nullptr, *sink = nullptr;
    int32_t* pos = nullptr;
    uint8_t* q8 = nullptr;
    void* scr = nullptr;
    const auto alloc = [](auto** p, size_t n, const char* what) {
        return check_cuda(cudaMalloc((void**) p, n), what) && check_cuda(cudaMemset(*p, 0, n), what);
    };
    bool ok = alloc(&y, (size_t) ink::kDecodeRows * ink::kProj * 4, "y") && alloc(&taps, 1024 * ink::kConv * 4, "taps") &&
              alloc(&qs, ink::kHeads * ink::kHeadDim * 4, "q scales") && alloc(&rs, ink::kHeads * ink::kRel * 4, "r scales") &&
              alloc(&kr, (size_t) ink::kConvRing * 1024 * 4, "k ring") && alloc(&vr, (size_t) ink::kConvRing * 1024 * 4, "v ring") &&
              alloc(&table, (size_t) ink::kRel * ink::kExtent * 4, "table") &&
              alloc(&q, (size_t) ink::kDecodeRows * ink::kHeads * ink::kHeadDim * 4, "q") &&
              alloc(&r, (size_t) ink::kDecodeRows * ink::kHeads * ink::kRel * 4, "r") && alloc(&sink, (size_t) ink::kDecodeRows * ink::kHeads * 4, "sink") &&
              alloc(&pos, ink::kDecodeRows * 4, "pos") && alloc(&q8, (size_t) ink::kDecodeRows * ink::kHeads * ink::kHeadDim / 32 * 36, "q8") &&
              alloc(&scr, ink::attention_decode_scratch(ink::kDecodeRows, 1 << 20), "scratch");
    cudaEvent_t t0{}, t1{};
    ok = ok && check_cuda(cudaEventCreate(&t0), "event") && check_cuda(cudaEventCreate(&t1), "event");
    for (const Fmt& f : fmts)
        for (const int64_t depth : depths) {
            if (!f.int8 && depth != 0 && depth != 8192 && depth != 65536 && depth != 131072 && depth != 262144) continue;
            const bool window = depth == 0;
            const int64_t cap = window ? ink::kExtent + ink::kDecodeRows : depth;
            const size_t row_bytes = (size_t) ink::kv_cache_row_bytes(f.fp8, f.int8), cache = (size_t) cap * ink::kKvHeads * row_bytes;
            uint8_t *kc = nullptr, *vc = nullptr;
            if (!ok || !check_cuda(cudaMalloc(&kc, cache), "k cache") || !check_cuda(cudaMalloc(&vc, cache), "v cache")) {
                ok = false;
                break;
            }
            cudaMemset(kc, 0x11, cache);
            cudaMemset(vc, 0x11, cache);
            for (const int rows : widths) {
                const bool compare = f.int8 && depth >= 32768;
                if (!compare && rows != 1 && rows != 3 && rows != 8) continue;
                const int64_t p0 = (window ? 8192 : depth) - rows;
                std::vector<int32_t> hp(rows);
                for (int i = 0; i < rows; ++i) hp[i] = (int32_t) (p0 + i);
                basalt::core::device_copy(pos, hp.data(), rows * 4, cudaMemcpyHostToDevice);
                ink::AttnPrep pa;
                pa.yq = y; pa.yk = y + ink::kProjQ; pa.yv = y + ink::kProjQ + ink::kProjKV; pa.ldq = pa.ldk = pa.ldv = ink::kProj;
                pa.pos = pos; pa.full = !window; pa.k_conv = taps; pa.v_conv = taps; pa.q_scale = qs; pa.r_scale = rs;
                pa.k_ring = kr; pa.v_ring = vr; pa.k_cache = kc; pa.v_cache = vc; pa.cap = cap; pa.fp8 = f.fp8; pa.int8 = f.int8;
                pa.q = q; pa.r = r; pa.sink = sink;
                ink::AttnArgs aa;
                aa.q = q; aa.r = r; aa.sink = sink; aa.rel_table = table; aa.pos = pos; aa.k_cache = kc; aa.v_cache = vc; aa.cap = cap;
                aa.fp8 = f.fp8; aa.int8 = f.int8; aa.window = window ? ink::kExtent : 0; aa.gate = y + ink::kOffGate; aa.gate_ld = ink::kProj;
                aa.q8 = q8;
                aa.min_full_keys = window ? 0 : p0 + 1;
                const int64_t keys = window ? ink::kExtent : depth;
                constexpr int reps = 50;
                ink::attention_window(pa, aa, rows, keys, scr, nullptr);
                cudaEventRecord(t0);
                for (int i = 0; i < reps; ++i) ink::attention_window(pa, aa, rows, keys, scr, nullptr);
                cudaEventRecord(t1);
                float ms = 0.f;
                ok = check_cuda(cudaEventSynchronize(t1), "bench") && check_cuda(cudaEventElapsedTime(&ms, t0, t1), "bench");
                const double us = (double) ms * 1000. / reps, kv = 2. * (double) keys * ink::kKvHeads * (double) row_bytes;
                std::printf("kv bench: %s %-7s %d rows: %8.1f us a window's attention, %6.0f GB/s of one row's K and V\n",
                            window ? "window ring" : "full cache ", f.name, rows, us, kv / us / 1000.);
                if (window) continue;
                const auto read_us = [&](bool mma, bool prepared_pdl = false, ink::Int8MmaPlan plan = ink::Int8MmaPlan::Auto,
                                         ink::Int8MmaPlanInfo* info = nullptr, ink::Int8MmaShards shards = ink::Int8MmaShards::Auto,
                                         const ink::AttnArgs* args = nullptr, int64_t bound = 0) {
                    const auto launch = [&](ink::Int8MmaPlanInfo* selected = nullptr) {
                        basalt::kernels::PdlScope pdl(prepared_pdl, true);
                        if (prepared_pdl) ink::decode_prepare(pa, rows, nullptr);
                        if (compare)
                            return ink::attention_decode_int8_test(args ? *args : aa, rows, bound ? bound : keys, scr, nullptr,
                                                                   nullptr, mma, plan, selected, shards);
                        ink::attention_decode(aa, rows, keys, scr, nullptr, nullptr);
                        return true;
                    };
                    if (!launch(info)) return 0.;
                    cudaEventRecord(t0);
                    for (int i = 0; i < reps; ++i) launch();
                    cudaEventRecord(t1);
                    float elapsed = 0.f;
                    ok = check_cuda(cudaEventSynchronize(t1), "bench read") &&
                         check_cuda(cudaEventElapsedTime(&elapsed, t0, t1), "bench elapsed") && ok;
                    return (double) elapsed * 1000. / reps;
                };
                double us2 = read_us(false);
                if (compare) {
                    ink::Int8MmaPlanInfo chosen;
                    const double first = read_us(true, false, ink::Int8MmaPlan::Auto, &chosen);
                    const double second = read_us(true);
                    us2 = (us2 + read_us(false)) / 2.;
                    const double mma_us = (first + second) / 2.;
                    std::printf("kv bench: depth %-6lld int8    %d rows: %8.1f us MMA read alone, %6.0f GB/s of one row's K and V "
                                "shards=%d waves=%.3f\n", (long long) depth, rows, mma_us, kv / mma_us / 1000.,
                                chosen.shards, (double) chosen.blocks / chosen.slots);
                    // The shard rule against the halving rule it falls back to, interleaved: at the cell's own bound, then at
                    // the serving graph's bucket bound with its proven lower bound (session.cpp's full_keys_of and its use).
                    using Shards = ink::Int8MmaShards;
                    const auto pair = [&](const ink::AttnArgs* args, int64_t bound, ink::Int8MmaPlanInfo& automatic,
                                          ink::Int8MmaPlanInfo& previous, double& auto_us, double& previous_us) {
                        const double a1 = read_us(true, false, ink::Int8MmaPlan::Auto, &automatic, Shards::Auto, args, bound);
                        const double b1 = read_us(true, false, ink::Int8MmaPlan::Auto, &previous, Shards::Previous, args, bound);
                        const double b2 = read_us(true, false, ink::Int8MmaPlan::Auto, nullptr, Shards::Previous, args, bound);
                        const double a2 = read_us(true, false, ink::Int8MmaPlan::Auto, nullptr, Shards::Auto, args, bound);
                        auto_us = (a1 + a2) / 2.;
                        previous_us = (b1 + b2) / 2.;
                    };
                    ink::Int8MmaPlanInfo ca, cp, ba, bp;
                    double ca_us = 0., cp_us = 0., ba_us = 0., bp_us = 0.;
                    pair(nullptr, 0, ca, cp, ca_us, cp_us);
                    std::printf("kv MMA cell: depth=%lld T=%d plan=%s stages=%d resident=%d slots=%d auto=%.1f us shards=%d "
                                "waves=%.3f busy=%.3f units=%.2f previous=%.1f us shards=%d waves=%.3f busy=%.3f units=%.2f "
                                "(%+.1f%%)\n", (long long) depth, rows, mma_ring_name(ca), ca.stages, ca.resident, ca.slots,
                                ca_us, ca.shards, (double) ca.blocks / ca.slots, (double) ca.busy / ca.slots, ca.units, cp_us,
                                cp.shards, (double) cp.blocks / cp.slots, (double) cp.busy / cp.slots, cp.units,
                                100. * (ca_us / cp_us - 1.));
                    int64_t bucket = 1024;
                    while (bucket < depth) bucket <<= 1;
                    int64_t below = bucket / 2;
                    if (rows == 1 && bucket == ink::kMmaDeepMinKeys) {
                        bucket *= 2;
                        below = ink::kMmaDeepMinKeys - 1;
                    }
                    ink::AttnArgs served = aa;
                    served.min_full_keys = below + 2 - rows;
                    served.cap = bucket;   // a context of at least the bucket; the rows read only the cache's first depth cells
                    pair(&served, bucket, ba, bp, ba_us, bp_us);
                    std::printf("kv MMA bucket: depth=%lld T=%d bound=%lld fewest=%lld auto=%.1f us shards=%d waves=%.3f "
                                "previous=%.1f us shards=%d waves=%.3f (%+.1f%%)\n", (long long) depth, rows,
                                (long long) bucket, (long long) served.min_full_keys, ba_us, ba.shards,
                                (double) ba.blocks / ba.slots, bp_us, bp.shards, (double) bp.blocks / bp.slots,
                                100. * (ba_us / bp_us - 1.));
                }
                std::printf("kv bench: depth %-6lld %-7s %d rows: %8.1f us the read alone,    %6.0f GB/s of one row's K and V\n",
                            (long long) depth, f.name, rows, us2, kv / us2 / 1000.);
                const auto bench_plans = [&](std::initializer_list<ink::Int8MmaPlan> plans) {
                    struct Sample {
                        ink::Int8MmaPlan plan;
                        ink::Int8MmaPlanInfo info;
                        double us;
                    };
                    std::vector<Sample> samples;
                    for (auto plan : plans) {
                        ink::Int8MmaPlanInfo info;
                        const double us = read_us(true, false, plan, &info);
                        samples.push_back({plan, info, us});
                    }
                    for (auto it = samples.rbegin(); it != samples.rend(); ++it)
                        if (it->info.resident != 0)
                            it->us = (it->us + read_us(true, false, it->plan)) / 2.;
                    for (const auto& sample : samples) {
                        const auto& info = sample.info;
                        const char* name = mma_plan_name(sample.plan);
                        if (info.resident == 0)
                            std::printf("kv MMA plan: depth=%lld T=%d %s unavailable\n", (long long) depth, rows, name);
                        else
                            std::printf("kv MMA plan: depth=%lld T=%d %s %.1f us %.0f GB/s stages=%d ping_pong=%d shared=%zu "
                                        "resident=%d slots=%d shards=%d waves=%.3f busy=%.3f units=%.2f\n", (long long) depth,
                                        rows, name, sample.us, kv / sample.us / 1000., info.stages, info.ping_pong, info.shared,
                                        info.resident, info.slots, info.shards, (double) info.blocks / info.slots,
                                        (double) info.busy / info.slots, info.units);
                    }
                };
                using Plan = ink::Int8MmaPlan;
                if (compare && rows >= 6 && depth >= kKeysPartTail)
                    bench_plans({Plan::Auto, Plan::TwoStageRing, Plan::ThreeStageRing, Plan::OneStagePingPong});
                if (compare && rows == 1 && depth >= 65536)
                    bench_plans({Plan::NarrowRing, Plan::Direct});
                if (compare && rows <= 2) {
                    const double scalar_a = read_us(false, true);
                    const double mma_b = read_us(true, true);
                    const double mma_c = read_us(true, true);
                    const double scalar_d = read_us(false, true);
                    std::printf("kv bench: depth %-6lld int8    %d rows: %8.1f -> %8.1f us scalar -> MMA prepare+read (PDL)\n",
                                (long long) depth, rows, (scalar_a + scalar_d) / 2., (mma_b + mma_c) / 2.);
                }
            }
            cudaFree(kc);
            cudaFree(vc);
        }
    if (t0) cudaEventDestroy(t0);
    if (t1) cudaEventDestroy(t1);
    for (void* p : {(void*) y, (void*) taps, (void*) qs, (void*) rs, (void*) kr, (void*) vr, (void*) table, (void*) q, (void*) r,
                    (void*) sink, (void*) pos, (void*) q8, scr})
        if (p) cudaFree(p);
    return ok;
}

// Repeat random cache records over a prime period so the CPU can fold the far keys by multiplicity.
// The GPU still reads every actual key, including unaligned tails and the full relative-bias band.
constexpr int kPeriod = 257;
constexpr int kOutput = ink::kHeads * ink::kHeadDim;
constexpr int kOutputQ8 = kOutput / 32 * 36;
constexpr int kCacheCell = ink::kKvHeads * ink::kKvInt8RowBytes;

static void random_cache(std::mt19937& rng, std::vector<uint8_t>& bytes, std::vector<double>& exact) {
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> magnitude(0.002f, 0.015f);
    for (int i = 0; i < kPeriod * ink::kKvHeads; ++i) {
        auto* row = bytes.data() + (size_t) i * ink::kKvInt8RowBytes;
        for (int group = 0; group < 2; ++group) {
            const __half scale = __float2half_rn(magnitude(rng));
            std::memcpy(row + ink::kHeadDim + group * sizeof(scale), &scale, sizeof(scale));
            for (int d = 64 * group; d < 64 * (group + 1); ++d) {
                const int x = code(rng);
                row[d] = (uint8_t) x;
                exact[(size_t) i * ink::kHeadDim + d] = (double) x * __half2float(scale);
            }
        }
    }
}

static void cpu_attention(int pos, const float* q, const float* r, const float* sink, const float* gate,
                          const float* table, const std::vector<double>& kc, const std::vector<double>& vc,
                          double scale, double* out) {
    const int near_first = std::max(0, pos + 1 - ink::kExtent);
    for (int h = 0; h < ink::kHeads; ++h) {
        const int kvh = h / (ink::kHeads / ink::kKvHeads);
        double base[kPeriod], near[ink::kExtent], weights[kPeriod] = {};
        double m = sink[h];
        for (int key = 0; key < kPeriod; ++key) {
            const double* k = kc.data() + (key * ink::kKvHeads + kvh) * ink::kHeadDim;
            double score = 0.;
            for (int d = 0; d < ink::kHeadDim; ++d) score += (double) q[h * ink::kHeadDim + d] * k[d];
            base[key] = score * scale;
            if (key < near_first) m = std::max(m, base[key]);
        }
        for (int key = near_first; key <= pos; ++key) {
            const int dist = pos - key;
            double bias = 0.;
            for (int d = 0; d < ink::kRel; ++d)
                bias += (double) r[h * ink::kRel + d] * table[d * ink::kExtent + dist];
            near[dist] = base[key % kPeriod] + bias;
            m = std::max(m, near[dist]);
        }
        for (int key = 0; key < kPeriod; ++key) {
            const int count = near_first / kPeriod + (key < near_first % kPeriod);
            if (count) weights[key] = count * std::exp(base[key] - m);
        }
        for (int key = near_first; key <= pos; ++key)
            weights[key % kPeriod] += std::exp(near[pos - key] - m);
        double denom = std::exp(sink[h] - m), numer[ink::kHeadDim] = {};
        for (int key = 0; key < kPeriod; ++key) {
            const double* v = vc.data() + (key * ink::kKvHeads + kvh) * ink::kHeadDim;
            denom += weights[key];
            for (int d = 0; d < ink::kHeadDim; ++d) numer[d] += weights[key] * v[d];
        }
        for (int d = 0; d < ink::kHeadDim; ++d) {
            const int j = h * ink::kHeadDim + d;
            out[j] = numer[d] / denom / (1. + std::exp(-(double) gate[j]));
        }
    }
}

struct DecodeResult {
    std::vector<float> fp32 = std::vector<float>(ink::kDecodeRows * kOutput);
    std::vector<uint8_t> q8 = std::vector<uint8_t>(ink::kDecodeRows * kOutputQ8);
};

static bool same_output(const DecodeResult& a, int ar, const DecodeResult& b, int br) {
    return std::memcmp(a.fp32.data() + ar * kOutput, b.fp32.data() + br * kOutput, kOutput * sizeof(float)) == 0 &&
           std::memcmp(a.q8.data() + ar * kOutputQ8, b.q8.data() + br * kOutputQ8, kOutputQ8) == 0;
}

static uint64_t output_hash(const DecodeResult& a, int row) {
    uint64_t hash = 14695981039346656037ull;
    const auto mix = [&](const unsigned char* bytes, size_t n) {
        for (size_t j = 0; j < n; ++j) hash = (hash ^ bytes[j]) * 1099511628211ull;
    };
    mix(reinterpret_cast<const unsigned char*>(a.fp32.data() + row * kOutput), kOutput * sizeof(float));
    mix(a.q8.data() + row * kOutputQ8, kOutputQ8);
    return hash;
}

// The --reduced MMA cases, for a race checker that serialises shared memory: few depths and widths, but every plan, both
// launch bounds and the extreme shard counts. A case's lines and hashes are the full run's for the same depth, width and row.
struct MmaVariant {
    ink::Int8MmaPlan plan;
    bool padded;   // the launch bound: the cache's cells, else the window's own keys
    int shards;    // 0: the rule's count; else forced (at most the launch's splits)
};
struct MmaCase {
    int depth, width;
    std::vector<MmaVariant> variants;
};

static std::vector<MmaCase> reduced_mma_cases() {
    using P = ink::Int8MmaPlan;
    const MmaVariant padded{P::Auto, true, 0}, fewest{P::Auto, false, 1}, most{P::Auto, false, 64};
    return {
        {4096, 1, {{P::Direct, true, 0}, fewest, most, {P::NarrowRing, false, 0}, {P::NarrowRing, true, 0}}},
        {4096, 2, {padded, fewest, most}},
        {4096, 3, {padded, fewest, most}},
        {4096, 6, {padded, fewest, most, {P::TwoStageRing, true, 0}, {P::ThreeStageRing, false, 0},
                   {P::OneStagePingPong, false, 0}}},
        {4096, 8, {padded, fewest, most, {P::ThreeStageRing, true, 0}}},
        {32768, 1, {padded, {P::NarrowRing, false, 0}}},
        {32768, 8, {fewest, {P::ThreeStageRing, false, 0}}},
        {32769, 2, {padded, fewest}},
        {32769, 6, {most, {P::ThreeStageRing, false, 0}}},
        {65537, 3, {padded, fewest}},
        {kKeys100Ki, 1, {{P::NarrowRing, false, 0}}},
        {kKeys100Ki + 1, 1, {{P::NarrowRing, false, 0}}},
        {131072, 2, {padded}},
    };
}

static const char* flag(int f) { return f < 0 ? "-" : f ? "1" : "0"; }

static bool check_attention_mma(bool reduced) {
    constexpr int rows = ink::kDecodeRows, cap = 262144 + rows;
    std::mt19937 rng(0x4b323062);
    std::normal_distribution<float> normal(0.f, 0.4f);
    std::vector<float> q(rows * kOutput), r(rows * ink::kHeads * ink::kRel), sink(rows * ink::kHeads),
        gate(rows * kOutput), table(ink::kRel * ink::kExtent);
    for (auto* a : {&q, &r, &sink, &gate, &table})
        for (float& x : *a) x = normal(rng);
    std::vector<uint8_t> kp(kPeriod * kCacheCell), vp(kPeriod * kCacheCell);
    std::vector<double> hk(kPeriod * ink::kKvHeads * ink::kHeadDim), hv(hk.size());
    random_cache(rng, kp, hk);
    random_cache(rng, vp, hv);

    int32_t* dp = nullptr;
    float *dq = nullptr, *dr = nullptr, *ds = nullptr, *dg = nullptr, *dt = nullptr, *dout = nullptr;
    uint8_t *dk = nullptr, *dv = nullptr, *dq8 = nullptr;
    void* scratch = nullptr;
    const auto alloc = [](auto** ptr, size_t bytes) {
        return check_cuda(cudaMalloc((void**) ptr, bytes), "attention parity allocate");
    };
    const auto upload = [&](float** ptr, const std::vector<float>& v) {
        const size_t bytes = v.size() * sizeof(float);
        return alloc(ptr, bytes) && check_cuda(basalt::core::device_copy(*ptr, v.data(), bytes, cudaMemcpyHostToDevice), "attention inputs");
    };
    const auto upload_cache = [&](uint8_t** ptr, const std::vector<uint8_t>& pattern) {
        std::vector<uint8_t> host((size_t) cap * kCacheCell);
        for (size_t i = 0; i < host.size(); i += pattern.size())
            std::memcpy(host.data() + i, pattern.data(), std::min(pattern.size(), host.size() - i));
        return alloc(ptr, host.size()) &&
               check_cuda(basalt::core::device_copy(*ptr, host.data(), host.size(), cudaMemcpyHostToDevice), "attention cache");
    };
    bool ok = upload(&dq, q) && upload(&dr, r) && upload(&ds, sink) && upload(&dg, gate) && upload(&dt, table) &&
              upload_cache(&dk, kp) && upload_cache(&dv, vp) && alloc(&dp, rows * sizeof(int32_t)) &&
              alloc(&dout, rows * kOutput * sizeof(float)) && alloc(&dq8, rows * kOutputQ8) &&
              alloc(&scratch, ink::attention_decode_scratch(rows, cap));
    const int mma_min_keys = ink::attention_decode_int8_test_min_keys();
    const std::vector<MmaCase> cases = reduced ? reduced_mma_cases() : std::vector<MmaCase>{};
    std::vector<int> depths;
    if (reduced) {
        for (const MmaCase& c : cases)
            if (depths.empty() || depths.back() != c.depth) depths.push_back(c.depth);
    } else {
        depths.push_back(4096);
        for (const int boundary : {32768, 65536, kKeys100Ki, 131072, 196608, 262144})
            depths.insert(depths.end(), {boundary - 1, boundary, boundary + 1});
    }
    bool schedules_ok = true;
    bool reported[static_cast<int>(ink::Int8MmaPlan::Direct) + 1] = {};
    for (const int depth : depths) {
        if (!ok) break;
        int32_t pos[rows];
        for (int j = 0; j < rows; ++j) pos[j] = depth - 4 + j;
        ok = check_cuda(basalt::core::device_copy(dp, pos, sizeof(pos), cudaMemcpyHostToDevice), "attention positions");
        if (!ok) break;
        enum class Reader { Auto, Scalar, Mma };
        const auto run = [&](int first, int width, Reader reader, DecodeResult& dst,
                             bool batch = false, bool swap_kv = false, int key_bound = 0,
                             ink::Int8MmaPlan plan = ink::Int8MmaPlan::Auto, ink::Int8MmaPlanInfo* info = nullptr,
                             ink::Int8MmaShards shards = ink::Int8MmaShards::Auto, int shard_count = 0) {
            ink::AttnArgs a;
            a.int8 = true;
            a.q = dq + first * kOutput;
            a.r = dr + first * ink::kHeads * ink::kRel;
            a.sink = ds + first * ink::kHeads;
            a.gate = dg + first * kOutput;
            a.gate_ld = kOutput;
            a.rel_table = dt;
            a.pos = dp + first;
            a.k_cache = swap_kv ? dv : dk;
            a.v_cache = swap_kv ? dk : dv;
            a.cap = cap;
            a.q8 = dq8;
            if (batch) {
                a.seg.at.n = 2;
                a.seg.at.row0[1] = 5;
                a.seg.at.row0[2] = width;
                a.seg.c[0].k = a.seg.c[1].v = dk;
                a.seg.c[0].v = a.seg.c[1].k = dv;
                a.k_cache = a.v_cache = nullptr;
            } else a.min_full_keys = pos[first] + 1;
            const int keys = key_bound ? key_bound : pos[first + width - 1] + 1;
            if (reader == Reader::Auto) {
                ink::attention_decode(a, width, keys, scratch, dout, nullptr);
            } else if (!ink::attention_decode_int8_test(a, width, keys, scratch, dout, nullptr, reader == Reader::Mma, plan, info,
                                                        shards, shard_count)) {
                return false;
            }
            return check_cuda(basalt::core::device_copy(dst.fp32.data(), dout, width * kOutput * sizeof(float), cudaMemcpyDeviceToHost),
                              "attention FP32 output") &&
                   check_cuda(basalt::core::device_copy(dst.q8.data(), dq8, width * kOutputQ8, cudaMemcpyDeviceToHost), "attention q8_1 output");
        };
        const auto matches_plan = [&](int first, int width, ink::Int8MmaPlan plan, int bound, const DecodeResult& expected,
                                      ink::Int8MmaShards shards = ink::Int8MmaShards::Auto, int shard_count = 0) {
            DecodeResult got;
            ink::Int8MmaPlanInfo info;
            const bool ran = run(first, width, Reader::Mma, got, false, false, bound, plan, &info, shards, shard_count);
            bool& printed = reported[static_cast<int>(plan)];
            if (plan != ink::Int8MmaPlan::Auto && !printed) {
                std::printf("kv MMA parity arm: %s available=%d\n", mma_plan_name(plan), info.resident != 0);
                printed = true;
            }
            if (!ran) return false;
            for (int j = 0; j < width; ++j)
                if (!same_output(got, j, expected, j)) return false;
            return true;
        };
        std::vector<double> ref(rows * kOutput);
        for (int j = 0; j < rows; ++j)
            cpu_attention(pos[j], q.data() + j * kOutput, r.data() + j * ink::kHeads * ink::kRel,
                          sink.data() + j * ink::kHeads, gate.data() + j * kOutput, table.data(), hk, hv,
                          ink::AttnArgs{}.scale, ref.data() + j * kOutput);
        // A window row's accuracy against the scalar reader and the CPU, its flags (-1: not checked) and its hash.
        const auto report = [&](int width, int first, int j, const DecodeResult& scalar, const DecodeResult& mma, int invariant,
                                int dispatched, int batch_invariant, int schedule_invariant, int shard_invariant) {
            double abs[3] = {}, rel[3] = {}, rel_sq[3] = {};
            int rel_count[3] = {};
            bool finite = true;
            for (int d = 0; d < kOutput; ++d) {
                const double ref_out = ref[(first + j) * kOutput + d];
                const double old_out = scalar.fp32[j * kOutput + d], mma_out = mma.fp32[j * kOutput + d];
                finite &= std::isfinite(old_out) && std::isfinite(mma_out);
                const double actual[] = {mma_out, mma_out, old_out};
                const double want[] = {old_out, ref_out, ref_out};
                for (int k = 0; k < 3; ++k) {
                    const double error = std::abs(actual[k] - want[k]);
                    abs[k] = std::max(abs[k], error);
                    if (std::abs(want[k]) >= 1e-2) {
                        const double fraction = error / std::abs(want[k]);
                        rel[k] = std::max(rel[k], fraction);
                        rel_sq[k] += fraction * fraction;
                        ++rel_count[k];
                    }
                }
            }
            double rms_rel[3];
            for (int k = 0; k < 3; ++k)
                rms_rel[k] = rel_count[k] ? std::sqrt(rel_sq[k] / rel_count[k]) : 0.;
            std::printf("kv MMA parity: depth=%d T=%d row=%d keys=%d max_abs[mma/scalar,mma/cpu,scalar/cpu]=%.7g/%.7g/%.7g "
                        "max_rel(|ref|>=1e-2)=%.7g/%.7g/%.7g rms_rel(|ref|>=1e-2)=%.7g/%.7g/%.7g "
                        "finite=%d invariant=%s dispatch=%s batch=%s schedule=%s shard_invariant=%s hash=%016llx\n",
                        depth, width, j, pos[first + j] + 1, abs[0], abs[1], abs[2], rel[0], rel[1], rel[2],
                        rms_rel[0], rms_rel[1], rms_rel[2], finite, flag(invariant), flag(dispatched), flag(batch_invariant),
                        flag(schedule_invariant), flag(shard_invariant), (unsigned long long) output_hash(mma, j));
            schedules_ok &= schedule_invariant != 0 && shard_invariant != 0;
            ok &= finite && invariant != 0 && dispatched != 0 && batch_invariant != 0 &&
                  *std::max_element(abs, abs + 3) < 0.002 && *std::max_element(rel, rel + 3) < 0.05;
        };
        if (reduced) {
            for (const MmaCase& c : cases) {
                if (c.depth != depth || !ok) continue;
                const auto start = std::chrono::steady_clock::now();
                const int first = 3 - (c.width - 1) / 2;
                DecodeResult scalar, mma;
                ok = run(first, c.width, Reader::Scalar, scalar) && run(first, c.width, Reader::Mma, mma);
                int invariant = -1, schedule = -1, shard_invariant = -1;
                for (const MmaVariant& v : c.variants) {
                    if (!ok) break;
                    const int same = matches_plan(first, c.width, v.plan, v.padded ? cap : 0, mma, ink::Int8MmaShards::Auto,
                                                  v.shards);
                    int& checked = v.plan != ink::Int8MmaPlan::Auto ? schedule : v.shards ? shard_invariant : invariant;
                    checked = checked != 0 && same;
                    if (v.padded) invariant = invariant != 0 && same;
                }
                if (!ok) break;
                // The first and the last row of the window: its fewest and most keys.
                report(c.width, first, 0, scalar, mma, invariant, -1, -1, schedule, shard_invariant);
                if (c.width > 1) report(c.width, first, c.width - 1, scalar, mma, invariant, -1, -1, schedule, shard_invariant);
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                std::printf("kv MMA reduced case: depth=%d T=%d launches=%zu %.1f s\n", depth, c.width, 2 + c.variants.size(),
                            seconds);
            }
            continue;
        }
        DecodeResult singleton[rows];
        bool singleton_schedule[rows] = {};
        for (int j = 0; j < rows && ok; ++j) {
            ok = run(j, 1, Reader::Mma, singleton[j], false, false, cap);
            // Compare the singleton schedules under tight and padded launch bounds.
            singleton_schedule[j] = ok;
            if (!ok) break;
            for (const int bound : {0, cap}) {
                singleton_schedule[j] &= matches_plan(j, 1, ink::Int8MmaPlan::NarrowRing, bound, singleton[j]);
                singleton_schedule[j] &= matches_plan(j, 1, ink::Int8MmaPlan::Direct, bound, singleton[j]);
            }
        }
        for (const int width : {1, 2, 3, 5, 6, 7, 8}) {
            if (!ok) break;
            const int first = 3 - (width - 1) / 2;
            DecodeResult scalar, mma, padded, automatic, auto_padded, segmented, separate;
            ok = run(first, width, Reader::Scalar, scalar) && run(first, width, Reader::Mma, mma) &&
                 run(first, width, Reader::Mma, padded, false, false, cap) &&
                 run(first, width, Reader::Auto, automatic) && run(first, width, Reader::Auto, auto_padded, false, false, cap);
            bool schedule = true, shard_invariant = true;
            if (ok) {
                for (const int bound : {0, cap}) {
                    shard_invariant &= matches_plan(first, width, ink::Int8MmaPlan::Auto, bound, mma, ink::Int8MmaShards::Previous);
                    // Every count the rule can take, and uneven ones, assign whole slices to blocks only.
                    for (const int count : {1, 2, 3, 4, 8, 16, 21, 32, 64})
                        shard_invariant &= matches_plan(first, width, ink::Int8MmaPlan::Auto, bound, mma,
                                                        ink::Int8MmaShards::Auto, count);
                    if (width >= 6) {
                        schedule &= matches_plan(first, width, ink::Int8MmaPlan::TwoStageRing, bound, mma);
                        schedule &= matches_plan(first, width, ink::Int8MmaPlan::ThreeStageRing, bound, mma);
                        schedule &= matches_plan(first, width, ink::Int8MmaPlan::OneStagePingPong, bound, mma);
                    }
                }
            }
            if (ok && width == rows)
                ok = run(first, width, Reader::Auto, segmented, true) &&
                     run(first + 5, width - 5, Reader::Auto, separate, false, true);
            if (!ok) break;
            for (int j = 0; j < width; ++j) {
                const bool invariant = same_output(mma, j, singleton[first + j], 0) && same_output(mma, j, padded, j);
                const auto& expected = pos[first + j] + 1 >= mma_min_keys ? mma : scalar;
                const bool dispatched = same_output(automatic, j, expected, j) && same_output(auto_padded, j, expected, j);
                const bool batch_invariant = width != rows ||
                    (j < 5 ? same_output(automatic, j, segmented, j) : same_output(separate, j - 5, segmented, j));
                report(width, first, j, scalar, mma, invariant, dispatched, batch_invariant,
                       schedule && singleton_schedule[first + j], shard_invariant);
            }
        }
    }
    if (reduced) std::printf("reduced case list: %zu cases\n", cases.size());
    for (void* p : {(void*) dp, (void*) dq, (void*) dr, (void*) ds, (void*) dg, (void*) dt, (void*) dout, (void*) dq8,
                    (void*) dk, (void*) dv, scratch})
        if (p) cudaFree(p);
    return ok && schedules_ok;
}

// The serving decode chain of a narrow window: the preparation writes the window's rows into the cache, the MMA reader
// (with programmatic dependent launch it copies its first tile of the proven prefix before it waits) and the merge,
// captured in a graph and replayed with fresh inputs and the window's cells poisoned (NaN scales) before each replay.
// The same graph captured with plain launches, which copies nothing early, is the reference: every replay's output must
// be finite and the same bits. With `tight` the proven bound is the first row's keys, else the serving bucket's.
static bool check_decode_graph(bool reduced) {
    constexpr int rows = 2, cap = 131072 + ink::kDecodeRows;
    const int replays = reduced ? 4 : 64;
    struct Shape {
        int depth;
        bool tight;
    };
    const std::vector<Shape> shapes = reduced ? std::vector<Shape>{{32769, true}}
                                              : std::vector<Shape>{{32769, false}, {32769, true}, {65537, false}, {65537, true},
                                                                   {kKeys100Ki, false}, {kKeys100Ki, true}, {131072, false},
                                                                   {131072, true}};
    std::mt19937 rng(0x67726170);
    std::normal_distribution<float> normal(0.f, 0.5f);
    const auto randoms = [&](size_t n, float base) {
        std::vector<float> v(n);
        for (float& x : v) x = base + normal(rng);
        return v;
    };
    std::vector<uint8_t> kp(kPeriod * kCacheCell), vp(kPeriod * kCacheCell);
    std::vector<double> exact(kPeriod * ink::kKvHeads * ink::kHeadDim);
    random_cache(rng, kp, exact);
    random_cache(rng, vp, exact);
    const std::vector<float> taps = randoms(1024 * ink::kConv, 0.f), qs = randoms(ink::kHeads * ink::kHeadDim, 1.f),
                             rs = randoms(ink::kHeads * ink::kRel, 1.f), ring0 = randoms((size_t) ink::kConvRing * 1024, 0.f),
                             table = randoms((size_t) ink::kRel * ink::kExtent, 0.f);
    std::vector<std::vector<float>> ys(replays + 1);
    for (auto& y : ys) y = randoms((size_t) rows * ink::kProj, 0.f);

    float *dy = nullptr, *dtaps = nullptr, *dqs = nullptr, *drs = nullptr, *dkr = nullptr, *dvr = nullptr, *dtable = nullptr;
    float *dq = nullptr, *dr = nullptr, *dsink = nullptr, *dout = nullptr;
    int32_t* dpos = nullptr;
    uint8_t *dk = nullptr, *dv = nullptr, *dq8 = nullptr;
    void* scratch = nullptr;
    cudaStream_t stream = nullptr;
    const auto alloc = [](auto** ptr, size_t bytes) {
        return check_cuda(cudaMalloc((void**) ptr, bytes), "graph parity allocate");
    };
    const auto upload = [&](float** ptr, const std::vector<float>& v) {
        return alloc(ptr, v.size() * 4) &&
               check_cuda(basalt::core::device_copy(*ptr, v.data(), v.size() * 4, cudaMemcpyHostToDevice), "graph inputs");
    };
    const auto upload_cache = [&](uint8_t** ptr, const std::vector<uint8_t>& pattern) {
        std::vector<uint8_t> host((size_t) cap * kCacheCell);
        for (size_t i = 0; i < host.size(); i += pattern.size())
            std::memcpy(host.data() + i, pattern.data(), std::min(pattern.size(), host.size() - i));
        return alloc(ptr, host.size()) &&
               check_cuda(basalt::core::device_copy(*ptr, host.data(), host.size(), cudaMemcpyHostToDevice), "graph cache");
    };
    bool ok = alloc(&dy, ys[0].size() * 4) && upload(&dtaps, taps) && upload(&dqs, qs) && upload(&drs, rs) &&
              upload(&dkr, ring0) && upload(&dvr, ring0) && upload(&dtable, table) &&
              alloc(&dq, (size_t) rows * kOutput * 4) && alloc(&dr, (size_t) rows * ink::kHeads * ink::kRel * 4) &&
              alloc(&dsink, (size_t) rows * ink::kHeads * 4) && alloc(&dout, (size_t) rows * kOutput * 4) &&
              alloc(&dq8, (size_t) rows * kOutputQ8) && alloc(&dpos, rows * 4) && upload_cache(&dk, kp) &&
              upload_cache(&dv, vp) &&
              alloc(&scratch, ink::attention_decode_scratch(rows, cap)) &&
              check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "graph stream");
    bool pass = true;
    for (const Shape& shape : shapes) {
        if (!ok) break;
        const int pos0 = shape.depth - rows;   // the last row reads `depth` keys
        int64_t bucket = 1024;                 // the serving graph's key bound (session.cpp full_keys_of)
        while (bucket < shape.depth) bucket <<= 1;
        const int64_t fewest = shape.tight ? pos0 + 1 : bucket / 2 + 2 - rows;
        const int32_t pos[rows] = {pos0, pos0 + 1};
        ok = check_cuda(basalt::core::device_copy(dpos, pos, sizeof(pos), cudaMemcpyHostToDevice), "graph positions");
        ink::AttnPrep pa;
        pa.yq = dy; pa.yk = dy + ink::kProjQ; pa.yv = dy + ink::kProjQ + ink::kProjKV; pa.ldq = pa.ldk = pa.ldv = ink::kProj;
        pa.pos = dpos; pa.full = true; pa.k_conv = dtaps; pa.v_conv = dtaps; pa.q_scale = dqs; pa.r_scale = drs;
        pa.k_ring = dkr; pa.v_ring = dvr; pa.k_cache = dk; pa.v_cache = dv; pa.cap = cap; pa.int8 = true;
        pa.q = dq; pa.r = dr; pa.sink = dsink;
        ink::AttnArgs aa;
        aa.q = dq; aa.r = dr; aa.sink = dsink; aa.rel_table = dtable; aa.pos = dpos; aa.k_cache = dk; aa.v_cache = dv;
        aa.cap = cap; aa.int8 = true; aa.gate = dy + ink::kOffGate; aa.gate_ld = ink::kProj; aa.q8 = dq8;
        aa.min_full_keys = fewest;
        const size_t cells = (size_t) pos0 * ink::kKvHeads * ink::kKvInt8RowBytes;
        const size_t window = (size_t) rows * ink::kKvHeads * ink::kKvInt8RowBytes;
        std::vector<float> fp32[2];
        std::vector<uint8_t> q8[2];
        long pdl_launches = 0;
        for (const bool pdl : {true, false}) {
            const auto chain = [&] {
                // The preparation follows the poisoning copies (a plain launch); the reader and the merge may start early.
                basalt::kernels::PdlScope scope(pdl, true);
                ink::decode_prepare(pa, rows, stream);
                ink::attention_decode(aa, rows, bucket, scratch, dout, stream);
            };
            // A shape's first window runs directly, as the session's does, before its graph is captured.
            ok = ok && check_cuda(basalt::core::device_copy(dy, ys[replays].data(), ys[replays].size() * 4, cudaMemcpyHostToDevice), "graph y");
            if (ok) chain();
            ok = ok && check_cuda(cudaStreamSynchronize(stream), "graph direct run");
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            const long before = basalt::kernels::pdl_count();
            ok = ok && check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "graph capture");
            if (ok) chain();
            ok = ok && check_cuda(cudaStreamEndCapture(stream, &graph), "graph capture end") &&
                 check_cuda(cudaGraphInstantiate(&exec, graph, 0), "graph instantiate");
            if (pdl) pdl_launches = basalt::kernels::pdl_count() - before;
            ok = ok && check_cuda(basalt::core::device_copy(dkr, ring0.data(), ring0.size() * 4, cudaMemcpyHostToDevice), "graph k ring") &&
                 check_cuda(basalt::core::device_copy(dvr, ring0.data(), ring0.size() * 4, cudaMemcpyHostToDevice), "graph v ring");
            fp32[pdl].resize((size_t) replays * rows * kOutput);
            q8[pdl].resize((size_t) replays * rows * kOutputQ8);
            for (int i = 0; i < replays && ok; ++i) {
                ok = check_cuda(cudaMemsetAsync(dk + cells, 0xff, window, stream), "graph poison k") &&
                     check_cuda(cudaMemsetAsync(dv + cells, 0xff, window, stream), "graph poison v") &&
                     check_cuda(cudaMemcpyAsync(dy, ys[i].data(), ys[i].size() * 4, cudaMemcpyHostToDevice, stream), "graph y") &&
                     check_cuda(cudaGraphLaunch(exec, stream), "graph launch") &&
                     check_cuda(cudaMemcpyAsync(fp32[pdl].data() + (size_t) i * rows * kOutput, dout, (size_t) rows * kOutput * 4,
                                                cudaMemcpyDeviceToHost, stream), "graph output") &&
                     check_cuda(cudaMemcpyAsync(q8[pdl].data() + (size_t) i * rows * kOutputQ8, dq8, (size_t) rows * kOutputQ8,
                                                cudaMemcpyDeviceToHost, stream), "graph q8") &&
                     check_cuda(cudaStreamSynchronize(stream), "graph replay");
            }
            if (exec) cudaGraphExecDestroy(exec);
            if (graph) cudaGraphDestroy(graph);
        }
        if (!ok) break;
        bool finite = true;
        for (const auto& out : fp32)
            for (const float x : out) finite &= std::isfinite(x);
        const bool identical = fp32[0] == fp32[1] && q8[0] == q8[1];
        // The reader's prefix slice: whole passes of its block, at most 64 slices, for the proven bound's keys.
        const int64_t units = (fewest + 128 * 64 - 1) / (128 * 64);
        const int prime_span = (int) std::max<int64_t>(units, 1) * 128;
        std::printf("kv MMA graph: depth=%d T=%d replays=%d prime_span=%d identical=%d bound=%lld fewest=%lld finite=%d "
                    "pdl_launches=%ld\n", shape.depth, rows, replays, prime_span, identical, (long long) bucket,
                    (long long) fewest, finite, pdl_launches);
        pass &= identical && finite && pdl_launches > 0;
    }
    if (stream) cudaStreamDestroy(stream);
    for (void* p : {(void*) dy, (void*) dtaps, (void*) dqs, (void*) drs, (void*) dkr, (void*) dvr, (void*) dtable, (void*) dq,
                    (void*) dr, (void*) dsink, (void*) dout, (void*) dpos, (void*) dk, (void*) dv, (void*) dq8, scratch})
        if (p) cudaFree(p);
    return ok && pass;
}

int main(int argc, char** argv) {
    const bool bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    const bool reduced = argc == 2 && std::strcmp(argv[1], "--reduced") == 0;
    if (argc > 2 || (argc == 2 && !bench && !reduced)) {
        std::fprintf(stderr, "usage: inkling2_kv_int8_parity [--bench | --reduced]\n");
        return 2;
    }

    constexpr int rows = 257;
    constexpr size_t row_bytes = ink::kKvInt8RowBytes;
    std::mt19937 rng(0x1a7e);
    std::normal_distribution<float> dist(0.f, 0.35f);
    std::vector<float> input((size_t) rows * ink::kHeadDim);
    for (float& x : input) x = dist(rng);
    for (int c = 0; c < ink::kHeadDim; ++c) input[c] = 0.f;
    for (int c = 0; c < ink::kHeadDim; ++c) input[(size_t) (rows - 2) * ink::kHeadDim + c] = (c % 9 == 0) ? 1.25f : -0.015f;
    std::copy_n(input.data(), ink::kHeadDim, input.data() + (size_t) (rows - 1) * ink::kHeadDim);

    float *di = nullptr;
    uint8_t* dout = nullptr;
    if (!check_cuda(cudaMalloc(&di, input.size() * sizeof(float)), "allocate input") ||
        !check_cuda(cudaMalloc(&dout, (size_t) rows * row_bytes), "allocate rows") ||
        !check_cuda(basalt::core::device_copy(di, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), "copy input") ||
        !check_cuda(cudaMemset(dout, 0xa5, (size_t) rows * row_bytes), "poison rows")) return 1;

    encode_rows<<<rows, 32>>>(di, dout, rows);
    if (!check_cuda(cudaGetLastError(), "launch") || !check_cuda(cudaDeviceSynchronize(), "synchronize")) return 1;

    std::vector<uint8_t> got((size_t) rows * row_bytes);
    if (!check_cuda(basalt::core::device_copy(got.data(), dout, got.size(), cudaMemcpyDeviceToHost), "copy rows")) return 1;

    double max_abs = 0., max_rel = 0., fp16_max_abs = 0., fp16_max_rel = 0.;
    int mismatches = 0;
    for (int r = 0; r < rows; ++r) {
        float scales[2]{};
        for (int g = 0; g < 2; ++g) {
            float m = 0.f;
            for (int c = g * 64; c < (g + 1) * 64; ++c)
                m = std::max(m, std::fabs(input[(size_t) r * ink::kHeadDim + c]));
            scales[g] = __half2float(__float2half_rn(m / 127.f));
            __half stored;
            std::memcpy(&stored, got.data() + (size_t) r * row_bytes + ink::kHeadDim + g * sizeof(__half), sizeof(stored));
            if (__half_as_ushort(stored) != __half_as_ushort(__float2half_rn(scales[g]))) ++mismatches;
        }
        for (int c = 0; c < ink::kHeadDim; ++c) {
            const float x = input[(size_t) r * ink::kHeadDim + c];
            const float s = scales[c / 64];
            const int q = s > 0.f ? std::clamp((int) std::nearbyint(x / s), -127, 127) : 0;
            const int got_q = (int) (int8_t) got[(size_t) r * row_bytes + c];
            if (got_q != q) ++mismatches;
            const float dequant = (float) got_q * s;
            const double abs = std::fabs((double) dequant - x);
            max_abs = std::max(max_abs, abs);
            const double ax = std::fabs((double) x);
            if (ax >= 1e-2) max_rel = std::max(max_rel, abs / ax);
            const float h = __half2float(__float2half_rn(x));
            const double h_abs = std::fabs((double) h - x);
            fp16_max_abs = std::max(fp16_max_abs, h_abs);
            if (ax >= 1e-2) fp16_max_rel = std::max(fp16_max_rel, h_abs / ax);
        }
        for (size_t b = ink::kHeadDim + 2 * sizeof(__half); b < row_bytes; ++b)
            if (got[(size_t) r * row_bytes + b] != 0) ++mismatches;
    }
    if (std::memcmp(got.data(), got.data() + (size_t) (rows - 1) * row_bytes, row_bytes) != 0) ++mismatches;
    std::printf("kv_int8_parity: rows=%d width=%d row_bytes=%zu mismatches=%d max_abs=%.7g max_rel(|x|>=1e-2)=%.7g fp16_max_abs=%.7g fp16_max_rel(|x|>=1e-2)=%.7g\n",
                rows, ink::kHeadDim, row_bytes, mismatches, max_abs, max_rel, fp16_max_abs, fp16_max_rel);

    bool ok = mismatches == 0 && check_attention_mma(reduced) && check_decode_graph(reduced);
    if (bench) {
        constexpr int bench_rows = 4096;
        constexpr int reps = 200;
        float* bi = nullptr;
        uint8_t* bo = nullptr;
        cudaEvent_t start{}, stop{};
        ok = ok && check_cuda(cudaMalloc(&bi, (size_t) bench_rows * ink::kHeadDim * sizeof(float)), "allocate bench input");
        ok = ok && check_cuda(cudaMalloc(&bo, (size_t) bench_rows * row_bytes), "allocate bench rows");
        ok = ok && check_cuda(cudaMemset(bi, 0, (size_t) bench_rows * ink::kHeadDim * sizeof(float)), "clear bench input");
        ok = ok && check_cuda(cudaEventCreate(&start), "create start event") && check_cuda(cudaEventCreate(&stop), "create stop event");
        if (ok) {
            encode_rows<<<bench_rows, 32>>>(bi, bo, bench_rows);
            ok = check_cuda(cudaGetLastError(), "bench warm-up") && check_cuda(cudaDeviceSynchronize(), "bench warm-up sync");
        }
        if (ok) ok = check_cuda(cudaEventRecord(start), "record start");
        for (int i = 0; ok && i < reps; ++i) encode_rows<<<bench_rows, 32>>>(bi, bo, bench_rows);
        if (ok) ok = check_cuda(cudaEventRecord(stop), "record stop") && check_cuda(cudaEventSynchronize(stop), "wait stop");
        float ms = 0.f;
        if (ok) ok = check_cuda(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
        if (ok) {
            const double us = (double) ms * 1000. / reps;
            const double bytes = (double) bench_rows * (ink::kHeadDim * sizeof(float) + row_bytes);
            std::printf("kv_int8_parity bench: rows=%d us/call=%.3f GB/s=%.1f\n", bench_rows, us, bytes / us / 1000.);
        }
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (bi) cudaFree(bi);
        if (bo) cudaFree(bo);
        ok = ok && bench_attention();
    }
    cudaFree(di);
    cudaFree(dout);
    return ok ? 0 : 1;
}
