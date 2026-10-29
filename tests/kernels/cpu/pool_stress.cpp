// tests/kernels/cpu/pool_stress.cpp - the expert pool under the load a big-VRAM card gives it.
//
// With most experts resident on the GPU, many layers hand the pool nothing, the workers go to sleep in the middle
// of a request, and the next tiny batch (one or two experts) is often finished by the host and the awake workers
// before a sleeper has even woken.  That late wake-up is what the park/claim protocol has to survive.  This drives
// hundreds of thousands of such batches - 0-4 jobs, run() and run_split(), with pauses long enough for the workers
// to sleep - and checks that every batch returns, with every job's output written and equal to a first run's.  On a
// CPU without the AVX-512 kernels the batches go through run_split_multi_native() on an i-quant format instead.  Idle
// work (idle_start) runs between half of the batches, with a barrier across the workers like the warm-up's, and must
// never be running once a batch or idle_stop has returned.  The active worker count changes at random between batches.  A watchdog thread turns a hang into a failure instead of a
// frozen test.
//
//   pool_stress [seconds] [workers]   (default 20 s, one worker per physical core but the host's)
//
// More workers than cores makes the OS deschedule workers in the middle of a batch, the case the claim protocol must
// survive without any wait for the workers to park.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = basalt::kernels::cpu;

int main(int argc, char** argv) {
    const double seconds = argc > 1 ? std::atof(argv[1]) : 20.0;
    const int workers = argc > 2 ? std::atoi(argv[2]) : (int) cpu::physical_cores(true).size();
    const cpu::CpuFeatures feat = cpu::cpu_features();
    const bool native = !feat.usable();
    if (native && !cpu::cpu_avx2_ok()) {
        std::printf("CPU lacks %s and AVX2: pool_stress SKIPPED\n", feat.reason());
        return 0;
    }
    constexpr int NJ = 4;
    std::vector<float> out((size_t) NJ * cpu::H);
    // the AVX-512 path: whole experts and row parts of one token
    std::vector<std::vector<uint8_t>> blobs(NJ, std::vector<uint8_t>(cpu::BLOB, 0));
    std::vector<float> x(cpu::H, 0.25f);
    cpu::ActQ act;
    std::vector<cpu::ExpertJob> jobs(NJ);
    // the native path: an IQ2_S gate/up + IQ4_NL down layer, two tokens on the odd experts
    cpu::NativeFmt fmt;
    std::vector<uint8_t> nact((size_t) 2 * cpu::kNativeActBytes);
    std::vector<float> out2((size_t) NJ * cpu::H);
    std::vector<cpu::ExpertJobMulti> mjobs(NJ);
    if (native) {
        std::string err;
        if (!cpu::native_fmt(GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_NL, cpu::H, cpu::FF, fmt, err)) {
            std::printf("pool_stress: native_fmt: %s\n", err.c_str());
            return 1;
        }
        std::mt19937_64 brng(7);
        for (int e = 0; e < NJ; ++e) {
            std::vector<uint8_t>& b = blobs[(size_t) e];
            b.resize(fmt.bytes);
            for (auto& v : b) v = (uint8_t) brng();
            auto scales = [&](size_t off, size_t len, int type) {   // a small valid fp16 scale at each block start
                const size_t bs = ggml_type_size((ggml_type) type);
                for (size_t o = 0; o + bs <= len; o += bs) {
                    const uint16_t d = (uint16_t) (0x2800 + (b[off + o] & 0xff));
                    std::memcpy(b.data() + off + o, &d, 2);
                }
            };
            scales(0, 2 * fmt.up_off, fmt.gu_type);
            scales(fmt.down_off, fmt.bytes - fmt.down_off, fmt.d_type);
        }
        for (int t = 0; t < 2; ++t) {
            for (int i = 0; i < cpu::H; ++i) x[(size_t) i] = 0.25f * (float) ((i * (t + 3)) % 17 - 8);
            cpu::native_quant_act(fmt, x.data(), nact.data() + (size_t) t * cpu::kNativeActBytes);
        }
        for (int e = 0; e < NJ; ++e) {
            cpu::ExpertJobMulti& j = mjobs[(size_t) e];
            j.blob = blobs[(size_t) e].data();
            j.nt = 1 + (e & 1);
            for (int t = 0; t < j.nt; ++t) {
                j.nact[t] = nact.data() + (size_t) t * cpu::kNativeActBytes;
                j.out[t] = (t == 0 ? out.data() : out2.data()) + (size_t) e * cpu::H;
            }
        }
    } else {
        cpu::act_quant_q8_1(x.data(), cpu::H, act);
        for (int e = 0; e < NJ; ++e) {
            jobs[(size_t) e].blob = blobs[(size_t) e].data();
            jobs[(size_t) e].act = &act;
            jobs[(size_t) e].out = out.data() + (size_t) e * cpu::H;
            jobs[(size_t) e].slot = e;
        }
    }

    cpu::ExpertPool pool((std::max)(1, workers));
    std::printf("pool_stress: %d workers + host, %.0f s, %s, parked workers %s\n", pool.workers(), seconds,
                native ? "run_split_multi_native" : "run and run_split", pool.yields() ? "yield" : "spin");
    if (native) pool.run_split_multi_native(fmt, mjobs.data(), NJ);
    else pool.run(jobs.data(), NJ);
    const std::vector<float> ref = out, ref2 = out2;

    // the idle work: every worker index arrives at a barrier, then spins until stopped
    struct Idle {
        std::atomic<int> active{0}, arrive{0};
        std::atomic<long long> runs{0};
    } idle;
    auto idle_fn = [](void* ctx, int w, int W, const std::atomic<bool>& stop) {
        Idle& d = *(Idle*) ctx;
        (void) w;
        d.active.fetch_add(1);
        d.runs.fetch_add(1);
        d.arrive.fetch_add(1);
        while (d.arrive.load() < W && !stop.load()) {}
        while (!stop.load()) {}
        d.active.fetch_sub(1);
    };
    long long overlaps = 0, wrong = 0;

    // the watchdog: a batch that does not return within 10 s is a hang
    std::atomic<long long> beat{0};
    std::atomic<bool> finished{false};
    std::thread dog([&] {
        long long last = -1;
        auto since = std::chrono::steady_clock::now();
        while (!finished.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const long long b = beat.load();
            const auto now = std::chrono::steady_clock::now();
            if (b != last) { last = b; since = now; continue; }
            if (now - since > std::chrono::seconds(10)) {
                std::printf("\npool_stress: HANG - batch %lld has not returned for 10 s\n", b);
                std::fflush(stdout);
                std::_Exit(3);
            }
        }
    });

    std::mt19937 rng(29);
    const float SENTINEL = -1.2345e33f;
    long long batches = 0, sleeps = 0, missed = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::duration<double>(seconds)) {
        // now and then a gap long enough for every worker to go to sleep (the pool sleeps after 20 ms)
        if (rng() % 64 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(21 + (int) (rng() % 10)));
            ++sleeps;
        } else if (rng() % 4 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(rng() % 300));
        }
        // the active count moves between batches, over its whole range
        if (rng() % 8 == 0) pool.set_active_workers(1 + (int) (rng() % (uint32_t) pool.workers()));
        if (rng() % 2) {
            idle.arrive.store(0);
            pool.idle_start(idle_fn, &idle);
            if (rng() % 3 == 0) std::this_thread::sleep_for(std::chrono::microseconds(rng() % 200));
            if (rng() % 4 == 0) {
                pool.idle_stop();
                if (idle.active.load() != 0) ++overlaps;
            }
        }
        const int n = (int) (rng() % (NJ + 1));
        std::fill(out.begin(), out.end(), SENTINEL);
        std::fill(out2.begin(), out2.end(), SENTINEL);
        if (native) pool.run_split_multi_native(fmt, mjobs.data(), n);
        else if (rng() % 2) pool.run(jobs.data(), n);
        else pool.run_split(jobs.data(), n);
        if (n > 0 && idle.active.load() != 0) ++overlaps;
        for (int e = 0; e < n; ++e) {
            const size_t o = (size_t) e * cpu::H;
            if (out[o] == SENTINEL || out[o + cpu::H - 1] == SENTINEL) ++missed;
            else if (std::memcmp(out.data() + o, ref.data() + o, cpu::H * sizeof(float)) != 0) ++wrong;
            if (!native || (e & 1) == 0) continue;
            if (out2[o] == SENTINEL || out2[o + cpu::H - 1] == SENTINEL) ++missed;
            else if (std::memcmp(out2.data() + o, ref2.data() + o, cpu::H * sizeof(float)) != 0) ++wrong;
        }
        beat.fetch_add(1);
        ++batches;
    }
    pool.idle_stop();
    pool.set_active_workers(pool.workers());
    if (idle.active.load() != 0) ++overlaps;
    finished.store(true);
    dog.join();
    std::printf("pool_stress: %lld batches (%lld after the workers slept), %lld jobs not run, %lld outputs not equal "
                "to the first run's, %lld idle jobs run, %lld still running after a batch or a stop\n",
                batches, sleeps, missed, wrong, idle.runs.load(), overlaps);
    if (missed || wrong || overlaps) {
        std::printf("pool_stress: FAILED\n");
        return 1;
    }
    std::printf("pool_stress OK\n");
    return 0;
}
