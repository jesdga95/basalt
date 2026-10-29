// tests/kernels/cpu/pool_test.cpp - test for the expert pool.
//
// The pool's correctness claims are small and specific, so they are checked directly rather than through a
// timing number:
//
//   1. THE SAME ANSWER AS SERIAL.  Every job's output must be bit-identical to running that expert serially.
//      Each worker owns its own `ExpertScratch`, and the jobs share one read-only activation, so there is no
//      legitimate source of difference - "close enough" here would be hiding a data race.
//   2. EVERY JOB RUNS EXACTLY ONCE.  Claiming is `head.fetch_add`, and an off-by-one in the bound is the
//      classic pool bug: it either drops a job or runs one twice.  Checked with a sentinel-filled output
//      buffer, so a dropped job is visible as an untouched slot rather than as a slightly wrong number.
//   3. REPEATED BATCHES.  A pool that works once and hangs or corrupts on the second `run()` is the failure
//      mode the park protocol exists to prevent, so `run()` is called many times in a row.
//   4. A BATCH BIGGER AND SMALLER THAN THE WORKER COUNT, because `n < workers` leaves most workers claiming
//      nothing and `n > workers` is the real case (10 experts, 5 workers).
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/platform/file.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {

/// A deterministic stand-in for the blob at an index: random 2-bit codes and fp16 scales in the range real packs use.
/// Distinct indices give distinct data, which is what the structural checks need.
void synth_blob(long long index, std::vector<uint8_t>& out) {
    out.assign(cpu::BLOB, 0);
    std::mt19937 rng((uint32_t) (index * 2654435761ull + 17));
    for (size_t i = 0; i < cpu::O_GU_SCALES; ++i) out[i] = (uint8_t) rng();
    for (size_t i = cpu::O_GU_SCALES; i < cpu::BLOB; i += 2) {
        const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);
        std::memcpy(out.data() + i, &h, 2);
    }
}

bool read_blob(const char* path, long long index, std::vector<uint8_t>& out) {
    if (path == nullptr) { synth_blob(index, out); return true; }
    out.assign(cpu::BLOB, 0);
    std::FILE* f = basalt::platform::open_file_utf8(path, "rb");
    if (!f) return false;
    // The blob a layer index points at can sit past 4 GiB, so the seek has to be a 64-bit one.
    if (basalt::platform::fseek64(f, index * (long long) cpu::BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    const char* path = nullptr;   // null: a synthetic fixture; --file reads real expert blobs
    long long layer = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--file" && i + 1 < argc) path = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoll(argv[++i]);
        else { std::fprintf(stderr, "usage: pool_test [--selftest] [--file experts.bin] [--layer N]\n"); return 2; }
    }

    const cpu::CpuFeatures feat = cpu::cpu_features();
    if (!feat.usable()) {
        std::printf("  CPU lacks %s - the VNNI path cannot run here; pool test SKIPPED, not passed.\n",
                    feat.reason());
        std::printf("\npool: 0 failures, 1 SKIPPED\n");
        return 0;
    }

    int bad = 0;

    // ---- ten experts off one layer, which is exactly what a token uses
    const int NEXP = 10;
    std::vector<std::vector<uint8_t>> blobs((size_t) NEXP);
    for (int e = 0; e < NEXP; ++e)
        if (!read_blob(path, layer * 512 + e, blobs[(size_t) e])) {
            std::fprintf(stderr, "cannot read expert %d of layer %lld from %s\n", e, layer, path ? path : "the fixture");
            return 2;
        }

    std::mt19937 rng(99);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x(cpu::H);
    for (auto& v : x) v = gauss(rng);

    cpu::ActQ act;
    cpu::act_quant_q8_1(x.data(), cpu::H, act);

    // ---- serial reference
    std::vector<float> ref((size_t) NEXP * cpu::H);
    {
        cpu::ExpertScratch ws;
        for (int e = 0; e < NEXP; ++e)
            cpu::s2_expert_vnni_q(blobs[(size_t) e].data(), act, ref.data() + (size_t) e * cpu::H, ws);
    }

    // ---- the pool
    const std::vector<int> cores = cpu::physical_cores(true);
    const int hw = (int) std::thread::hardware_concurrency();
    std::printf("  %-44s %d logical, %d physical (skipping the first)\n", "cores the pool will use",
                hw, (int) cores.size());

    cpu::ExpertPool pool((std::max)(1, (int) cores.size()));
    std::printf("  %-44s %d\n", "workers", pool.workers());
    // A parked worker yields only when the pool's threads, the host's included, outnumber the physical cores.
    {
        const int c = (int) cores.size();
        cpu::ExpertPool over(c + 1);
        const bool ok = pool.yields() == (pool.workers() > c) && over.yields();
        std::printf("  %-44s %s (%d workers: %s, %d workers: %s)\n", "parked workers yield only when oversubscribed",
                    ok ? "yes" : "*** NO ***", pool.workers(), pool.yields() ? "yield" : "spin", over.workers(),
                    over.yields() ? "yield" : "spin");
        if (!ok) ++bad;
    }

    std::vector<cpu::ExpertJob> jobs((size_t) NEXP);
    std::vector<float> got((size_t) NEXP * cpu::H);
    std::vector<float> weights((size_t) NEXP);
    for (int e = 0; e < NEXP; ++e) weights[(size_t) e] = 0.1f * (float) (e + 1);

    // ---- 1 + 2: same answer as serial, and every job exactly once.
    // The output buffer starts at a sentinel no real result can equal, so a job that never runs shows up as
    // an untouched slot instead of as a plausible number.
    const float SENTINEL = -1.2345e33f;
    for (int e = 0; e < NEXP; ++e) {
        jobs[(size_t) e].blob = blobs[(size_t) e].data();
        jobs[(size_t) e].act = &act;
        jobs[(size_t) e].out = got.data() + (size_t) e * cpu::H;
        jobs[(size_t) e].weight = weights[(size_t) e];
        jobs[(size_t) e].slot = e;
        std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
    }
    pool.run(jobs.data(), NEXP);

    long long not_run = 0, diff = 0;
    float worst = 0.f;
    for (int e = 0; e < NEXP; ++e)
        for (int i = 0; i < cpu::H; ++i) {
            const float a = ref[(size_t) e * cpu::H + i], b = got[(size_t) e * cpu::H + i];
            if (b == SENTINEL) { ++not_run; continue; }
            if (std::memcmp(&a, &b, 4) != 0) ++diff;
            worst = std::fmax(worst, std::fabs(a - b));
        }
    std::printf("  %-44s %s (%lld of %d outputs untouched)\n", "every job ran exactly once",
                not_run ? "*** NO ***" : "yes", not_run, NEXP * cpu::H);
    if (not_run) ++bad;
    // BIT-IDENTICAL, not "close".  Each worker has a private scratch and the activation is shared read-only,
    // so any difference at all is a race or a scratch collision - and a tolerance would hide exactly that.
    std::printf("  %-44s %s (%lld of %d differ, worst |d| %.3e)\n", "identical to the serial run",
                diff ? "*** NO ***" : "yes", diff, NEXP * cpu::H, (double) worst);
    if (diff) ++bad;

    // ---- 3: repeated batches.  A park protocol that works once and corrupts on the second call is the
    //        exact failure this loop is here to catch.
    {
        int repeats_bad = 0;
        for (int r = 0; r < 200; ++r) {
            for (int e = 0; e < NEXP; ++e) std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
            pool.run(jobs.data(), NEXP);
            for (int e = 0; e < NEXP && !repeats_bad; ++e)
                for (int i = 0; i < cpu::H; ++i)
                    if (std::memcmp(&ref[(size_t) e * cpu::H + i], &got[(size_t) e * cpu::H + i], 4) != 0) {
                        ++repeats_bad;
                        break;
                    }
        }
        std::printf("  %-44s %s (200 consecutive batches)\n", "repeated batches stay correct",
                    repeats_bad ? "*** NO ***" : "yes");
        if (repeats_bad) ++bad;
    }

    // ---- 4: batch sizes either side of the worker count
    {
        int size_bad = 0;
        for (int n : {1, 2, pool.workers() - 1 > 0 ? pool.workers() - 1 : 1, pool.workers(),
                      pool.workers() + 1, NEXP}) {
            if (n < 1 || n > NEXP) continue;
            for (int e = 0; e < n; ++e) std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
            pool.run(jobs.data(), n);
            for (int e = 0; e < n && !size_bad; ++e)
                for (int i = 0; i < cpu::H; ++i)
                    if (std::memcmp(&ref[(size_t) e * cpu::H + i], &got[(size_t) e * cpu::H + i], 4) != 0) {
                        ++size_bad;
                        break;
                    }
        }
        std::printf("  %-44s %s (1, w-1, w, w+1, 10)\n", "batch sizes around the worker count",
                    size_bad ? "*** NO ***" : "yes");
        if (size_bad) ++bad;
    }

    // ---- 5: the active worker count changes live.  1, 2, half, all, and back, with batches in between; every output
    //        is bit-identical to the serial run at every count, for whole-expert jobs and for the row-split ones.
    {
        const int W = pool.workers();
        std::vector<int> counts = {1, 2, (std::max)(1, W / 2), W, 1, W, (std::max)(1, W / 2), 2, W};
        int count_bad = 0, wrong_count = 0;
        for (int k : counts) {
            const int in_effect = pool.set_active_workers(k);
            if (in_effect != (std::min)(k, W) || pool.active_workers() != in_effect) ++wrong_count;
            for (int rep = 0; rep < 20; ++rep) {
                for (int mode = 0; mode < 2; ++mode) {
                    for (int e = 0; e < NEXP; ++e) std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
                    if (mode == 0) pool.run(jobs.data(), NEXP);
                    else pool.run_split(jobs.data(), NEXP);
                    if (std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) != 0) ++count_bad;
                }
            }
        }
        std::printf("  %-44s %s (counts 1, 2, half, all and back; run and run_split)\n",
                    "bitwise at every active worker count", (count_bad || wrong_count) ? "*** NO ***" : "yes");
        if (count_bad || wrong_count) ++bad;
        pool.set_active_workers(W);
    }

    // ---- 6: one NVFP4 expert (real geometry) at every count, against a plain serial CPU reference
    {
        cpu::NativeFmt f;
        std::string ferr;
        constexpr int kNvfp4 = 40, kBlock = 36;
        if (!cpu::native_fmt(kNvfp4, kNvfp4, cpu::H, cpu::FF, f, ferr) || f.gu_row % kBlock != 0) {
            std::printf("  %-44s SKIPPED (%s)\n", "NVFP4 expert at every active worker count", ferr.c_str());
        } else {
            std::vector<uint8_t> nb(f.bytes);
            std::mt19937_64 r64(7);
            for (size_t i = 0; i + 8 <= nb.size(); i += 8) { const uint64_t v = r64(); std::memcpy(nb.data() + i, &v, 8); }
            for (size_t b = 0; b + kBlock <= nb.size() - cpu::kNvfp4Tail; b += kBlock)
                for (int k = 0; k < 4; ++k) nb[b + (size_t) k] = (uint8_t) (0x30 + (nb[b + (size_t) k] & 7));
            const float tail[4] = {1.f, 1.f, 1.f, 0.f};
            std::memcpy(nb.data() + f.bytes - sizeof tail, tail, sizeof tail);
            std::vector<uint8_t> nact(cpu::kNativeActBytes), hq(cpu::kNativeHBytes);
            cpu::native_quant_act(f, x.data(), nact.data());
            std::vector<float> ff(cpu::FF), nref(cpu::H), nout(cpu::H);
            {
                const void* a[1] = {nact.data()};
                float* fp[1] = {ff.data()};
                cpu::native_gu_rows(f, nb.data(), a, 1, fp, 0, cpu::FF);
                cpu::native_quant_h(f, ff.data(), hq.data());
                const void* h[1] = {hq.data()};
                float* op[1] = {nref.data()};
                cpu::native_down_rows(f, nb.data(), h, 1, op, 0, cpu::H);
            }
            cpu::ExpertJobMulti mj;
            mj.blob = nb.data();
            mj.nt = 1;
            mj.nact[0] = nact.data();
            mj.out[0] = nout.data();
            const int W = pool.workers();
            int nbad = 0;
            for (int k : {1, 2, (std::max)(1, W / 2), W, 1, W}) {
                pool.set_active_workers(k);
                for (int rep = 0; rep < 10; ++rep) {
                    std::fill(nout.begin(), nout.end(), SENTINEL);
                    pool.run_split_multi_native(f, &mj, 1);
                    if (std::memcmp(nref.data(), nout.data(), nout.size() * sizeof(float)) != 0) ++nbad;
                }
            }
            std::printf("  %-44s %s\n", "NVFP4 expert, every count, vs serial CPU", nbad ? "*** NO ***" : "yes");
            if (nbad) ++bad;

            // microbench: one expert, best-of median per count; each pick is a different expert far past the caches
            constexpr int NB = 192;
            const size_t stride = (f.bytes + 4095) & ~(size_t) 4095;
            std::vector<uint8_t> mem(stride * NB + 4096);
            uint8_t* base = (uint8_t*) (((uintptr_t) mem.data() + 4095) & ~(uintptr_t) 4095);
            for (int e = 0; e < NB; ++e) std::memcpy(base + stride * (size_t) e, nb.data(), f.bytes);
            std::vector<int> cs;
            for (int k = 1; k < W; k *= 2) cs.push_back(k);
            cs.push_back(W);
            std::printf("\n  NVFP4 expert (%.2f MB) through the pool, cold, median of 40:\n", (double) f.bytes / 1e6);
            for (int k : cs) {
                pool.set_active_workers(k);
                std::vector<double> us;
                for (int it = 0; it < 40; ++it) {
                    mj.blob = base + stride * (size_t) (it % NB);
                    const double t0 = now_ms();
                    pool.run_split_multi_native(f, &mj, 1);
                    us.push_back((now_ms() - t0) * 1e3);
                }
                std::sort(us.begin(), us.end());
                std::printf("    %3d workers + host: %8.1f us   (%.1f GB/s)\n", k, us[us.size() / 2],
                            (double) f.bytes / (us[us.size() / 2] * 1e3));
            }
            pool.set_active_workers(W);
        }
    }

    // ---- and the number that matters: c for one layer, 10 experts over 48 layers.
    //
    // BEST OF N, not a mean.  A mean over a contended machine measures the contention; a best-of measures the
    // instrument, and every other timing tool in this project takes the best.  The spread is printed
    // so a contended run is visible rather than being read as a regression.
    {
        const int REPS = 20;
        std::vector<double> t((size_t) REPS, 0.0);
        for (int r = 0; r < REPS; ++r) {
            const double t0 = now_ms();
            pool.run(jobs.data(), NEXP);
            t[(size_t) r] = now_ms() - t0;
        }
        std::vector<double> sorted = t;
        std::sort(sorted.begin(), sorted.end());
        const double best = sorted.front(), median = sorted[sorted.size() / 2], worst = sorted.back();
        const double gbs = (double) NEXP * cpu::BLOB / (best * 1e-3) / 1e9;
        std::printf("\n  %-44s %7.3f ms   (%.1f GB/s)\n", "10 experts, one layer (best of 20)",
                    best, gbs);
        std::printf("  %-44s %7.3f / %7.3f ms   (spread %.2fx)\n", "median / worst", median, worst,
                    median / (best > 0 ? best : 1));
        std::printf("  %-44s %7.3f ms   -> %.1f tok/s for the full 48 layers\n",
                    "extrapolated to 48 layers", best * 48, 1000.0 / (best * 48));
        std::printf("  this pool uses %d workers (core 0 is left to the host loop), so %.1f GB/s x 6/%d = %.1f GB/s\n"
                    "  is the 6-core comparison.\n",
                    pool.workers(), gbs, pool.workers(), gbs * 6.0 / pool.workers());
        if (median / (best > 0 ? best : 1) > 1.5)
            std::printf("  *** the spread is over 1.5x: this machine was CONTENDED and the median is not a\n"
                        "      property of the pool.  Re-run on a quiet machine before quoting it. ***\n");
    }

    std::printf("\npool: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("pool_test OK\n");
    return 0;
}
