// prefill_layout_test - the prompt path's buffer layout against its sizing, without allocating device memory.
// `Prefill::bytes_needed` counts every buffer `init` carves, and the attention half's, the MoE half's and the
// hyper-connection rows' scratch share one region sized as the largest of them; `Prefill::layout_fits` carves the
// same sequence over a region of exactly that count and fails when a buffer or an aliased set overruns it.  Swept over
// chunk sizes around every threshold (the FP4 minimum, the streamed walk, the largest chunk), with and without an
// expert source, a resident and a streamed KV, and the ring depths.  The expert layout is process-wide, so the layout
// is the argument: nvfp4 (every layer NVFP4 with a tail: the FP4 path), mixed (NVFP4 and K-quant layers),
// kquant (Q4_K / Q5_1) and q2 (no native layout, as in the IQ and Q2_0 packs).  Needs a CUDA device for the MMQ plan (exit 77 without).
#include "basalt/core/layout.hpp"
#include "basalt/core/session.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/prefill/prefill.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {
namespace cpu = basalt::kernels::cpu;
using basalt::prefill::Prefill;

constexpr int kNvfp4 = 40, kQ4K = 12, kQ51 = 7;

bool load_layout(const std::string& mode, int64_t layers, std::string& err) {
    if (mode == "q2") return true;   // no layout: the IQ and Q2_0 packs'
    std::ostringstream t;
    t << "# experts v6: test\n";
    uint64_t off = 0;
    for (int64_t l = 0; l < layers; ++l) {
        int gt = kNvfp4, dt = kNvfp4;
        if (mode == "kquant") { gt = kQ4K; dt = kQ51; }
        if (mode == "mixed" && l % 2 == 1) { gt = kQ4K; dt = kQ51; }
        cpu::NativeFmt f;
        if (!cpu::native_fmt(gt, dt, 2560, 640, f, err)) return false;
        t << l << ' ' << gt << ' ' << dt << ' ' << off << ' ' << f.bytes << '\n';
        off += f.bytes * 512;
    }
    return cpu::expert_layout_load_text(t.str(), layers, 512, err);
}
}  // namespace

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    const std::string mode = argc > 1 ? argv[1] : "nvfp4";
    basalt::core::ModelGeometry g;
    std::string err;
    if (!load_layout(mode, g.n_layers, err)) {
        std::printf("layout %s: %s\n", mode.c_str(), err.c_str());
        return 1;
    }
    int fails = 0, cases = 0;
    uint64_t worst_headroom = ~0ull;
    for (int kv_mode : {0, 1}) {
        basalt::core::SessionState ss;
        basalt::core::QsaState qs;
        qs.max_cells = 300000;
        qs.kv_mode = kv_mode;
        qs.kv_int8 = true;
        qs.n_pages = 1024;
        ss.qsa_states = &qs;
        for (int64_t stream_min : {int64_t{1024}, int64_t{6144}}) {
            Prefill::set_stream_all_min(stream_min);
            for (int ring : {9, 40, 256, 384}) {
                Prefill::set_ring_depth(ring);
                for (int64_t T : {1, 63, 256, 1023, 1024, 1025, 2048, 4096, 6143, 6144, 8192, 16384, 16385, 32768}) {
                    for (bool src : {true, false}) {
                        uint64_t used = 0, cap = 0;
                        const bool ok = Prefill::layout_fits(g, ss, T, src, &used, &cap);
                        ++cases;
                        if (cap >= used) worst_headroom = std::min<uint64_t>(worst_headroom, cap - used);
                        if (!ok) {
                            ++fails;
                            std::printf("  FAIL layout=%s kv_mode=%d stream_min=%lld ring=%d T=%lld src=%d: used %llu of %llu\n",
                                        mode.c_str(), kv_mode, (long long) stream_min, ring, (long long) T, (int) src,
                                        (unsigned long long) used, (unsigned long long) cap);
                        }
                    }
                }
            }
        }
        ss.qsa_states = nullptr;
    }
    std::printf("layout %s: %d cases, %d failed, least headroom %llu B\n", mode.c_str(), cases, fails,
                (unsigned long long) (worst_headroom == ~0ull ? 0 : worst_headroom));
    return fails ? 1 : 0;
}
