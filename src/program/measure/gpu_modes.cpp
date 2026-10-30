// basalt/program/measure/gpu_modes.cpp - see gpu_modes.hpp.
//
// The three replay modes and the per-stage event names were lifted out of `main()` unchanged; the aliases at the top
// of each function are what the surrounding function used to provide, so the moved code reads as it did.
#include "gpu_modes.hpp"

#include "basalt/core/layer.hpp"      // Doorbell, stage_timing_*, is_qsa_layer
#include "basalt/core/session.hpp"    // session_replay*, session_graphs_free
#include "basalt/log.hpp"
#include "basalt/program/options.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace basalt::program::measure {
namespace {

using Clock = std::chrono::steady_clock;

/// The `basalt error:` line a start-ending failure prints, with the exit code `main` returned for it.
int fail(const std::string& text) {
    basalt::log::error("basalt", "%s", text.c_str());
    return 1;
}

}  // namespace

int arm_stage_timing(const Options& o) {
    if (o.stage_timing) {
        if (!o.no_capture) {
            basalt::log::error("basalt generate", "--stage-timing records CUDA events inside the layer path, "
                                 "and an event record inside a stream capture is silently dropped. Pass "
                                 "--no-capture as well.");
            return 2;
        }
        if (!basalt::core::stage_timing_enable()) {
            basalt::log::error("basalt generate", "stage_timing_enable failed");
            return 1;
        }
        basalt::core::stage_timing_name(0, "gr_read (attn)");
        basalt::core::stage_timing_name(1, "attention block");
        basalt::core::stage_timing_name(2, "gr_write (attn)");
        basalt::core::stage_timing_name(3, "gr_read (ffn)");
        basalt::core::stage_timing_name(4, "moe_route");
        basalt::core::stage_timing_name(5, "moe_finish");
        basalt::core::stage_timing_name(6, "gr_write (ffn)");
        // The GDN block's internals: a single "attention block" number cannot say which of its many small
        // latency-bound launches costs anything.
        basalt::core::stage_timing_name(8, "  gdn: quantize x");
        basalt::core::stage_timing_name(9, "  gdn: qkv gemv");
        basalt::core::stage_timing_name(10, "  gdn: conv+silu");
        basalt::core::stage_timing_name(11, "  gdn: l2 norms");
        basalt::core::stage_timing_name(12, "  gdn: alpha/beta/gate");
        basalt::core::stage_timing_name(13, "  gdn: gdn_step");
        basalt::core::stage_timing_name(14, "  gdn: z + out_norm");
        basalt::core::stage_timing_name(15, "  gdn: out gemv");
    }
    return 0;
}

int run_gpu_modes(const GpuModeContext& c) {
    const Options& o = *c.o;
    const basalt::core::ModelGeometry& g = *c.g;
    basalt::core::SessionState& ss = *c.ss;
    basalt::core::SessionGraphs& gr = *c.gr;
    basalt::core::Doorbell& db = *c.db;
    void* const main_cs = c.stream;
    int* const d_next = (int*) c.d_next;
    const std::function<bool(void*)>& run_head = c.run_head;
    std::string& err = *c.err;

    // ---- THE PURE-GPU MEASUREMENT.  `session_replay` launches all 48 `pre` graphs back to back on one stream
    // with NO host work between them - no doorbell poll, no pool, no parts copy - so what it times is the GPU
    // executing the layer sequence and nothing else.  It had been declared, defined and never called since the
    // day it was written.
    //
    // **THIS IS THE MEASUREMENT THAT SAYS WHETHER THE ENGINE IS HOST-BOUND OR GPU-BOUND**, and the stage table
    // cannot answer it: those events measure the interval between two marks on a stream, which includes every
    // gap where the GPU sat idle waiting for the host to enqueue the next kernel.  In `--no-capture` those gaps
    // are the host's launch latency and they are proportional to the KERNEL COUNT rather than to any work, so
    // the no-capture stage shares are shares of kernel count - which is why the attention block, with the most
    // kernels, looks like 55% of the token there.
    // ---- THE PER-STAGE TABLE ON THE CAPTURED GRAPH.
    if (o.gpu_stages) {
        basalt::core::doorbell_reset(db);
        double mix = 0, ffn = 0, post = 0;
        if (!basalt::core::session_replay_stages(g, 0, 0, ss, gr, main_cs, mix, ffn, post, err)) {
            // no per-layer graphs are captured; the serve path times its window's stages
            basalt::log::error("basalt generate", "%s%s", err.c_str(),
                         " - there are no per-layer graphs to replay. The decode window's GPU "
                         "stages are printed per request by the serve path instead: --profile "
                         "--decode-timing with --serve");
            return 1;
        }
        const int reps = 20;
        double t_mix = 0, t_ffn = 0, t_post = 0;
        for (int r = 0; r < reps; ++r) {
            if (!basalt::core::session_replay_stages(g, 0, 0, ss, gr, main_cs, mix, ffn, post, err)) {
                return fail(err);
            }
            t_mix += mix;
            t_ffn += ffn;
            t_post += post;
        }
        const double tot = t_mix + t_ffn + t_post;
        std::printf("\nper-stage GPU time on the CAPTURED graph, one token over %lld layers\n",
                    (long long) g.n_layers);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "mixer (gr_read+attn+gr_write)",
                    t_mix / reps, t_mix / reps / (double) g.n_layers, 100.0 * t_mix / tot);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "ffn front + router",
                    t_ffn / reps, t_ffn / reps / (double) g.n_layers, 100.0 * t_ffn / tot);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "post (moe_finish+gr_write)",
                    t_post / reps, t_post / reps / (double) g.n_layers, 100.0 * t_post / tot);
        std::printf("  %-22s %9.3f ms/token\n", "sum of the three", tot / reps);

        // ---- AND THE MIXER BY LAYER KIND, because 36 of the 48 layers are GDN and 12 are QSA and a total cannot
        // separate them.
        {
            // ACCUMULATED OVER `reps`, NOT MEASURED ONCE AND THEN DIVIDED: the per-layer replay is timed over all
            // repetitions, so the per-kind figures add up to the mixer total.
            std::vector<double> acc((size_t) g.n_layers, 0.0), per;
            double f2 = 0, p2 = 0;
            for (int r = 0; r < reps; ++r) {
                if (!basalt::core::session_replay_stages_per_layer(g, 0, 0, ss, gr, main_cs, per, f2, p2, err)) {
                    return fail(err);
                }
                for (int64_t l = 0; l < g.n_layers; ++l) acc[(size_t) l] += per[(size_t) l];
            }
            double gdn = 0, qsa = 0, worst = 0;
            int64_t ng = 0, nq = 0, worst_l = 0;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                const double v = acc[(size_t) l] / reps;
                if (basalt::core::is_qsa_layer(g, l)) { qsa += v; ++nq; }
                else { gdn += v; ++ng; }
                if (v > worst) { worst = v; worst_l = l; }
            }
            std::printf("\n  the mixer by layer kind, averaged over %d runs\n", reps);
            std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  (%lld layers)\n", "GDN layers",
                        gdn, ng ? gdn / (double) ng : 0.0, (long long) ng);
            std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  (%lld layers)\n", "QSA layers",
                        qsa, nq ? qsa / (double) nq : 0.0, (long long) nq);
            std::printf("  %-22s layer %lld at %.3f ms\n", "worst mixer layer", (long long) worst_l, worst);
            std::printf("  %-22s %9.3f ms/token (must equal the mixer above)\n", "GDN + QSA", gdn + qsa);
        }

        // ================================ THE FIVE STAGES SEPARATELY ================================
        //
        // Prefixes 1..5 are replayed per layer with the residual restored between them, and consecutive
        // differences are the per-stage times.  **THE CHECK IS THAT THE FIVE SUM TO THE THREE-GRAPH TOTAL** -
        // an independent restatement, and the only reason to believe a table built out of
        // differences.
        {
            std::vector<double> acc5(5, 0.0), per, s5;
            for (int r = 0; r < reps; ++r) {
                if (!basalt::core::session_replay_stage_prefixes(g, 0, 0, ss, gr, main_cs, s5, per, err)) {
                    return fail(err);
                }
                for (int k = 0; k < 5; ++k) acc5[(size_t) k] += s5[(size_t) k];
            }
            static const char* sn[5] = {"0 gr_read (attn)", "1 attention", "2 gr_write (attn)",
                                        "3 gr_read (ffn)", "4 moe_route (router)"};
            const char* kind[5] = {"GR", "ATTN", "GR", "GR", "ROUTER"};
            double tot5 = 0, gr_ms = 0;
            std::printf("\n  the five stages separately, by differencing prefixes\n");
            std::printf("  %-24s %-8s %10s %10s %8s\n", "stage", "kind", "ms/token", "ms/layer", "share");
            for (int k = 0; k < 5; ++k) {
                const double v = acc5[(size_t) k] / reps;
                tot5 += v;
                if (k != 1 && k != 4) gr_ms += v;
                std::printf("  %-24s %-8s %10.3f %10.4f %7.1f%%\n", sn[k], kind[k], v,
                            v / (double) g.n_layers, 0.0);
            }
            for (int k = 0; k < 5; ++k) {
                const double v = acc5[(size_t) k] / reps;
                (void) v;
            }
            std::printf("  %-24s %-8s %10.3f\n", "sum of the five", "", tot5);
            std::printf("  %-24s %-8s %10.3f\n",
                        "GR passes (0,2,3)", "GR", gr_ms);
            std::printf("\n  the three-graph total above was %.3f ms/token; the five must account for it.\n",
                        tot / reps);

            // ================================ THE SAME TABLE, A DIFFERENT WAY ================================
            //
            // The differencing table above mixes five graphs per layer, so stage 4's interval carries the launch
            // of the FULL five-stage graph while stage 3's carries a four-stage one.  This sweep launches ONE
            // graph type per layer and nothing else, so that bias cannot exist.  **If the two disagree, the
            // difference IS the bias and this one is right** - and stage 4 is the router, so it is exactly the
            // number that must not be wrong.
            {
                double sweep[6] = {0, 0, 0, 0, 0, 0};
                for (int k = 1; k <= 5; ++k) {
                    double acc = 0, one = 0;
                    for (int r = 0; r < reps; ++r) {
                        if (!basalt::core::session_replay_stage_sweep(g, 0, 0, ss, gr, main_cs, k, one, err)) {
                            return fail(err);
                        }
                        acc += one;
                    }
                    sweep[k] = acc / reps;
                }
                std::printf("\n  the same five stages, by SWEEPING each prefix back to back (no graph switching)\n");
                std::printf("  %-24s %10s %10s %12s\n", "stage", "prefix sweep", "difference", "bias");
                static const char* sn2[5] = {"0 gr_read (attn)", "1 attention", "2 gr_write (attn)",
                                             "3 gr_read (ffn)", "4 moe_route (router)"};
                for (int k = 0; k < 5; ++k) {
                    const double sw = sweep[k + 1] - sweep[k];
                    const double df = acc5[(size_t) k] / reps;
                    std::printf("  %-24s %10.3f %10.3f %11.1f%%\n", sn2[k], sw, df,
                                sw != 0.0 ? 100.0 * (df / sw - 1.0) : 0.0);
                }
                std::printf("  %-24s %10.3f   (full pre, 48 layers)\n", "prefix 5 total", sweep[5]);
            }
        }
        std::printf("\n  compare `--gpu-only-full`, which replays the same work as TWO graphs per layer.  The\n");
        std::printf("  three sum slightly above it because each launch carries the driver's gap.\n");
        basalt::core::session_graphs_free(gr);
        basalt::core::doorbell_free(db);
        cudaFree(d_next);
        return 0;
    }

    if (o.graph_only) {
        basalt::core::doorbell_reset(db);
        // one warm pass so the first launch does not pay for page mapping
        if (!basalt::core::session_replay(g, 0, 0, ss, gr, main_cs, err)) {
            basalt::log::error("basalt generate", "session_replay warm: %s", err.c_str());
            return 1;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            basalt::log::error("basalt generate", "session_replay warm faulted");
            return 1;
        }
        const int reps = 20;
        const Clock::time_point t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            if (!basalt::core::session_replay(g, 0, 0, ss, gr, main_cs, err)) {
                basalt::log::error("basalt generate", "session_replay: %s", err.c_str());
                return 1;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            basalt::log::error("basalt generate", "session_replay faulted: %s", cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / (double) reps;
        std::printf("pre graphs only   %8.2f ms per token over %lld layers  ->  %.2f tok/s of GPU work\n", ms,
                    (long long) g.n_layers, ms > 0 ? 1000.0 / ms : 0.0);

        return 0;
    }

    // ---- THE TRUE PER-TOKEN GPU FLOOR.
    //
    // `--graph-only` above launches ONLY `gr.execs[l]`, the `pre` graphs.  It omits the 48 `post` graphs - the shared
    // expert, the combine and the second `gr_write` - and the LM head.  This flag measures the whole GPU work of a
    // token: no pool runs, so `parts` keeps whatever the buffer holds and the timing is GPU work alone.
    if (o.gpu_only_full) {
        basalt::core::doorbell_reset(db);
        const int reps = 20;
        double ms_layers = 0, ms_head = 0;
        for (int r = -1; r < reps; ++r) {          // r == -1 is the warm pass, not counted
            const Clock::time_point t0 = Clock::now();
            if (!basalt::core::session_replay_full(g, 0, 0, ss, gr, main_cs, err)) {
                basalt::log::error("basalt generate", "session_replay_full: %s", err.c_str());
                return 1;
            }
            // The sync is INSIDE the interval on purpose: it is the wait for the GPU, so t1 - t0 is GPU time.
            if (cudaStreamSynchronize((cudaStream_t) main_cs) != cudaSuccess) {
                basalt::log::error("basalt generate", "gpu-only-full layers faulted: %s",
                             cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            const Clock::time_point t1 = Clock::now();
            if (!run_head(main_cs)) {
                basalt::log::error("basalt generate", "gpu-only-full lm_head: %s", err.c_str());
                return 1;
            }
            if (cudaStreamSynchronize((cudaStream_t) main_cs) != cudaSuccess) {
                basalt::log::error("basalt generate", "gpu-only-full head faulted: %s",
                             cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            const Clock::time_point t2 = Clock::now();
            if (r < 0) continue;
            ms_layers += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ms_head += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        ms_layers /= (double) reps;
        ms_head /= (double) reps;
        const double ms = ms_layers + ms_head;
        std::printf("GPU floor pre+post+head %7.2f ms per token  ->  %.2f tok/s of GPU work\n", ms,
                    ms > 0 ? 1000.0 / ms : 0.0);
        std::printf("                  %8.3f ms layers (%lld x pre+post)\n", ms_layers, (long long) g.n_layers);
        std::printf("                  %8.3f ms per layer\n", ms_layers / (double) g.n_layers);
        std::printf("                  %8.3f ms LM head\n", ms_head);
        return 0;
    }

    return kNotRequested;
}

}  // namespace basalt::program::measure
