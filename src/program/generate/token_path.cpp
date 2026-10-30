#include "context.hpp"

namespace basalt::program::generate {

int token_path(Context& context) {
    auto& o = *context.o;
    auto& multi_gpu = *context.multi_gpu;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& K = *context.K;
    auto& profile = *context.profile;
    auto& wt = *context.wt;
    auto& ss = *context.ss;
    auto& main_cs = *context.main_cs;
    auto& db = *context.db;
    auto& d_parts = *context.d_parts;
    auto& stages = *context.stages;
    auto& stage_of = context.stage_of;
    auto& native_head = *context.native_head;
    auto& d_logits = *context.d_logits;
    auto& srcp = *context.srcp;
    auto& xcache = *context.xcache;
    auto& remote_experts = *context.remote_experts;
    auto& remote_opt = *context.remote_opt;

    // ---- Multi-GPU: the second GPU's expert tier (filled after the RAM complement below, from it), and the further
    // cards' tiers beside it
    basalt::core::PeerExperts peer;
    std::vector<std::unique_ptr<basalt::core::PeerExperts>> xpeers;
    if (o.peer_device >= 1 && (profile.empty() || srcp == nullptr || o.expert_cache <= 0)) {
        basalt::log::error("basalt generate", "a second GPU as expert tier needs an expert profile and the expert cache; run on one GPU (--gpu 0)");
        return 1;
    }

    if (remote_opt && !remote_opt->init(err)) { basalt::log::error("basalt generate", "%s", err.c_str()); return 1; }
    Drive drive;
    for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
        drive.d.remote[drive.d.remote_count++] = &remote_experts[(size_t) r];
    drive.d.src = srcp;
    drive.d.n_expert = g.n_expert;
    drive.d.jobs.resize((size_t) K);
    // ---- THE HIT PATH.  Every one of these is required for `hits_ready()`, which is all-or-nothing on
    // purpose: a half-configured hit path would compute some experts twice and others not at all, and a token
    // built on that is wrong rather than refused.
    void* hit_scratch = nullptr;
    int32_t* d_hit_slot = nullptr;
    int32_t* d_hit_dst = nullptr;
    uint8_t* d_hit_q8 = nullptr;
    float* d_hit_q8_scale = nullptr;   ///< the fp32 activation scales the CPU path also uses
    float* d_hit_out = nullptr;
    if (o.expert_cache > 0 && !o.no_pool) {
        const uint64_t sb = basalt::kernels::moe_hit_grouped_scratch_bytes(K, g.n_embd, basalt::kernels::cpu::FF);
        if (cudaMalloc(&hit_scratch, (size_t) sb) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_slot, (size_t) K * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_dst, (size_t) K * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_q8, (size_t) (g.n_embd / 32) * 34) != cudaSuccess ||
            // the fp32 activation scales.  Without this the GPU's hits use the block's fp16 `d`
            // while the CPU's misses use `ActQ::scale`, which is fp32 - a relative disagreement of up to 2^-11 on
            // every chunk that would change the tokens when the cache is enabled.
            cudaMalloc((void**) &d_hit_q8_scale, (size_t) (g.n_embd / 32) * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_out, (size_t) K * g.n_embd * 4) != cudaSuccess) {
            basalt::log::error("basalt generate", "the hit path could not allocate its device buffers");
            return 1;
        }
        drive.d.cache = &xcache;
        drive.d.cache_stream = main_cs;
        drive.d.cache_base = (const uint8_t*) xcache.device_slot(0);
        drive.d.cache_blob = (int64_t) basalt::kernels::cpu::expert_layout().max_blob;
        drive.d.cache_slot_off = xcache.slot_offsets();
        drive.d.hit_scratch = hit_scratch;
        drive.d.parts_out = d_parts;
        drive.d.hit_out = d_hit_out;
        drive.d.parts_elems = K * g.n_embd;
        drive.d.mixed = ss.block.mixed;
        drive.d.x_q8_0_hit = d_hit_q8;
        drive.d.x_q8_0_hit_scale = d_hit_q8_scale;
        drive.d.d_slot = d_hit_slot;
        drive.d.d_dst = d_hit_dst;
        drive.d.h_slot.resize((size_t) K);
        cudaEvent_t hit_done = nullptr;
        if (cudaEventCreate(&hit_done) != cudaSuccess) {
            basalt::log::error("basalt generate", "the hit path could not create its probe event");
            return 1;
        }
        drive.d.hit_done = (void*) hit_done;
        drive.d.h_dst.resize((size_t) K);
        basalt::log::info("basalt generate", "R4 hit path ON - resident experts are computed on the GPU");
    }
    // ---- the routing trace.  Only meaningful with the pool running, because the ids arrive through the doorbell that
    // the pool consumes - so `--no-pool` is refused rather than silently producing an empty file that would read as
    // "the router selected nothing".
    // ---- PER-STAGE TIMING.  `--no-capture` only: an event recorded inside a stream capture is silently dropped, so a
    // captured graph cannot carry these events and the numbers would be zeros that read as "every stage is free".
    // Refusing is the fix.
    if (const int rc = basalt::program::measure::arm_stage_timing(o); rc != 0) return rc;
    std::FILE* routing = nullptr;
    if (!o.dump_routing.empty()) {
        if (o.no_pool) {
            basalt::log::error("basalt generate", "--dump-routing needs the expert pool; the routed ids reach "
                                 "the host through the doorbell the pool reads. Drop --no-pool.");
            return 2;
        }
        routing = std::fopen(o.dump_routing.c_str(), "wb");
        if (routing == nullptr) {
            basalt::log::error("basalt generate", "cannot write %s", o.dump_routing.c_str());
            return 1;
        }
        drive.routing = routing;
    }
    basalt::core::PoolFn pool_fn = o.no_pool ? nullptr : &drive_pool;
    // The hit hook rides the same switch as the pool: with no pool there is no `parts` staging to
    // write into, and a hit path with nowhere to write is a wrong token rather than an error.
    basalt::core::HitFn hit_fn =
        (o.no_pool || o.expert_cache <= 0) ? nullptr : &basalt::core::expert_hit_run;
    void* pool_user = o.no_pool ? nullptr : (void*) &drive;

    // THE LOOP RUNS A CAPTURED PAIR PER LAYER: `pre[l]` ends with the router and the doorbell, then the CPU pool runs,
    // then `post[l]` combines those experts with THAT layer's weights - so layer `l`'s experts meet layer `l`'s
    // routing.  The window for the CPU pool is the GPU work that follows the ring inside `pre[l]`, which is the shared
    // expert and nothing else; a per-layer CPU expert pool cannot be hidden behind a strictly serial residual chain,
    // and the CPU term is answered by the VRAM expert cache, not by this pipeline.

    // ---- the graphs
    basalt::core::SessionGraphs gr;
    // **`--no-capture` AND THE EXPERTS ARE MUTUALLY EXCLUSIVE, AND SILENTLY SO.**
    //
    // The CPU expert pool is wired into `session_loop` - the host loop around the captured graphs - and
    // `session_token` has no pool hook at all.  So `--no-capture` did not merely change HOW the layers were
    // launched: it ran the whole model with `parts` left at whatever the buffer held, which is ZERO, and the
    // only symptom was `expert blobs 0` in a stats line nobody had to read.  A run that silently omits the
    // routed experts is not a slow measurement of this model, it is a measurement of a different model.
    //
    // Refusing is the fix.  `--no-pool` is the explicit way to say "I want the GPU-only floor".
    if (o.no_capture && !o.no_pool) {
        basalt::log::error("basalt generate", "--no-capture runs `session_token`, which has NO CPU expert pool hook, so "
                     "the routed experts would silently contribute nothing. Pass --no-pool as well if the "
                     "GPU-only floor is what you want.");
        return 2;
    }
    // The ladder is written by `session_loop`, which this build never runs; nothing fills the staging buffer, so the
    // two flags that asked for it are gone (see the PR that removed them).

    basalt::log::info("basalt generate", "session is up (engine %s)", BASALT_VERSION);
    auto run_head = [&](void* stream) -> bool {
        if (!native_head.loaded())
            return basalt::core::lm_head(wt, g, ss.block, d_logits, stream, err);
        return basalt::core::lm_head_mix(wt, g, ss.block, stream, err) &&
               native_head.run(ss.block.mixed, d_logits, stream, err);
    };
    float* d_emb = nullptr;
    if (cudaMalloc(&d_emb, (size_t) g.n_embd * 4) != cudaSuccess) {
        basalt::log::error("basalt generate", "the embedding buffer failed");
        return 1;
    }
    // **`sample_tokens` TAKES DEVICE POINTERS.**  It is a kernel launch; `logits` and `out` are both read and
    // written on the device.  Passing `logits.data()` - the host vector - faults inside the kernel and the
    // error surfaces at the NEXT synchronising call, which here was the next token's `embed_row`, reporting an
    // illegal access on a weight plane.  Nothing in the parameter names said device.
    int* d_next = nullptr;
    if (cudaMalloc(&d_next, sizeof(int)) != cudaSuccess) {
        basalt::log::error("basalt generate", "the sampler output buffer failed");
        return 1;
    }

    // **`R` IS BOTH THE INPUT AND THE OUTPUT, SO THE NEW TOKEN'S EMBEDDING HAS TO REPLACE THE OLD RESIDUAL.**
    // At `pos == 0` that is `session_zero`, which is the reference's own initial condition - the embedding
    // broadcast to all `hc` streams.  After that `session_zero` would also wipe the recurrence, so the
    // broadcast is done directly.  Getting this wrong is invisible for exactly one token.
    auto put_input = [&](int64_t tok, int64_t pos) -> bool {
        if (!basalt::core::embed_row(wt, g, tok, d_emb, main_cs, err)) {
            basalt::log::error("basalt generate", "%s", err.c_str());
            return false;
        }
        if (pos == 0) {
            basalt::core::session_zero(ss, g, d_emb, main_cs);
        } else {
            for (int64_t c = 0; c < g.hc; ++c)
                if (cudaMemcpyAsync(ss.R + (size_t) c * g.n_embd, d_emb, (size_t) g.n_embd * 4,
                                    cudaMemcpyDeviceToDevice, (cudaStream_t) main_cs) != cudaSuccess) {
                    basalt::log::error("basalt generate", "the residual broadcast failed");
                    return false;
                }
        }
        return true;
    };

    const auto sp = cli::sampling_options(o);

    // ---- THE LAYER ORACLE: ONE RESIDUAL SNAPSHOT PER LAYER PER POSITION, so the engine can be bisected against
    // `llama-debug`'s `l_last-<il>` node instead of against a single end-to-end perplexity.  The buffer is
    // PINNED because `session_loop` enqueues a device-to-host copy into it after every layer and the transfer
    // would otherwise be staged through a pageable bounce buffer on the critical path.
    // ---- prefill here is the DECODE PATH ONE TOKEN AT A TIME: the prompt goes through the decode-style graphs.  The
    // batched prompt path is a separate component.
    //
    // THE LOOP IS `feed -> 48 layers -> head -> sample -> feed`, AND THE FIRST GENERATED TOKEN COMES FROM THE LAST
    // *PROMPT* POSITION.  Sampling happens on the decode positions only, so `produced` is still empty when the first
    // generated position asks for `produced.back()`; teacher forcing below removes the special case: for every
    // position before the last prompt one, the next input is the PROMPT's next token, and after that it is the
    // sampled one.
    std::vector<int64_t> produced;
    double total_ms = 0;
    double prefill_ms = 0;   // positions 0 .. n_prompt-2: prompt tokens that only condition
    const Clock::time_point t_start = Clock::now();
    double ttft_ms = 0;
    const int64_t n_prompt = (int64_t) o.tokens.size();
    int64_t tok = o.tokens[0];

    // ---- THE GPU-ONLY MEASUREMENT MODES.  `--gpu-stages`, `--graph-only` and `--gpu-only-full` replay the captured
    // graphs and print a table instead of decoding; they live in measure/gpu_modes.cpp.
    {
        basalt::program::measure::GpuModeContext mc;
        mc.o = &o;
        mc.g = &g;
        mc.ss = &ss;
        mc.gr = &gr;
        mc.stream = main_cs;
        mc.db = &db;
        mc.d_next = d_next;
        mc.run_head = run_head;
        mc.err = &err;
        const int rc = basalt::program::measure::run_gpu_modes(mc);
        if (rc != basalt::program::measure::kNotRequested) return rc;
    }

    // ---- THE TOKEN PATH ALLOCATES NOTHING.
    //
    // `session_loop` must not allocate its pinned staging buffer, its probe event and its host pin on every token:
    // `cudaFreeHost` implicitly synchronises the device, so every token would finish with a device-wide sync.  The
    // scratch is created once here and reused; it also owns the host pin for the whole session.
    basalt::core::SessionLoopScratch loop_scratch;
    struct ScratchFree {
        basalt::core::SessionLoopScratch* p;
        ~ScratchFree() { if (p != nullptr) p->free(); }
    } scratch_free{&loop_scratch};
    // Initialised unconditionally, including under --no-pool: the loop validates the scratch it is handed, so
    // passing a default-constructed one is an error rather than a fallback.  (It was, and the guard caught it -
    // which is the point of the guard.)  One allocation at startup either way.
    if (!loop_scratch.init((size_t) K * g.n_embd * 4, err)) {
        return fatal_start_error(err);
    }
    // The whole token as ONE graph whenever nothing needs a host step between the ring and post[l] (the VRAM expert
    // tier and the per-layer dumps do).
    basalt::core::TokenGraph tgraph;
    struct TokenGraphFree {
        basalt::core::TokenGraph* p;
        ~TokenGraphFree() { basalt::core::token_graph_free(*p); }
    } tgraph_free{&tgraph};
    // With a PROFILE-filled cache the residency is static, so the hit decision moves onto the device and the token
    // graph keeps it.  (A cache filled on demand still needs the per-layer host path.)
    std::vector<int32_t> host_res;
    // The residency table is read by kernels on non-blocking streams, which do not order against the legacy stream a
    // plain pageable copy runs on, and such a copy can return before its data lands: each upload waits for its own
    // copy on the current device's legacy stream.
    auto res_put = [&host_res](int32_t* dst) -> cudaError_t {
        const cudaError_t e = cudaMemcpy(dst, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        return e != cudaSuccess ? e : cudaStreamSynchronize(cudaStreamLegacy);
    };
    int32_t* d_res = nullptr;
    int32_t* d_hit_count = nullptr;
    basalt::core::TokenHits thits;
    const bool graph_hits = hit_fn != nullptr && !profile.empty() && !o.no_pool;
    if (graph_hits && !o.no_capture) {
        host_res.assign((size_t) (g.n_layers * g.n_expert), basalt::core::kNotResident);
        int64_t resident = 0;
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const int st = multi_gpu ? stage_of(l) : 0;
                const int32_t slot = st > 0 ? stages[(size_t) st - 1]->cache.slot_of(l, e) : xcache.slot_of(l, e);
                host_res[(size_t) (l * g.n_expert + e)] = slot;
                if (slot != basalt::core::kNotResident) ++resident;
            }
        if (cudaMalloc((void**) &d_res, host_res.size() * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_count, sizeof(int32_t)) != cudaSuccess ||
            res_put(d_res) != cudaSuccess) {
            basalt::log::error("basalt generate", "the device residency table could not be staged");
            return 1;
        }
        thits.d_res = d_res;
        thits.n_expert = g.n_expert;
        // a file-backed arena: the experts no GPU holds are the ones the CPU pool and the prompt path will read - start
        // reading them now instead of faulting them in 4 KB at a time mid-request
        // ... and the ones a GPU holds are handed back first: the fill read the whole arena, and left mapped and
        // referenced it crowds every other allocation into swap
        {
            uint64_t released = 0;
            for (size_t i = 0; i < host_res.size(); ++i)
                if (host_res[i] != basalt::core::kNotResident)
                    released += srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
            if (released > 0)
                basalt::log::info("basalt generate", "expert arena: %.2f GiB of VRAM-held experts handed back to the OS",
                             (double) released / (1024.0 * 1024.0 * 1024.0));
            int64_t pf = 0;
            for (size_t i = 0; i < host_res.size(); ++i)
                if (host_res[i] == basalt::core::kNotResident) {
                    srcp->prefetch((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
                    ++pf;
                }
            (void) pf;
        }
        for (auto& st : stages) {   // layer split across GPUs: the same table on every device
            const basalt::core::OnDevice on(st->dev);
            if (cudaMalloc((void**) &st->d_res, host_res.size() * sizeof(int32_t)) != cudaSuccess ||
                res_put(st->d_res) !=
                    cudaSuccess) {
                basalt::log::error("basalt generate", "layer split: CUDA%d residency table failed", st->dev);
                return 1;
            }
        }
        thits.cache_base = drive.d.cache_base;
        thits.blob = drive.d.cache_blob;
        thits.d_slot = drive.d.d_slot;
        thits.d_dst = drive.d.d_dst;
        thits.d_count = d_hit_count;
        thits.x_q8 = drive.d.x_q8_0_hit;
        thits.x_scale = drive.d.x_q8_0_hit_scale;
        thits.scratch = drive.d.hit_scratch;
        thits.hit_out = drive.d.hit_out;
        drive.d.host_res = host_res.data();
        basalt::log::info("basalt generate", "token graph hit path: %lld resident experts, decided on the device",
                     (long long) resident);
    }

    // ================================ WHERE THE HOST TERM GOES, PER TOKEN ================================
    //
    // `--gpu-only-full` MEASURES THE 48 LAYER GRAPHS AND THE LM HEAD AND NOTHING ELSE.  It never enters this loop, so
    // it does not run `ple_stage_token`, `embed_row`, the whole-vocabulary logits readback, the NaN scan or the
    // sampler - and part of the difference between `--no-pool --stats` and that floor is per TOKEN, not per layer.
    //
    // Six accumulators, because the six have different fixes.  Reported in `--stats` as ms/token.  The boundary after
    // the layer loop is the one that matters: without it the head's interval would swallow all 48 layers.
    double ms_ple = 0, ms_embed = 0, ms_layers = 0, ms_head = 0, ms_readback = 0, ms_sample = 0;
    int64_t phase_tokens = 0;


    context.peer = &peer;
    context.xpeers = &xpeers;
    context.drive = &drive;
    context.routing = &routing;
    context.gr = &gr;
    context.d_emb = &d_emb;
    context.d_next = &d_next;
    context.sp = &sp;
    context.produced = &produced;
    context.total_ms = &total_ms;
    context.prefill_ms = &prefill_ms;
    context.t_start = &t_start;
    context.ttft_ms = &ttft_ms;
    context.n_prompt = &n_prompt;
    context.tok = &tok;
    context.tgraph = &tgraph;
    context.host_res = &host_res;
    context.res_put = res_put;
    context.d_res = &d_res;
    context.thits = &thits;
    context.graph_hits = &graph_hits;
    context.ms_ple = &ms_ple;
    context.ms_embed = &ms_embed;
    context.ms_layers = &ms_layers;
    context.ms_head = &ms_head;
    context.ms_readback = &ms_readback;
    context.ms_sample = &ms_sample;
    context.phase_tokens = &phase_tokens;
    return resident_plan(context);
}

}
