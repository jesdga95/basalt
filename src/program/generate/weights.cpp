#include "context.hpp"

namespace basalt::program::generate {

int weights(Context& context) {
    auto& o = *context.o;
    auto& srcs = *context.srcs;
    auto& pack_files = *context.pack_files;
    auto& multi_gpu = *context.multi_gpu;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& K = *context.K;
    auto& skip = *context.skip;
    auto& hc8_names = *context.hc8_names;
    auto& pool_bytes = *context.pool_bytes;

    // ---- the bytes this start places, once, from the model file's tables: the dense weights, the 8-bit copies, the
    // draft layer, the head and every expert (the GPU cache plus its complement in RAM).  Each term is the total its own
    // load phase reports.  A layer split places bytes per stage: no total, the progress lines stand alone.
    uint64_t model_bytes = 0, expert_total_bytes = 0, cuda0_fixed_bytes = 0;   // the latter: all but the cache fill
    for (int64_t l = 0; l < g.n_layers; ++l)
        expert_total_bytes += (uint64_t) g.n_expert * basalt::kernels::cpu::expert_layout().blob_bytes(l);
    // The default expert mode copies every expert into RAM and keeps it there, so a machine that cannot hold them
    // would swap until it froze.  Refuse before anything is allocated.  (A shared arena may already be in RAM.)
    if (!o.mmap_experts && o.shared_expert_arena.empty()) {
        bool known = false;
        const uint64_t budget = basalt::platform::ram_budget(&known);
        if (known && expert_total_bytes > budget) {
            basalt::log::error("basalt", "the experts need %.1f GiB of RAM, %.1f GiB is available; use "
                                 "--resident-experts or --mmap-experts",
                         (double) expert_total_bytes / 1073741824.0,
                         (double) basalt::platform::available_memory() / 1073741824.0);
            return 1;
        }
    }
    if (!multi_gpu) {
        uint64_t dense_bytes = 0, mtp_bytes = 0, head_bytes = 0;
        std::set<std::string> index_names;
        for_each_index_name(pack_files.index, [&](const std::string& n) { index_names.insert(n); });
        const bool planned =
            basalt::core::NativeDense::plan_bytes(o.native_shards, true, 0, -1, dense_bytes, err) &&
            basalt::core::NativeHead::plan_bytes(o.native_shards, head_bytes, err) &&
            (o.mtp.empty() || o.spec < 2 ||
             basalt::core::MtpDrafter::plan_bytes(basalt::core::MtpFiles::from(srcs), g, mtp_bytes, err));
        if (planned) {
            cuda0_fixed_bytes = pool_bytes + dense_bytes + head_bytes + mtp_bytes +
                                basalt::core::Verifier::hc8_plan_bytes(
                                    [&](const std::string& n) { return index_names.count(n) > 0; }, g);
            model_bytes = cuda0_fixed_bytes + expert_total_bytes;
            basalt::core::LoadProgress::announce_total(model_bytes);
            // the experts' places follow once the cache is sized; with the arena every expert goes to RAM first
            basalt::core::LoadProgress::plan("cuda0", cuda0_fixed_bytes);
            if (!o.mmap_experts) basalt::core::LoadProgress::plan("ram", expert_total_bytes);
        } else {
            basalt::log::warn("basalt generate", "load total not computed: %s", err.c_str());
        }
    }
    void* arena = nullptr;
    if (const cudaError_t ce = cudaMalloc(&arena, pool_bytes); ce != cudaSuccess) {
        // the arena is the first large allocation and its size does not depend on the context, so what is
        // missing is held by something else: say how much was free
        cudaGetLastError();
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        basalt::log::error("basalt generate", "cudaMalloc(%llu) for the weight arena failed (%s): %llu MiB of %llu "
                             "MiB VRAM free on this GPU. The arena is allocated first, before the KV and expert "
                             "caches: another program (or an engine that is still exiting) holds the rest - "
                             "nvidia-smi / rocm-smi lists them",
                     (unsigned long long) pool_bytes, cudaGetErrorString(ce), (unsigned long long) (free_b >> 20),
                     (unsigned long long) (total_b >> 20));
        return 1;
    }
    auto load_t0 = std::chrono::steady_clock::now();
    auto load_s = [&] {
        const auto t = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t - load_t0).count();
        load_t0 = t;
        return s;
    };
    basalt::core::WeightTable wt;
    if (!wt.load(pack_files, arena, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        return fatal_start_error(err);
    }
    // every tensor's shape and engine form against what the kernels read (skipped rows keep their metadata)
    if (!basalt::core::check_all(wt, g, err)) {
        return fatal_start_error(o.model + ": " + err);
    }
    basalt::log::info("basalt generate", "%llu MiB of weights loaded from %s in %.1f s (%zu tensors "
                         "skipped: served from the source tensors)",
                 (unsigned long long) (pool_bytes >> 20), o.model.c_str(), load_s(), skip.size());
    if (!hc8_names.empty()) {
        // the copies the model file holds are registered as they are; the rest are quantized from their bf16 matrices,
        // staged for the purpose and dropped
        std::set<std::string> staged_names;
        size_t n_done = 0;
        for (const std::string& n : hc8_names) {
            const basalt::core::WeightRef* r0 = wt.find(n);
            const void* q = r0 ? basalt::core::hc8_table_adopt(n, nullptr, (int) r0->ne1, (int) r0->ne0) : nullptr;
            if (q == nullptr) { staged_names.insert(n); continue; }
            if (!wt.rebind(n, q)) {
                basalt::log::error("basalt generate", "hc8: no 8-bit copy of %s", n.c_str());
                return 1;
            }
            ++n_done;
        }
        uint64_t stage_bytes = 0;
        if (!staged_names.empty()) {
        std::set<std::string> others;
        for (const auto& kv : wt.all())
            if (!staged_names.count(kv.first)) others.insert(kv.first);
        void* stage = nullptr;
        basalt::core::WeightTable staged;
        if (!basalt::core::WeightTable::pool_bytes(pack_files, stage_bytes, err, &others) ||
            cudaMalloc(&stage, stage_bytes) != cudaSuccess || !staged.load(pack_files, stage, stage_bytes, err, &others, false)) {
            basalt::log::error("basalt generate", "hc8: staging the hyper-connection matrices failed%s%s",
                         err.empty() ? "" : ": ", err.c_str());
            return 1;
        }
        for (const std::string& n : staged_names) {
            const basalt::core::WeightRef* r = staged.find(n);
            const void* q = r && r->data && r->kind == basalt::core::WeightKind::Bf16InF32
                                ? basalt::kernels::hc8_adopt((const uint16_t*) r->data, (int) r->ne1, (int) r->ne0)
                                : nullptr;
            if (q == nullptr || !wt.rebind(n, q)) {
                basalt::log::error("basalt generate", "hc8: no 8-bit copy of %s", n.c_str());
                return 1;
            }
            ++n_done;
        }
        cudaFree(stage);
        }
        for (const auto& kv : wt.all())   // the injections stay bf16 in the arena, with copies beside them
            if (kv.first.ends_with("_inject.weight") && kv.first.find(".hc_") != std::string::npos && kv.second.data) {
                if (basalt::core::hc8_table_adopt(kv.first, (const uint16_t*) kv.second.data, (int) kv.second.ne1,
                                                  (int) kv.second.ne0) == nullptr)
                    basalt::kernels::hc8_prepare((const uint16_t*) kv.second.data, (int) kv.second.ne1, (int) kv.second.ne0);
            }
        basalt::log::info("basalt hc8", "%zu hyper-connection matrices held only as 8-bit copies (%.0f MiB of bf16 "
                             "left out of VRAM, %.0f MiB of copies) in %.1f s",
                     n_done, (double) stage_bytes / 1048576.0, (double) basalt::kernels::hc8_bytes() / 1048576.0,
                     load_s());
    }

    basalt::core::NativeDense native_dense;
    {
        if (!native_dense.load(o.native_shards, wt, err, true)) {
            basalt::log::error("basalt generate", "native dense projections: %s", err.c_str());
            return 1;
        }
        basalt::log::info("basalt generate", "%zu native projection matrices, %.2f MiB of weights, in %.1f s",
                     native_dense.tensor_count(), (double) native_dense.weight_bytes() / (1024.0 * 1024.0), load_s());
    }

    // The fused hyper-connection read needs the layout where R lags by one pending write; the per-stage measurement
    // needs the unfused layout of R.
    basalt::core::layer_set_fused_gr(!o.gpu_stages && !o.stage_timing);
    // The vision path: every rope kernel reads a cell's (t, h, w) from this table (basalt/kernels/mrope.hpp).  It is
    // the identity until an image request, and it is set here, before any CUDA graph captures a rope kernel.
    int32_t* d_mrope = nullptr;
    std::vector<int32_t> mrope_host;
    if (o.vision) {
        const int64_t cells = o.max_context + 64;
        mrope_host.resize((size_t) cells * 3);
        for (int64_t c = 0; c < cells; ++c)
            mrope_host[(size_t) c * 3] = mrope_host[(size_t) c * 3 + 1] = mrope_host[(size_t) c * 3 + 2] = (int32_t) c;
        if (cudaMalloc(&d_mrope, mrope_host.size() * sizeof(int32_t)) != cudaSuccess ||
            cudaMemcpy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            basalt::log::error("basalt generate", "cannot allocate the image position table");
            return 1;
        }
        basalt::kernels::mrope_table_set(d_mrope);
    }
    // before session_init: every graph captured from here on has the vector's kernels where it applies
    std::string cvec_summary = "0";
    if (!o.cvec_files.empty()) {
        std::string ce;
        if (!cli::load_control_vectors(o, g, cvec_summary, ce)) {
            basalt::log::error("basalt generate", "control vector: %s", ce.c_str());
            return 2;
        }
    }
    if (o.max_context < (int64_t) o.tokens.size() + o.max_new) {
        basalt::log::error("basalt generate", "--max-context %lld cannot hold %zu prompt + %lld new tokens",
                     (long long) o.max_context, o.tokens.size(), (long long) o.max_new);
        return 2;
    }

    basalt::core::SessionState ss;
    void* sbuf = nullptr;   // allocated after the layer-split search, sized to CUDA0's own layer range (the carve)
    // The engine runs its captured graphs and kernels on its own stream, not the legacy default stream: every
    // session call - `session_capture`, `session_replay`, `session_token` and `session_loop` - is handed that stream,
    // because a launch through stream 0 carries a large fixed per-launch cost.
    cudaStream_t main_stream = nullptr;
    if (cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking) != cudaSuccess) {
        basalt::log::error("basalt generate", "cannot create the main stream");
        return 1;
    }
    void* const main_cs = (void*) main_stream;

    basalt::core::Doorbell db;
    if (basalt::core::doorbell_init(g, K, db) == 0) {
        basalt::log::error("basalt generate", "doorbell_init failed");
        return 1;
    }
    ss.db = &db;


    context.model_bytes = &model_bytes;
    context.expert_total_bytes = &expert_total_bytes;
    context.cuda0_fixed_bytes = &cuda0_fixed_bytes;
    context.arena = &arena;
    context.wt = &wt;
    context.d_mrope = &d_mrope;
    context.mrope_host = &mrope_host;
    context.cvec_summary = &cvec_summary;
    context.ss = &ss;
    context.sbuf = &sbuf;
    context.main_stream = &main_stream;
    context.main_cs = &main_cs;
    context.db = &db;
    return vision(context);
}

}
