#include "context.hpp"

namespace basalt::program::generate {

int vision(Context& context) {
    auto& o = *context.o;
    auto& multi_gpu = *context.multi_gpu;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& K = *context.K;
    auto& wt = *context.wt;
    auto& ss = *context.ss;

    // ================================ THE PLE ================================
    //
    // Layer 1 carries six `blk.1.ple_*` tensors and the model is wrong without the module.  Everything below is construction - the table
    // is a mapping of the ORIGINAL second GGUF shard, the six weights are already loaded in the arena, and the
    // three buffers are the only allocation.
    basalt::kernels::PleTable ple_table;
    std::vector<float> ple_emb_host((size_t) basalt::kernels::NG_N_EMBD);
    float* ple_emb_dev = nullptr;
    float* ple_scratch = nullptr;
    {
        basalt::kernels::PleIoOptions pio;
        pio.mode = o.ple_io == "ram" ? basalt::kernels::PleIo::Mmap : basalt::kernels::PleIo::Direct;
        pio.lock = o.ple_io == "ram";
        const auto tpl = Clock::now();
        pio.max_inflight = (uint32_t) o.ple_inflight;
        pio.cache_rows = (uint64_t) o.ple_row_cache;
        // Keep the SSD awake while rows are being asked for (PleReader::set_keepalive): some SSDs stall the first reads
        // after a short idle period.  The keepalive reads one page after a read-free interval, from the table open
        // on and for a window after the last row request; then the SSD may sleep until the next request.  The
        // interval is well under the idle period after which the SSD drops its power state.
        {
            pio.keepalive_ms = 40.0;
            pio.keepalive_window_s = 60.0;
            if (pio.mode != basalt::kernels::PleIo::Direct) pio.keepalive_ms = 0;
        }
        if (!ple_table.open(o.model, err, pio)) {
            return fatal_start_error(err);
        }
        if (pio.mode == basalt::kernels::PleIo::Direct) {
            // --ple-io direct's random reads (up to --ple-inflight at once) are for SSDs; a rotational disk can take
            // longer than the stall watchdog's limit for one prompt chunk's rows.  Recommended, not forced.
            std::string dev;
            if (basalt::platform::disk_rotational(o.model, dev) == 1)
                basalt::log::warn("basalt generate", "the n-gram table (%s) is on a rotational disk (%s): "
                                     "--ple-io direct reads it at random and can stall a prompt for minutes. "
                                     "--ple-io ram (it then needs RAM for the table) or the model on an SSD is "
                                     "recommended", o.model.c_str(), dev.c_str());
        }
        if (pio.lock)
            basalt::log::info("basalt generate", "PLE table %s (--ple-io ram) in %.1f s",
                         ple_table.locked() ? "locked in RAM" : "loaded (not locked)",
                         std::chrono::duration<double>(Clock::now() - tpl).count());
        if (pio.keepalive_ms > 0)
            basalt::log::info("basalt generate", "the SSD is kept awake while rows are read: one page of the table after "
                                 "%.0f ms without a read, until %.0f s after the last request ", pio.keepalive_ms, pio.keepalive_window_s);
        else if (pio.mode == basalt::kernels::PleIo::Direct)
            basalt::log::info("basalt generate", "SSD keep-alive off: the SSD may fall asleep between reads");
        const basalt::core::WeightRef* wk = wt.find("blk.1.ple_key.weight");
        const basalt::core::WeightRef* wv = wt.find("blk.1.ple_value.weight");
        const basalt::core::WeightRef* wnk = wt.find("blk.1.ple_norm_key.weight");
        const basalt::core::WeightRef* wnq = wt.find("blk.1.ple_norm_query.weight");
        const basalt::core::WeightRef* wnc = wt.find("blk.1.ple_norm_conv.weight");
        const basalt::core::WeightRef* wc = wt.find("blk.1.ple_conv1d.weight");
        if (!wk || !wv || !wnk || !wnq || !wnc || !wc) {
            basalt::log::error("basalt generate", "the model has no blk.1.ple_* tensors, so the PLE cannot be wired");
            return 1;
        }
        // `ple_key` is S2 and the loader has already widened its scales to f32, so the two planes are located
        // by the sizes the `WeightRef` records rather than re-derived - the same rule `plane_ptrs` follows.
        if (!wk->quantized()) {
            // a BF16 key
            ss.ple.w.key_bf16 = (const uint16_t*) wk->data;
        } else if (wk->data != nullptr) {
            ss.ple.w.key_codes = (const uint8_t*) wk->data;
            ss.ple.w.key_scales = (const float*) ((const uint8_t*) wk->data + wk->codes_bytes);
        }
        if (wk->quantized()) {
            if (!wk->native_data || (wk->native_type != 42 && wk->native_type != 18 && wk->native_type != 23 &&
                                     wk->native_type != 8) || !wk->native_q8_1) {
                basalt::log::error("basalt generate", "native PLE key is absent or incompatible");
                return 1;
            }
            ss.ple.w.key_native_data = wk->native_data;
            ss.ple.w.key_native_type = wk->native_type;
            ss.ple.w.key_native_q8_1 = wk->native_q8_1;
        }
        ss.ple.w.value_bf16 = (const uint16_t*) wv->data;
        ss.ple.w.norm_key = (const float*) wnk->data;
        ss.ple.w.norm_query = (const float*) wnq->data;
        ss.ple.w.norm_conv = (const float*) wnc->data;
        // The conv1d kernel reads F16, so this cast is a claim about the stored form.  A tensor kept as F32 would hand
        // the kernel the low halves of the f32 words: a plausible wrong layer-1 routing, not an error.  The converter
        // narrows it (index kind 3); a file that was not converted that way is refused here.  BF16 (kind 4) has the
        // same size and is not F16.
        const bool f16 = wc->kind == basalt::core::WeightKind::F16InF32 ||
                         (wc->kind == basalt::core::WeightKind::Verbatim && wc->code_bits == 0);
        if (!f16 || wc->bytes != (uint64_t) wc->elements * 2) {
            basalt::log::error("basalt generate", "blk.1.ple_conv1d.weight is not stored as F16 (index kind %d, "
                                 "%llu B for %lld values); the PLE conv1d kernel reads F16 - convert the model again "
                                 "with tools/convert/convert.py",
                         (int) wc->kind, (unsigned long long) wc->bytes, (long long) wc->elements);
            return 1;
        }
        ss.ple.w.conv1d_f16 = (const uint16_t*) wc->data;
        ss.ple.consts = basalt::kernels::ple_artifact_consts();
        if (o.ple_delay_us > 0) ple_table.set_injected_delay_us(o.ple_delay_us);
        ss.ple.table = &ple_table;
        ss.ple.token = &ss.ple_token;
        ss.ple.prev = ss.ple_prev;
        // `ss.ple.hist` and the ready() check wait for `session_init`, which carves the history - the session is
        // allocated after the layer-split search (see the carve), and the wiring lands there
        ss.ple.emb_host = ple_emb_host.data();
        if (cudaMalloc((void**) &ple_emb_dev, (size_t) basalt::kernels::NG_N_EMBD * 4) != cudaSuccess ||
            cudaMalloc((void**) &ple_scratch, basalt::core::ple_run_scratch_bytes()) != cudaSuccess) {
            basalt::log::error("basalt generate", "the PLE buffers failed");
            return 1;
        }
        ss.ple.emb_dev = ple_emb_dev;
        ss.ple.scratch = ple_scratch;
    }

    // ---- --vision-mmproj: the image encoder's weights staged in pinned host memory, its stream and product handle made,
    // and one full-size encode run in a temporary region (the product library loads its kernels on first use), all
    // before the expert cache is sized: afterwards the encoder holds no VRAM until an ENC borrows cache slots
    basalt::vision::CudaEncoder vis_enc;
    const int vis_max_tokens = basalt::vision::Geometry{}.max_tokens;
    if (!o.vision_mmproj.empty() && o.serve) {
        const auto t0 = Clock::now();
        std::string verr;
        basalt::vision::Mmproj mp;
        if (!mp.open(o.vision_mmproj, verr) || !vis_enc.stage(mp, true, verr) || !vis_enc.create(verr)) {
            basalt::log::error("basalt generate", "the image encoder: %s", verr.c_str());
            return 1;
        }
        mp.release();
        void* tmp = nullptr;
        const size_t tmp_bytes = vis_enc.arena_bytes(vis_max_tokens);
        if (cudaMalloc(&tmp, tmp_bytes) != cudaSuccess || !vis_enc.attach(tmp, tmp_bytes, vis_max_tokens, verr)) {
            basalt::log::error("basalt generate", "the image encoder warm-up: %s", verr.empty() ? "out of device memory" : verr.c_str());
            return 1;
        }
        const basalt::vision::Image gray{2048, 2048, std::vector<uint8_t>((size_t) 2048 * 2048 * 3, 128)};
        const basalt::vision::Geometry geo;
        const basalt::vision::Size sz = basalt::vision::fit_size(gray.w, gray.h, geo);
        std::vector<float> planar, patches, out;
        basalt::vision::normalize_planar(basalt::vision::to_canvas(gray, sz), planar);
        basalt::vision::patchify(planar, sz, geo, patches);
        const bool warm = vis_enc.encode(patches, sz, out);
        vis_enc.detach();
        cudaFree(tmp);
        if (!warm) {
            basalt::log::error("basalt generate", "the image encoder warm-up failed: %s", vis_enc.last_error().c_str());
            return 1;
        }
        basalt::log::info("basalt serve", "image encoder staged: %.0f MiB of weights in pinned memory, an image borrows up "
                             "to %.2f GiB of expert-cache slots while it is read (%.1f s)",
                     (double) vis_enc.weights_bytes() / 1048576.0, (double) tmp_bytes / 1073741824.0,
                     std::chrono::duration<double>(Clock::now() - t0).count());
    }

    float* d_parts = nullptr;
    if (cudaMalloc(&d_parts, (size_t) K * g.n_embd * 4) != cudaSuccess ||
        cudaMemset(d_parts, 0, (size_t) K * g.n_embd * 4) != cudaSuccess) {
        basalt::log::error("basalt generate", "the parts buffer failed");
        return 1;
    }

    basalt::core::Verifier::set_commit_async(!multi_gpu);   // see Verifier::set_commit_async

    context.ple_table = &ple_table;
    context.vis_enc = &vis_enc;
    context.d_parts = &d_parts;
    return sessions(context);
}

}
