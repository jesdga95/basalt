#include "context.hpp"

namespace basalt::program::generate {

int dispatch(Context& context) {
    auto& o = *context.o;
    auto& ppl_base = *context.ppl_base;
    auto& ppl_out = *context.ppl_out;
    auto& model_file = *context.model_file;
    auto& peer_streamed_share = *context.peer_streamed_share;
    auto& split_at = *context.split_at;
    auto& split_devs = *context.split_devs;
    auto& split_same = *context.split_same;
    auto& multi_gpu = *context.multi_gpu;
    auto& pipe_want = *context.pipe_want;
    auto& no_spec = *context.no_spec;
    auto& err = *context.err;
    auto& main_link_gbps = *context.main_link_gbps;
    auto& g = *context.g;
    auto& K = *context.K;
    auto& profile = *context.profile;
    auto& peer_hot_share = *context.peer_hot_share;
    auto& profile_loaded = *context.profile_loaded;
    auto& pf_borrow = *context.pf_borrow;
    auto& wt = *context.wt;
    auto& d_mrope = *context.d_mrope;
    auto& mrope_host = *context.mrope_host;
    auto& cvec_summary = *context.cvec_summary;
    auto& ss = *context.ss;
    auto& main_stream = *context.main_stream;
    auto& main_cs = *context.main_cs;
    auto& ple_table = *context.ple_table;
    auto& vis_enc = *context.vis_enc;
    auto& stages = *context.stages;
    auto& last_st = *context.last_st;
    auto& stage_of = context.stage_of;
    auto& profile_all = *context.profile_all;
    auto& kv_floor = *context.kv_floor;
    auto& bslot_ss = *context.bslot_ss;
    auto& slot_cap = *context.slot_cap;
    auto& slot_kv_cuda0 = *context.slot_kv_cuda0;
    auto& slot0_is_solo = *context.slot0_is_solo;
    auto& mtp = *context.mtp;
    auto& mtp_on = *context.mtp_on;
    auto& mtp_kv = *context.mtp_kv;
    auto& n_vocab = *context.n_vocab;
    auto& grammar_vocab = *context.grammar_vocab;
    auto& native_head = *context.native_head;
    auto& mtp_head = *context.mtp_head;
    auto& src = *context.src;
    auto& arena_src = *context.arena_src;
    auto& srcp = *context.srcp;
    auto& xcache = *context.xcache;
    auto& remote_experts = *context.remote_experts;
    auto& remote_opt = *context.remote_opt;
    auto& peer = *context.peer;
    auto& xpeers = *context.xpeers;
    auto& drive = *context.drive;
    auto& host_res = *context.host_res;
    auto& res_put = context.res_put;
    auto& d_res = *context.d_res;
    auto& thits = *context.thits;
    auto& graph_hits = *context.graph_hits;
    auto& auto_lend_pct = *context.auto_lend_pct;
    auto& auto_ceiling = *context.auto_ceiling;
    auto& prompt_file_bytes = *context.prompt_file_bytes;
    auto& pool = *context.pool;
    auto& ram_floor_bytes = *context.ram_floor_bytes;
    auto& pstages = *context.pstages;
    auto& read_cap = context.read_cap;
    auto& kSplitRefCells = *context.kSplitRefCells;
    auto& split_hand_rows = *context.split_hand_rows;
    auto& split_peer_rows = *context.split_peer_rows;
    auto& split_price = context.split_price;
    auto& split_layers_text = context.split_layers_text;
    auto& split_stages_text = context.split_stages_text;
    auto& split_stages = *context.split_stages;

    // Where the routed experts live, fastest tier first, and a warning when a share of them is only in the file (decode
    // reads those from the SSD).  Called once the tiers are final.
    uint64_t tier_file_bytes = 0;   // the experts only the file holds, once the tiers are final
    auto report_expert_tiers = [&]() {
        uint64_t b_vram = 0, b_peer = 0, b_ram = 0, b_top = 0, b_file = 0;
        const bool ram_ok = srcp == &src && src.complement_ready();
        // the arena holds every expert in RAM (opening it fails when it cannot), so nothing it serves is file-backed
        const bool arena_ok = srcp == &arena_src && arena_src.mapped();
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const uint64_t blob = basalt::kernels::cpu::expert_layout().blob_bytes(l);
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const size_t i = (size_t) (l * g.n_expert + e);
                bool on_gpu = (i < host_res.size() && host_res[i] >= 0);
                for (const auto& st : stages) on_gpu = on_gpu || st->cache.slot_of(l, e) >= 0;
                for (const auto& r : remote_experts) on_gpu = on_gpu || (r.resident() > 0 && r.owns((int64_t) i));
                bool on_peer = peer.valid() && peer.has(l, e);
                for (const auto& xpe : xpeers) on_peer = on_peer || (xpe->valid() && xpe->has(l, e));
                if (on_gpu) b_vram += blob;
                else if (on_peer) b_peer += blob;
                else if (arena_ok) b_ram += blob;
                else if (ram_ok && src.has_resident(l, e)) (src.resident_in_top_up(l, e) ? b_top : b_ram) += blob;
                else b_file += blob;
            }
        }
        tier_file_bytes = b_file;
        const double total = (double) (b_vram + b_peer + b_ram + b_top + b_file);
        if (total <= 0.0) return;
        basalt::core::draft_cum_cpu_mb() = (float) ((double) (b_ram + b_top) / (double) (g.n_layers * g.n_expert) / 1e6);
        constexpr double kGiB = 1073741824.0;
        auto pct = [&](uint64_t b) { return 100.0 * (double) b / total; };
        // RAM is the loaded copy (the bytes the loader reports); the top-up added after the load is shown beside it
        char top[48] = "";
        if (b_top > 0) std::snprintf(top, sizeof top, " + %.1f GiB topped up", (double) b_top / kGiB);
        basalt::log::info("basalt generate", "experts live in VRAM %.1f GiB (%.0f%%), peer GPU %.1f GiB (%.0f%%), RAM "
                             "%.1f GiB%s (%.0f%%), file (SSD) %.1f GiB (%.0f%%); %.1f GiB in all",
                     (double) b_vram / kGiB, pct(b_vram), (double) b_peer / kGiB, pct(b_peer), (double) b_ram / kGiB, top,
                     pct(b_ram + b_top), (double) b_file / kGiB, pct(b_file), total / kGiB);
        // The file holds the coldest experts (the tiers fill in profile order), so its share of the bytes is well above
        // its share of the reads; 2% is where a decode window starts to meet SSD reads that cost far more than a RAM hit.
        if (pct(b_file) < 2.0) return;
        std::string levers = "a smaller quant, more RAM";
        if (!o.resident_cpu_experts) levers += ", --resident-experts";
        if (o.peer_device < 1) levers += ", a second GPU as the peer tier (--gpu 0,1)";
        if (o.batch > 1) levers += ", fewer --parallel slots or no --slot-full-context (the slots hold " +
                                   std::to_string((slot_kv_cuda0 + (1u << 29)) >> 30) + " GiB of KV beyond the solo one)";
        basalt::log::warn("basalt", "%.0f%% of the experts (%.1f GiB) are only in the file: decode reads them "
                             "from the SSD and may be much slower. To fit them: %s",
                     pct(b_file), (double) b_file / kGiB, levers.c_str());
    };
    if (!o.serve) report_expert_tiers();
    if (o.serve) {
        basalt::program::serve::Context context{
            .o = o,
            .ppl_base = ppl_base,
            .ppl_out = ppl_out,
            .model_file = model_file,
            .peer_streamed_share = peer_streamed_share,
            .split_at = split_at,
            .split_devs = split_devs,
            .split_same = split_same,
            .multi_gpu = multi_gpu,
            .pipe_want = pipe_want,
            .no_spec = no_spec,
            .err = err,
            .main_link_gbps = main_link_gbps,
            .g = g,
            .K = K,
            .profile = profile,
            .peer_hot_share = peer_hot_share,
            .profile_loaded = profile_loaded,
            .pf_borrow = pf_borrow,
            .wt = wt,
            .d_mrope = d_mrope,
            .mrope_host = mrope_host,
            .cvec_summary = cvec_summary,
            .ss = ss,
            .main_stream = main_stream,
            .main_cs = main_cs,
            .ple_table = ple_table,
            .vis_enc = vis_enc,
            .stages = stages,
            .last_st = last_st,
            .stage_of = stage_of,
            .profile_all = profile_all,
            .kv_floor = kv_floor,
            .bslot_ss = bslot_ss,
            .slot_cap = slot_cap,
            .slot0_is_solo = slot0_is_solo,
            .mtp = mtp,
            .mtp_on = mtp_on,
            .mtp_kv = mtp_kv,
            .n_vocab = n_vocab,
            .grammar_vocab = grammar_vocab,
            .native_head = native_head,
            .mtp_head = mtp_head,
            .src = src,
            .srcp = srcp,
            .xcache = xcache,
            .remote_experts = remote_experts,
            .remote_opt = remote_opt,
            .peer = peer,
            .xpeers = xpeers,
            .drive = drive,
            .host_res = host_res,
            .res_put = res_put,
            .d_res = d_res,
            .thits = thits,
            .graph_hits = graph_hits,
            .auto_lend_pct = auto_lend_pct,
            .auto_ceiling = auto_ceiling,
            .prompt_file_bytes = prompt_file_bytes,
            .pool = pool,
            .ram_floor_bytes = ram_floor_bytes,
            .pstages = pstages,
            .read_cap = read_cap,
            .kSplitRefCells = kSplitRefCells,
            .split_hand_rows = split_hand_rows,
            .split_peer_rows = split_peer_rows,
            .split_price = split_price,
            .split_layers_text = split_layers_text,
            .split_stages_text = split_stages_text,
            .split_stages = split_stages,
            .tier_file_bytes = tier_file_bytes,
            .report_expert_tiers = report_expert_tiers,
        };
        return basalt::program::serve::run(context);
    }


    return prefill(context);
}

}
