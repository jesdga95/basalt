// The flags Flash-Next takes on top of the common ones.
#include "models/qwen4exp/adapter.hpp"

#include "basalt/core/diag.hpp"
#include "basalt/log.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/models/qwen4exp/options.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>

namespace basalt::models::qwen4exp {
namespace {
constexpr float kRelaxedDraftingDefault = 0.5f;
constexpr int kRouteTailSkipDefault = 9;
constexpr float kRoutePriorDefault = 0.5f;
}  // namespace

bool parse_flag(const std::string& a, model::FlagArgs& args, Options& o, int& code) {
    if (a == "--rope-scaling") o.rope_scaling = args.next("--rope-scaling");
    else if (a == "--rope-scale") o.rope_scale = std::atof(args.next("--rope-scale"));
    else if (a == "--rope-freq-base") o.rope_freq_base = std::atof(args.next("--rope-freq-base"));
    else if (a == "--rope-freq-scale") o.rope_freq_scale = std::atof(args.next("--rope-freq-scale"));
    else if (a == "--yarn-orig-ctx") o.yarn_orig_ctx = std::atof(args.next("--yarn-orig-ctx"));
    else if (a == "--yarn-ext-factor") o.yarn_ext_factor = std::atof(args.next("--yarn-ext-factor"));
    else if (a == "--yarn-attn-factor") o.yarn_attn_factor = std::atof(args.next("--yarn-attn-factor"));
    else if (a == "--yarn-beta-fast") o.yarn_beta_fast = std::atof(args.next("--yarn-beta-fast"));
    else if (a == "--yarn-beta-slow") o.yarn_beta_slow = std::atof(args.next("--yarn-beta-slow"));
    else if (a == "--prompt-cache-every") o.prompt_cache_every = std::max(0LL, std::atoll(args.next("--prompt-cache-every")));
    else if (a == "--dump-residual") o.dump_residual = args.next("--dump-residual");
    else if (a == "--dump-mixed") o.dump_mixed = args.next("--dump-mixed");
    else if (a == "--dump-routing") o.dump_routing = args.next("--dump-routing");
    else if (a == "--ple-io") o.ple_io = args.next("--ple-io");
    else if (a == "--ple-row-cache") o.ple_row_cache = std::atoll(args.next("--ple-row-cache"));
    else if (a == "--ple-inflight") o.ple_inflight = std::atoi(args.next("--ple-inflight"));
    else if (a == "--ple-delay-us") o.ple_delay_us = std::atof(args.next("--ple-delay-us"));
    else if (a == "--kv") o.kv = args.next("--kv");
    else if (a == "--kv-resident") o.kv_resident = std::atoll(args.next("--kv-resident"));
    else if (a == "--no-capture") o.no_capture = true;
    else if (a == "--no-pool") o.no_pool = true;
    else if (a == "--stage-timing") o.stage_timing = true;
    else if (a == "--graph-only") { o.graph_only = true; o.generates = false; }
    else if (a == "--gpu-only-full") o.gpu_only_full = true;
    else if (a == "--prefill") {
        const std::string v = args.next("--prefill");
        o.prefill_given = true;
        o.prefill_auto = v == "auto" || v.rfind("auto:", 0) == 0;
        // auto:N (N = 16384 or 32768) lets auto take chunks above 8192
        const long long want_max = v.rfind("auto:", 0) == 0 ? std::atoll(v.c_str() + 5) : 8192;
        o.prefill_auto_fit = v == "auto";
        o.prefill_auto_max = o.prefill_auto_fit || want_max >= 32768 ? 32768 : want_max >= 16384 ? 16384 : 8192;
        o.prefill_chunk = o.prefill_auto ? o.prefill_auto_max : std::atoll(v.c_str());
    }
    else if (a == "--vram-elastic") o.vram_elastic = true;
    else if (a == "--vram-segment-mib") o.vram_segment_mib = std::atoll(args.next("--vram-segment-mib"));
    else if (a == "--prefill-until") o.prefill_until = std::atoll(args.next("--prefill-until"));
    else if (a == "--dump-final-r") o.dump_final_r = args.next("--dump-final-r");
    else if (a == "--spec") { o.spec = std::atoi(args.next("--spec")); o.spec_given = true; }
    else if (a == "--spec-oracle") o.spec_oracle = args.next("--spec-oracle");
    else if (a == "--spec-corrupt") o.spec_corrupt = std::atoi(args.next("--spec-corrupt"));
    else if (a == "--mtp-window") o.mtp_window = std::atoll(args.next("--mtp-window"));
    else if (a == "--adapt-every") { o.adapt_every = std::atoi(args.next("--adapt-every")); o.adapt_every_given = true; }
    else if (a == "--adapt-decay") { o.adapt_decay = (float) std::atof(args.next("--adapt-decay")); o.adapt_decay_set = true; }
    else if (a == "--prefill-cpu-share") o.prefill_cpu_share = std::atof(args.next("--prefill-cpu-share"));
    else if (a == "--relaxed-drafting") {   // the value is optional: a number in (0, 1] when one follows
        o.relaxed_drafting = kRelaxedDraftingDefault;
        if (args.peek() != nullptr) {
            char* e = nullptr;
            const float v = std::strtof(args.peek(), &e);
            if (e != args.peek() && *e == '\0') {
                args.take();
                if (!(v > 0.0f && v <= 1.0f)) { basalt::log::error("basalt", "--relaxed-drafting: P must be in (0, 1]"); { code = 2; return true; } }
                o.relaxed_drafting = v;
            }
        }
    }
    else if (a == "--route-tail-skip") {   // the value is optional: an integer 1..9 when one follows
        o.route_tail_skip = kRouteTailSkipDefault;
        if (args.peek() != nullptr) {
            char* e = nullptr;
            const long v = std::strtol(args.peek(), &e, 10);
            if (e != args.peek() && *e == '\0') {
                args.take();
                if (v < 1 || v > 9) { basalt::log::error("basalt", "--route-tail-skip: R must be 1..9"); { code = 2; return true; } }
                o.route_tail_skip = (int) v;
            }
        }
    }
    else if (a == "--route-prior") {   // the value is optional: a number in (0, 16] when one follows
        o.route_prior = kRoutePriorDefault;
        if (args.peek() != nullptr) {
            char* e = nullptr;
            const float v = std::strtof(args.peek(), &e);
            if (e != args.peek() && *e == '\0') {
                args.take();
                if (!(v > 0.0f && v <= 16.0f)) { basalt::log::error("basalt", "--route-prior: LAMBDA must be in (0, 16]"); { code = 2; return true; } }
                o.route_prior = v;
            }
        }
    }
    else if (a == "--short-read") o.short_read = std::max(0LL, std::atoll(args.next("--short-read")));
    else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(args.next("--suffix-draft")));
    else if (a == "--mtp-max-t") o.mtp_max_t = std::max(0, std::atoi(args.next("--mtp-max-t")));
    else if (a == "--control-vector") o.cvec_files.push_back({args.next("--control-vector"), 1.0f});
    else if (a == "--control-vector-scaled") {
        // FILE:SCALE, comma-separated; the LAST colon splits
        std::stringstream list(args.next("--control-vector-scaled"));
        std::string item;
        while (std::getline(list, item, ',')) {
            const size_t colon = item.rfind(':');
            char* end = nullptr;
            const float sc = colon == std::string::npos ? 0.0f : std::strtof(item.c_str() + colon + 1, &end);
            if (colon == std::string::npos || colon == 0 || end == item.c_str() + colon + 1 || *end != '\0') {
                basalt::log::error("basalt", "--control-vector-scaled: expected FILE:SCALE, got '%s'", item.c_str());
                { code = 2; return true; }
            }
            o.cvec_files.push_back({item.substr(0, colon), sc});
        }
    }
    else if (a == "--control-vector-layer-range") {
        o.cvec_first = std::atoi(args.next("--control-vector-layer-range"));
        o.cvec_last = std::atoi(args.next("--control-vector-layer-range"));
    }
    else if (a == "--cvec-mode") {
        const std::string m = args.next("--cvec-mode");
        if (m == "project") o.cvec_mode = 0;
        else if (m == "add") o.cvec_mode = 1;
        else { basalt::log::error("basalt", "--cvec-mode: add or project, got '%s'", m.c_str()); { code = 2; return true; } }
    }
    else if (a == "--cvec-dir") {
        const std::string d = args.next("--cvec-dir");
        if (d == "per-layer") o.cvec_single = -1;
        else if (d.rfind("single:", 0) == 0) o.cvec_single = std::atoi(d.c_str() + 7);
        else { basalt::log::error("basalt", "--cvec-dir: per-layer or single:L, got '%s'", d.c_str()); { code = 2; return true; } }
    }
    else if (a == "--adapt-swaps") o.adapt_swaps = std::atoi(args.next("--adapt-swaps"));
    else if (a == "--peer-exclusive") { o.peer_exclusive = true; o.peer_exclusive_given = true; }
    else if (a == "--peer-adapt-swaps") o.peer_adapt_swaps = std::atoi(args.next("--peer-adapt-swaps"));
    else if (a == "--peer-prefill-rows") o.peer_prefill_rows = std::atoll(args.next("--peer-prefill-rows"));
    else if (a == "--expert-profile-save") o.expert_profile_save = args.next("--expert-profile-save");
    else if (a == "--expert-profile-save-every")
        o.expert_profile_save_min = std::atof(args.next("--expert-profile-save-every"));
    else if (a == "--gpu-stages") o.gpu_stages = true;
    else if (a == "--mmap-experts") o.mmap_experts = true;
    else if (a == "--shared-expert-arena") o.shared_expert_arena = args.next("--shared-expert-arena");
    else if (a == "--resident-cpu-experts") o.resident_cpu_experts = o.resident_cpu_explicit = true;
    else if (a == "--resident-experts") {
        o.mmap_experts = o.resident_cpu_experts = o.resident_pin = o.resident_soft = true;
        o.resident_headroom = 0;
    }
    else if (a == "--stats") o.stats = true;
    else if (a == "--profile") basalt::core::diagnostics().profile = true;
    else if (a == "--wait-stats") basalt::core::diagnostics().wait_stats = true;
    else if (a == "--force-zero-doorbell") basalt::core::diagnostics().zero_doorbell = true;
    else if (a == "--decode-timing") basalt::core::diagnostics().decode_timing = true;
    else if (a == "--draft-log") basalt::core::diagnostics().draft_log = args.next("--draft-log");
    else if (a == "--exp") {
        const std::string list = args.next("--exp");   // NAME=VALUE[,NAME=VALUE...]
        for (size_t at = 0; at < list.size();) {
            size_t end = list.find(',', at);
            if (end == std::string::npos) end = list.size();
            const std::string kv = list.substr(at, end - at);
            at = end + 1;
            const size_t eq = kv.find('=');
            if (eq == std::string::npos || eq == 0) { basalt::log::error("basalt generate", "--exp needs NAME=VALUE"); { code = 2; return true; } }
            basalt::core::diagnostics().exp[kv.substr(0, eq)] = std::atof(kv.c_str() + eq + 1);
        }
    }
    else return false;
    return true;
}

std::optional<int> finish_flags(Options& o) {
    if (o.resident_budget_given) {   // the resident mode with a RAM budget
        o.mmap_experts = o.resident_cpu_experts = o.resident_pin = true;
        o.resident_headroom = 0;
    }
    if (o.route_tail_skip > 0 || o.route_prior > 0.0f || o.relaxed_drafting > 0.0f) {
        basalt::log::warn("basalt", "lossy flags enabled, these flags will affect the accuracy of the served model");
        if (o.relaxed_drafting > 0.0f)
            basalt::log::warn("basalt", "--relaxed-drafting %g accepts draft tokens the model would not pick", (double) o.relaxed_drafting);
        if (o.route_prior > 0.0f)
            basalt::log::warn("basalt", "--route-prior %g changes expert routing", (double) o.route_prior);
        if (o.route_tail_skip > 0)
            basalt::log::warn("basalt", "--route-tail-skip %d drops low-ranked missed experts", o.route_tail_skip);
    }
    return std::nullopt;
}

}  // namespace basalt::models::qwen4exp
