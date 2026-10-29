#include "arguments.hpp"
#include "help.hpp"
#include "tokens.hpp"

#include "basalt/core/diag.hpp"
#include "basalt/log.hpp"
#include "basalt/program/options.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <sstream>

namespace basalt::program::cli {
namespace {
constexpr float kRelaxedDraftingDefault = 0.5f;
}

std::optional<int> parse_options(int argc, char** argv, Options& o, std::string& ppl_base, std::string& ppl_out) {
    bool have_tokens = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { basalt::log::error("basalt", "%s needs a value", what); std::exit(2); }
            return argv[++i];
        };
        bool parsed = true;
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--model") o.model = next("--model");
        else if (a == "--dump-hc8") o.dump_hc8 = next("--dump-hc8");
        else if (a == "--tokens") {
            if (have_tokens) { basalt::log::error("basalt", "supply one token input only"); return 2; }
            std::string e;
            if (!parse_i64_list(next("--tokens"), o.tokens, e)) { basalt::log::error("basalt", "%s", e.c_str()); return 2; }
            have_tokens = true;
        }
        else if (a == "--tokens-file") {
            if (have_tokens) { basalt::log::error("basalt", "supply one token input only"); return 2; }
            if (const auto code = load_tokens_file(next("--tokens-file"), o.tokens)) return *code;
            have_tokens = true;
        }
        else if (a == "--max-new") o.max_new = std::atoll(next("--max-new"));
        else if (a == "--max-context") o.max_context = std::atoll(next("--max-context"));
        else if (a == "--rope-scaling") o.rope_scaling = next("--rope-scaling");
        else if (a == "--rope-scale") o.rope_scale = std::atof(next("--rope-scale"));
        else if (a == "--rope-freq-base") o.rope_freq_base = std::atof(next("--rope-freq-base"));
        else if (a == "--rope-freq-scale") o.rope_freq_scale = std::atof(next("--rope-freq-scale"));
        else if (a == "--yarn-orig-ctx") o.yarn_orig_ctx = std::atof(next("--yarn-orig-ctx"));
        else if (a == "--yarn-ext-factor") o.yarn_ext_factor = std::atof(next("--yarn-ext-factor"));
        else if (a == "--yarn-attn-factor") o.yarn_attn_factor = std::atof(next("--yarn-attn-factor"));
        else if (a == "--yarn-beta-fast") o.yarn_beta_fast = std::atof(next("--yarn-beta-fast"));
        else if (a == "--yarn-beta-slow") o.yarn_beta_slow = std::atof(next("--yarn-beta-slow"));
        else if (a == "--greedy") o.greedy = true;
        else if (a == "--seed") { o.seed = (uint64_t) std::atoll(next("--seed")); o.greedy = false; }
        else if (a == "--top-k") o.top_k = std::atoi(next("--top-k"));
        else if (a == "--top-p") o.top_p = (float) std::atof(next("--top-p"));
        else if (a == "--temperature") o.temperature = (float) std::atof(next("--temperature"));
        else if (a == "--dump-residual") o.dump_residual = next("--dump-residual");
        else if (a == "--dump-mixed") o.dump_mixed = next("--dump-mixed");
        else if (a == "--dump-routing") o.dump_routing = next("--dump-routing");
        else if (a == "--ple-io") o.ple_io = next("--ple-io");
        else if (a == "--ple-row-cache") o.ple_row_cache = std::atoll(next("--ple-row-cache"));
        else if (a == "--ple-inflight") o.ple_inflight = std::atoi(next("--ple-inflight"));
        else if (a == "--ple-delay-us") o.ple_delay_us = std::atof(next("--ple-delay-us"));
        else if (a == "--kv") o.kv = next("--kv");
        else if (a == "--kv-resident") o.kv_resident = std::atoll(next("--kv-resident"));
        else if (a == "--no-capture") o.no_capture = true;
        else if (a == "--no-pool") o.no_pool = true;
        else if (a == "--stage-timing") o.stage_timing = true;
        else if (a == "--graph-only") o.graph_only = true;
        else if (a == "--gpu-only-full") o.gpu_only_full = true;
        else if (a == "--pool-workers") { o.pool_workers = std::atoi(next("--pool-workers")); o.pool_workers_given = true; }
        else if (a == "--pool-affinity") {
            const std::string v = next("--pool-affinity");
            o.pool_affinity_given = true;
            if (v == "auto") o.pool_affinity = basalt::kernels::cpu::PoolAffinity::Auto;
            else if (v == "p-cores" || v == "pcores") o.pool_affinity = basalt::kernels::cpu::PoolAffinity::PCores;
            else if (v == "all") o.pool_affinity = basalt::kernels::cpu::PoolAffinity::All;
            else {
                basalt::log::error("basalt generate", "unknown --pool-affinity value '%s' (expected auto, p-cores, or all)", v.c_str());
                return 2;
            }
        }
        else parsed = false;
        // The chain continues here in a second statement: the compiler limits how deeply `else if` blocks may nest.
        if (!parsed) {
        if (a == "--expert-cache") {
            const std::string v = next("--expert-cache");
            o.expert_cache = (v == "auto") ? -1 : std::atoi(v.c_str());
        }
        else if (a == "--gpu") o.gpu = next("--gpu");
        else if (a == "--gpu-plan") o.gpu_plan = true;
        else if (a == "--vram-reserve-mib") { o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib")); o.vram_reserve_given = true; }
        else if (a == "--prefill") {
            const std::string v = next("--prefill");
            o.prefill_given = true;
            o.prefill_auto = v == "auto" || v.rfind("auto:", 0) == 0;
            // auto:N (N = 16384 or 32768) lets auto take chunks above 8192
            const long long want_max = v.rfind("auto:", 0) == 0 ? std::atoll(v.c_str() + 5) : 8192;
            o.prefill_auto_fit = v == "auto";
            o.prefill_auto_max = o.prefill_auto_fit || want_max >= 32768 ? 32768 : want_max >= 16384 ? 16384 : 8192;
            o.prefill_chunk = o.prefill_auto ? o.prefill_auto_max : std::atoll(v.c_str());
        }
        else if (a == "--vram-elastic") o.vram_elastic = true;
        else if (a == "--vram-segment-mib") o.vram_segment_mib = std::atoll(next("--vram-segment-mib"));
        else if (a == "--prefill-until") o.prefill_until = std::atoll(next("--prefill-until"));
        else if (a == "--dump-final-r") o.dump_final_r = next("--dump-final-r");
        else if (a == "--spec") { o.spec = std::atoi(next("--spec")); o.spec_given = true; }
        else if (a == "--batch") o.batch = std::atoi(next("--batch"));
        else if (a == "--slots") o.batch = std::atoi(next("--slots"));   // the same as --batch
        else if (a == "--spec-oracle") o.spec_oracle = next("--spec-oracle");
        else if (a == "--spec-corrupt") o.spec_corrupt = std::atoi(next("--spec-corrupt"));
        else if (a == "--mtp-window") o.mtp_window = std::atoll(next("--mtp-window"));
        else if (a == "--adapt-every") { o.adapt_every = std::atoi(next("--adapt-every")); o.adapt_every_given = true; }
        else if (a == "--adapt-decay") { o.adapt_decay = (float) std::atof(next("--adapt-decay")); o.adapt_decay_set = true; }
        else if (a == "--prefill-cpu-share") o.prefill_cpu_share = std::atof(next("--prefill-cpu-share"));
        else if (a == "--relaxed-drafting") {   // the value is optional: a number in (0, 1] when one follows
            o.relaxed_drafting = kRelaxedDraftingDefault;
            if (i + 1 < argc) {
                char* e = nullptr;
                const float v = std::strtof(argv[i + 1], &e);
                if (e != argv[i + 1] && *e == '\0') {
                    ++i;
                    if (!(v > 0.0f && v <= 1.0f)) { basalt::log::error("basalt", "--relaxed-drafting: P must be in (0, 1]"); return 2; }
                    o.relaxed_drafting = v;
                }
            }
        }
        else if (a == "--grammar") {
            const std::string v = next("--grammar");
            if (v != "on" && v != "off") { basalt::log::error("basalt", "--grammar: on or off"); return 2; }
            o.grammar = v == "on";
        }
        else if (a == "--stop-eos") o.stop_eos = true;
        else if (a == "--serve") o.serve = true;
        else if (a == "--vision") o.vision = true;
        else if (a == "--vision-mmproj") { o.vision_mmproj = next("--vision-mmproj"); o.vision = true; }
        else if (a == "--prompt-cache") o.prompt_cache = std::max(0, std::atoi(next("--prompt-cache")));
        else if (a == "--conversation-cache-dir") o.conversation_cache_dir = next("--conversation-cache-dir");
        else if (a == "--conversation-cache-mib" || a == "--conversation-cache-slots" ||
                 a == "--conversation-cache-min-free-mib" || a == "--conversation-cache-disk-mib") {
            const std::string value = next(a.c_str());
            int64_t number = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
            const int64_t limit = a == "--conversation-cache-slots" ? INT32_MAX : INT64_MAX / (1024 * 1024);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || number < 0 || number > limit) {
                basalt::log::error("basalt", "%s needs a nonnegative integer within range", a.c_str());
                return 2;
            }
            if (a == "--conversation-cache-mib") o.conversation_cache_mib = number;
            else if (a == "--conversation-cache-min-free-mib") o.conversation_cache_min_free_mib = number;
            else if (a == "--conversation-cache-disk-mib") o.conversation_cache_disk_mib = number;
            else o.conversation_cache_slots = (int) number;
        }
        else if (a == "--prompt-cache-every") o.prompt_cache_every = std::max(0LL, std::atoll(next("--prompt-cache-every")));
        else if (a == "--prompt-cache-root") o.prompt_cache_root = std::max(0LL, std::atoll(next("--prompt-cache-root")));
        else if (a == "--turn-token") o.turn_token = std::atoll(next("--turn-token"));
        else if (a == "--tail-role-token") o.tail_role_token = std::atoll(next("--tail-role-token"));
        else if (a == "--short-read") o.short_read = std::max(0LL, std::atoll(next("--short-read")));
        else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(next("--suffix-draft")));
        else if (a == "--mtp-max-t") o.mtp_max_t = std::max(0, std::atoi(next("--mtp-max-t")));
        else if (a == "--control-vector") o.cvec_files.push_back({next("--control-vector"), 1.0f});
        else if (a == "--control-vector-scaled") {
            // FILE:SCALE, comma-separated; the LAST colon splits
            std::stringstream list(next("--control-vector-scaled"));
            std::string item;
            while (std::getline(list, item, ',')) {
                const size_t colon = item.rfind(':');
                char* end = nullptr;
                const float sc = colon == std::string::npos ? 0.0f : std::strtof(item.c_str() + colon + 1, &end);
                if (colon == std::string::npos || colon == 0 || end == item.c_str() + colon + 1 || *end != '\0') {
                    basalt::log::error("basalt", "--control-vector-scaled: expected FILE:SCALE, got '%s'", item.c_str());
                    return 2;
                }
                o.cvec_files.push_back({item.substr(0, colon), sc});
            }
        }
        else if (a == "--control-vector-layer-range") {
            o.cvec_first = std::atoi(next("--control-vector-layer-range"));
            o.cvec_last = std::atoi(next("--control-vector-layer-range"));
        }
        else if (a == "--cvec-mode") {
            const std::string m = next("--cvec-mode");
            if (m == "project") o.cvec_mode = 0;
            else if (m == "add") o.cvec_mode = 1;
            else { basalt::log::error("basalt", "--cvec-mode: add or project, got '%s'", m.c_str()); return 2; }
        }
        else if (a == "--cvec-dir") {
            const std::string d = next("--cvec-dir");
            if (d == "per-layer") o.cvec_single = -1;
            else if (d.rfind("single:", 0) == 0) o.cvec_single = std::atoi(d.c_str() + 7);
            else { basalt::log::error("basalt", "--cvec-dir: per-layer or single:L, got '%s'", d.c_str()); return 2; }
        }
        else if (a == "--eos-ids") {
            std::string e;
            if (!parse_i64_list(next("--eos-ids"), o.eos_ids, e)) { basalt::log::error("basalt", "--eos-ids: %s", e.c_str()); return 2; }
            o.stop_eos = true;
        }
        else if (a == "--adapt-swaps") o.adapt_swaps = std::atoi(next("--adapt-swaps"));
        else if (a == "--peer-exclusive") { o.peer_exclusive = true; o.peer_exclusive_given = true; }
        else if (a == "--peer-reserve-mib") o.peer_reserve_mib = std::atoi(next("--peer-reserve-mib"));
        else if (a == "--peer-adapt-swaps") o.peer_adapt_swaps = std::atoi(next("--peer-adapt-swaps"));
        else if (a == "--peer-prefill-rows") o.peer_prefill_rows = std::atoll(next("--peer-prefill-rows"));
        else if (a == "--expert-profile-save") o.expert_profile_save = next("--expert-profile-save");
        else if (a == "--expert-profile-save-every")
            o.expert_profile_save_min = std::atof(next("--expert-profile-save-every"));
        else if (a == "--gpu-stages") o.gpu_stages = true;
        else if (a == "--mmap-experts") o.mmap_experts = true;
        else if (a == "--shared-expert-arena") o.shared_expert_arena = next("--shared-expert-arena");
        else if (a == "--resident-cpu-experts") o.resident_cpu_experts = o.resident_cpu_explicit = true;
        else if (a == "--resident-experts") {
            o.mmap_experts = o.resident_cpu_experts = o.resident_pin = o.resident_soft = true;
            o.resident_headroom = 0;
        }
        else if (a == "--resident-budget-gib") {
            const double gib = std::atof(next("--resident-budget-gib"));
            if (!(gib > 0.0)) { basalt::log::error("basalt generate", "--resident-budget-gib needs N > 0"); return 2; }
            o.resident_budget = (uint64_t) (gib * 1073741824.0);
            o.resident_budget_given = true;
            o.mmap_experts = o.resident_cpu_experts = o.resident_pin = true;
            o.resident_headroom = 0;
        }
        else if (a == "--stats") o.stats = true;
        else if (a == "--profile") basalt::core::diagnostics().profile = true;
        else if (a == "--wait-stats") basalt::core::diagnostics().wait_stats = true;
        else if (a == "--decode-timing") basalt::core::diagnostics().decode_timing = true;
        else if (a == "--draft-log") basalt::core::diagnostics().draft_log = next("--draft-log");
        else if (a == "--exp") {
            const std::string list = next("--exp");   // NAME=VALUE[,NAME=VALUE...]
            for (size_t at = 0; at < list.size();) {
                size_t end = list.find(',', at);
                if (end == std::string::npos) end = list.size();
                const std::string kv = list.substr(at, end - at);
                at = end + 1;
                const size_t eq = kv.find('=');
                if (eq == std::string::npos || eq == 0) { basalt::log::error("basalt generate", "--exp needs NAME=VALUE"); return 2; }
                basalt::core::diagnostics().exp[kv.substr(0, eq)] = std::atof(kv.c_str() + eq + 1);
            }
        }
        else if (a == "--ppl-base") ppl_base = next("--ppl-base");
        else if (a == "--ppl-out") ppl_out = next("--ppl-out");
        else {
            // An unknown flag is an ERROR and not a warning: a typo'd `--max-neww` that silently generated 16
            // tokens would look like a working run.
            basalt::log::error("basalt", "unknown argument: %s", a.c_str());
            usage();
            return 2;
        }
        }
    }
    return std::nullopt;
}

}  // namespace basalt::program::cli
