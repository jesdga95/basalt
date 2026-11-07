#include "arguments.hpp"
#include "help.hpp"
#include "tokens.hpp"

#include "basalt/log.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/program/options.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <sstream>

namespace basalt::program::cli {
std::optional<int> parse_options(int argc, char** argv, CommonOptions& o, const model::ModelAdapter* adapter) {
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
        else if (a == "--max-context") { o.max_context = std::atoll(next("--max-context")); }
        else if (a == "--diag-dump") o.diag_dump = next("--diag-dump");
        else if (a == "--diag-tier-ram") o.diag_tier_ram = std::atoi(next("--diag-tier-ram"));
        else if (a == "--diag-tier-vram") o.diag_tier_vram = std::atoi(next("--diag-tier-vram"));
        else if (a == "--diag-serial-prompt") o.diag_serial_prompt = true;
        else if (a == "--diag-logits") o.diag_logits = next("--diag-logits");
        else if (a == "--image") o.images.push_back(next("--image"));
        else if (a == "--image-at") o.image_at.push_back(std::atoll(next("--image-at")));
        else if (a == "--image-pad-id") { o.image_pad_id = std::atoll(next("--image-pad-id")); o.image_pad_given = true; }
        else if (a == "--greedy") o.greedy = true;
        else if (a == "--seed") { o.seed = (uint64_t) std::atoll(next("--seed")); o.greedy = false; }
        else if (a == "--top-k") o.top_k = std::atoi(next("--top-k"));
        else if (a == "--top-p") o.top_p = (float) std::atof(next("--top-p"));
        else if (a == "--temperature") o.temperature = (float) std::atof(next("--temperature"));
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
        // The chain continues in a second statement: the compiler limits how deeply `else if` blocks may nest.
        if (!parsed) {
        if (a == "--expert-cache") {
            const std::string v = next("--expert-cache");
            o.expert_cache = (v == "auto") ? -1 : std::atoi(v.c_str());
        }
        else if (a == "--gpu") o.gpu = next("--gpu");
        else if (a == "--gpu-plan") o.gpu_plan = true;
        else if (a == "--ppl-base") o.ppl_base = next("--ppl-base");
        else if (a == "--ppl-out") o.ppl_out = next("--ppl-out");
        else if (a == "--vram-reserve-mib") { o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib")); o.vram_reserve_given = true; }
        else if (a == "--batch") o.batch = std::atoi(next("--batch"));
        else if (a == "--slots") o.batch = std::atoi(next("--slots"));   // the same as --batch
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
                 a == "--conversation-cache-min-free-mib" || a == "--conversation-cache-disk-mib" ||
                 a == "--conversation-cache-min-tokens" || a == "--ram-reserve-mib") {
            const std::string value = next(a.c_str());
            int64_t number = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
            const int64_t limit = a == "--conversation-cache-slots" || a == "--conversation-cache-min-tokens"
                                  ? INT32_MAX : INT64_MAX / (1024 * 1024);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || number < 0 || number > limit) {
                basalt::log::error("basalt", "%s needs a nonnegative integer within range", a.c_str());
                return 2;
            }
            if (a == "--conversation-cache-mib") o.conversation_cache_mib = number;
            else if (a == "--conversation-cache-min-free-mib") o.conversation_cache_min_free_mib = number;
            else if (a == "--conversation-cache-disk-mib") o.conversation_cache_disk_mib = number;
            else if (a == "--conversation-cache-min-tokens") o.conversation_cache_min_tokens = (int) number;
            else if (a == "--ram-reserve-mib") o.ram_reserve_mib = number;
            else o.conversation_cache_slots = (int) number;
        }
        else if (a == "--prompt-cache-root") o.prompt_cache_root = std::max(0LL, std::atoll(next("--prompt-cache-root")));
        else if (a == "--turn-token") { o.turn_token = std::atoll(next("--turn-token")); o.turn_token_given = true; }
        else if (a == "--tail-role-token") o.tail_role_token = std::atoll(next("--tail-role-token"));
        else if (a == "--eos-ids") {
            std::string e;
            if (!parse_i64_list(next("--eos-ids"), o.eos_ids, e)) { basalt::log::error("basalt", "--eos-ids: %s", e.c_str()); return 2; }
            o.stop_eos = true;
        }
        else if (a == "--peer-reserve-mib") o.peer_reserve_mib = std::atoi(next("--peer-reserve-mib"));
        else if (a == "--expert-profile") o.expert_profile = next("--expert-profile");
        else if (a == "--resident-budget-gib") {
            const double gib = std::atof(next("--resident-budget-gib"));
            if (!(gib > 0.0)) { basalt::log::error("basalt generate", "--resident-budget-gib needs N > 0"); return 2; }
            o.resident_budget = (uint64_t) (gib * 1073741824.0);
            o.resident_budget_given = true;
        }
        else {
            // An unknown flag is an ERROR and not a warning: a typo'd `--max-neww` that silently generated 16
            // tokens would look like a working run.  A flag another model takes is unknown to this one.
            int code = 0;
            model::FlagArgs args(argc, argv, i);
            if (adapter == nullptr || !adapter->parse_flag(a, args, o, code)) {
                basalt::log::error("basalt", "unknown argument: %s", a.c_str());
                usage();
                return 2;
            }
            if (code != 0) return code;
        }
        }
    }
    basalt::platform::set_ram_reserve((uint64_t) o.ram_reserve_mib << 20);
    if (o.serve && o.ram_reserve_mib == 0 && !o.gpu_plan)
        basalt::log::warn("basalt", "RAM guard off (--ram-reserve-mib 0): low RAM can stop the engine or the machine");
    if (adapter != nullptr) return adapter->finish_flags(o);
    return std::nullopt;
}

}  // namespace basalt::program::cli
