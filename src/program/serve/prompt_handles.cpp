// src/program/serve/prompt_handles.cpp - prompts held by handle, and what the idle slots hold of them (--batch).
#include "prefix_anchor.hpp"
#include "state.hpp"

namespace basalt::program::serve {

const std::vector<int32_t>* Runtime::idle_ids(int b) const {
    if (b < 0 || (size_t) b >= bs.size() || bs[(size_t) b].active) return nullptr;
    // slot 0 beside a solo session holds that session's conversation in place
    if (b == 0 && slot0_solo() && live_ok && !live.empty()) return &live;
    return bs[(size_t) b].cached && !bs[(size_t) b].ids.empty() ? &bs[(size_t) b].ids : nullptr;
}

std::vector<Runtime::SlotMeet> Runtime::slot_meets(const std::vector<int32_t>& prompt, bool media) const {
    std::vector<SlotMeet> out(bs.size());
    const auto bytes = [this](int32_t id) { return token_bytes(id); };
    for (size_t b = 0; b < bs.size(); ++b) {
        const std::vector<int32_t>* h = idle_ids((int) b);
        if (h == nullptr) continue;
        out[b].held = (int64_t) h->size();
        out[b].match = meet_held(prompt.data(), (int64_t) prompt.size(), h->data(), (int64_t) h->size(),
                                 !media && text_tokenizer_note.empty(), bytes).held;
    }
    return out;
}

uint64_t Runtime::hold_prompt(HeldPrompt&& p) {
    std::lock_guard<std::mutex> lk(gent_mu);
    const uint64_t handle = ++gent_seq;
    while (prompts.size() >= kPromptsMax) prompts.erase(prompts.begin());   // a request that never freed its prompt
    prompts[handle] = std::move(p);
    return handle;
}

void Runtime::take_prompt_ids(const std::string& cmd) {
    HeldPrompt p;
    const char* at = cmd.c_str() + 7;
    while (*at == ' ') ++at;
    if (std::strncmp(at, "img", 3) == 0) {
        p.media = true;
        at += 3;
    }
    std::vector<int64_t> v;
    std::string pe;
    if (!cli::parse_i64_list(at, v, pe)) {
        std::printf("PROMPT ERR bad request: %s\n", pe.c_str());
        std::fflush(stdout);
        return;
    }
    p.ids.assign(v.begin(), v.end());
    const size_t n = p.ids.size();
    const uint64_t handle = hold_prompt(std::move(p));
    std::printf("PROMPT %llu %zu\n", (unsigned long long) handle, n);
    std::fflush(stdout);
}

void Runtime::prompt_command(const std::string& cmd) {
    const uint64_t handle = std::strtoull(cmd.c_str() + 6, nullptr, 10);
    if (cmd.rfind("PFREE ", 0) == 0) {
        std::lock_guard<std::mutex> lk(gent_mu);
        prompts.erase(handle);
        return;
    }
    HeldPrompt p;
    {
        std::lock_guard<std::mutex> lk(gent_mu);
        const auto it = prompts.find(handle);
        if (it == prompts.end()) {
            std::printf("ERR the held prompt is gone\n");
            std::fflush(stdout);
            return;
        }
        p = it->second;
    }
    std::string reply = "PMATCH " + std::to_string((unsigned long long) handle);
    for (const SlotMeet& m : slot_meets(p.ids, p.media))
        reply += " " + std::to_string((long long) m.match) + ":" + std::to_string((long long) m.held);
    std::printf("%s\n", reply.c_str());
    std::fflush(stdout);
}

}  // namespace basalt::program::serve
