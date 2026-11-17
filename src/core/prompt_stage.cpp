// src/core/prompt_stage.cpp - see include/basalt/core/prompt_stage.hpp.
#include "basalt/core/prompt_stage.hpp"

#include "basalt/log.hpp"

#include <cstdio>

namespace basalt::core {

void log_prompt_split(const char* area, const std::vector<PromptStage>& stages) {
    std::string t;
    for (const PromptStage& st : stages) {
        char b[128];
        std::snprintf(b, sizeof b, "%s%d-%d on CUDA%d (%.2f GiB %s)", t.empty() ? "" : ", ", st.lo, st.hi - 1, st.device,
                      (double) st.bytes / 1073741824.0, st.home == StageHome::Card ? "on the card" : "in page-locked RAM");
        t += b;
    }
    basalt::log::info(area, "prompt split: layers %s", t.c_str());
}

void log_prompt_split_off(const char* area, const std::string& why, bool lost) {
    if (lost) basalt::log::warn(area, "prompt split: off (%s)", why.c_str());
    else basalt::log::info(area, "prompt split: off (%s)", why.c_str());
}

}  // namespace basalt::core
