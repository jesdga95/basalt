// src/models/inkling2/prompt.cpp - the prompt path of an Inkling 2 session. None yet: prompts are read a decode window at
// a time.
#include "models/inkling2/prompt.hpp"

namespace basalt::models::inkling2 {

std::unique_ptr<PromptRunner> make_prompt(InklingModel&, const std::vector<VisitCache>&, void*, std::string&) {
    return nullptr;
}

uint64_t prompt_bytes(const Geometry&) { return 0; }

}  // namespace basalt::models::inkling2
