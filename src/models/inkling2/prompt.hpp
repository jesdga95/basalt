// src/models/inkling2/prompt.hpp - the prompt path of an Inkling 2 session: reads of more rows than a decode window, into
// the session's caches.
#pragma once

#include "basalt/model/adapter.hpp"
#include "models/inkling2/caches.hpp"
#include "models/inkling2/model.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::inkling2 {

class PromptRunner {
public:
    virtual ~PromptRunner() = default;
    /// Reads r.tokens at r.pos.. into the caches; `hist` takes each token read.
    virtual model::Result read(const model::PrefillRequest& r, std::vector<int32_t>& hist) = 0;
    /// Reads `n` embedding rows (host fp32 [n][n_embd], taken as they are: no embedding norm) at pos..; `hist` takes `pad`
    /// for each row.
    virtual model::Result read_rows(const float* rows, int n, int64_t pos, int32_t pad, std::vector<int32_t>& hist) = 0;
};

/// The prompt path over a session's caches and stream; null when the build has none (the session then reads prompts a
/// decode window at a time).
std::unique_ptr<PromptRunner> make_prompt(InklingModel& m, const std::vector<VisitCache>& caches, void* stream,
                                          std::string& err);

/// The device bytes the prompt path adds to a session.
uint64_t prompt_bytes(const Geometry& g);

}  // namespace basalt::models::inkling2
