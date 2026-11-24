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

class DsparkHook;

/// The most rows one prompt chunk runs (the drafter's tap holds as many).
constexpr int kPromptChunk = 4096;

class PromptRunner {
public:
    virtual ~PromptRunner() = default;
    /// Reads r.tokens at r.pos.. into a session's caches and drafter, on its stream; `hist` takes each token read.
    virtual model::Result read(const model::PrefillRequest& r, std::vector<int32_t>& hist, const std::vector<VisitCache>& caches,
                               DsparkHook& dspark, void* stream) = 0;
    /// Reads `n` embedding rows (host fp32 [n][n_embd], taken as they are: no embedding norm) at pos..; `hist` takes `pad`
    /// for each row.
    virtual model::Result read_rows(const float* rows, int n, int64_t pos, int32_t pad, std::vector<int32_t>& hist,
                                    const std::vector<VisitCache>& caches, DsparkHook& dspark, void* stream) = 0;
};

/// The prompt path every session of the model shares (one read at a time; each read syncs before it returns), set up on
/// `stream`; null when the build has none (a session then reads prompts a decode window at a time). With the drafter on, a
/// chunk fills the tap (up to kPromptChunk rows) and commits its rows' context, as a decode window does.
std::unique_ptr<PromptRunner> make_prompt(InklingModel& m, void* stream, std::string& err);

/// The device bytes of the prompt path (one for every session).
uint64_t prompt_bytes(const Geometry& g);

}  // namespace basalt::models::inkling2
