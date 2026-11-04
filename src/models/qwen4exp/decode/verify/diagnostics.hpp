#pragma once

#include "common.hpp"
#include "basalt/core/verify.hpp"
#include <atomic>
#include <cstdio>
#include <string>

namespace basalt::core::verify_detail {
constexpr int kLiveMax = 16;
extern std::atomic<Verifier*> g_live[kLiveMax];
extern std::atomic<const Verifier*> g_diag_verifier;
void diag_active_verifier(std::FILE* f);
void release_live_verifiers(std::FILE* f);
std::string released_note(bool drained);
}
