#pragma once

#include "basalt/core/expert_tier.hpp"
#include "basalt/model/adapter.hpp"
#include "models/deepseek4/pack.hpp"

#include <string>

namespace basalt::models::deepseek4 {

/// Build the shared tier's per-layer expert layout and copy the load's tier/routing controls.
/// `experts_fd` is an open descriptor for the pack; its lifetime must cover the ExpertTier.
bool make_expert_tier_spec(const Pack& pack, const std::string& path, int experts_fd,
                           const model::LoadRequest& request, core::ExpertTierSpec& spec, std::string& err);

}  // namespace basalt::models::deepseek4
