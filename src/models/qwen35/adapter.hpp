// src/models/qwen35/adapter.hpp - the qwen35 (Qwen3.5-family dense hybrid) adapter.
#pragma once

#include "basalt/model/adapter.hpp"

#include <memory>

namespace basalt::models::qwen35 {
std::unique_ptr<model::ModelAdapter> make_adapter();
/// The routed-expert variant (`qwen35moe`: the same hybrid layers with a routed-expert FFN), registered as `qwen35moe` from this library.
std::unique_ptr<model::ModelAdapter> make_moe_adapter();
}  // namespace basalt::models::qwen35
