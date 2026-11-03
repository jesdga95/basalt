// src/models/qwen35/adapter.hpp - the qwen35 (Qwen3.5-family dense hybrid) adapter.
#pragma once

#include "basalt/model/adapter.hpp"

#include <memory>

namespace basalt::models::qwen35 {
std::unique_ptr<model::ModelAdapter> make_adapter();
}  // namespace basalt::models::qwen35
