// src/models/qwen35moe/adapter.cpp - the qwen35moe (Qwen3.5-family hybrid with routed-expert FFN layers) model: the dense stack
// and the qwen35 adapter's code with its routed-expert variant.
#include "models/qwen35/adapter.hpp"

namespace basalt::models::qwen35moe {

std::unique_ptr<model::ModelAdapter> make_adapter() { return qwen35::make_moe_adapter(); }

}  // namespace basalt::models::qwen35moe
