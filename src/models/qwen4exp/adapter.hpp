#pragma once
#include "basalt/model/adapter.hpp"

namespace basalt::models::qwen4exp {
std::unique_ptr<model::ModelAdapter> make_adapter();
}
