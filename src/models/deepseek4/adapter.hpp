#pragma once

#include "basalt/model/adapter.hpp"

namespace basalt::models::deepseek4 {

std::unique_ptr<model::ModelAdapter> make_adapter();

}  // namespace basalt::models::deepseek4
