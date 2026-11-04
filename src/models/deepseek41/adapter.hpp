#pragma once
#include "basalt/model/adapter.hpp"

namespace basalt::models::deepseek41 {
std::unique_ptr<model::ModelAdapter> make_adapter();
void pretok_joyai(const char* text, size_t size, std::vector<size_t>& ends);
}
