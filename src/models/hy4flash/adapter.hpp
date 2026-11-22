#pragma once
#include "basalt/model/adapter.hpp"

namespace basalt::models::hy4flash {
std::unique_ptr<model::ModelAdapter> make_adapter();
void pretok_hy4(const char* text, size_t size, std::vector<size_t>& ends);
}
