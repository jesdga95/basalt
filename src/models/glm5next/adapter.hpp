// src/models/glm5next/adapter.hpp - the glm5next adapter contract: the factory the engine's model registry calls.
// adapter.cpp holds what the model says about its tokenizer, tool grammar and GPU modes.
#pragma once

#include "basalt/model/adapter.hpp"

#include <memory>

namespace basalt::models::glm5next {

// The architecture names live in meta.hpp, which uses them to read the pack's `<arch>.*` keys and to refuse a file
// whose `general.architecture` is neither of them.

std::unique_ptr<model::ModelAdapter> make_adapter();

}  // namespace basalt::models::glm5next
