// src/models/nemotronh/adapter.hpp - the Nemotron-H adapter's entry points.
#pragma once

#include "basalt/model/adapter.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace basalt::models::nemotronh {

std::unique_ptr<model::ModelAdapter> make_adapter();
/// The nemotronh pre-tokenizer (Mistral's tekken split): `ends` gets where each piece of the text ends.
void pretok_nemotronh(const char* text, size_t size, std::vector<size_t>& ends);

}  // namespace basalt::models::nemotronh
