// src/models/ernie55/adapter.hpp - ERNIE-5.5-Flash adapter factory and pre-tokenizer entry point.
#pragma once

#include "basalt/model/adapter.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace basalt::models::ernie55 {

std::unique_ptr<model::ModelAdapter> make_adapter();

/// Pre-tokenizer matching ERNIE 5.5's tokenizer.json Sequence (right-aligned 3-digit runs + CJK + word/punct/space).
void pretok_ernie55(const char* text, size_t size, std::vector<size_t>& ends);

}  // namespace basalt::models::ernie55
