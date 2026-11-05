// src/models/deepseek41/session.hpp - one conversation's state and the single-token decode loop.
#pragma once

#include "basalt/model/adapter.hpp"

#include <memory>
#include <string>

namespace basalt::models::deepseek41 {

class DsModel;

/// A session over `model`; null with `err` set when the layer ops cannot be built.
std::unique_ptr<model::ModelSession> make_session_impl(DsModel& model, std::string& err);

}  // namespace basalt::models::deepseek41
