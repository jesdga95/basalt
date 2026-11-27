// src/models/ernie55/session.hpp - per-conversation session factory for ERNIE-5.5-Flash.
#pragma once

#include "models/ernie55/model.hpp"

#include <memory>
#include <string>

namespace basalt::models::ernie55 {

std::unique_ptr<model::ModelSession> make_session(ErnieModel& model, std::string& err);

}  // namespace basalt::models::ernie55
