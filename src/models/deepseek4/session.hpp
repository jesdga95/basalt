#pragma once

#include "basalt/model/adapter.hpp"
#include "models/deepseek4/model.hpp"

namespace basalt::models::deepseek4 {

std::unique_ptr<model::ModelSession> make_session_impl(DsModel& model, std::string& err);

}  // namespace basalt::models::deepseek4
