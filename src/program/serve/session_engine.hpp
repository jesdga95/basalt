// src/program/serve/session_engine.hpp - serving a model that offers only a ModelSession.
#pragma once

#include "request_base.hpp"
#include "basalt/model/adapter.hpp"

namespace basalt::program::serve {

/// Serves until QUIT or stdin closes; returns the process exit code.
int run_session(CommonOptions& o, basalt::core::ModelFile& file, basalt::model::LoadedModel& model);

}  // namespace basalt::program::serve
