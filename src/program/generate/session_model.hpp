#pragma once

#include "basalt/model/adapter.hpp"
#include "basalt/program/options.hpp"

namespace basalt::program::generate {

/// A model served through its sessions, to its exit code.
int run_session_model(basalt::program::CommonOptions& o, const basalt::model::ModelAdapter& adapter);

}  // namespace basalt::program::generate
