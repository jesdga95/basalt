// basalt/program/measure/dump_set.hpp - the activation dumps that run once, after a request.
//
// `--dump-mixed` writes the head's input and `--dump-residual` the final residual: both read one buffer back from the
// device and write it with a small header, and both exist so a Python side can bisect the head against the layers.
// They run at the end of a request, touch nothing else, and are the last of the dump modes to leave `generate.cpp`.
#pragma once

#include <string>

namespace basalt::core {
struct ModelGeometry;
struct SessionState;
}  // namespace basalt::core

namespace basalt::program {

struct Options;

namespace measure {

struct HostDumpContext {
    const Options* o = nullptr;
    const basalt::core::ModelGeometry* g = nullptr;
    const basalt::core::SessionState* ss = nullptr;
};

/// `--dump-mixed`, `--dump-residual`: read the activation back and write it.  Returns 0, or the exit code on failure
/// (the failure line is logged here, exactly as it was logged inline).
int write_host_dumps(const HostDumpContext& c, std::string& err);

}  // namespace measure
}  // namespace basalt::program
