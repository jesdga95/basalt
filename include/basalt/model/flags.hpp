// include/basalt/model/flags.hpp - the argument cursor a model's flags read their values from.
#pragma once

#include "basalt/log.hpp"

#include <cstdlib>

namespace basalt::model {

class FlagArgs {
public:
    FlagArgs(int argc, char** argv, int& i) : argc_(argc), argv_(argv), i_(i) {}
    /// The next argument, the value of flag `what`; a flag without one ends the program.
    const char* next(const char* what) {
        if (i_ + 1 >= argc_) {
            basalt::log::error("basalt", "%s needs a value", what);
            std::exit(2);
        }
        return argv_[++i_];
    }
    /// The next argument, for a flag whose value is optional, or null; take() consumes it.
    const char* peek() const { return i_ + 1 < argc_ ? argv_[i_ + 1] : nullptr; }
    void take() { ++i_; }

private:
    int argc_;
    char** argv_;
    int& i_;
};

}  // namespace basalt::model
