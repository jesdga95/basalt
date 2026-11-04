// include/basalt/models/builtin.hpp - the composition root: registers every model adapter that ships.
// Programs call it once at startup; shared libraries never do.
#pragma once

namespace basalt::models {
void register_builtin_models();
}
