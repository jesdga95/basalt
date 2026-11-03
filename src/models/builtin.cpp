#include "basalt/models/builtin.hpp"

#include "basalt/model/adapter.hpp"

#include <memory>

// One BASALT_BUILT_MODEL(<arch>) line per model in BASALT_MODELS, written by cmake/models.cmake. Each model's adapter
// factory is basalt::models::<arch>::make_adapter(), declared here so the model's headers stay out of this file.
namespace basalt::models {
#define BASALT_BUILT_MODEL(arch) \
    namespace arch { std::unique_ptr<model::ModelAdapter> make_adapter(); }
#include "builtin_models.inc"
#undef BASALT_BUILT_MODEL

void register_builtin_models() {
#define BASALT_BUILT_MODEL(arch) \
    if (model::find_adapter(#arch) == nullptr) model::register_adapter(arch::make_adapter());
#include "builtin_models.inc"
#undef BASALT_BUILT_MODEL
}

}  // namespace basalt::models
