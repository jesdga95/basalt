// tests/model/meta_adapters.hpp - registers an adapter per architecture that has the model's metadata rules and nothing
// else, so a CPU test links no model and no runtime.
#pragma once

#include "basalt/model/adapter.hpp"
#include "models/deepseek41/meta.hpp"
#include "models/glm5next/meta.hpp"
#include "models/qwen4exp/meta.hpp"

#include <memory>
#include <string>

namespace basalt::test {

class MetaAdapter final : public model::ModelAdapter {
public:
    using Validate = std::string (*)(const GgufFile&);
    using Describe = model::ModelDesc (*)(const GgufFile&);
    MetaAdapter(const char* arch, Validate validate, Describe describe) : arch_(arch), validate_(validate), describe_(describe) {}
    const char* arch() const override { return arch_; }
    std::string validate(const GgufFile& g) const override { return validate_(g); }
    model::ModelDesc describe(const GgufFile& g) const override { return describe_(g); }
    model::TokenizerSpec tokenizer_spec() const override { return {}; }
    program::GpuModes gpu_modes() const override { return {}; }
    model::TokenNames token_names() const override { return {}; }
    PreTokenizer pretokenizer() const override { return {}; }
    program::grammar::ToolForm tool_form() const override { return {}; }

private:
    const char* arch_;
    Validate validate_;
    Describe describe_;
};

inline void register_meta_adapters() {
    if (model::find_adapter("qwen4exp") == nullptr)
        model::register_adapter(std::make_unique<MetaAdapter>("qwen4exp", models::qwen4exp::validate_meta, models::qwen4exp::describe_meta));
    if (model::find_adapter("deepseek41") == nullptr)
        model::register_adapter(std::make_unique<MetaAdapter>("deepseek41", models::deepseek41::validate_meta, models::deepseek41::describe_meta));
    // A GLM pack's gguf_kv keeps the source GGUF's architecture, which is what the registry looks an adapter up by; the
    // same rules answer to both names, as they do for hy4flash.
    if (model::find_adapter(models::glm5next::kGgufArch) == nullptr)
        model::register_adapter(std::make_unique<MetaAdapter>(models::glm5next::kGgufArch, models::glm5next::validate_meta,
                                                             models::glm5next::describe_meta));
    if (model::find_adapter("glm5next") == nullptr)
        model::register_adapter(std::make_unique<MetaAdapter>("glm5next", models::glm5next::validate_meta, models::glm5next::describe_meta));
}

}  // namespace basalt::test
