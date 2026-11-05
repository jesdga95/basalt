// Adapter registry: architecture name -> ModelAdapter.
#include "basalt/model/adapter.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/program/options.hpp"

#include <algorithm>
#include <map>

namespace basalt::model {

std::unique_ptr<program::CommonOptions> ModelAdapter::make_options() const {
    return std::make_unique<program::CommonOptions>();
}

int ModelAdapter::planned_spec(const program::CommonOptions& o) const { return o.spec >= 2 ? o.spec : 0; }

namespace {
std::map<std::string, std::unique_ptr<ModelAdapter>>& table() {
    static std::map<std::string, std::unique_ptr<ModelAdapter>> t;
    return t;
}
}  // namespace

void register_adapter(std::unique_ptr<ModelAdapter> a) {
    const std::string name = a->arch();
    table()[name] = std::move(a);
}

const ModelAdapter* find_adapter(const std::string& arch) {
    const auto it = table().find(arch);
    return it == table().end() ? nullptr : it->second.get();
}

int max_expert_cards() {
    int most = 0;
    for (const std::string& name : adapter_names()) most = std::max(most, find_adapter(name)->max_expert_cards());
    return most;
}

std::vector<std::string> adapter_names() {
    std::vector<std::string> v;
    for (const auto& kv : table()) v.push_back(kv.first);
    return v;
}

const ModelAdapter* adapter_for(const GgufFile& meta, std::string& err) {
    const MetaValue* arch = meta.get("general.architecture");
    if (!arch) {
        err = "missing general.architecture";
        return nullptr;
    }
    const ModelAdapter* a = find_adapter(arch->s);
    if (!a) {
        std::string known;
        for (const auto& n : adapter_names()) known += (known.empty() ? "" : ", ") + n;
        err = "architecture is '" + arch->s + "', no model adapter is registered for it (known: " +
              (known.empty() ? "none" : known) + ")";
    }
    return a;
}

std::string check_architecture(const GgufFile& meta) {
    std::string err;
    const ModelAdapter* a = adapter_for(meta, err);
    return a ? a->validate(meta) : err;
}

}  // namespace basalt::model
