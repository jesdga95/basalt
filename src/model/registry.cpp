// Adapter registry: architecture name -> ModelAdapter.
#include "basalt/model/adapter.hpp"

#include "basalt/artifact/gguf_reader.hpp"

#include <map>

namespace basalt::model {

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
