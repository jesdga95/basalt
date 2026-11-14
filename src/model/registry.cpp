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

// BASALT_KNOWN_MODELS (every model Basalt ships) and BASALT_BUILT_MODELS (the ones this build holds) are comma lists
// that cmake/models.cmake bakes in.
std::vector<std::string> models_of(const std::string& csv) {
    std::vector<std::string> v;
    for (size_t i = 0; i < csv.size();) {
        const size_t j = std::min(csv.find(',', i), csv.size());
        v.push_back(csv.substr(i, j - i));
        i = j + 1;
    }
    return v;
}

bool contains(const std::vector<std::string>& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); }
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

std::string unavailable_architecture(const std::string& arch) {
    const std::vector<std::string> known = models_of(BASALT_KNOWN_MODELS);
    const std::vector<std::string> built = models_of(BASALT_BUILT_MODELS);
    if (contains(known, arch) && !contains(built, arch)) {
        // a Basalt model this build left out: what the build holds, and the option that adds this one
        std::string held, rebuild;
        for (const std::string& m : built) held += (held.empty() ? "" : ";") + m;
        for (const std::string& m : known)
            if (m == arch || contains(built, m)) rebuild += (rebuild.empty() ? "" : ";") + m;
        return "architecture is '" + arch + "', a Basalt model this build left out (built with BASALT_MODELS=" + held +
               "); rebuild with cmake -DBASALT_MODELS=\"" + rebuild + "\" to load it";
    }
    std::string registered;
    for (const std::string& n : adapter_names()) registered += (registered.empty() ? "" : ", ") + n;
    return "architecture is '" + arch + "', no model adapter is registered for it (known: " +
           (registered.empty() ? "none" : registered) + ")";
}

const ModelAdapter* adapter_for(const GgufFile& meta, std::string& err) {
    const MetaValue* arch = meta.get("general.architecture");
    if (!arch) {
        err = "missing general.architecture";
        return nullptr;
    }
    const ModelAdapter* a = find_adapter(arch->s);
    if (!a) err = unavailable_architecture(arch->s);
    return a;
}

std::string check_architecture(const GgufFile& meta) {
    std::string err;
    const ModelAdapter* a = adapter_for(meta, err);
    return a ? a->validate(meta) : err;
}

}  // namespace basalt::model
