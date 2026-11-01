#pragma once

#include "basalt/core/model_adapter.hpp"
#include "basalt/core/weights.hpp"

#include <memory>
#include <set>

namespace basalt::core {

/// Fixed-address weight replacement between drained requests; model zero is the base.
class RuntimeAdapters {
public:
    RuntimeAdapters();
    ~RuntimeAdapters();
    RuntimeAdapters(const RuntimeAdapters&) = delete;
    RuntimeAdapters& operator=(const RuntimeAdapters&) = delete;

    bool open(const ModelFile& base, const std::vector<std::vector<std::string>>& groups, std::string& err);
    size_t model_count() const;
    const std::set<std::string>& names() const;
    const AdapterTensor* source(size_t model, const std::string& name) const;
    bool changed(size_t from, size_t to, const std::string& name) const;
    /// Bind each stage after native and derived forms are installed, before cache sizing.
    bool bind_weights(WeightTable& weights, const PackFiles& pack, int device, std::string& err);
    bool select(size_t model, std::string& err);
    size_t current() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace basalt::core
