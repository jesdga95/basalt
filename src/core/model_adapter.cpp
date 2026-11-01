#include "basalt/core/model_adapter.hpp"

#include "basalt/artifact/gguf_reader.hpp"

#include <initializer_list>
#include <limits>
#include <utility>

namespace basalt::core {
namespace {

bool fields(const Json* value, std::initializer_list<const char*> names) {
    if (!value || !value->is_object() || value->members.size() != names.size()) return false;
    for (const char* name : names) {
        if (!value->get(name)) return false;
    }
    return true;
}

bool uint_value(const Json* value, uint64_t& out) {
    if (!value || value->type != Json::Type::Int || (value->neg && value->u != 0)) return false;
    out = value->u;
    return true;
}

bool string_value(const Json* value, const char* expected) {
    return value && value->is_string() && value->s == expected;
}

bool valid_name(const std::string& name) {
    return !name.empty() && name.find('\0') == std::string::npos;
}

}  // namespace

bool AdapterFile::open(const std::string& path, std::string& err) {
    replacements_.clear();
    err.clear();
    ModelFile file;
    if (!file.open(path, err) || !file.load_meta(err)) return false;
    const auto fail = [&](const std::string& message) {
        err = path + ": adapter: " + message;
        return false;
    };
    const Json& meta = file.meta();
    uint64_t version = 0;
    if (!string_value(meta.get("format"), "basalt") ||
        !uint_value(meta.get("version"), version) || version != 1 ||
        !string_value(meta.get("kind"), "adapter")) {
        return fail("expected basalt version 1 adapter metadata");
    }
    if (const Json* name = meta.get("name")) {
        if (!name->is_string() || name->s.find('\0') != std::string::npos) {
            return fail("name must be a string without NUL");
        }
    }
    const Json* adapter = meta.get("adapter");
    if (!fields(adapter, {"version", "entries"}) ||
        !uint_value(adapter->get("version"), version) || version != 1) {
        return fail("expected adapter version 1 and entries only");
    }
    const Json* entries = adapter->get("entries");
    if (!entries->is_object() || entries->members.empty()) return fail("entries must be a nonempty object");
    const ModelFile::Section* section = file.find("adapter/data");
    if (!section) return fail("missing adapter/data section");

    constexpr uint64_t max = (std::numeric_limits<uint64_t>::max)();
    std::map<std::string, AdapterTensor> replacements;
    for (const auto& item : entries->members) {
        const std::string& id = item.first;
        const Json& entry = item.second;
        const auto invalid = [&](const std::string& message) { return fail("entry " + id + ": " + message); };
        if (!valid_name(id)) return fail("entry IDs must be nonempty strings without NUL");
        if (!fields(&entry, {"target", "storage"})) return invalid("expected target and storage only");
        const Json* target = entry.get("target");
        const Json* storage = entry.get("storage");
        if (!fields(target, {"tensor", "shape", "region"})) return invalid("invalid target fields");
        const Json* tensor = target->get("tensor");
        if (!tensor->is_string() || !valid_name(tensor->s)) return invalid("invalid tensor name");
        const Json* region = target->get("region");
        if (!fields(region, {"kind"}) || !string_value(region->get("kind"), "whole")) {
            return invalid("only whole-tensor regions are supported");
        }
        if (!fields(storage, {"kind", "type_id", "section", "offset", "bytes"}) ||
            !string_value(storage->get("kind"), "replace") ||
            !string_value(storage->get("section"), "adapter/data")) {
            return invalid("only replace storage in adapter/data is supported");
        }
        uint64_t type_id = 0, offset = 0, bytes = 0;
        if (!uint_value(storage->get("type_id"), type_id) || type_id > (std::numeric_limits<uint32_t>::max)() ||
            !uint_value(storage->get("offset"), offset) || !uint_value(storage->get("bytes"), bytes)) {
            return invalid("type_id, offset and bytes must be unsigned integers");
        }
        int block_elements = 0, block_bytes = 0;
        if (!basalt::block_geometry(static_cast<uint32_t>(type_id), block_elements, block_bytes)) {
            return invalid("unsupported tensor type");
        }
        const Json* shape = target->get("shape");
        if (!shape->is_array() || shape->count < 1 || shape->count > 4) {
            return invalid("shape must have 1..4 dimensions");
        }
        AdapterTensor replacement;
        replacement.shape.reserve(static_cast<size_t>(shape->count));
        uint64_t elements = 1;
        for (const Json& extent : shape->items) {
            uint64_t dimension = 0;
            if (!uint_value(&extent, dimension) || dimension == 0 || elements > max / dimension) {
                return invalid("invalid or overflowing shape");
            }
            elements *= dimension;
            replacement.shape.push_back(dimension);
        }
        if (replacement.shape[0] % static_cast<uint64_t>(block_elements) != 0) {
            return invalid("fastest dimension must contain whole dtype blocks");
        }
        const uint64_t blocks = elements / static_cast<uint64_t>(block_elements);
        if (blocks > max / static_cast<uint64_t>(block_bytes) ||
            bytes != blocks * static_cast<uint64_t>(block_bytes)) {
            return invalid("bytes must equal the packed tensor size without overflow");
        }
        if (offset % kTensorAlign != 0 || offset > section->bytes || bytes > section->bytes - offset ||
            offset > max - section->offset || bytes > max - (section->offset + offset)) {
            return invalid("unaligned or out-of-bounds payload");
        }
        replacement.tensor = tensor->s;
        replacement.type_id = static_cast<uint32_t>(type_id);
        replacement.data = FileRange(path, section->offset + offset, bytes);
        if (!replacements.emplace(tensor->s, std::move(replacement)).second) {
            return invalid("duplicate target tensor");
        }
    }
    replacements_ = std::move(replacements);
    return true;
}

const AdapterTensor* AdapterFile::find(const std::string& tensor) const {
    const auto it = replacements_.find(tensor);
    return it == replacements_.end() ? nullptr : &it->second;
}

bool AdapterFile::read_tensor(const std::string& tensor, std::vector<uint8_t>& out, std::string& err) const {
    out.clear();
    err.clear();
    const AdapterTensor* replacement = find(tensor);
    if (!replacement) {
        err = "unknown adapter target: " + tensor;
        return false;
    }
    if (replacement->data.bytes > (std::numeric_limits<size_t>::max)()) {
        err = "adapter tensor is too large to read: " + tensor;
        return false;
    }
    if (!read_range(replacement->data, out, err)) {
        out.clear();
        return false;
    }
    return true;
}

}  // namespace basalt::core
