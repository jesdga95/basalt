// src/program/grammar/json_schema.cpp - JSON schema -> SchemaNode tree. See schema_internal.hpp.
#include <climits>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

#include "schema_internal.hpp"

namespace basalt::program::grammar {

namespace {

using Type = JsonValue::Type;

std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);
    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }
    parts.push_back(str.substr(start));
    return parts;
}

class Builder {
    const JsonValue& root_;
    SchemaDocument& doc_;
    // the targets built here, moved into doc_ once the whole schema is built
    std::map<std::string, SchemaPtr> refs_;
    // ref nodes get their target once every $ref is built, a cycle would otherwise need it too early
    std::vector<SchemaNode*> pending_;

    [[noreturn]] static void fail(const std::string& path, const std::string& msg) {
        throw std::runtime_error("JSON schema error at " + path + ": " + msg);
    }

    static int get_count(const JsonValue& schema, const std::string& key, const std::string& path, int def) {
        const JsonValue* value = schema.get(key);
        if (!value) return def;
        if (!value->is_integer() || (int) value->as_int() < 0) fail(path, key + " must be a non-negative integer");
        return (int) value->as_int();
    }

    // a fractional bound is rounded inwards, towards the integers it still admits
    static int64_t get_bound(const JsonValue& schema, const std::string& key, const std::string& path, bool round_up) {
        const JsonValue& value = *schema.get(key);
        if (value.is_integer()) return value.as_int();
        if (value.type != Type::Number) fail(path, key + " must be a number");
        double d = value.as_double();
        return (int64_t) (round_up ? std::ceil(d) : std::floor(d));
    }

    static SchemaNode::Format get_format(const JsonValue& schema, const std::string& path) {
        const JsonValue* value = schema.get("format");
        if (!value) return SchemaNode::FormatNone;
        if (value->type != Type::String) fail(path, "format must be a string");
        const std::string& format = value->s;
        if (format == "date") return SchemaNode::FormatDate;
        if (format == "time") return SchemaNode::FormatTime;
        if (format == "date-time") return SchemaNode::FormatDateTime;
        if (format == "uuid" || (format.size() == 5 && format.compare(0, 4, "uuid") == 0 && format[4] >= '1' && format[4] <= '5')) {
            return SchemaNode::FormatUuid;
        }
        return SchemaNode::FormatNone;
    }

    const JsonValue& resolve_ref(const std::string& ref, const std::string& path) {
        const JsonValue* target = &root_;
        auto tokens = split(ref.substr(1), "/");
        for (size_t i = 1; i < tokens.size(); i++) {
            const std::string& sel = tokens[i];
            if (target->type == Type::Object && target->contains(sel)) {
                target = target->get(sel);
            } else if (target->type == Type::Array) {
                size_t idx;
                try {
                    idx = (size_t) std::stoull(sel);
                } catch (const std::logic_error&) {
                    idx = target->items.size();
                }
                if (idx >= target->items.size()) {
                    fail(path, "cannot resolve $ref " + ref + ", " + sel + " is out of range");
                }
                target = &target->items[idx];
            } else {
                fail(path, "cannot resolve $ref " + ref + ", " + sel + " not found");
            }
        }
        return *target;
    }

    // the schema `false` (also through $ref links): no value satisfies it
    bool is_never(const JsonValue& schema) {
        const JsonValue* cur = &schema;
        for (int guard = 0; guard < 32; ++guard) {
            if (cur->type == Type::Bool) return !cur->b;
            const JsonValue* ref = cur->type == Type::Object ? cur->get("$ref") : nullptr;
            if (!ref || ref->type != Type::String || ref->s.compare(0, 2, "#/") != 0) return false;
            cur = &resolve_ref(ref->s, "#");
        }
        return false;
    }

    SchemaPtr build_ref(const JsonValue& value, const std::string& path) {
        if (value.type != Type::String) fail(path, "$ref must be a string");
        const std::string& ref = value.s;
        if (ref.compare(0, 2, "#/") != 0) {
            fail(path, "unsupported $ref " + ref + ", only references into the same document are supported");
        }
        if (refs_.find(ref) == refs_.end()) {
            // reserve the key first, so that a cycle back to this $ref stops here
            refs_[ref] = nullptr;
            refs_[ref] = build_node(resolve_ref(ref, path), ref);
        }
        auto node = std::make_unique<SchemaNode>();
        node->kind = SchemaNode::Ref;
        node->ref = ref;
        pending_.push_back(node.get());
        return node;
    }

    SchemaPtr build_alternatives(SchemaNode::Kind kind, const JsonValue& alts, const std::string& path) {
        if (alts.type != Type::Array) fail(path, "must be an array of schemas");
        if (alts.items.empty()) fail(path, "must not be empty");
        auto node = std::make_unique<SchemaNode>();
        node->kind = kind;
        size_t i = 0;
        for (const auto& alt : alts.items) {
            const std::string alt_path = path + "/" + std::to_string(i++);
            // an alternative that is `false` never matches; for allOf it makes the whole schema unsatisfiable
            if (kind == SchemaNode::AnyOf && is_never(alt)) continue;
            node->children.push_back(build_node(alt, alt_path));
        }
        if (node->children.empty()) fail(path, "every alternative is the schema false, no value is valid");
        return node;
    }

    SchemaPtr build_object(const JsonValue& schema, const std::string& path) {
        auto node = std::make_unique<SchemaNode>();
        node->kind = SchemaNode::Object;

        std::unordered_set<std::string> required;
        if (const JsonValue* req = schema.get("required")) {
            if (req->type == Type::Array) {
                for (const auto& name : req->items) {
                    if (name.type == Type::String) required.insert(name.s);
                }
            }
        }

        if (const JsonValue* properties = schema.get("properties")) {
            if (properties->type != Type::Object) fail(path, "properties must be an object");
            for (const auto& kv : properties->members) {
                const std::string prop_path = path + "/properties/" + kv.first;
                const bool is_required = required.count(kv.first) > 0;
                if (is_never(kv.second)) {
                    // an optional property that no value satisfies is never written
                    if (is_required) fail(prop_path, "required property " + kv.first + " has the schema false, no value is valid");
                    continue;
                }
                SchemaProperty prop;
                prop.name = kv.first;
                prop.schema = build_node(kv.second, prop_path);
                prop.required = is_required;
                node->properties.push_back(std::move(prop));
            }
        }

        if (const JsonValue* additional = schema.get("additionalProperties")) {
            if (additional->type == Type::Bool) {
                if (additional->b) node->additional_properties = std::make_unique<SchemaNode>();
            } else if (additional->type == Type::Object) {
                node->additional_properties = build_node(*additional, path + "/additionalProperties");
            } else {
                fail(path, "additionalProperties must be a boolean or a schema");
            }
        } else if (!schema.contains("properties")) {
            // {"type": "object"} on its own accepts any object
            node->additional_properties = std::make_unique<SchemaNode>();
        }
        return node;
    }

    SchemaPtr build_array(const JsonValue& schema, const std::string& path) {
        auto node = std::make_unique<SchemaNode>();
        node->kind = SchemaNode::Array;
        if (schema.contains("items") || schema.contains("prefixItems")) {
            // "items" wins when both are present; as in the converter, a schema instead of an array is the item schema
            // "items": false after "prefixItems" only closes the tuple, which is already exact
            const JsonValue* pre = schema.get("prefixItems");
            const bool closed_tuple = pre && pre->type == Type::Array && schema.contains("items") && is_never(*schema.get("items"));
            const std::string key = schema.contains("items") && !closed_tuple ? "items" : "prefixItems";
            const JsonValue& items = *schema.get(key);
            if (items.type == Type::Array) {
                auto tuple = std::make_unique<SchemaNode>();
                tuple->kind = SchemaNode::Tuple;
                size_t i = 0;
                for (const auto& item : items.items) {
                    if (is_never(item)) fail(path + "/" + key + "/" + std::to_string(i), "tuple item has the schema false, no value is valid");
                    tuple->children.push_back(build_node(item, path + "/" + key + "/" + std::to_string(i++)));
                }
                return tuple;
            }
            if (is_never(items)) {
                // "items": false leaves only the empty array
                node->items = std::make_unique<SchemaNode>();
                node->min_items = get_count(schema, "minItems", path, 0);
                if (node->min_items > 0) fail(path, "items is the schema false but minItems asks for items");
                node->max_items = 0;
                return node;
            }
            node->items = build_node(items, path + "/" + key);
        } else {
            node->items = std::make_unique<SchemaNode>();
        }
        node->min_items = get_count(schema, "minItems", path, 0);
        node->max_items = get_count(schema, "maxItems", path, -1);
        return node;
    }

    SchemaPtr build_string(const JsonValue& schema, const std::string& path) {
        auto node = std::make_unique<SchemaNode>();
        node->kind = SchemaNode::String;
        if (const JsonValue* pattern = schema.get("pattern")) {
            if (pattern->type != Type::String) fail(path, "pattern must be a string");
            node->pattern = pattern->s;
        }
        node->format = get_format(schema, path);
        node->min_length = get_count(schema, "minLength", path, 0);
        node->max_length = get_count(schema, "maxLength", path, -1);
        return node;
    }

    SchemaPtr build_integer(const JsonValue& schema, const std::string& path) {
        auto node = std::make_unique<SchemaNode>();
        node->kind = SchemaNode::Integer;
        if (schema.contains("minimum")) {
            node->minimum = get_bound(schema, "minimum", path, /* round_up */ true);
        } else if (schema.contains("exclusiveMinimum")) {
            node->minimum = get_bound(schema, "exclusiveMinimum", path, /* round_up */ false) + 1;
        }
        if (schema.contains("maximum")) {
            node->maximum = get_bound(schema, "maximum", path, /* round_up */ false);
        } else if (schema.contains("exclusiveMaximum")) {
            node->maximum = get_bound(schema, "exclusiveMaximum", path, /* round_up */ true) - 1;
        }
        return node;
    }

    static SchemaPtr simple(SchemaNode::Kind kind) {
        auto node = std::make_unique<SchemaNode>();
        node->kind = kind;
        return node;
    }

    SchemaPtr build_node(const JsonValue& schema, const std::string& path) {
        // a boolean schema: `true` is the empty schema, `false` has no valid value
        if (schema.type == Type::Bool) {
            if (!schema.b) fail(path, "the schema false has no valid value");
            return simple(SchemaNode::Any);
        }
        if (schema.type != Type::Object) fail(path, "schema must be an object or a boolean");
        if (const JsonValue* ref = schema.get("$ref")) return build_ref(*ref, path);
        if (schema.contains("oneOf") || schema.contains("anyOf")) {
            const std::string key = schema.contains("oneOf") ? "oneOf" : "anyOf";
            return build_alternatives(SchemaNode::AnyOf, *schema.get(key), path + "/" + key);
        }

        JsonValue type;
        if (const JsonValue* t = schema.get("type")) type = *t;
        if (type.type == Type::Array) {
            // {"type": ["a", "b"], ...} is {"anyOf": [{"type": "a", ...}, {"type": "b", ...}]}
            if (type.items.empty()) fail(path, "type must not be empty");
            auto node = simple(SchemaNode::AnyOf);
            size_t i = 0;
            for (const auto& t : type.items) {
                JsonValue alt = schema;
                alt.set("type", t);
                node->children.push_back(build_node(alt, path + "/type/" + std::to_string(i++)));
            }
            return node;
        }
        if (const JsonValue* c = schema.get("const")) {
            auto node = simple(SchemaNode::Const);
            node->values.push_back(*c);
            return node;
        }
        if (const JsonValue* values = schema.get("enum")) {
            if (values->type != Type::Array || values->items.empty()) fail(path, "enum must be a non-empty array");
            auto node = simple(SchemaNode::Enum);
            for (const auto& value : values->items) node->values.push_back(value);
            return node;
        }
        if (type.type != Type::Null && type.type != Type::String) {
            fail(path, "type must be a string or an array of strings");
        }

        const std::string type_name = type.type == Type::String ? type.s : "";
        const JsonValue* addl = schema.get("additionalProperties");
        const bool has_properties = schema.contains("properties") || (addl && !addl->is_true());

        if (type_name.empty()) {
            // without a type the structural keywords decide, in the same order as the converter
            if (has_properties) return build_object(schema, path);
            if (schema.contains("allOf")) return build_alternatives(SchemaNode::AllOf, *schema.get("allOf"), path + "/allOf");
            if (schema.contains("items") || schema.contains("prefixItems")) return build_array(schema, path);
            if (schema.contains("pattern") || schema.contains("minLength") || schema.contains("maxLength") ||
                get_format(schema, path) != SchemaNode::FormatNone) {
                return build_string(schema, path);
            }
            return simple(SchemaNode::Any);
        }
        if (type_name == "object") {
            if (!has_properties && schema.contains("allOf")) {
                return build_alternatives(SchemaNode::AllOf, *schema.get("allOf"), path + "/allOf");
            }
            return build_object(schema, path);
        }
        if (type_name == "string") {
            if (schema.contains("allOf")) return build_alternatives(SchemaNode::AllOf, *schema.get("allOf"), path + "/allOf");
            return build_string(schema, path);
        }
        if (type_name == "array") return build_array(schema, path);
        if (type_name == "integer") return build_integer(schema, path);
        if (type_name == "number") return simple(SchemaNode::Number);
        if (type_name == "boolean") return simple(SchemaNode::Boolean);
        if (type_name == "null") return simple(SchemaNode::Null);
        fail(path, "unrecognized type " + type_name);
    }

public:
    Builder(const JsonValue& root, SchemaDocument& doc) : root_(root), doc_(doc) {}

    SchemaPtr build(const JsonValue& node) {
        auto out = build_node(node, "#");
        for (auto& entry : refs_) doc_.refs[entry.first] = std::move(entry.second);
        for (auto* ref : pending_) ref->target = doc_.refs.at(ref->ref).get();
        return out;
    }
};

}  // namespace

SchemaDocument build_schema_document(const JsonValue& schema, const JsonValue* root) {
    SchemaDocument doc;
    doc.root = Builder(root ? *root : schema, doc).build(schema);
    return doc;
}

}  // namespace basalt::program::grammar
