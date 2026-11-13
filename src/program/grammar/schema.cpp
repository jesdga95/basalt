// src/program/grammar/schema.cpp - see schema.hpp. The JSON-schema conversion is a port of llama.cpp's
// json-schema-to-grammar (schema_convert.cpp, json_schema.cpp); this file adds the tool-call and request grammars.
#include "basalt/program/grammar/schema.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

#include "schema_internal.hpp"

namespace basalt::program::grammar {

namespace {

using Type = JsonValue::Type;

std::string literal(const std::string& s) { return gbnf_format_literal(s); }

std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string o;
    for (size_t i = 0; i < v.size(); ++i) o += (i ? sep : "") + v[i];
    return o;
}

bool read_json(const std::string& text, JsonValue& v, std::string& err) { return JsonReader::parse(text, v, err); }

// Follows local $ref links ("#/a/b") to the schema they name; null if one does not resolve.
const JsonValue* deref(const JsonValue* root, const JsonValue* s) {
    for (int guard = 0; s && s->is(Type::Object) && guard < 32; ++guard) {
        const JsonValue* r = s->get("$ref");
        if (!r || !r->is(Type::String)) break;
        if (r->s.compare(0, 2, "#/") != 0 || !root) return nullptr;
        const JsonValue* cur = root;
        size_t pos = 2;
        while (cur && pos <= r->s.size()) {
            size_t e = r->s.find('/', pos);
            std::string part = r->s.substr(pos, e == std::string::npos ? std::string::npos : e - pos);
            if (cur->is(Type::Object)) {
                cur = cur->get(part);
            } else if (cur->is(Type::Array) && !part.empty() && part.find_first_not_of("0123456789") == std::string::npos &&
                       std::stoul(part) < cur->items.size()) {
                cur = &cur->items[std::stoul(part)];
            } else {
                cur = nullptr;
            }
            if (e == std::string::npos) break;
            pos = e + 1;
        }
        s = cur;
    }
    return s;
}

}  // namespace

std::string json_schema_to_gbnf(const std::string& schema_json, std::string& err) {
    JsonValue schema;
    if (!read_json(schema_json, schema, err)) return "";
    try {
        return schema_document_to_gbnf(build_schema_document(schema));
    } catch (const std::invalid_argument& e) {
        err = std::string("schema: ") + e.what();
    } catch (const std::runtime_error& e) {
        err = std::string("schema: JSON schema conversion failed:\n") + e.what();
    }
    return "";
}

// ------------------------------------------------------------------------------------------ tool calls

namespace {

// True when the schema is a type that is raw text in a parameter: "string", or a type list naming "string".
bool is_text_type(const JsonValue& t) {
    if (t.is(Type::String)) return t.s == "string";
    return t.is(Type::Array) && std::any_of(t.items.begin(), t.items.end(),
                                            [](const JsonValue& v) { return v.is(Type::String) && v.s == "string"; });
}

// One branch of an anyOf/oneOf/allOf: a plain string (type "string" or a list of string and null, no enum/const/
// structure) or null. `has_text` is set when it can be text.
bool plain_text_branch(const JsonValue* b, bool& has_text) {
    if (!b || !b->is(Type::Object)) return false;
    for (const char* k : {"enum", "const", "anyOf", "oneOf", "allOf", "properties", "items", "$ref"})
        if (b->get(k)) return false;
    const JsonValue* t = b->get("type");
    if (!t) return false;
    if (t->is(Type::String)) {
        if (t->s == "string") has_text = true;
        return t->s == "string" || t->s == "null";
    }
    if (!t->is(Type::Array)) return false;
    for (const JsonValue& v : t->items) {
        if (!v.is(Type::String) || (v.s != "string" && v.s != "null")) return false;
        if (v.s == "string") has_text = true;
    }
    return true;
}

// A parameter written as anyOf/oneOf (or a one-branch allOf) whose every branch is a plain string or null, with at
// least one string: it takes raw text like a string, and `null` is just text it may write.
bool text_union(const JsonValue* ps, const JsonValue* root) {
    if (!ps || !ps->is(Type::Object) || ps->get("type") || ps->get("enum") || ps->get("const") ||
        ps->get("properties") || ps->get("items"))
        return false;
    const JsonValue* u = nullptr;
    for (const char* k : {"anyOf", "oneOf", "allOf"}) {
        if (const JsonValue* v = ps->get(k)) {
            if (u || !v->is(Type::Array) || v->items.empty()) return false;
            if (k[0] == 'a' && k[1] == 'l' && v->items.size() != 1) return false;
            u = v;
        }
    }
    if (!u) return false;
    bool has_text = false;
    for (const JsonValue& b : u->items)
        if (!plain_text_branch(deref(root, &b), has_text)) return false;
    return has_text;
}

std::string hex_escape(unsigned char c) {
    static const char* h = "0123456789abcdef";
    return std::string("\\x") + h[c >> 4] + h[c & 15];
}

// The UTF-8 characters of `s`: a grammar's literals and classes read characters, not bytes.
std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        size_t n = 1;
        while (i + n < s.size() && ((unsigned char) s[i + n] & 0xC0) == 0x80) ++n;
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

// Text that never contains the closing tag: a KMP automaton over the tag's characters, one rule per matched length.
// The tag's last character is the one move the automaton refuses.
std::string add_free_text(SchemaConverter& c, const std::string& tag) {
    const std::vector<std::string> chars = utf8_chars(tag);
    size_t n = chars.size();
    std::set<std::string> alpha(chars.begin(), chars.end());
    std::string excluded;
    for (const std::string& a : alpha) excluded += a.size() == 1 ? hex_escape((unsigned char) a[0]) : a;
    for (size_t i = 0; i < n; ++i) {
        std::vector<std::string> alts;
        for (const std::string& a : alpha) {
            std::vector<std::string> seen(chars.begin(), chars.begin() + (std::ptrdiff_t) i);
            seen.push_back(a);
            size_t k = (std::min)(seen.size(), n);
            while (k > 0 && !std::equal(chars.begin(), chars.begin() + (std::ptrdiff_t) k, seen.end() - (std::ptrdiff_t) k)) --k;
            if (k == n) continue;
            alts.push_back(literal(a) + " free-text-" + std::to_string(k));
        }
        alts.push_back("[^" + excluded + "] free-text-0");
        alts.push_back("\"\"");
        c.add_rule("free-text-" + std::to_string(i), join(alts, " | "));
    }
    return "free-text-0";
}

// The object variants of a oneOf/anyOf union, nested unions flattened.
void collect_variants(const JsonValue* root, const JsonValue* s, std::vector<const JsonValue*>& out, int depth) {
    s = deref(root, s);
    if (!s || !s->is(Type::Object) || depth > 8) return;
    bool is_union = false;
    for (const char* k : {"oneOf", "anyOf"}) {
        if (const JsonValue* u = s->get(k); u && u->is(Type::Array)) {
            is_union = true;
            for (const JsonValue& alt : u->items) collect_variants(root, &alt, out, depth + 1);
        }
    }
    const JsonValue* p = s->get("properties");
    if (!is_union && p && p->is(Type::Object)) out.push_back(s);
}

bool string_enum(const JsonValue& s) {
    const JsonValue* e = s.is(Type::Object) ? s.get("enum") : nullptr;
    return e && e->is(Type::Array) &&
           std::all_of(e->items.begin(), e->items.end(), [](const JsonValue& v) { return v.is(Type::String); });
}

JsonValue json_string(const std::string& text) {
    JsonValue v;
    v.type = Type::String;
    v.s = text;
    return v;
}

// Two schemas for one property name in different variants: string enums are united, anything else becomes an anyOf.
JsonValue merge_property(const JsonValue& cur, const JsonValue& add) {
    if (cur.dump() == add.dump()) return cur;
    if (string_enum(cur) && string_enum(add)) {
        JsonValue merged = cur;
        JsonValue en = *cur.get("enum");
        for (const JsonValue& e : add.get("enum")->items) {
            auto same = [&](const JsonValue& x) { return x.s == e.s; };
            if (std::none_of(en.items.begin(), en.items.end(), same)) en.items.push_back(e);
        }
        merged.set("enum", en);
        return merged;
    }
    JsonValue alts;
    alts.type = Type::Array;
    const JsonValue* any = cur.get("anyOf");
    if (any && any->is(Type::Array) && cur.members.size() == 1) alts = *any;
    else alts.items.push_back(cur);
    auto same = [&](const JsonValue& x) { return x.dump() == add.dump(); };
    if (std::none_of(alts.items.begin(), alts.items.end(), same)) alts.items.push_back(add);
    JsonValue merged;
    merged.type = Type::Object;
    merged.set("anyOf", alts);
    return merged;
}

// Parameters sent as a union of objects ({"properties": {}, "oneOf": [...]}, as zod-generated tools do) become one
// object: every property of every variant, and required only what every variant requires (the client still checks
// the variant). False when the parameters are not such a union.
bool flatten_union(const JsonValue* params, JsonValue& out) {
    if (!params || !params->is(Type::Object)) return false;
    if (const JsonValue* p = params->get("properties"); p && p->is(Type::Object) && !p->members.empty()) return false;
    std::vector<const JsonValue*> variants;
    for (const char* k : {"oneOf", "anyOf"}) {
        if (const JsonValue* u = params->get(k); u && u->is(Type::Array))
            for (const JsonValue& alt : u->items) collect_variants(params, &alt, variants, 0);
    }
    if (variants.empty()) return false;

    JsonValue props;
    props.type = Type::Object;
    std::map<std::string, size_t> required_by;   // property -> number of variants that require it
    for (const JsonValue* v : variants) {
        for (const auto& m : v->get("properties")->members) {
            const JsonValue* r = deref(params, &m.second);
            JsonValue value;
            if (r) value = *r;
            else value.type = Type::Object;
            const JsonValue* cur = props.get(m.first);
            props.set(m.first, cur ? merge_property(*cur, value) : value);
        }
        if (const JsonValue* req = v->get("required"); req && req->is(Type::Array)) {
            for (const JsonValue& n : req->items)
                if (n.is(Type::String)) ++required_by[n.s];
        }
    }

    JsonValue required;
    required.type = Type::Array;
    for (const auto& m : props.members) {
        auto it = required_by.find(m.first);
        if (it != required_by.end() && it->second == variants.size()) required.items.push_back(json_string(m.first));
    }
    out = JsonValue();
    out.type = Type::Object;
    out.set("type", json_string("object"));
    out.set("properties", props);
    out.set("required", required);
    for (const char* k : {"$defs", "definitions"})
        if (const JsonValue* d = params->get(k)) out.set(k, *d);
    return true;
}

}  // namespace

namespace {
ToolForm g_tool_form;
}

void set_tool_form(const ToolForm& form) { g_tool_form = form; }
const ToolForm& tool_form() { return g_tool_form; }

std::string tool_call_gbnf(const std::string& tools_json, const ToolGrammarOptions& opt, std::string& err) {
    const ToolForm& form = tool_form();
    if (form.call_open.empty()) {
        err = "tools: this model has no tool-call grammar";
        return "";
    }
    JsonValue tools;
    if (!read_json(tools_json, tools, err)) return "";
    if (!tools.is(Type::Array) || tools.items.empty()) {
        err = "tools: expected a non-empty array";
        return "";
    }
    SchemaConverter c;
    std::string text_rule;
    std::vector<std::string> fns;
    for (size_t ti = 0; ti < tools.items.size(); ++ti) {
        const JsonValue* tool = &tools.items[ti];
        if (const JsonValue* f = tool->is(Type::Object) ? tool->get("function") : nullptr) tool = f;
        const JsonValue* nm = tool->is(Type::Object) ? tool->get("name") : nullptr;
        if (!nm || !nm->is(Type::String) || nm->s.empty()) {
            err = "tools: every tool needs a name";
            return "";
        }
        const JsonValue* params = tool->get("parameters");
        if (!params) params = tool->get("input_schema");
        JsonValue flat;
        if (flatten_union(params, flat)) params = &flat;
        const JsonValue* props = params && params->is(Type::Object) ? params->get("properties") : nullptr;
        std::set<std::string> required;
        if (const JsonValue* r = params && params->is(Type::Object) ? params->get("required") : nullptr) {
            for (const JsonValue& n : r->items) if (n.is(Type::String)) required.insert(n.s);
        }
        std::string fn_id = "fn" + std::to_string(ti);
        std::vector<std::string> req_items, opt_items;
        if (props && props->is(Type::Object)) {
            for (const auto& m : props->members) {
                const JsonValue* ps = deref(params, &m.second);
                JsonValue any_schema;   // `true` and a missing target are the empty schema
                if (ps && ps->is(Type::Bool) && !ps->b) {   // `false`: no value is valid
                    if (required.count(m.first)) {
                        err = "tools: " + nm->s + "." + m.first + ": required parameter has the schema false, no value is valid";
                        return "";
                    }
                    continue;
                }
                if (!ps || ps->is(Type::Bool)) {
                    any_schema.type = Type::Object;
                    ps = &any_schema;
                }
                const JsonValue* ty = ps && ps->is(Type::Object) ? ps->get("type") : nullptr;
                const JsonValue* en = ps && ps->is(Type::Object) ? ps->get("enum") : nullptr;
                bool raw = false;
                std::string value;
                bool typed = ps && ps->is(Type::Object) &&
                             (ps->get("anyOf") || ps->get("oneOf") || ps->get("allOf") || ps->get("const") ||
                              ps->get("properties") || ps->get("items"));
                if (en && en->is(Type::Array) && !en->items.empty()) {
                    bool all_str = std::all_of(en->items.begin(), en->items.end(),
                                               [](const JsonValue& v) { return v.is(Type::String); });
                    if (all_str) {
                        std::vector<std::string> vals;
                        for (const JsonValue& v : en->items) vals.push_back(literal(v.s));
                        value = c.add_rule(fn_id + "-" + m.first + "-enum", "(" + join(vals, " | ") + ")");
                        raw = true;
                    }
                } else if (ty && is_text_type(*ty)) {
                    raw = true;
                } else if (text_union(ps, params)) {
                    raw = true;
                } else if (!ty && !typed) {
                    raw = true;
                }
                if (raw && value.empty()) {
                    if (text_rule.empty()) text_rule = add_free_text(c, form.text_end);
                    value = text_rule;
                }
                if (!raw) {
                    try {
                        SchemaDocument d = build_schema_document(*ps, params);
                        value = c.add_schema(fn_id + "-" + m.first, *d.root);
                    } catch (const std::exception& e) {
                        err = "tools: " + nm->s + "." + m.first + ": " + e.what();
                        return "";
                    }
                }
                if (value.empty()) {
                    err = "tools: " + nm->s + "." + m.first + ": no rule";
                    return "";
                }
                (required.count(m.first) ? req_items : opt_items)
                    .push_back(literal(form.param_open + m.first + (raw ? form.param_mid_text : form.param_mid_value)) + " " + value + " " + literal(form.param_close));
            }
        }
        // As llama.cpp's Qwen3-Coder parser (common/chat-peg-parser.cpp permute, parsers/qwen3-coder.cpp): the required
        // parameters in any order, each once (one rule per set still to write; past max_permute they keep the declared
        // order), then the optional ones in any order, repeatable.
        std::string body;
        const size_t n = req_items.size();
        if (n == 1 || n > opt.max_permute) {
            for (const std::string& it : req_items) body += it + " ";
        } else if (n > 1) {
            auto rule_of = [&](size_t remaining) { return fn_id + "-req" + std::to_string(remaining); };
            for (size_t remaining = 1; remaining < ((size_t) 1 << n); ++remaining) {
                std::vector<std::string> alts;
                for (size_t i = 0; i < n; ++i) {
                    const size_t bit = (size_t) 1 << i;
                    if (!(remaining & bit)) continue;
                    alts.push_back(req_items[i] + (remaining & ~bit ? " " + rule_of(remaining & ~bit) : ""));
                }
                c.add_rule(rule_of(remaining), join(alts, " | "));
            }
            body = rule_of(((size_t) 1 << n) - 1) + " ";
        }
        if (!opt_items.empty()) body += "(" + c.add_rule(fn_id + "-opt", join(opt_items, " | ")) + ")* ";
        fns.push_back(c.add_rule(fn_id, literal(form.fn_open + nm->s + form.fn_mid) + " " + body + literal(form.fn_close)));
    }
    const std::string one = "(" + join(fns, " | ") + ")";
    const bool shared_block = form.calls_share_block && opt.multiple_calls;
    c.add_rule("call", literal(form.call_open) + " " + (shared_block ? one + " (" + literal("\n") + " " + one + ")*" : one) + " " + literal(form.call_close));
    std::string root = opt.multiple_calls && !form.calls_share_block ? "call ([ \\t\\n]* call)*" : "call";
    if (!opt.answer_schema.empty()) {
        JsonValue answer;
        if (!read_json(opt.answer_schema, answer, err)) return "";
        try {
            SchemaDocument d = build_schema_document(answer);
            root = "(" + root + ") | " + c.add_schema("answer", *d.root);
        } catch (const std::exception& e) {
            err = std::string("schema: ") + e.what();
            return "";
        }
    }
    c.add_rule("root", root);
    try {
        c.check_errors();
    } catch (const std::exception& e) {
        err = std::string("tools: ") + e.what();
        return "";
    }
    return c.format_grammar();
}

RequestGrammar grammar_from_request(const std::string& grammar_json, std::string& err) {
    RequestGrammar out;
    JsonValue req;
    if (!read_json(grammar_json, req, err)) return out;
    const JsonValue* kind = req.is(Type::Object) ? req.get("kind") : nullptr;
    if (!kind || !kind->is(Type::String)) {
        err = "grammar: missing kind";
        return out;
    }
    std::string gbnf;
    if (kind->s == "json_object" || kind->s == "json_schema") {
        std::string schema = R"({"type":"object"})";
        if (kind->s == "json_schema") {
            const JsonValue* sch = req.get("schema");
            if (!sch) {
                err = "grammar: json_schema needs a schema";
                return out;
            }
            schema = sch->dump();
        }
        if (const JsonValue* tools = req.get("tools")) {   // tool calls or the JSON answer, from the first answer token
            if (!tools->is(Type::Array)) {
                err = "grammar: tools needs an array";
                return out;
            }
            ToolGrammarOptions opt;
            opt.answer_schema = schema;
            gbnf = tool_call_gbnf(tools->dump(), opt, err);
        } else {
            gbnf = json_schema_to_gbnf(schema, err);
        }
    } else if (kind->s == "tools") {
        const JsonValue* tools = req.get("tools");
        if (!tools || !tools->is(Type::Array)) {
            err = "grammar: tools needs an array";
            return out;
        }
        std::string only;   // a named choice
        if (const JsonValue* tc = req.get("tool_choice")) {
            if (tc->is(Type::String) && tc->s == "none") return out;
            if (tc->is(Type::Object)) {
                const JsonValue* fn = tc->get("function");
                const JsonValue* nm = fn && fn->is(Type::Object) ? fn->get("name") : tc->get("name");
                if (nm && nm->is(Type::String)) only = nm->s;
            }
        }
        JsonValue list = *tools;
        if (!only.empty()) {
            list.items.clear();
            for (const JsonValue& t : tools->items) {
                const JsonValue* f = t.is(Type::Object) && t.get("function") ? t.get("function") : &t;
                const JsonValue* nm = f->is(Type::Object) ? f->get("name") : nullptr;
                if (nm && nm->is(Type::String) && nm->s == only) list.items.push_back(t);
            }
        }
        gbnf = tool_call_gbnf(list.dump(), ToolGrammarOptions{}, err);
        out.triggers.words = {tool_form().trigger};
    } else {
        err = "grammar: unknown kind '" + kind->s + "'";
        return out;
    }
    if (gbnf.empty()) return out;
    out.grammar = Grammar::parse(gbnf, err);
    return out;
}

}  // namespace basalt::program::grammar
