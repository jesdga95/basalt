// tests/program/grammar_schema_test.cpp - schema-to-GBNF and the tool-call grammar, checked by running JSON text and
// tool-call text through the parsed grammar.
#include "basalt/program/grammar/grammar.hpp"
#include "basalt/program/grammar/schema.hpp"

#include <cstdio>
#include <string>

using namespace basalt::program::grammar;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

static std::shared_ptr<const Grammar> build(const std::string& schema) {
    std::string err;
    std::string gbnf = json_schema_to_gbnf(schema, err);
    if (gbnf.empty()) { std::printf("schema error: %s\n", err.c_str()); return nullptr; }
    auto g = Grammar::parse(gbnf, err);
    if (!g) std::printf("grammar error: %s\n--- grammar ---\n%s\n", err.c_str(), gbnf.c_str());
    return g;
}
static bool ok(const std::shared_ptr<const Grammar>& g, const std::string& text) {
    GrammarState s(g);
    for (char c : text) if (!s.accept(std::string_view(&c, 1))) return false;
    return s.accepting();
}
static bool pre(const std::shared_ptr<const Grammar>& g, const std::string& text) {
    GrammarState s(g);
    for (char c : text) if (!s.accept(std::string_view(&c, 1))) return false;
    return true;
}

static void test_scalars() {
    auto b = build(R"({"type":"boolean"})");
    CHECK(b && ok(b, "true") && ok(b, "false") && !pre(b, "True") && !pre(b, "tru e"));
    auto n = build(R"({"type":"null"})");
    CHECK(n && ok(n, "null") && !pre(n, "None"));
    auto i = build(R"({"type":"integer"})");
    CHECK(i && ok(i, "-12") && ok(i, "0") && !pre(i, "1.") && !pre(i, "01"));
    auto f = build(R"({"type":"number"})");
    CHECK(f && ok(f, "1.5e-3") && ok(f, "-0.25") && !pre(f, "1,5"));
    auto s = build(R"({"type":"string"})");
    CHECK(s && ok(s, "\"a b\\n\\u00e9\"") && !pre(s, "\"a\nb") && !pre(s, "a"));
    auto sl = build(R"({"type":"string","minLength":2,"maxLength":3})");
    CHECK(sl && !ok(sl, "\"a\"") && ok(sl, "\"ab\"") && ok(sl, "\"abc\"") && !pre(sl, "\"abcd"));
    auto d = build(R"({"type":"string","format":"date"})");
    CHECK(d && ok(d, "\"2026-10-21\"") && !pre(d, "\"2026-13"));
    auto u = build(R"({"type":"string","format":"uuid"})");
    CHECK(u && ok(u, "\"123e4567-e89b-12d3-a456-426614174000\"") && !pre(u, "\"123g"));
}

static void test_enum_const() {
    auto e = build(R"({"enum":["red","green",3,null,true]})");
    CHECK(e && ok(e, "\"red\"") && ok(e, "3") && ok(e, "null") && ok(e, "true") && !pre(e, "\"blue") && !pre(e, "4"));
    auto c = build(R"({"const":{"a":[1,"x"]}})");
    CHECK(c && ok(c, "{\"a\":[1,\"x\"]}") && !pre(c, "{\"a\": ["));
    auto t = build(R"({"type":["string","null"]})");
    CHECK(t && ok(t, "null") && ok(t, "\"x\"") && !pre(t, "1"));
    auto a = build(R"({"anyOf":[{"type":"integer"},{"type":"boolean"}]})");
    CHECK(a && ok(a, "5") && ok(a, "false") && !pre(a, "\"x"));
}

static void test_object() {
    auto g = build(R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"tag":{"type":"string"}},"required":["name","age"]})");
    CHECK(g);
    CHECK(ok(g, R"({"name":"a","age":3})"));
    CHECK(ok(g, R"({"name":"a","age":3,"tag":"x"})"));
    CHECK(!ok(g, R"({"name":"a"})"));                                   // age is required
    CHECK(!pre(g, R"({"age":3)"));                                      // order follows the schema
    CHECK(!pre(g, R"({"name":"a","age":"3")"));                         // wrong type
    CHECK(!pre(g, R"({"name":"a","age":3,"extra":1)"));                 // no additional properties
    auto only_opt = build(R"({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}})");
    CHECK(only_opt && ok(only_opt, "{}") && ok(only_opt, R"({"a":1})") && ok(only_opt, R"({"b":2})") &&
          ok(only_opt, R"({"a":1,"b":2})") && !pre(only_opt, R"({"b":2,"a")"));
    auto open = build(R"({"type":"object","additionalProperties":{"type":"integer"}})");
    CHECK(open && ok(open, R"({"x":1,"y":2})") && !pre(open, R"({"x":"1")"));
    auto any = build(R"({"type":"object"})");
    CHECK(any && ok(any, R"({"x":[1,{"y":null}]})"));
}

static void test_array_nested() {
    auto a = build(R"({"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":3})");
    CHECK(a && ok(a, "[1]") && ok(a, "[1,2,3]") && !ok(a, "[]") && !pre(a, "[1,2,3,4") && !pre(a, "[\"x\""));
    auto free = build(R"({"type":"array"})");
    CHECK(free && ok(free, "[]") && ok(free, "[1,\"a\",[null]]"));
    auto tup = build(R"({"type":"array","prefixItems":[{"type":"integer"},{"type":"string"}]})");
    CHECK(tup && ok(tup, "[1,\"a\"]") && !pre(tup, "[\"a\""));
    auto nested = build(R"({"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"id":{"type":"integer"},"ok":{"type":"boolean"}},"required":["id"]}}},"required":["items"]})");
    CHECK(nested && ok(nested, R"({"items":[{"id":1},{"id":2,"ok":true}]})") && !pre(nested, R"({"items":[{"id":1,"ok":1)"));
}

static void test_ref() {
    auto r = build(R"({"$defs":{"pt":{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},"required":["x","y"]}},
        "type":"object","properties":{"a":{"$ref":"#/$defs/pt"},"b":{"$ref":"#/$defs/pt"}},"required":["a"]})");
    CHECK(r && ok(r, R"({"a":{"x":1,"y":2}})") && ok(r, R"({"a":{"x":1,"y":2},"b":{"x":3,"y":4}})") &&
          !pre(r, R"({"a":{"x":1,"y":"2")"));
    // a recursive reference
    auto tree = build(R"({"$ref":"#/definitions/node","definitions":{"node":{"type":"object","properties":{"v":{"type":"integer"},"kids":{"type":"array","items":{"$ref":"#/definitions/node"}}},"required":["v"]}}})");
    CHECK(tree && ok(tree, R"({"v":1,"kids":[{"v":2},{"v":3,"kids":[{"v":4}]}]})") && !pre(tree, R"({"v":1,"kids":[{"kids")"));
    std::string err;
    CHECK(json_schema_to_gbnf(R"({"$ref":"#/nope"})", err).empty() && !err.empty());
    CHECK(json_schema_to_gbnf("{", err).empty());
}

static const char* kTools = R"([
  {"name":"get_weather","description":"x","parameters":{"type":"object","properties":{
      "city":{"type":"string"},"days":{"type":"integer"},"metric":{"type":"boolean"},
      "unit":{"type":"string","enum":["c","f"]},"tags":{"type":"array","items":{"type":"string"}}},
    "required":["city","metric"]}},
  {"name":"ping","parameters":{"type":"object","properties":{}}},
  {"type":"function","function":{"name":"set_flag","parameters":{"type":"object","properties":{"on":{"type":"boolean"},"note":{}},"required":["on"]}}}
])";

static bool call_ok(const std::shared_ptr<const Grammar>& g, const std::string& t) { return ok(g, t); }

static void test_tools() {
    std::string err;
    std::string gbnf = tool_call_gbnf(kTools, ToolGrammarOptions{}, err);
    CHECK(!gbnf.empty());
    auto g = Grammar::parse(gbnf, err);
    if (!g) std::printf("%s\n%s\n", err.c_str(), gbnf.c_str());
    CHECK(g);
    if (!g) return;
    const std::string head = "<tool_call>\n<function=get_weather>\n";
    const std::string city = "<parameter=city>\nParis\n</parameter>\n";
    const std::string metric = "<parameter=metric>\ntrue\n</parameter>\n";
    CHECK(call_ok(g, head + city + metric + "</function>\n</tool_call>"));
    CHECK(call_ok(g, head + city + metric + "<parameter=days>\n-3\n</parameter>\n</function>\n</tool_call>"));
    CHECK(call_ok(g, head + city + metric + "<parameter=unit>\nc\n</parameter>\n</function>\n</tool_call>"));
    CHECK(call_ok(g, head + city + metric + "<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n</function>\n</tool_call>"));
    // a string may hold anything but the closing tag, newlines and angle brackets included
    CHECK(call_ok(g, head + "<parameter=city>\nline 1\nline <2> </param </parameter\n</parameter>\n" + metric + "</function>\n</tool_call>"));
    CHECK(!pre(g, head + "<parameter=city>\nab</parameter>"));
    // a boolean must be JSON
    CHECK(!pre(g, head + city + "<parameter=metric>\nTrue"));
    CHECK(!pre(g, head + city + "<parameter=metric>\nyes"));
    CHECK(pre(g, head + city + "<parameter=metric>\ntrue"));
    // an integer must be a number
    CHECK(!pre(g, head + city + "<parameter=days>\n\"3\""));
    // enum values are the listed ones
    CHECK(!pre(g, head + city + metric + "<parameter=unit>\nk"));
    // unknown function and unknown parameter
    CHECK(!pre(g, "<tool_call>\n<function=get_wether>"));
    CHECK(pre(g, "<tool_call>\n<function=get_w"));
    CHECK(!pre(g, head + "<parameter=zip>"));
    // required parameters in any order, each once; optional ones after them, any order, repeatable
    CHECK(call_ok(g, head + metric + city + "</function>\n</tool_call>"));
    CHECK(call_ok(g, head + metric + city + "<parameter=days>\n2\n</parameter>\n<parameter=unit>\nc\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"));
    CHECK(!call_ok(g, head + city + "</function>\n</tool_call>"));                    // metric is required
    CHECK(!pre(g, head + city + "<parameter=city>"));                                // a required one is written once
    CHECK(!pre(g, head + "<parameter=days>"));                                       // optional ones follow the required
    CHECK(!pre(g, head + city + metric + city));
    // no parameters, wrapper form, untyped parameter takes free text
    CHECK(call_ok(g, "<tool_call>\n<function=ping>\n</function>\n</tool_call>"));
    CHECK(call_ok(g, "<tool_call>\n<function=set_flag>\n<parameter=on>\nfalse\n</parameter>\n<parameter=note>\nhi there\n</parameter>\n</function>\n</tool_call>"));
    // two calls
    CHECK(call_ok(g, "<tool_call>\n<function=ping>\n</function>\n</tool_call>\n<tool_call>\n<function=ping>\n</function>\n</tool_call>"));
    // stays inactive until the trigger
    Triggers tr;
    tr.words = {tool_call_trigger()};
    GrammarState s(g, tr);
    CHECK(!s.active());
    CHECK(s.accept("I will check the weather. Let me <call"));
    CHECK(!s.active());
    CHECK(s.accept("<tool_call>"));
    CHECK(s.active());
    CHECK(s.accept("\n<function=ping>\n</function>\n</tool_call>"));
    CHECK(s.accepting());
    // one call only
    ToolGrammarOptions one;
    one.multiple_calls = false;
    auto g1 = Grammar::parse(tool_call_gbnf(kTools, one, err), err);
    CHECK(g1 && ok(g1, "<tool_call>\n<function=ping>\n</function>\n</tool_call>") &&
          !pre(g1, "<tool_call>\n<function=ping>\n</function>\n</tool_call>\n<"));
    // past max_permute the required parameters keep the declared order
    ToolGrammarOptions big;
    big.max_permute = 1;
    auto gb = Grammar::parse(tool_call_gbnf(kTools, big, err), err);
    CHECK(gb && ok(gb, head + city + metric + "</function>\n</tool_call>") && !pre(gb, head + metric));
    CHECK(tool_call_gbnf("[]", ToolGrammarOptions{}, err).empty());
    CHECK(tool_call_gbnf("[{\"parameters\":{}}]", ToolGrammarOptions{}, err).empty());
}

static void test_text_unions() {
    std::string err;
    auto tool = [&](const std::string& param) {
        return Grammar::parse(tool_call_gbnf(R"([{"name":"w","parameters":{"type":"object","properties":{"p":)" + param +
                                                 R"(},"required":["p"]}}])", ToolGrammarOptions{}, err), err);
    };
    const std::string head = "<tool_call>\n<function=w>\n<parameter=p>\n";
    const std::string tail = "\n</parameter>\n</function>\n</tool_call>";
    // anyOf/oneOf of a string and null (a nullable string): raw text, `null` and constraints included
    for (const char* form : {R"({"anyOf":[{"type":"string","minLength":1,"maxLength":1000},{"type":"null"}],"title":"P"})",
                             R"({"oneOf":[{"type":"string"},{"type":"null"}]})",
                             R"({"anyOf":[{"type":"null"},{"type":"string"}]})",
                             R"({"anyOf":[{"type":"string"}]})",
                             R"({"anyOf":[{"type":"string"},{"type":["string","null"]}]})",
                             R"({"allOf":[{"type":"string"}]})"}) {
        auto g = tool(form);
        CHECK(g);
        if (!g) continue;
        CHECK(call_ok(g, head + "notes/diagnostics/x.txt" + tail));
        CHECK(call_ok(g, head + "null" + tail));
        CHECK(call_ok(g, head + "123" + tail));
        CHECK(call_ok(g, head + "nope\nline 2 <b>" + tail));
        CHECK(!pre(g, head + "a</parameter>\n<parameter=q>"));      // the closing tag still ends the value
    }
    // not a plain string: stays the JSON value
    for (const char* form : {R"({"anyOf":[{"type":"string"},{"type":"integer"}]})",
                             R"({"anyOf":[{"type":"integer"},{"type":"null"}]})",
                             R"({"anyOf":[{"type":"string","enum":["a","b"]},{"type":"null"}]})",
                             R"({"anyOf":[{"type":"string"},{"type":"array","items":{"type":"string"}}]})",
                             R"({"allOf":[{"type":"string"},{"minLength":1}]})"}) {
        auto g = tool(form);
        CHECK(g);
        if (!g) continue;
        CHECK(!pre(g, head + "notes/x.txt"));
    }
    auto mixed = tool(R"({"anyOf":[{"type":"string"},{"type":"integer"}]})");
    CHECK(mixed && call_ok(mixed, head + "\"x\"" + tail) && call_ok(mixed, head + "5" + tail));
    auto num = tool(R"({"anyOf":[{"type":"integer"},{"type":"null"}]})");
    CHECK(num && call_ok(num, head + "null" + tail) && call_ok(num, head + "7" + tail) && !pre(num, head + "x"));
    // the type-list forms are unchanged
    auto lst = tool(R"({"type":["string","null"]})");
    CHECK(lst && call_ok(lst, head + "notes/x.txt" + tail) && call_ok(lst, head + "null" + tail));
    auto plain = tool(R"({"type":"string"})");
    CHECK(plain && call_ok(plain, head + "notes/x.txt" + tail));
    // a $ref to a nullable string
    auto ref = Grammar::parse(tool_call_gbnf(R"([{"name":"w","parameters":{"type":"object","properties":{"p":{"$ref":"#/$defs/s"}},
        "required":["p"],"$defs":{"s":{"type":"string"}}}}])", ToolGrammarOptions{}, err), err);
    CHECK(ref && call_ok(ref, head + "notes/x.txt" + tail));
}

static void test_request() {
    std::string err;
    auto r = grammar_from_request(R"({"kind":"tools","tool_choice":"auto","tools":[{"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}}},{"type":"function","function":{"name":"pong","parameters":{"type":"object","properties":{}}}}]})", err);
    CHECK(r.grammar && err.empty() && r.triggers.words.size() == 1);
    GrammarState s(r.grammar, r.triggers);
    CHECK(!s.active() && s.accept("ok <tool_call>\n<function=ping>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>") && s.accepting());
    CHECK(!s.allows("<tool_call>\n<function=pong>\n<parameter=n>"));
    auto named = grammar_from_request(R"({"kind":"tools","tool_choice":{"type":"function","function":{"name":"pong"}},"tools":[{"type":"function","function":{"name":"ping"}},{"type":"function","function":{"name":"pong"}}]})", err);
    GrammarState n(named.grammar, named.triggers);
    CHECK(named.grammar && !n.allows("<tool_call>\n<function=ping>") && n.allows("<tool_call>\n<function=pong>"));
    auto none = grammar_from_request(R"({"kind":"tools","tool_choice":"none","tools":[{"type":"function","function":{"name":"ping"}}]})", err);
    CHECK(!none.grammar && err.empty());
    auto js = grammar_from_request(R"({"kind":"json_schema","schema":{"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"]}})", err);
    GrammarState j(js.grammar, js.triggers);
    CHECK(js.grammar && j.active() && j.accept("{\"a\":true}") && j.accepting());
    GrammarState j2(js.grammar, js.triggers);
    CHECK(!j2.allows("{\"a\":True"));
    auto jo = grammar_from_request(R"({"kind":"json_object"})", err);
    GrammarState o(jo.grammar, jo.triggers);
    CHECK(jo.grammar && o.accept("{\"x\":[1,{\"y\":null}]}") && o.accepting() && !o.allows("[1]"));
    CHECK(!grammar_from_request(R"({"kind":"nope"})", err).grammar && !err.empty());
}

// A pattern or an enum string can decode (a "\u0000" JSON escape) to raw control bytes. They must reach the
// grammar text as \xNN escapes: the parser walks the text as a C string, so a raw NUL ends it mid-rule.
static void test_control_escapes() {
    std::string err;
    // a nested-object parameter whose string property has a control-byte character class (pi's interactive_shell)
    std::string gbnf = tool_call_gbnf(R"([
      {"name":"send","parameters":{"type":"object","properties":{
          "cfg":{"type":"object","properties":{
              "input":{"type":"string","pattern":"^[^\u0000-\u001f\u007f;&|`$<>]+$"},
              "k":{"type":"string","enum":["a\u0000b","c"]}},
            "required":["input"]}},
        "required":["cfg"]}}
    ])", ToolGrammarOptions{}, err);
    CHECK(!gbnf.empty() && err.empty());
    CHECK(gbnf.find('\0') == std::string::npos);
    auto g = Grammar::parse(gbnf, err);
    if (!g) std::printf("%s\n--- grammar ---\n%s\n", err.c_str(), gbnf.c_str());
    CHECK(g);
    if (g) {
        const std::string head = "<tool_call>\n<function=send>\n<parameter=cfg>\n";
        CHECK(ok(g, head + R"({"input":"hello world","k":"c"})" + "\n</parameter>\n</function>\n</tool_call>"));
        CHECK(!pre(g, head + R"({"input":"bad|pipe")"));   // the pattern excludes |
    }
    // a schema pattern naming raw control points directly
    auto p = build(R"({"type":"string","pattern":"^[A\u0000]+$"})");
    CHECK(p);
    if (p) {
        const std::string nul = std::string(1, '\0');
        CHECK(ok(p, "\"" + nul + "A\"") && ok(p, "\"A\"") && !pre(p, "\"x\""));
    }
    // Grammar::parse says what a raw NUL is instead of "unexpected end of input"
    err.clear();
    CHECK(!Grammar::parse("root ::= \"" + std::string(1, '\0') + "\"", err) && err.find("NUL") != std::string::npos);
}

int main() {
    test_request();
    test_scalars();
    test_enum_const();
    test_object();
    test_array_nested();
    test_ref();
    test_tools();
    test_text_unions();
    test_control_escapes();
    if (g_fail) { std::printf("grammar_schema_test: %d failures\n", g_fail); return 1; }
    std::printf("grammar_schema_test: ok\n");
    return 0;
}
