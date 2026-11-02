"""Finite schema compilation. Reject domains that cannot be represented exactly."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal

TEXT_MAX = 1 << 20


def require(ok, message):
    if not ok:
        raise ValueError(message)


def keys(obj, allowed, where):
    require(isinstance(obj, dict), f"{where}: expected an object")
    require(not obj.keys() - set(allowed), f"{where}: unsupported members {sorted(obj.keys() - set(allowed))}")


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def text(value, where, empty=True):
    require(isinstance(value, str) and (empty or bool(value)), f"{where}: expected text")
    require(len(value.encode("utf-8")) <= TEXT_MAX, f"{where}: exceeds 1 MiB")
    return value


def integer(value, lo, hi, where):
    require(type(value) is int and lo <= value <= hi, f"{where}: expected an integer in {lo}..{hi}")
    return value


def decimal(value):
    require(type(value) in (int, float) and abs(value) <= 1e100 and math.isfinite(value), "grid: expected finite numbers")
    require(type(value) is not int or int(float(value)) == value, "grid: integer loses precision as a double")
    result = Decimal(str(value))
    require(result.as_tuple().exponent >= -9, "grid: at most nine decimal places")
    return result


@dataclass
class Field:
    name: str
    description: str
    values: list
    encoded: list[str]
    aggregate: str = "mode"
    question: str = ""


def field(name, spec, wrapped=False):
    keys(spec, ("type", "description", "title", "enum", "choices", "minimum", "maximum", "step",
                "multipleOf", "aggregate", "x-aggregate"), "schema field")
    require(isinstance(name, str) and 0 < len(name.encode("utf-8")) <= 4096, "field name: 1..4096 bytes")
    desc = text(spec.get("description", ""), "description", wrapped)
    kind = spec.get("type")
    require(not ("enum" in spec and "choices" in spec), "use enum or choices, not both")
    if "enum" in spec:
        require(kind in (None, "string", "enum"), "enum: expected a string domain")
        kind = "enum"
    choice = kind in ("enum", "choice", "selection")
    require(choice or "choices" not in spec, "choices require an enum")
    require(kind in ("integer", "number") or not {"minimum", "maximum"} & spec.keys(), "bounds require numbers")
    require(kind == "number" or not {"step", "multipleOf"} & spec.keys(), "step requires a number grid")
    require(not ("aggregate" in spec and "x-aggregate" in spec), "use aggregate or x-aggregate")
    encoded = None
    if kind == "boolean":
        values = [True, False]
    elif choice:
        values = spec.get("enum", spec.get("choices"))
        require(isinstance(values, list) and 1 <= len(values) <= 255, "choices: 1..255 strings")
        require(all(isinstance(v, str) and len(v.encode("utf-8")) <= 4096 for v in values), "choices: strings of at most 4096 bytes")
    elif kind == "integer":
        lo = integer(spec.get("minimum"), -(1 << 63), (1 << 63) - 1, "minimum")
        hi = integer(spec.get("maximum"), lo, min((1 << 63) - 1, lo + 254), "maximum")
        values = list(range(lo, hi + 1))
    elif kind == "number":
        step_key = "multipleOf" if wrapped else "step"
        require(step_key in spec and ("step" if wrapped else "multipleOf") not in spec,
                f"number: use {step_key}")
        lo, hi, step = (decimal(spec.get(k)) for k in ("minimum", "maximum", step_key))
        require(step > 0 and hi >= lo, "grid: ordered bounds and a positive step")
        count = (hi - lo) / step
        require(count == count.to_integral_value() and count <= 254, "grid: both bounds must lie on a 1..255 value grid")
        require(not wrapped or lo % step == 0, "minimum must lie on the multipleOf grid")
        numbers = [lo + step * i for i in range(int(count) + 1)]
        encoded = [format(v, "f") for v in numbers]
        values = [json.loads(v) for v in encoded]
        require(all(math.isfinite(v) for v in values), "grid values must be finite")
        require(all(Decimal(str(v)) == n for v, n in zip(values, numbers)), "grid values lose precision")
    else:
        raise ValueError("field type: boolean, enum, integer or number")
    encoded = encoded or [dump(v) for v in values]
    require(len(set(encoded)) == len(encoded), "duplicate field values")
    agg = spec.get("aggregate", spec.get("x-aggregate", "mode"))
    require(agg == "mode" or kind in ("integer", "number") and agg in ("mean", "median"), "invalid aggregate")
    return Field(name, desc, values, encoded, agg)


def parse(body, default_tree=128):
    keys(body, ("model", "instructions", "schema", "contexts", "mode", "tree_max", "cache_prompt"), "decision")
    instructions = text(body.get("instructions", ""), "instructions")
    if "model" in body:
        text(body["model"], "model", False)
    mode = body.get("mode", "auto")
    require(mode in ("auto", "tree", "greedy"), "mode: auto, tree or greedy")
    threshold = integer(body.get("tree_max", default_tree), 1, 255, "tree_max")
    cache = body.get("cache_prompt", True)
    require(type(cache) is bool, "cache_prompt: true or false")
    contexts = body.get("contexts")
    require(isinstance(contexts, list) and 1 <= len(contexts) <= 256, "contexts: 1..256 strings")
    for context in contexts:
        text(context, "context")
    require(sum(len(c.encode("utf-8")) for c in contexts) <= TEXT_MAX, "contexts: exceeds 1 MiB")
    schema = body.get("schema")
    require(isinstance(schema, dict), "schema: expected an object")
    properties = schema.get("properties")
    wrapped = schema.get("type") == "object" or (isinstance(properties, dict) and
               all(isinstance(v, dict) for v in properties.values()))
    if wrapped:
        keys(schema, ("$schema", "type", "properties", "required", "additionalProperties", "description", "title"), "schema")
        require(schema.get("type", "object") == "object", "schema type: object")
        props = schema.get("properties")
        required = schema.get("required", [])
        require(isinstance(required, list) and all(isinstance(k, str) and k in (props or {}) for k in required), "invalid required fields")
        require(type(schema.get("additionalProperties", False)) is bool, "additionalProperties: boolean")
    else:
        props = schema
    require(isinstance(props, dict) and 1 <= len(props) <= 32, "schema: 1..32 fields")
    fields = [field(k, v, wrapped) for k, v in props.items()]
    require(len(dump(schema).encode("utf-8")) + len(instructions.encode("utf-8")) <= TEXT_MAX, "catalogue: exceeds 1 MiB")
    for f in fields:
        require(f.aggregate == "mode" or mode != "greedy" and (mode == "tree" or len(f.values) <= threshold),
                "mean/median require tree mode")
    return instructions, fields, contexts, mode, threshold, cache
