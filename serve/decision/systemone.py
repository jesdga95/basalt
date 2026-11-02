"""TypeSafe-shaped questions, compiled locally; Noul is always P(true)."""
from __future__ import annotations

import math

from serve.decision.schema import TEXT_MAX, dump, field, keys, require, text
from serve.engine.errors import EngineError


def entry(value):
    require(value is None or isinstance(value, (str, dict, list)), "entry: text, object, array or null")
    return value if isinstance(value, str) else dump(value)


def parse(body):
    keys(body, ("state", "questions", "model"), "systemone")
    require("state" in body, "state is required")
    state = entry(body["state"])
    qs = body.get("questions")
    require(isinstance(qs, dict) and 1 <= len(qs) <= 32, "questions: 1..32 named questions")
    if "model" in body:
        text(body["model"], "model", False)
    fields, questions = [], []
    size = len(state.encode("utf-8"))
    for name, q in qs.items():
        keys(q, ("type", "instructions", "criteria"), "question")
        kind, criteria = q.get("type"), q.get("criteria")
        entry(q.get("instructions"))
        if kind == "choice":
            require(isinstance(criteria, dict) and 1 <= len(criteria) <= 255, "choice: 1..255 named criteria")
            for label, desc in criteria.items():
                require(isinstance(label, str) and bool(label), "choice: empty label")
                entry(desc)
            spec = {"type": "enum", "choices": list(criteria)}
        elif kind == "noul":
            criteria = {} if criteria is None else criteria
            keys(criteria, ("true", "false"), "noul criteria")
            for desc in criteria.values():
                entry(desc)
            spec = {"type": "boolean"}
        elif kind == "score":
            require(isinstance(criteria, list) and 2 <= len(criteria) <= 10, "score: 2..10 ordered levels")
            for desc in criteria:
                entry(desc)
            spec = {"type": "integer", "minimum": 0, "maximum": len(criteria) - 1}
        else:
            raise ValueError("question type: choice, noul or score")
        f = field(name, {**spec, "description": "Answer the isolated question."})
        f.question = dump({"type": kind, "instructions": q.get("instructions"), "criteria": criteria})
        size += len(f.question.encode("utf-8"))
        fields.append(f)
        questions.append((kind, criteria))
    require(size <= TEXT_MAX, "state and questions: exceeds 1 MiB")
    return ("Evaluate each isolated question against the state and its instructions and criteria. "
            "Return its label, true/false, or zero-based level as the JSON answer."), fields, ["State:\n" + state], "tree", 255, True, questions


def format_response(result, fields, questions, model):
    answers = {}
    details = result["results"][0]["fields"]
    for f, (kind, criteria) in zip(fields, questions):
        ps = details[f.name]["probabilities"]
        if not (len(ps) == len(f.values) and all(math.isfinite(p) and 0 <= p <= 1 for p in ps) and
                abs(math.fsum(ps) - 1) < 1e-8):
            raise EngineError("backend returned an invalid distribution")
        winner = max(range(len(ps)), key=ps.__getitem__)
        if kind == "noul":
            answers[f.name] = {"type": kind, "noul": ps[0]}
            continue
        n = len(ps)
        if kind == "score":
            uniform = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
            confidence = max(0.0, 1 - sum(p * abs(i - winner) for i, p in enumerate(ps)) / uniform)
        else:
            confidence = 1.0 if n == 1 else max(0.0, min(1.0, (max(ps) * n - 1) / (n - 1)))
        answer = {"type": kind, "confidence": confidence,
                  "probabilities": {str(i) if kind == "score" else v: p for i, (v, p) in enumerate(zip(f.values, ps))}}
        if kind == "score":
            answer.update(score=math.fsum(i * p for i, p in enumerate(ps)), legend={str(i): v for i, v in enumerate(criteria)})
        else:
            answer["choice"] = f.values[winner]
        answers[f.name] = answer
    return {"model": model, "answers": answers,
            "usage": {"input_tokens": result["usage"]["prompt_tokens"], "output_tokens": 0},
            **{key: result[key] for key in ("execution", "timings") if key in result}}
