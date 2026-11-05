"""TypeSafe-shaped questions, compiled locally; Noul is always P(true)."""
from __future__ import annotations

import math

from serve.decision.schema import TEXT_MAX, dump, field, keys, require, text
from serve.engine.errors import EngineError

# A Box answer is four grid coordinates scored on their own; 0.01 is the precision Jev reports them at.
EDGES = ("top", "bottom", "left", "right")
BOX_STEP = 0.01
IMAGES_MAX = 8


def entry(value):
    require(value is None or isinstance(value, (str, dict, list)), "entry: text, object, array or null")
    return value if isinstance(value, str) else dump(value)


def parse(body):
    keys(body, ("state", "questions", "model", "images"), "systemone")
    require("state" in body, "state is required")
    state = entry(body["state"])
    qs = body.get("questions")
    require(isinstance(qs, dict) and 1 <= len(qs) <= 32, "questions: 1..32 named questions")
    images = body.get("images", [])
    require(isinstance(images, list) and len(images) <= IMAGES_MAX and
            all(isinstance(source, str) and bool(source) for source in images),
            f"images: up to {IMAGES_MAX} image sources")
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
            spec, edges = {"type": "enum", "choices": list(criteria)}, (None,)
        elif kind == "noul":
            criteria = {} if criteria is None else criteria
            keys(criteria, ("true", "false"), "noul criteria")
            for desc in criteria.values():
                entry(desc)
            spec, edges = {"type": "boolean"}, (None,)
        elif kind == "score":
            require(isinstance(criteria, list) and 2 <= len(criteria) <= 10, "score: 2..10 ordered levels")
            for desc in criteria:
                entry(desc)
            spec, edges = {"type": "integer", "minimum": 0, "maximum": len(criteria) - 1}, (None,)
        elif kind == "box":
            require(bool(images), "box questions are asked of an image: send images")
            if criteria in (None, {}):
                spec = {"type": "number", "minimum": 0.0, "maximum": 1.0, "step": BOX_STEP}
            else:
                keys(criteria, ("step",), "box criteria")
                spec = {"type": "number", "minimum": 0.0, "maximum": 1.0, "step": criteria["step"]}
            edges = EDGES
        else:
            raise ValueError("question type: choice, noul, score or box")
        for edge in edges:
            f = field(name if edge is None else f"{name}\u0000{edge}",
                      {**spec, "description": "Answer the isolated question."})
            f.question = dump({"type": kind, "instructions": q.get("instructions"), "criteria": criteria}) \
                if edge is None else dump(
                    {"type": "box", "edge": edge, "instructions": q.get("instructions"),
                     "grid": f"a normalized coordinate, 0 at the picture's top-left edge and 1 at its "
                             f"bottom-right one, on a {spec['step']} grid"})
            size += len(f.question.encode("utf-8"))
            fields.append(f)
        questions.append((name, kind, criteria, edges))
    require(size <= TEXT_MAX, "state and questions: exceeds 1 MiB")
    instruction = ("Evaluate each isolated question against the state and its instructions and criteria. Return its "
                   "label, true/false or zero-based level as the JSON answer.")
    if any(kind == "box" for _, kind, _, _ in questions):
        instruction += " A coordinate answer names where the asked-for thing sits in the picture, 0 at its "\
                       "top-left and 1 at its bottom-right."
    return instruction, fields, ["State:\n" + state], "tree", 255, True, questions, tuple(images)


def edge_summary(values, ps):
    """One grid coordinate: its distribution's mean, and a confidence that is one minus the expected distance from
    the modal value divided by the uniform distance from the grid's centre, floored at zero."""
    modal = values[max(range(len(ps)), key=ps.__getitem__)]
    centre = (values[0] + values[-1]) / 2
    uniform = sum(abs(v - centre) for v in values) / len(values)
    spread = sum(p * abs(v - modal) for v, p in zip(values, ps))
    return {"value": math.fsum(v * p for v, p in zip(values, ps)), "modal": modal,
            "confidence": max(0.0, 1 - spread / uniform)}


def format_response(result, fields, questions, model):
    answers, details = {}, result["results"][0]["fields"]
    at = 0
    for name, kind, criteria, edges in questions:
        group, at = fields[at:at + len(edges)], at + len(edges)
        require(at <= len(fields), "Jev returned fewer fields than the questions")
        distributions = []
        for f in group:
            ps = details[f.name]["probabilities"]
            if not (len(ps) == len(f.values) and all(math.isfinite(p) and 0 <= p <= 1 for p in ps) and
                    abs(math.fsum(ps) - 1) < 1e-8):
                raise EngineError("backend returned an invalid distribution")
            distributions.append(ps)
        if kind == "noul":
            answers[name] = {"type": kind, "noul": distributions[0][0], "confidence": distributions[0][0]}
            continue
        if kind == "box":
            # the four edges are scored on their own, so each pair is sorted: an answer never inverts the box
            summary = {edge: edge_summary(f.values, ps) for edge, f, ps in zip(EDGES, group, distributions)}
            pair = {edge: sorted((summary[a]["value"], summary[b]["value"]))
                    for edge, (a, b) in (("top", ("top", "bottom")), ("left", ("left", "right")))}
            answer = {"type": kind,
                      "box": {"top": pair["top"][0], "bottom": pair["top"][1],
                              "left": pair["left"][0], "right": pair["left"][1]},
                      "confidence": min(s["confidence"] for s in summary.values())}
            answer["edges"] = {edge: {"probability": max(ps), "modal": summary[edge]["modal"],
                                      "moved": summary[edge]["value"] != answer["box"][edge]}
                               for edge, ps in zip(EDGES, distributions)}
            answers[name] = answer
            continue
        f, ps, winner = group[0], distributions[0], max(range(len(distributions[0])), key=distributions[0].__getitem__)
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
        answers[name] = answer
    return {"model": model, "answers": answers,
            "usage": {"input_tokens": result["usage"]["prompt_tokens"], "output_tokens": 0},
            **{key: result[key] for key in ("execution", "timings") if key in result}}
