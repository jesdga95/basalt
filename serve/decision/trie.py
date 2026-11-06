"""Compressed finite-token tries; only divergent next-token rows need a target-head read."""
from __future__ import annotations

import math
from dataclasses import dataclass

from serve.decision.schema import require
from serve.engine.errors import EngineError


def probabilities(logits):
    if not logits or not all(math.isfinite(v) for v in logits):
        raise EngineError("backend returned invalid logits")
    peak = max(logits)
    weights = [math.exp(v - peak) for v in logits]
    total = math.fsum(weights)
    return [v / total for v in weights]


@dataclass
class Node:
    prompt: tuple[int, ...]
    options: list[int]
    children: list


class Trie:
    def __init__(self, paths):
        self.prefix, self.paths = None, []
        for full in paths:
            full = tuple(full)
            if self.prefix is None:
                self.prefix = full
                self.paths.append(())
                continue
            common = common_size(self.prefix, full)
            if common < len(self.prefix):
                tail = self.prefix[common:]
                self.paths = [tail + p for p in self.paths]
                self.prefix = self.prefix[:common]
            self.paths.append(full[common:])
        require(self.prefix is not None and len(set(self.paths)) == len(self.paths), "candidate tokenizations collide")
        self.nodes = []
        self.root = self._build(list(range(len(self.paths))), 0)

    def _build(self, members, depth):
        if len(members) == 1:
            return -members[0] - 1
        first = self.paths[members[0]]
        while all(len(self.paths[i]) > depth and self.paths[i][depth] == first[depth] for i in members):
            depth += 1
            require(depth < len(first), "one candidate token path prefixes another")
        require(all(len(self.paths[i]) > depth for i in members), "one candidate token path prefixes another")
        groups = {}
        for i in members:
            groups.setdefault(self.paths[i][depth], []).append(i)
        node = Node(tuple(first[:depth]), list(groups), [])
        index = len(self.nodes)
        self.nodes.append(node)
        for group in groups.values():
            node.children.append(self._build(group, depth + 1))
        return index

    def distribution(self, scores):
        out = [0.0] * len(self.paths)
        pending = [(self.root, 1.0)]
        while pending:
            index, weight = pending.pop()
            if index < 0:
                out[-index - 1] = weight
            else:
                node = self.nodes[index]
                pending.extend(zip(node.children, [weight * p for p in probabilities(scores[index])]))
        total = math.fsum(out)
        if not total > 0 or not math.isfinite(total):
            raise EngineError("candidate distribution underflow")
        return [p / total for p in out]


def common_size(left, right):
    n = 0
    while n < min(len(left), len(right)) and left[n] == right[n]:
        n += 1
    return n


def summary(field, ps, count, tree):
    winner = max(range(len(ps)), key=ps.__getitem__)
    selected = winner
    if field.aggregate == "median":
        total = 0.0
        for i, p in enumerate(ps):
            total += p
            if total >= 0.5:
                selected = i
                break
    elif field.aggregate == "mean":
        base = field.values[0]
        expected = math.fsum(p * (v - base) for p, v in zip(ps, field.values))
        selected = min(range(len(ps)), key=lambda i: abs((field.values[i] - base) - expected))
    value = field.values[selected]
    result = {"value": value, "probability": ps[selected], "scored_nodes": count, "tree": tree}
    if tree and type(value) in (int, float):
        def quantile(q):
            total = 0.0
            for v, p in zip(field.values, ps):
                total += p
                if total >= q:
                    return v
            return field.values[-1]
        result["interval_p10_p90"] = [quantile(0.1), quantile(0.9)]
    return result
