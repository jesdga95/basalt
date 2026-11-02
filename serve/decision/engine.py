"""Host decision assembly over native prefix checkpoints and exact target logits."""
from __future__ import annotations

import contextlib
import time

from serve.decision.admission import Control, Gate
from serve.decision.batches import Batches
from serve.decision import schema, systemone
from serve.decision.trie import Trie, common_size, probabilities, summary
from serve.engine.errors import EngineError


class Decisions:
    def __init__(self, service, pending=16, timeout=300, max_nodes=8192, tree_max=128):
        self.svc = service
        self.gate = Gate(pending)
        self.timeout, self.max_nodes, self.tree_max = timeout, max_nodes, tree_max
        self.batches = Batches(self)

    @contextlib.contextmanager
    def lease(self, control):
        with self.gate.exclusive(control):
            with contextlib.ExitStack() as stack:
                while not self.svc.fifo.acquire(timeout=0.1):
                    control.check()
                stack.callback(self.svc.fifo.release)
                control.check()
                self.svc.ensure_loaded()
                try:
                    yield
                finally:
                    self.svc.last_request_at = time.time()

    def _score(self, rows, control):
        return self.batches.score(rows, control)

    def run(self, body, cancel, system_one=False):
        started = time.monotonic()
        control = Control(cancel, self.timeout)
        with self.gate.admit():
            parsed = systemone.parse(body) if system_one else schema.parse(body, self.tree_max)
            if not callable(getattr(self.svc.engine, "score", None)):
                raise EngineError("Jev requires the native Basalt scoring engine")
            if not self.svc.loaded() or getattr(self.svc.engine, "asleep", False) is True:
                with self.lease(control):
                    pass
            result = self.evaluate(parsed[:6], control, started, system_one)
            return systemone.format_response(result, parsed[1], parsed[6], self.svc.model) if system_one else result

    def evaluate(self, parsed, control, started, retain=False):
        instructions, fields, contexts, mode, threshold, cache = parsed
        if not retain:
            catalogue = [{"name": f.name, "description": f.description, "values": f.values} for f in fields]
            instructions += "\nAnswer each field independently as JSON, using its allowed values.\n" + schema.dump(catalogue)
        single = retain and len(fields) == 1
        if single:
            instructions += "\n\nQuestion:\n" + fields[0].question

        def render(context):
            control.check()
            messages = [{"role": "system", "content": instructions}, {"role": "user", "content": context}]
            return self.svc.render_text(messages, None, {"enable_thinking": False})

        def tokenize(prompt, tail):
            control.check()
            rendered, plain = prompt
            # Complete continuations are encoded together: BPE may cross the JSON value boundary.
            tokens = self.svc.tok.encode(rendered + tail, parse_special=True, plain=plain)
            schema.require(bool(tokens) and len(tokens) + 9 <= self.svc.engine.max_context,
                           "decision prompt exceeds the context; requests are never truncated")
            return tokens

        static = tokenize(render(""), "{\n")
        usage = {"prompt_tokens": 0, "cached_tokens": 0, "computed_tokens": 0, "context_tokens": 0, "scored_rows": 0}
        results, states, prefill_ms, forward, rounds = [], [], 0.0, 0, 0
        largest, workspace, deduplicated = 0, 0, 0
        native_ms = 0.0
        fresh = not cache
        planned = 0
        for context in contexts:
            control.check()
            tries = []
            for f in fields:
                local = context if not retain or single else context + "\n\nQuestion:\n" + f.question
                key = "answer" if retain else f.name
                opening = "{\n  " + schema.dump(key) + ": "
                prompt = render(local)
                trie = Trie(tokenize(prompt, opening + value + "\n}") for value in f.encoded)
                planned += len(trie.nodes)
                schema.require(planned <= self.max_nodes, "decision exceeds jev-max-nodes")
                tries.append(trie)
            trunk = tries[0].prefix
            for trie in tries[1:]:
                trunk = trunk[:common_size(trunk, trie.prefix)]
            root = common_size(trunk, static)
            context_n = len(trunk) - root
            usage["context_tokens"] += context_n
            result = {"decision": {}, "fields": {}, "usage": {"context_tokens": context_n, "scored_rows": 0}}
            results.append(result)
            for f, trie in zip(fields, tries):
                tree = mode == "tree" or mode == "auto" and len(f.values) <= threshold
                points = sorted({root, len(trunk), len(trie.prefix)})
                states.append(dict(field=f, trie=trie, tree=tree, points=points, scores={}, current=trie.root,
                                   path_p=1.0, count=0, result=result, done=False))
        while True:
            ready, owners = [], []
            for state in states:
                if state["done"] or state["current"] < 0:
                    continue
                trie = state["trie"]
                indices = range(len(trie.nodes)) if state["tree"] else [state["current"]]
                for index in indices:
                    if index in state["scores"]:
                        continue
                    if len(ready) == 256:
                        break
                    node = trie.nodes[index]
                    ready.append((trie.prefix + node.prompt, node.options, state["points"], fresh))
                    fresh = False
                    owners.append((state, index))
            if not ready:
                break
            rounds += 1
            # Cache opt-out establishes its fresh prefix before any dependent rows can be reordered.
            scored = []
            if ready[0][3]:
                scored.extend(self._score(ready[:1], control))
                ready_tail = ready[1:]
            else:
                ready_tail = ready
            if ready_tail:
                scored.extend(self._score(ready_tail, control))
            for row, (state, index), (values, stats) in zip(ready, owners, scored):
                ids = row[0]
                state["scores"][index] = values
                state["count"] += 1
                state["result"]["usage"]["scored_rows"] += 1
                reused = stats.get("reused", 0)
                if not isinstance(reused, int) or not 0 <= reused <= len(ids):
                    raise EngineError("Jev returned invalid prefix accounting")
                usage["prompt_tokens"] += len(ids)
                usage["cached_tokens"] += reused
                usage["computed_tokens"] += len(ids) - reused
                usage["scored_rows"] += 1
                prefill_ms += stats.get("prompt_ms", 0.0)
                native_ms += stats.get("decode_ms", 0.0)
                forward += stats.get("scored_windows", 1)
                largest = max(largest, stats.get("batch_size", 1))
                workspace = max(workspace, stats.get("workspace_bytes", 0))
                deduplicated += stats.get("deduplicated_rows", 0)
            for state in states:
                if state["done"] or state["current"] < 0:
                    continue
                if state["tree"]:
                    state["done"] = len(state["scores"]) == len(state["trie"].nodes)
                else:
                    if state["current"] not in state["scores"]:
                        continue
                    ps_edge = probabilities(state["scores"][state["current"]])
                    edge = max(range(len(ps_edge)), key=ps_edge.__getitem__)
                    state["path_p"] *= ps_edge[edge]
                    state["current"] = state["trie"].nodes[state["current"]].children[edge]
        for state in states:
            f, trie = state["field"], state["trie"]
            if state["tree"]:
                ps = trie.distribution(state["scores"])
            else:
                ps = [0.0] * len(f.values)
                ps[-state["current"] - 1] = state["path_p"]
            info = summary(f, ps, state["count"], state["tree"])
            if retain:
                info["probabilities"] = ps
            state["result"]["fields"][f.name] = info
            state["result"]["decision"][f.name] = info["value"]
        control.check()
        total = (time.monotonic() - started) * 1000
        return {"object": "decision", "results": results, "usage": usage,
                "execution": {"requested_lanes": getattr(self.svc.engine, "info", {}).get("jev", 1),
                              "effective_lanes": max(1, largest), "dedicated_kv_bytes": workspace,
                              "kv": getattr(self.svc.engine, "info", {}).get("kv")},
                "timings": {"prefill_ms": prefill_ms, "scoring_ms": max(0, total - prefill_ms), "total_ms": total,
                            "native_scoring_ms": native_ms,
                            "rounds": rounds, "scored_windows": forward, "largest_batch": largest,
                            "deduplicated_rows": deduplicated,
                            "per_decision_ms": total / len(contexts)}}
