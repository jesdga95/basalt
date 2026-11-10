"""Mantle structured handoff between BIG and LITTLE models.

Only the canonical transcript reference, plan, evidence and task DAG cross models.
Raw KV / recurrent / draft state is model specific and is rejected anywhere in the
payload; the destination re-prefills from the transcript under its own fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1
MAX_BYTES = 4 * 1024 * 1024
MAX_EVIDENCE = 512
MAX_TASKS = 1024
MAX_DEPTH = 32

# Keys that indicate raw model-specific state. Matched case-insensitively at any depth.
FORBIDDEN_KEYS = frozenset({
    "kv", "kv_cache", "kvcache", "past_key_values", "key_cache", "value_cache",
    "gdn_state", "recurrent_state", "ssm_state", "conv_state", "hidden_states",
    "logits", "ple", "mtp_draft", "draft_state", "pages", "page_table",
    "state_blob", "snapshot", "tensor", "tensors",
})
_FINGERPRINT = re.compile(r"^[0-9a-f]{8,128}$")


class HandoffError(ValueError):
    """Handoff failed validation or integrity checks."""


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _scan(obj: Any, path: str, depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        raise HandoffError(f"{path}: nesting deeper than {MAX_DEPTH}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise HandoffError(f"{path}: non-string key")
            if k.lower() in FORBIDDEN_KEYS:
                raise HandoffError(f"{path}.{k}: raw model state is not allowed in a handoff")
            _scan(v, f"{path}.{k}", depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _scan(v, f"{path}[{i}]", depth + 1)
    elif isinstance(obj, (bytes, bytearray, memoryview)):
        raise HandoffError(f"{path}: binary data is not allowed in a handoff")
    elif obj is not None and not isinstance(obj, (str, int, float, bool)):
        raise HandoffError(f"{path}: unsupported type {type(obj).__name__}")


def _check_dag(dag: Any) -> None:
    if not isinstance(dag, dict) or not isinstance(dag.get("nodes"), list):
        raise HandoffError("task_dag must be an object with a 'nodes' list")
    nodes = dag["nodes"]
    if len(nodes) > MAX_TASKS:
        raise HandoffError(f"task_dag has more than {MAX_TASKS} nodes")
    deps: dict[str, list[str]] = {}
    for n in nodes:
        if not isinstance(n, dict) or not isinstance(n.get("id"), str) or not n["id"]:
            raise HandoffError("every task_dag node needs a non-empty string 'id'")
        d = n.get("deps", [])
        if not isinstance(d, list) or not all(isinstance(x, str) for x in d):
            raise HandoffError(f"task {n['id']!r}: deps must be a list of ids")
        if n["id"] in deps:
            raise HandoffError(f"duplicate task id {n['id']!r}")
        deps[n["id"]] = d
    for tid, d in deps.items():
        for x in d:
            if x not in deps:
                raise HandoffError(f"task {tid!r} depends on unknown task {x!r}")
    # Kahn's algorithm: anything left over is on (or behind) a cycle.
    indeg = {t: len(d) for t, d in deps.items()}
    rev: dict[str, list[str]] = {t: [] for t in deps}
    for t, d in deps.items():
        for x in d:
            rev[x].append(t)
    ready = [t for t, n in indeg.items() if n == 0]
    seen = 0
    while ready:
        t = ready.pop()
        seen += 1
        for c in rev[t]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if seen != len(deps):
        raise HandoffError("task_dag contains a cycle")


@dataclass(frozen=True)
class Handoff:
    plan: Any
    evidence: list = field(default_factory=list)
    task_dag: dict = field(default_factory=lambda: {"nodes": []})
    transcript_ref: str = ""
    source_fingerprint: str = ""
    target_model: str = ""

    def to_body(self) -> dict:
        return {
            "plan": self.plan,
            "evidence": self.evidence,
            "task_dag": self.task_dag,
            "transcript_ref": self.transcript_ref,
            "source_fingerprint": self.source_fingerprint,
            "target_model": self.target_model,
        }

    def validate(self) -> "Handoff":
        if not isinstance(self.plan, (str, dict)) or not self.plan:
            raise HandoffError("plan must be a non-empty string or object")
        if not isinstance(self.evidence, list):
            raise HandoffError("evidence must be a list")
        if len(self.evidence) > MAX_EVIDENCE:
            raise HandoffError(f"more than {MAX_EVIDENCE} evidence items")
        if not isinstance(self.transcript_ref, str) or not 0 < len(self.transcript_ref) <= 512:
            raise HandoffError("transcript_ref must be a non-empty string (<=512 chars)")
        if not isinstance(self.source_fingerprint, str) or not _FINGERPRINT.fullmatch(self.source_fingerprint):
            raise HandoffError("source_fingerprint must be lowercase hex")
        if not isinstance(self.target_model, str) or not self.target_model:
            raise HandoffError("target_model is required")
        _scan(self.to_body(), "handoff")
        _check_dag(self.task_dag)
        try:
            size = len(_canonical(self.to_body()))
        except ValueError as exc:
            raise HandoffError(f"not JSON serializable: {exc}") from exc
        if size > MAX_BYTES:
            raise HandoffError(f"handoff exceeds {MAX_BYTES} bytes")
        return self

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.to_body())).hexdigest()

    def dumps(self) -> str:
        """Validated JSON envelope with schema version and sha256 integrity."""
        self.validate()
        env = {"schema_version": SCHEMA_VERSION, "sha256": self.digest, "handoff": self.to_body()}
        return json.dumps(env, sort_keys=True, ensure_ascii=False, allow_nan=False)

    @classmethod
    def loads(cls, text: "str | bytes") -> "Handoff":
        raw = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        if len(raw) > MAX_BYTES + 4096:
            raise HandoffError("handoff envelope too large")
        try:
            env = json.loads(raw)
        except (ValueError, RecursionError) as exc:
            raise HandoffError(f"invalid JSON: {exc}") from exc
        if not isinstance(env, dict):
            raise HandoffError("envelope must be an object")
        if env.get("schema_version") != SCHEMA_VERSION:
            raise HandoffError(f"unsupported schema_version {env.get('schema_version')!r}")
        body = env.get("handoff")
        if not isinstance(body, dict):
            raise HandoffError("missing handoff body")
        extra = set(body) - {"plan", "evidence", "task_dag", "transcript_ref",
                             "source_fingerprint", "target_model"}
        if extra:
            raise HandoffError(f"unknown handoff fields: {sorted(extra)}")
        try:
            h = cls(plan=body.get("plan"), evidence=body.get("evidence", []),
                    task_dag=body.get("task_dag", {"nodes": []}),
                    transcript_ref=body.get("transcript_ref", ""),
                    source_fingerprint=body.get("source_fingerprint", ""),
                    target_model=body.get("target_model", ""))
            _scan(h.to_body(), "handoff")  # reject state fields before hashing odd types
            if h.digest != env.get("sha256"):
                raise HandoffError("sha256 mismatch: handoff corrupted or tampered")
        except (TypeError, ValueError) as exc:
            if isinstance(exc, HandoffError):
                raise
            raise HandoffError(f"malformed handoff: {exc}") from exc
        return h.validate()
