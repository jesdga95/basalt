"""Per-architecture conversion rules for tools/convert/convert.py, keyed by the GGUF's `general.architecture`.

An architecture says how its source tensors are stored in the pack (the float form the engine reads, tensors that get
a section of their own), which optional sections it carries, and the license, sampling defaults and extra `meta` keys.
Everything else (tokenizer, dense/gguf, experts, container) is shared.
"""
from __future__ import annotations

import re


class Arch:
    name = ""
    # float form the engine reads each dense tensor in (name without `blk.N.`); empty = tensors are stored as they are
    form_table: dict[str, str] = {}
    # tensors kept out of the dense pool (they are served from `gguf` or from a section of their own)
    not_in_dense: frozenset = frozenset()
    # tensor name -> section that holds it alone (2 MiB aligned); such tensors are left out of `gguf`
    own_sections: dict[str, str] = {}
    # drafter, hc8 and vision sections
    side_sections = True
    # only the vision projector section (vision/mmproj), written when --mmproj is given
    mmproj_only = False
    projector_type = ""
    # takes a DSpark drafter GGUF (--dspark) as the dspark/gguf section
    dspark_ok = False
    # the model's sampling defaults, for a GGUF that does not carry them (llama.cpp's general.sampling.* keys)
    sampling_kv: dict = {}

    def form_of(self, name: str):
        return self.form_table.get(re.sub(r"^blk\.\d+\.", "", name))

    def native(self, name: str, type_name: str) -> bool:
        """A quantized tensor the engine reads from the source bytes as they are."""
        return self.form_of(name) is None

    def license(self, kv: dict) -> dict:
        raise NotImplementedError

    def default_name(self, first_gguf, kv: dict, sections_from: bool = False) -> str:
        raise NotImplementedError

    def meta_extra(self, kv: dict, tensors: dict, own: dict) -> dict:
        """Keys added to `meta`: `tensors` are the `gguf` tensors' entries, `own` the own-section tensors' ones."""
        return {}

    def check(self, kv: dict, own: dict) -> None:
        """Raise ConvertError when the source does not fit this architecture."""


REGISTRY: dict[str, Arch] = {}


def register(arch: Arch, *names: str) -> None:
    for n in names or (arch.name,):
        REGISTRY[n] = arch


def for_architecture(name) -> Arch:
    """The rules for `general.architecture`; an architecture without its own entry is converted as Flash-Next."""
    from . import qwen4exp, deepseek41  # noqa: F401  (register themselves)
    return REGISTRY.get(name) or REGISTRY["qwen4exp"]
