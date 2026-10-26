"""The server's console lines in the compact key=value style: `[server] <label>  <key=value ...>`."""
from __future__ import annotations

LABEL_WIDTH = 8

LEGEND = ("pp=prompt read  tg=generation  c=cached  acc=drafts accepted  dl=tokens per window  "
          "hit=expert cache  pcie=routed experts over PCIe")


def tagged(label: str, text: str) -> str:
    """A startup line: the label padded to a column, then the text."""
    return f"[server] {label:<{LABEL_WIDTH}} {text}"

