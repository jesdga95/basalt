"""Shared helpers for quality suite v2: token roles from Gemma 4 chat markup, reference-file IO."""
import numpy as np

# role codes per token: 0 = BOS/none, 1 = assistant (generated, scored headline), 2 = user/system/template/tool-response
ROLE_ASSISTANT, ROLE_OTHER = 1, 2


def special_ids(tok):
    n = ["<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|tool_response>", "<tool_response|>", "<bos>"]
    return {k: tok.convert_tokens_to_ids(k) for k in n}


def token_roles(ids, S, model_id):
    """Role per token. Assistant span: after "<|turn> model \\n" (and an optional "<|channel>...<channel|>"
    empty-thought header) up to and including "<turn|>". Tool-response blocks inside an assistant turn are
    role 2 (they are fed by the harness, not generated). Everything else (except BOS) is role 2."""
    roles = np.full(len(ids), ROLE_OTHER, dtype=np.uint8)
    i, n = 0, len(ids)
    while i < n:
        if ids[i] == S["<bos>"]:
            roles[i] = 0
        if ids[i] == S["<|turn>"] and i + 1 < n and ids[i + 1] == model_id:
            j = i + 2
            if j < n and ids[j] != S["<|channel>"]:
                j += 1  # the newline after the role name
            elif j < n:
                pass
            # newline token follows "model"; then optional empty thought header
            j = i + 3 if (i + 2 < n) else n
            if j < n and ids[j] == S["<|channel>"]:
                while j < n and ids[j] != S["<channel|>"]:
                    j += 1
                j += 1
            in_tool = False
            while j < n:
                t = ids[j]
                if t == S["<|tool_response>"] and not in_tool:
                    roles[j] = ROLE_ASSISTANT  # generated stop token that triggers the tool
                    in_tool = True
                    j += 1
                    continue
                if in_tool:
                    roles[j] = ROLE_OTHER
                    if t == S["<tool_response|>"]:
                        in_tool = False
                    j += 1
                    continue
                roles[j] = ROLE_ASSISTANT
                j += 1
                if t == S["<turn|>"]:
                    break
            i = j
            continue
        i += 1
    return roles


def smoke_convs(conv_off, stride=20, max_len=5000):
    """--smoke: a fixed deterministic slice of the corpus, every `stride`-th conversation shorter than max_len tokens
    (the corpus is ordered chat, code, reasoning, multilingual, tools, long, so this mixes the kinds)."""
    n = len(conv_off) - 1
    return [c for c in range(n) if c % stride == 0 and conv_off[c + 1] - conv_off[c] < max_len]
