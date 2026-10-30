# Sampling and thinking

The per-request sampling fields, the defaults a request inherits, and how thinking ends.

A request carries its own sampling fields. A field the request leaves out takes the launch option of the same name
(`--temperature`, `--top-p`, ... in the [server options](index.md#server); command line over the config file's `server`
block), then the model file's `sampling` default (below), then the engine's. Request fields always win. `/props`
reports the defaults in effect. A value out of range is refused with a 400 naming the field; a value in range reaches
the sampler as sent.

| Field | Type, values | What it does |
| --- | --- | --- |
| `temperature` | number >= 0 | `0` is greedy (argmax); higher is more random. Greedy requests stay exact under every other setting. |
| `top_p` | number in (0, 1] | Nucleus sampling: keep the smallest set of tokens whose probability adds up to this. `1` is off. |
| `top_k` | integer >= 0 | Keep only the K likeliest tokens. `0` (or `-1`) is off: every token of the vocabulary. |
| `min_p` | number in [0, 1] | Sampling cutoff: drop tokens less likely than `min_p` times the likeliest token's probability. `0` is off. It acts on the model's tokens. |
| `repetition_penalty` | number > 0 | Multiplicative penalty on tokens seen in the window; `1` is off. |
| `presence_penalty` | number | Flat penalty on any token seen in the window; `0` is off. |
| `frequency_penalty` | number | Penalty that grows with how often a token was seen; `0` is off. |
| `penalty_last_n` | integer 0..4096 | The window, in tokens, the three penalties look back over (`0`: none). Applies only when a penalty is set; 64 when omitted. Penalties apply in batch windows too; pipelined batch groups refuse them (400). |
| `seed` | integer | Fixes the random draw for a reproducible sampled reply. |
| `relaxed_drafting` | number in [0, 1], or `false` | This request's relaxed-drafting threshold; `false` or `0` forces exact sampling. Needs speculation; see [Speculative decoding](index.md#speculative-decoding-mtp). |

Other request fields the API reads: `max_tokens` / `max_completion_tokens` (the Anthropic `max_tokens`; `0` or
`-1` means the rest of the context), `stop` / `stop_sequences`, `logit_bias`, `reasoning_effort`, `enable_thinking`
and `reasoning_budget_tokens` (a cap on thinking tokens; `0` is none). Anthropic's `thinking.budget_tokens` and
llama.cpp's `thinking_budget_tokens` are the same budget, and `reasoning_budget_tokens` wins when more than one is
sent. A request that names none of them takes the server's [`reasoning-budget-tokens`](index.md#server).

With `answer-room` on and an effective output cap of at least 1024 tokens, thinking is limited to the cap minus
`min(max(512, cap // 4), cap // 2)`. An 8192-token cap allows up to 6144 thinking tokens; the remaining 2048
include the closing of the thinking and the answer. Smaller positive budgets stay unchanged. Context fitting uses the reduced
cap. Explicit request budgets of `0` disable both the server default and automatic budgeting; smaller output caps
keep their existing behavior. Reserved space reduces empty answers but does not guarantee a complete answer.

### Thinking loops

When the model repeats one token 256 times in a row inside its thinking, the server ends the thinking with the
template's own close (`\n</think>\n\n`) and the model answers. The reply's `finish_reason` stays honest: `stop` when
the answer completes, `length` only at `max_tokens`. The server logs `thinking repeated one token N times, ending it`.
Only the thinking is watched; separators, padding, base64 and digit strings in an answer are never cut. Set
[`thinking-loop-tokens`](index.md#server) to another count, or `0` to turn the safeguard off.

### The model file's sampling defaults

The model file's `meta` section carries a `sampling` object. Its keys are the request fields above, with the same
meaning:

| Key | Allowed values |
| --- | --- |
| `temperature` | number >= 0 |
| `top_p` | 0 < value <= 1 |
| `top_k` | integer >= 0 (`0` or `-1` = off) |
| `min_p` | 0 <= value <= 1 |
| `presence_penalty` | number |
| `frequency_penalty` | number |
| `repetition_penalty` | number > 0 (`1` = off) |
| `penalty_last_n` | integer 0..4096 |
| `seed` | integer |

A bad value stops the server at startup; an unknown key is named in a warning and ignored. The defaults in effect
are printed at startup as `[server] sampling ...`, each marked `(cli)` or `(config)` when a launch
option set it and unmarked when it comes from the model file. A launch option overrides the model file's value for
that key; the others stay.
