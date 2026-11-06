# Sampling and thinking

A request's sampling fields win. A field it leaves out takes the server option of the same name, then the model file's `sampling`
default, then the engine's. `/props` reports what is in effect. An out-of-range value is a 400 that names the field.

| Field | Type, values | What it does |
| --- | --- | --- |
| `temperature` | number >= 0 | `0` is greedy. Greedy requests stay exact under every other setting. |
| `top_p` | (0, 1] | Keep the smallest set reaching this probability. `1` is off. |
| `top_k` | integer >= 0 | Keep the K likeliest tokens. `0` or `-1` is off. |
| `min_p` | [0, 1] | Drop tokens below `min_p` times the top token's probability. |
| `repetition_penalty` | > 0 | Multiplicative penalty on tokens in the window. `1` is off. |
| `presence_penalty`, `frequency_penalty` | number | Flat, and count-based, penalties. `0` is off. |
| `penalty_last_n` | 0..4096 | Window for the penalties (default 64). |
| `seed` | integer | Fixes the draw for a reproducible sampled reply. |
| `relaxed_drafting` | [0, 1] or `false` | This request's relaxed-drafting threshold. `false` or `0` forces exact sampling. |

Also read: `max_tokens` / `max_completion_tokens` (`0` or `-1` = the rest of the context), `stop`, `logit_bias`, `reasoning_effort`,
`enable_thinking`, `reasoning_budget_tokens` (`0` = no thinking budget). Anthropic's `thinking.budget_tokens` is the same budget.

## Thinking

- **Budget:** with `answer-room` on and a cap of at least 1,024 tokens, thinking is limited to the cap minus `min(max(512, cap // 4), cap // 2)`.
  An 8,192-token cap allows 6,144 thinking tokens, and 2,048 remain for the close and the answer. Reserved space does not guarantee a
  complete answer.

## Thinking loops

- **Loops:** when one token repeats 256 times inside the thinking, the server closes the thinking and the model answers
  (`thinking-loop-tokens`). Only the thinking is watched. `finish_reason` stays honest.

## The model file's defaults

The model's GGUF metadata keys `general.sampling.temp`, `top_p`, `top_k`, `min_p`, `penalty_repeat` and `penalty_last_n` are the defaults of
the matching fields. A bad value stops the start. The defaults in effect print on the model's line (`temp=1.0 top_p=0.95 ...`).
