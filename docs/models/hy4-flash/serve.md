# Hy4-flash serving

What the server does for `hy4flash` (the engine side is in [Hy4-flash](../hy4-flash.md#serve)). The server reads the architecture from the
pack's `gguf_kv`: `hy_v4_flash` (the quantized test pack) and `hy4flash` both pick the dialect in `serve/models/hy4flash.py`.

## Prompt

The chat template of the pack is rendered by the shared Jinja path (the server's settings are transformers'). The Hugging Face release
(`tencent/Hy4-flash`) and a quantized release write the same template with a different token suffix (`:opensource` and `:6124c78e`), and the
quantized one also accepts `reasoning_effort: low`. The dialect reads the suffix from the template text it is given, so the stop token
(`hy_end`), the thinking close, the image tokens, the tool tags and the plain-text marks follow whichever template the pack carries.
`serve/tests/fixtures` holds the official template and its rendering of a corpus (system, multi-turn, tools, parallel calls and results,
thinking history, images, effort errors) by transformers 5.23's `render_jinja_template`; the tests render the corpus with the server and with
the derived quantized template and compare the strings.

Token text in a message (`<｜hy_end:...｜>`, `<tool_calls:...>`, ...) stays text: only the template writes the real tokens.

## Reasoning controls

| API | Template |
|---|---|
| default, `reasoning_effort` `low`, `medium`, `high`, `reasoning_budget` N > 0 | `reasoning_effort: high`, the turn opens with `<think>` |
| `max`, `xhigh`, `reasoning_budget` -1 | `max` |
| `none`, `off`, `no_think`, `enable_thinking: false`, `reasoning_budget` 0 | `no_think`, the turn opens with an empty `<think></think>` |
| a number | refused (400) |
| `chat_template_kwargs.preserve_reasoning` | `preserved_thinking` (the template's own default is on only when tools are offered) |

`low` is rendered as `high` so one mapping serves both templates. A thinking budget closes the thinking with `</think>`.

## Tools

The model writes `<tool_calls><tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>...</tool_calls>`. A string value is
raw text and any other value is JSON, so the parser decodes each value by the request's schema for that parameter (text where the schema
allows a string, JSON otherwise, as for Flash-Next), streams string arguments as they arrive, and closes a call cut off by the budget
without reporting it. A named `tool_choice` writes the opening of the call itself (`<tool_calls><tool_call>NAME`).

## Images, sampling, context

- Images take Flash-Next's path through `--vision dynamic` (the only mode: a sidecar does not know the newline column): the template
  writes `image_start`, one `image_pad` and `image_end` per image and the server expands the pad to the image's rows, `oh * (ow + 1)`
  of them. The model reads no video yet.
- The pack's sampling keys give the defaults: temperature 0.9, top_p 1.0 and `top_k` -1, which is off. A request's fields win.
- `--max-context` as for the other models; the engine refuses a context the cards cannot hold, and the server plans with the context the
  engine reports.
- `--spec`, `--vram-reserve-mib` and `POST /v1/vram` are the engine's flags and endpoint, as for DeepSeek.

## Sleep, park, swap

`--idle-sleep`, `--idle-unload`, the conversation cache and the router's swap are the server's one state machine for every model; the
engine says whether it can sleep. Not yet run against the model: the unit tests use a scripted engine. The load bars and `[server] planning weight allocation` follow the engine's `basalt load:` lines.
