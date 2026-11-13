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

llama.cpp clients also work: `/props` reports `chat_template_caps` for the model (tool calls, parallel calls, reasoning effort,
system role, preserve reasoning), `chat_template_kwargs.preserve_reasoning` keeps earlier turns' thinking in the prompt, and
`reasoning_budget` is read on the chat endpoint: `0` is thinking off, `-1` is no cap at the highest effort, and N > 0 is an N-token
budget at the level it names (up to 2048 low, up to 8192 medium, else high; DeepSeek has no medium of its own, so it renders its default, high). When fields disagree, off from any of them wins, then
`chat_template_kwargs.reasoning_effort`, then `reasoning_effort`, and last the level a `reasoning_budget` implies. For the token
cap, `reasoning_budget_tokens` wins over `thinking_budget_tokens`, then `reasoning_budget`, then `thinking.budget_tokens`.

## Logprobs

A request can ask for each generated token's log probability and the likeliest tokens at its place. The engine computes them
on the GPU from the row's full head logits: the natural log of the token's probability under the model's own distribution,
before temperature, penalties, `logit_bias`, top-k/top-p and any grammar mask. The reply's tokens are the same with or without
them, and decode speed stays within a few percent.

- **Chat Completions:** `logprobs: true` and `top_logprobs` (0 to 20). `choices[0].logprobs.content` has one entry per answer
  token, `{token, logprob, bytes, top_logprobs: [{token, logprob, bytes}]}`, likeliest first. Streaming: each chunk carries the
  entries of its tokens.
- **Thinking and tool calls:** Basalt adds `logprobs.reasoning_content` (the thinking's tokens) and `logprobs.tool_calls`
  (`[{index, content}]`, each call's tokens as the model wrote them, markup included): the least likely token of a call is where
  the model guessed.
- **Responses:** `include: ["message.output_text.logprobs"]` with `top_logprobs` fills `output_text.logprobs` and each
  `response.output_text.delta`; Basalt adds `logprobs` to reasoning and call items (and to `response.reasoning_text.delta` and
  `response.function_call_arguments.delta`).
- **Anthropic Messages** (a Basalt extension; the API has no such field): the same `logprobs` and `top_logprobs`. Each
  `content_block_delta` carries `logprobs`, the entries of its tokens; the message, or else `message_delta`, carries the
  `logprobs` object above.
- **Uses:** with `response_format` set to a schema whose field is an `enum`, the `top_logprobs` at the value's first
  token give each label's probability (a classifier); in an agent, a tool call whose least likely token sits far below
  its alternative is one the model was unsure of.
- Tokens the server writes itself (the close of a thinking budget) have none. Refused with parallel slots split over GPUs by
  layer (pipelined batch groups).

## Thinking

- **Budget:** `answer-room` keeps output tokens for the close and the answer by limiting the thinking to the cap minus the reserve. A smaller
  budget (the request's, else the entry's `reasoning-budget-tokens`) still wins. Reserved space does not guarantee a complete answer.
  - `auto` (the default): the reserve is `min(max(512, cap // 4), cap // 2)`. An 8,192-token cap allows 6,144 thinking tokens and keeps
    2,048; a 700-token cap allows 350 and keeps 350. A cap under 32 tokens is left as it is.
  - A whole number `N`: a fixed reserve at any cap, larger or smaller than `auto`'s, so thinking is limited to `cap - N` (`1024` with a
    4,096-token cap leaves 3,072). `0` reserves nothing. An `N` of the cap or more leaves thinking one token.
  - In a config file the value is `"auto"` or a whole number, written `100` or `"100"`; on the command line, `--answer-room auto` or
    `--answer-room 100`. A negative number, `true`, `false` or any other text is refused at start.
  Examples: `"answer-room": "auto"`, `"answer-room": 2048` (think longer at large caps), `"answer-room": 0` (no reserve).
  - A request's explicit budget of 0 or less opts out in every mode.

## Answer now

A client can end a reply's thinking while it runs: `POST /v1/answer_now` closes the thinking and the reply goes on to its answer
in the same response. The engine finishes the decode window under way, so a few more thinking tokens may arrive first; the close
then follows them, the way the budget closes, and the engine reads only the close, however long the thinking was. It works on
`/v1/chat/completions`, `/v1/messages` and `/v1/responses`, streaming or not, and with `--parallel` slots. A request nobody asks
runs exactly as before.

- **Naming the request:** every chat response carries `X-Request-Id`, the response's own id (`chatcmpl-...`, `msg_...`, `resp_...`,
  also the `id` of a stream's chunks). Send `{"request_id": "<id>"}`. A client that sends `x-client-request-id` with its request can
  use that value instead, before any reply arrives. Or name the session: `{"session_id": "<id>"}` or the request's
  [session header](../running.md#session-affinity) on the call, which picks the session's newest running request.
- **Reply:** `200 {"object": "answer_now", "request_id": "...", "state": "thinking"}`. A request still `queued` or reading its
  `prompt` takes it too, and closes its thinking at its first token.
- **Refusals:** `409` with `error.code` `not_thinking` and `error.state` `answering` or `finished`, or a message naming thinking off;
  `404` with `unknown_request` or `unknown_session`; `400` `no_target` without a request id or session.

```sh
curl -s localhost:8080/v1/answer_now -H 'Content-Type: application/json' -d '{"request_id": "chatcmpl-..."}'
```

The call takes the API key like every `/v1` route. A page on another origin reads `X-Request-Id` through `--cors-origin`.

### Detecting Basalt

`/props` (llama.cpp's route, unchanged for its clients) carries one extra key, `basalt`: `{"version": "...", "capabilities":
["answer_now", "logprobs", "documents"]}`. A client that finds `basalt` there talks to Basalt; `"answer_now" in basalt.capabilities` says the
endpoint above is served, `"logprobs"` that logprobs come back on all three APIs with the thinking's and tool calls' tokens
([logprobs](#logprobs)), `"documents"` that messages may carry PDF and text documents ([documents](documents.md)), and `"video"` (only
while video is on) that this model takes video parts now (image input on, a model that reads video, FFmpeg found:
[video](../running.md#video)). Later features add their names to the same list.

## Thinking loops

- **Loops:** when one token repeats 256 times inside the thinking, the server closes the thinking and the model answers
  (`thinking-loop-tokens`). Only the thinking is watched. `finish_reason` stays honest.

## The model file's defaults

The model's GGUF metadata keys `general.sampling.temp`, `top_p`, `top_k`, `min_p`, `penalty_repeat` and `penalty_last_n` are the defaults of
the matching fields. A bad value stops the start. The defaults in effect print on the model's line (`temp=1.0 top_p=0.95 ...`).
