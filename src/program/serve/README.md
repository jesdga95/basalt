# Persistent serving

`Runtime` (`state.hpp`) holds what every model shares: the input, the conversation chain, the slots. `Request`
(`request.hpp`) runs one request's phases in order. A model answers through their virtual calls, which run per request,
per command or per batch window; `Request::decode_loop` (`decode_loop.hpp`) is instantiated once per model, so a window
pays no indirect call. `Flow::next` returns to command dispatch; `Flow::error` ends serving; `Flow::quit` runs shutdown.

| Concern | Implementation |
| --- | --- |
| Lifecycle and shutdown | `runtime.cpp` |
| Command dispatch and scheduling | `commands.cpp` |
| Input framing, tokenization and watchdog | `protocol.cpp` |
| Request order | `request.cpp` |
| Request keys, token ids and grammar | `request_parse.cpp` |
| Prompt limits | `request_image.cpp` |
| Which source holds the prompt's start | `request_restore.cpp` |
| Sampling setup, turn and root parts | `request_prefill.cpp` |
| The window's order | `decode_loop.hpp`, `reply_rows.hpp` |
| Completion, slot admission and reports | `request_finish.cpp` |
| A model that offers only sessions | `session_engine.cpp` |

A model with its own serving derives from both classes in its own folder; see `src/models/qwen4exp/serve/`.

The protocol text and error paths are shared contracts with the Python server and the benchmark client. A structural
change preserves those contracts and is checked against a frozen engine with fixed expert placement, including batch
slots, cancellation and conversation restoration.
