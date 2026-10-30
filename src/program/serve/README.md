# Persistent serving

`run(Context&)` borrows the loaded engine from `generate.cpp`. `Context` holds
references and startup callbacks; it does not own the model, sessions or expert
tiers. The implementation headers in this directory are private to the program.

`Runtime` holds state shared across requests. `Request` holds one request's state
and runs parsing, image preparation, restoration, prompt processing, decoding and
completion in order. `Flow::next` returns to command dispatch; `Flow::error` ends
serving; `Flow::quit` runs shutdown and flushes the disk cache.

| Concern | Implementation |
| --- | --- |
| Lifecycle and shutdown | `runtime.cpp` |
| Command dispatch and scheduling | `commands.cpp` |
| Input framing, tokenization and watchdog | `protocol.cpp` |
| Request orchestration and state | `request.cpp`, `request.hpp` |
| Request keys, token ids and grammar | `request_parse.cpp` |
| Image conditioning | `request_image.cpp` |
| Admission and conversation restoration | `request_restore.cpp` |
| Prompt processing and sampling setup | `request_prefill.cpp` |
| Decode and speculative verification | `request_decode.cpp` |
| Completion, slot admission and reports | `request_finish.cpp` |
| Checkpoints, parking and disk persistence | `conversation.cpp` |
| Prompt setup, loans and refills | `prompt.cpp`, `loans.cpp`, `loan_moves.cpp` |
| Prompt reads, splits and verify windows | `prompt_read.cpp`, `prompt_split.cpp`, `prompt_windows.cpp` |
| Verify setup and graph capture | `verify.cpp` |
| RAM admission and VRAM controls | `memory.cpp` |
| Elastic KV and cache resizing | `kv.cpp` |
| Expert adaptation and exchanges | `tiers.cpp` |
| Profile persistence and probe adaptation | `tier_profile.cpp` |
| Batch windows and initialization | `batch.cpp` |
| Slot sessions and state transfers | `slots.cpp` |
| Pipelined batch groups | `batch_pipeline.cpp` |
| In-engine image encoding | `vision.cpp` |

State declarations retain resource lifetime order. In particular, checkpoint
deferral ends before checkpoint storage is destroyed, and job runners finish
before the state they use is released. Constructors leave buffers empty; the
initialization operations allocate buffers and capture graphs in execution order.
Parameterized caches and draft policies are emplaced at their initialization
point. Request-local references remain borrowed views of the runtime's sessions.

The protocol text and error paths are shared contracts with the Python server and
benchmark client. A structural change preserves those contracts and is checked
against a frozen engine with fixed expert placement, including batch slots,
cancellation and conversation restoration.
