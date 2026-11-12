# Documents

A message can carry a document, as on the hosted APIs: a PDF or a text file, in a user turn or in a tool result. The model reads it on
every model and every API. Claude Code's Read tool sends a PDF this way, so asking Claude Code to read a paper, a datasheet or a spec
works against Basalt.

| API | Part | Bytes |
|---|---|---|
| Anthropic Messages | `{"type": "document", "source": {...}, "title": "...", "context": "..."}` | `source` of type `base64` (PDF or text), `text` or `content` |
| Chat Completions | `{"type": "file", "file": {"filename": "...", "file_data": "data:application/pdf;base64,..."}}` | `file_data` |
| Responses | `{"type": "input_file", "filename": "...", "file_data": "data:application/pdf;base64,..."}` | `file_data` |

## What the model sees

- A PDF's text layer, page by page, inside `<document name="..." pages="N">` with a `[page K]` line before each page, so the model can
  name pages. A `context` (Anthropic) comes first, in `<context>`.
- A page with no text whose content is an image (a scan) is that image, read by the vision encoder like any image in a message
  (`[page K: scanned, as an image]`). It needs `--vision`; a system prompt cannot hold one. A page with neither says `[page K: no text]`.
- A text document (`text/*`, JSON, YAML, XML) is its text. A `content` document (Anthropic) is its own text and image blocks.

The same bytes give the same text every time, and a parsed PDF is kept, so a conversation that resends its document on every turn keeps
its cached prefix and parses it once.

## Refusals

Each is a 400 that names the document: a password-protected or unreadable PDF, more than 600 pages, more than 64 MiB, a media type
other than PDF or text, text that is not UTF-8. The server keeps no files and fetches none: a `file_id`, a `file_url` or a `url`
source is refused with the field to send instead. Citations are not produced.

## Example

```sh
curl -s 127.0.0.1:8080/v1/messages -H 'content-type: application/json' -d @- <<EOF
{"model": "model.basalt", "max_tokens": 1024, "messages": [{"role": "user", "content": [
  {"type": "document", "title": "paper.pdf",
   "source": {"type": "base64", "media_type": "application/pdf", "data": "$(base64 -w0 paper.pdf)"}},
  {"type": "text", "text": "What does section 3 measure?"}]}]}
EOF
```
