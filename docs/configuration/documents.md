# Documents

A message can carry a document, as on the hosted APIs: a PDF, an Office file or a text file, in a user turn or in a tool result. The model reads it on
every model and every API. Claude Code's Read tool sends a PDF this way, so asking Claude Code to read a paper, a datasheet or a spec
works against Basalt.

| API | Part | Bytes |
|---|---|---|
| Anthropic Messages | `{"type": "document", "source": {...}, "title": "...", "context": "..."}` | `source` of type `base64` (PDF or text), `text`, `content` or `url` |
| Chat Completions | `{"type": "file", "file": {"filename": "...", "file_data": "data:application/pdf;base64,..."}}` | `file_data` |
| Responses | `{"type": "input_file", "filename": "...", "file_data": "data:application/pdf;base64,..."}` | `file_data` or `file_url` |

A `url` source or a `file_url` is an `http(s)` URL, a `file:` URL or a path, read under the same rules as an image named that way
([image sources](image-sources.md)): from this machine any of them, from elsewhere only hosts and directories the server allows. A
harness on the same machine can name the file instead of sending its bytes on every turn.

## Pages

Basalt adds `"pages"` beside a document's other fields (`file.pages` in Chat Completions): pages and ranges such as `"1-5,8"`. Only
those pages are read, under their own numbers, and the frame says which (`read="1-5,8"`). With a path, a harness can hand the model
pages 40-55 of a long manual without parsing it.

## What the model sees

- A PDF's text layer, page by page, inside `<document name="..." pages="N">` with a `[page K]` line before each page, so the model can
  name pages. A `context` (Anthropic) comes first, in `<context>`.
- A page with no text whose content is an image (a scan) is that image, read by the vision encoder like any image in a message
  (`[page K: scanned, as an image]`). A page with under 200 characters of text (a slide, a stamped scan) gives its text and then its
  largest image, when that has at least 512 x 512 pixels (`[page K, with its image]`). Images need `--vision`; a system prompt cannot
  hold one. A page with neither text nor image says `[page K: no text]`.
- A Word (`.docx`) file is its paragraphs and tables (cells split by ` | `), a PowerPoint (`.pptx`) file its slides' text in
  presentation order (`[slide K]`, `slides="N"`), an Excel (`.xlsx`) file each sheet's rows (`[sheet NAME]`, `sheets="N"`).
- A text document (`text/*`, JSON, YAML, XML) is its text. A `content` document (Anthropic) is its own text and image blocks.

The same bytes give the same text every time, and a parsed PDF is kept, so a conversation that resends its document on every turn keeps
its cached prefix and parses it once.

## Refusals

Each is a 400 that names the document: a password-protected or unreadable PDF, more than 600 pages read, a page past the end, `pages`
on a text document, more than 64 MiB, a media type
other than PDF or text, text that is not UTF-8, a URL or path the image rules refuse. The server keeps no files: a `file_id` or a
`file` source is refused with the field to send instead. Citations are not produced.

## Example

```sh
curl -s 127.0.0.1:8080/v1/messages -H 'content-type: application/json' -d @- <<EOF
{"model": "model.basalt", "max_tokens": 1024, "messages": [{"role": "user", "content": [
  {"type": "document", "title": "paper.pdf",
   "source": {"type": "base64", "media_type": "application/pdf", "data": "$(base64 -w0 paper.pdf)"}},
  {"type": "text", "text": "What does section 3 measure?"}]}]}
EOF
```
