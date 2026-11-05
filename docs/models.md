# Models

What Basalt runs, and what each model supports. A model is one `.basalt` file; the server reads its architecture from the
file and picks the chat format, tool-call syntax and thinking switch to match. Several models can sit behind one address
([models and the router](configuration/models.md)).

## Supported models

| | Qwen3.8-Flash-Next | DeepSeek-V4.1-Flash |
| --- | --- | --- |
| Packs | [jesdga/Qwen3.8-Flash-Next-Basalt](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt): nine, from NVFP4 to Q2_0 (names in the [README](../README.md#qwen38-flash-next)) | [jesdga/DeepSeek-V4.1-Flash-1017-Basalt](https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt): one, `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS.basalt`, with the DSpark drafter and the vision projector in the file |
| Layers | 48 (36 GDN, 12 QSA) | 40 |
| Experts | 512 per layer, top-10 | 384 per layer, top-6, plus one shared |
| Pack size | per pack on the Hugging Face card | about 240 GiB |
| Context | `--max-context`, default 4,096; the YaRN flags apply | `--max-context` up to 1,048,576 (from the pack), default 4,096; the YaRN flags are refused |
| Architecture id | `qwen4exp` | `deepseek41` |

Pack formats and how they are built: [model file](model-file.md). Geometry and precision: [precision](precision/index.md)
and [DeepSeek-V4.1 precision](precision/deepseek41.md). Both are served by the same command, `basalt serve PACK`.

## What each model supports

| | Qwen3.8-Flash-Next | DeepSeek-V4.1-Flash |
| --- | --- | --- |
| Chat on OpenAI, Anthropic and Responses APIs | yes | yes |
| Images | yes: `--vision dynamic`, a GPU number or `cpu` | `--vision dynamic` only; the sidecar modes do not apply |
| Video | no | no |
| Tools | yes, `<tool_call>` form | yes, DSML form |
| Grammar and strict tools | yes ([`grammar`](configuration/index.md#server)) | yes |
| Thinking | yes | yes |
| Reasoning effort | `none`, `low`, `medium`, `high`, `max`; `high` and `max` are the same; a number is refused | `none`, `low` (50), `medium` and `high` (75), `max` (100), or a number 1 to 100 |
| Drafting | MTP drafter in the pack, on by default (`--spec`, default 8) | DSpark in the pack, off unless `--spec N` |
| Parallel slots (`--parallel`) | yes | yes; draft nothing |
| Conversation cache in RAM and on disk | yes | yes |
| Idle sleep and wake | yes (Linux) | yes (Linux); see limits |
| One card | yes | yes |
| Two cards (`--gpu 0,1`) | yes: second card is an expert tier; up to three tier cards | yes: second card is an expert tier; one second card only |
| Layer split (`--gpu-stages`) | yes, for matched cards ([GPUs](configuration/gpus.md#what-several-cards-do-together)) | no, the flag is not there |
| `--kv` formats | fp16 (default), int8, q4_0, k8v4 | fp16 only; `--kv` is the unknown-argument error |
| Prompt skip (`--no-prompt-skip`) | no | yes, on by default |
| Lossy flags (`--relaxed-drafting`, `--route-prior`, `--route-tail-skip`) | yes, off by default | no |
| `basalt-bench --quality` | yes, against a BF16 reference | yes, against a BF16 reference, one row per window |
| `basalt-bench` and `basalt-probe` | yes | yes, with the model's own fixtures in `data/bench/deepseek41` |
| `POST /v1/vram` | yes | yes |
| Router model switching | yes | yes |

Reasoning effort in the config and on the API: [sampling](configuration/sampling.md). Settings by flag:
[configuration](configuration/index.md).

## Qwen3.8-Flash-Next

- **Quality:** each pack is measured against the BF16 reference on the Hugging Face card; the method and the numbers are in
  [quality](precision/quality.md) and [precision](precision/index.md).
- **Drafting does not change the output** unless `--relaxed-drafting` is set ([precision](precision/index.md#summary)).
- **Two-card idle sleep:** a sleep the engine refuses can drop the second card's experts until the next restart, at a context
  of 8,192 without an elastic KV cache. This was seen at that setting and not at 64K or 262,144.

## DeepSeek-V4.1-Flash

How it uses the memory, long prompts and drafting: [DeepSeek-V4.1 packs](deepseek41.md).

- **Quality:** `basalt-bench --quality` scores it against a BF16 reference of the model made by @giveen (100 chunks of 512,
  base PPL 2.7394), through the plain decode path, one row per window ([quality](precision/deepseek41.md#quality)).
- **Output depends on placement.** Experts on the card and experts on the CPU round their activations differently, so which
  experts are resident can change a token. A fixed prompt on a fixed memory budget repeats exactly; a different RAM tier
  size can change the text ([precision](precision/deepseek41.md#routed-and-shared-experts)).
- **Prompt skip and read shape.** A real run differs from a full read by the rounding of another batch shape, as another chunk
  size does ([prompt skip](precision/deepseek41.md#prompt-skip)). `--no-prompt-skip` reads every row through every layer and
  is slower.
- **DSpark is a loss in most cases measured.** On one 400 W card and one 180 W card, on long agentic conversations
  (125-141K tokens, thinking and tools), `--spec 8` gave 51.62 tok/s against 64.06 plain; on short greedy prose it also
  lost, and on short code it won. It takes 7.47 GiB of the main card, which the expert tier then lacks. Numbers and the one
  case where it paid: [what it costs](deepseek41.md#what-it-costs).
- **Parallel slots add no throughput.** At 400 W + 180 W, one stream gave 60.1 tok/s, two slots 58.4 and four 54.9 in
  total. Slots share one decode stream and draft nothing.
- **Two-card idle sleep may not happen.** The server sleeps only when the GPU state fits in the free RAM with 8 GiB to spare.
  On a box with 121 GiB of RAM, two cards held 38 GiB of GPU state against 21 GiB free, so the model stayed awake; one card
  slept in 1.9 s.
- **One second card.** More than one expert-tier card is not used.
- **Not measured:** restore drift (whether a conversation read back from the disk tier differs from the live one), and a
  second-card sleep on a larger machine.
- **First turn after a long prompt:** a read of 4,096 tokens or more (on two cards, more than 2,048) gives the main card's copies back and reopens them
  after it; on one card the first tokens after it decode slower ([long prompts](deepseek41.md)). A cold server's first turn
  reads the experts it routes from the pack and costs more than later turns; a restart with the disk tier on restores a
  stored conversation instead of reading it again.
- **Fixed by the model:** no `--kv`, no `--rope-*` or `--yarn-*`, no sidecar image modes, no layer split, no lossy flags.

## Which to run

Two differences to weigh:

- Choose Qwen3.8-Flash-Next for a choice of nine packs and KV formats, and drafting that is on by default and leaves the output
  unchanged.
- Choose DeepSeek-V4.1-Flash for a context up to 1M tokens in one pack, with drafting off and one KV format.

Speed numbers are not repeated here; the release notes and the Hugging Face cards carry them ([changelog](changelog/index.md)).
Add a model: [adding a model](adding-a-model.md).
