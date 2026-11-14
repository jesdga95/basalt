# Models

A model is one `.basalt` file. The server reads the architecture from the file and picks the chat format, tool syntax and thinking
switch. Several models can share one address ([router](../configuration/models.md)). What each model supports:

| | Qwen3.8-Flash-Next | DeepSeek-V4.1-Flash |
| --- | --- | --- |
| Architecture id | `qwen4exp` | `deepseek41` |
| Packs | default `Qwen3.8-FN-NVFP4-DQ6_K`, [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) | default `DeepSeek-V4.1-Flash-1017-UD-IQ1_M`, [Hugging Face](https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt) |
| Layers, experts | 48 (36 GDN, 12 QSA); 512 per layer, top-10 | 40; 384 per layer, top-6, plus one shared |
| Context | default 8,192; `--max-context` sets it | default 8,192; `--max-context` sets it, 1,024 to 1,048,576 |
| Drafting | MTP drafter in the pack, on; `--spec` default 8 | DSpark drafter in the pack, on; `--spec` default 6 |
| Chat, tools, thinking | yes; `<tool_call>` tools | yes; DSML tools |
| Reasoning effort | `none`, `low`, `medium`, `high`, `max` (`high` = `max`); a number is refused | `none`, `low` (50), `medium` and `high` (75), `max` (100), or 1 to 100 |
| Images | `--vision dynamic`, a GPU number or `cpu` | `--vision dynamic` only |
| Video | with images, and FFmpeg on the machine ([video](../running.md#video)) | no |
| `--kv` | fp16 (default), int8, q4_0, k8v4 | fp8 (default: FP8 window, FP4 compressed rows), fp16 |
| Cards | main card plus up to three expert-tier cards | main card plus one expert-tier card |
| Layer split | yes, matched cards, priced at start | no |
| Parallel slots | yes; each slot drafts | yes; each slot drafts |
| Lossy flags | `--relaxed-drafting`, `--route-prior`, `--route-tail-skip` (off) | none |
| Prompt skip | no | yes (`--no-prompt-skip` turns it off) |
| Idle sleep, wake | yes (Linux) | yes (Linux); a second-card sleep may be refused |
| `basalt-bench --quality` | yes | yes, one row per window |
| Guide | [guide](qwen38-flash-next.md) | [guide](deepseek41.md) |
| Precision | [precision](../precision/qwen38-flash-next.md) | [precision](../precision/deepseek41.md) |
