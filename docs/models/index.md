# Models

A model is one `.basalt` file, mixture-of-experts or dense. The server reads the architecture from the
file and picks the chat format, tool syntax and thinking switch. Several models can share one address ([router](../configuration/models.md)). What each model supports:

| | Qwen3.8-Flash-Next | DeepSeek-V4.1-Flash | Qwen3.8-27B | Hy4-flash | Inkling 2 Mini | Nemotron 3.5 Super |
| --- | --- | --- | --- | --- | --- | --- |
| Architecture id | `qwen4exp` | `deepseek41` | `qwen35` | `hy4flash` | `inkling2` | `nemotronh` |
| Packs | default `Qwen3.8-FN-NVFP4-DQ6_K`, [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) | default `DeepSeek-V4.1-Flash-1017-UD-IQ1_M`, [Hugging Face](https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt) | `Qwen3.8-27B-QUASAR-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-27B-Basalt), built from QUASAR's checkpoint ([model file](../model-file.md#qwen38-27b-pack)) | `Hy4-flash-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Hy4-flash-Basalt), built from Tencent's BF16 checkpoint ([model file](../model-file.md#hy4-flash-nvfp4-pack)) | `Inkling-2-Mini-NVFP4-q8`, [Hugging Face](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt), built from the official release ([guide](inkling2.md#pack)) | `Nemotron-3.5-Super-NVFP4`, built from NVIDIA's NVFP4 checkpoint ([model file](../model-file.md#nemotron-35-super-pack)) |
| Layers, experts | 48 (36 GDN, 12 QSA); 512 per layer, top-10 | 40; 384 per layer, top-6, plus one shared | 64 (48 GDN, 16 gated attention); dense | 42 (12 sparse MLA with an indexer, 30 sliding-window); layer 0 dense, then 512 per layer, top-16, plus one shared | 24 visited 48 times a token (a core of 12 run three times; 4 full attention, 20 sliding-window); 128 per layer, top-8 | 88 (40 Mamba-2, 8 attention, 40 MoE); 512 per layer, top-22, plus one shared |
| Context | default 8,192; `--max-context` sets it | default 8,192; `--max-context` sets it, 1,024 to 1,048,576 | default 8,192; `--max-context` sets it, up to 262,144, past it with `--rope-scaling yarn` | default 8,192; `--max-context` sets it, up to 1,048,576 | default 8,192; `--max-context` sets it, up to 1,048,576 | default 8,192; `--max-context` sets it (tested to 16,384) |
| Drafting | MTP drafter in the pack, on; `--spec` default 8 | DSpark drafter in the pack, on; `--spec` default 6 | DFlash 2 block drafter in the pack (else the MTP layer), on; `--spec` default 16 | DFly2 block drafter in the pack, on; `--spec` default 6 | DSpark drafter in the pack, on; `--spec` default 8 | MTP head in the pack, on; `--spec` default 4; prompt lookup (`--suffix-draft`, default 3) |
| Chat, tools, thinking | yes; `<tool_call>` tools | yes; DSML tools | yes; `<tool_call>` tools | yes; `<tool_calls>` tools | yes; JSON tool calls | yes; `<tool_call>` tools |
| Reasoning effort | `none`, `low`, `medium`, `high`, `max` (`high` = `max`); a number is refused | `none`, `low` (50), `medium` and `high` (75), `max` (100), or 1 to 100 | as Flash-Next | `none`, `low`, `medium`, `high`, `max` (all but `none` and `max` are `high`); a number is refused | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or 1 to 100 | `none`, `low` and `medium` (the template's efficient reasoning), `high` and `max`; a number is refused |
| Images | `--vision dynamic`, a GPU number or `cpu` | `--vision dynamic` only | `--vision dynamic` | `--vision dynamic` only | `--vision dynamic` only | no |
| Video | with images, and FFmpeg on the machine ([video](../running.md#video)) | no | with images, and FFmpeg on the machine | no | no | no |
| `--kv` | fp16 (default), int8, q4_0, k8v4 | fp8 (default: FP8 window, FP4 compressed rows), fp16 | fp16 (default), int8 | fp16 (default), fp8 | fp16 (default), fp8 | fp16 |
| Cards | main card plus up to three expert-tier cards | main card plus one expert-tier card | main card only | main card plus up to three expert-tier cards | main card; experts in RAM when the context needs their room | main card; the experts in RAM |
| Layer split | yes, matched cards, priced at start | no | no | no | no | no |
| Parallel slots | yes; each slot drafts | yes; each slot drafts | yes; each slot drafts | not measured | not measured | not measured |
| Lossy flags | `--relaxed-drafting`, `--route-prior`, `--route-tail-skip` (off) | none | none | none | none | none |
| Prompt skip | no | yes (`--no-prompt-skip` turns it off) | no | no | no | no |
| Idle sleep, wake | yes (Linux) | yes (Linux); a second-card sleep may be refused | no | yes (Linux) | no | no |
| `basalt-bench --quality` | yes | yes, one row per window | yes | yes | yes | not yet (no base); [precision](../precision/nemotron.md) has its float64 parity |
| Guide | [guide](qwen38-flash-next.md) | [guide](deepseek41.md) | [guide](qwen38-27b.md) | [guide](hy4-flash.md) | [guide](inkling2.md) | [guide](nemotron.md) |
| Precision | [precision](../precision/qwen38-flash-next.md) | [precision](../precision/deepseek41.md) | [precision](../precision/qwen38-27b.md) | [precision](../precision/hy4-flash.md) | [precision](../precision/inkling2.md) | [precision](../precision/nemotron.md) |
