# Precision

Each model has one page for what it computes in and what each rounding costs. The method is shared.

- [Qwen3.8-Flash-Next](qwen38-flash-next.md): stages, prefill, the measured costs, drafting and the lossy flags.
- [DeepSeek-V4.1-Flash](deepseek41.md): stages, the prompt path, verify windows, quality against its BF16 reference and the lossy
  routing flags.
- [Qwen3.8-27B](qwen38-27b.md): the dense stack's roundings and quality against its BF16 reference.
- [Hy4-flash](hy4-flash.md): weights, stages, the attention caches, drafting, quality against its BF16 reference and the lossy
  routing flags.
- [Inkling 2 Mini](inkling2.md): weights, stages, the attention caches and quality against its reference (the release with
  its experts dequantized to BF16, not a true BF16 model).
- [ERNIE-5.5-Flash](ernie55.md): weights, stages, the HCA/MLA-DSA attention caches, drafting and the lossy routing flags.
- GLM-5.3-Flash ([parity](../models/glm-5-3-flash.md#parity)): what it computes in and what each rounding costs, in its
  guide rather than a page of its own - the gate has no BF16 base for this model, so quality is scored against
  llama.cpp reading the same IQ1_S bytes, and the tensor audit of the pack sits beside it.
- [How quality is measured](quality.md): the gate, the BF16 references, the lossy routing flags, greedy identity and reproduction
  commands.
- [Quality suite](quality-suite.md): scoring instruction-tuned models on chat conversations against a stored BF16 reference
  (assistant rows, conversation-bootstrap error bars, paired A/B, the smoke slice).
