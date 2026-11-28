# Precision

Each model has one page for what it computes in and what each rounding costs. The method is shared: the
[quality suite](quality-suite.md).

- [Qwen3.8-Flash-Next](qwen38-flash-next.md): stages, prefill, the measured costs, drafting and the lossy flags.
- [DeepSeek-V4.1-Flash](deepseek41.md): stages, the prompt path, verify windows, quality against its BF16 reference and the lossy
  routing flags.
- [Qwen3.8-27B](qwen38-27b.md): the dense stack's roundings and quality against its BF16 reference.
- [Hy4-flash](hy4-flash.md): weights, stages, the attention caches, drafting, quality against its BF16 reference and the lossy
  routing flags.
- [Inkling 2 Mini](inkling2.md): weights, stages, the attention caches and quality against its reference (the release with
  its experts dequantized to BF16, not a true BF16 model).
- [Gemma 4](gemma4.md): weights, stages, the attention caches, quality against its BF16 reference (the CPU experts included).
- [ERNIE-5.5-Flash](ernie55.md): weights, stages, the HCA/MLA-DSA attention caches, drafting and the lossy routing flags.
- [How quality is measured](quality.md): the suite in brief, the lossy routing flags, greedy identity and wikitext KLD
  (its BF16 bases and reproduction commands).
- [Quality suite](quality-suite.md): the quality measure. A stored BF16 reference on chat conversations, scored by KLD, top-1
  and PPL ratio with conversation-bootstrap error bars, paired A/B and a smoke slice.
