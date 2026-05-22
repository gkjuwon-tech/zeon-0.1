---
name: Bug report
about: Something doesn't work the way the docs say it does.
title: "[bug] "
labels: bug
assignees: ''
---

### What broke

<!-- One or two sentences. -->

### Repro

<!--
  The most useful field. Ideally a complete, runnable snippet.

  Example:

  ```python
  import torch
  from zeon import ZeonConfig, ZeonForCausalLM
  m = ZeonForCausalLM(ZeonConfig(vocab_size=32, hidden_size=16))
  m(input_ids=torch.zeros(1, 4, dtype=torch.long))   # explodes here
  ```

  If your repro is "I trained for 3 weeks and the loss went up at step
  47k", that's still useful — just say so.
-->

### Expected behavior

<!-- What you thought would happen. -->

### Actual behavior

<!-- Stack trace / wrong output / silent NaNs / etc. -->

### Environment

- Python:
- PyTorch:
- transformers:
- OS:
- GPU (if any):
- ZEON commit hash:

### Anything else
