# ZEON

> Recurrent-depth transformer with adaptive compute.
> **Trade context length for reasoning depth.**

Everyone is racing toward 2M-token context windows while real reasoning
quality plateaus. ZEON goes the other way: keep the context modest
(~16k), then make every token *think harder* by recurring the same
transformer block in latent space, with a learned halting mechanism
that spends more compute on harder tokens.

## Core ideas

1. **Latent recursion.** A `RecurrentCore` block is applied up to `K`
   times to the same hidden state, before producing the next token.
   This lets the model perform chain-of-thought *in latent space*
   without emitting any tokens — so the context window doesn't grow.
2. **Adaptive compute (PonderNet-style).** A learned halting head
   decides per token how many recurrent steps to spend. Easy tokens
   halt quickly, hard tokens (operators, conclusions, math) loop longer.
3. **Memory / reasoning separation.** Embedding, FFN and `lm_head` can
   be transplanted (and frozen) from a strong open base model. Only the
   recurrent core is trained — so we inherit world knowledge and grow
   reasoning depth on top of it.
4. **Surgical transplant.** `zeon.transplant` ships a generic procedure
   that loads any HF causal LM (Qwen, Llama, …), copies embeddings /
   FFN / `lm_head` into a `ZeonForCausalLM` of matching shape, and
   leaves the recurrent attention block ready for distillation.

## Status

PoC scaffold. Forward pass is wired and shape-correct. Training scripts
exist as runnable stubs. **Not yet trained** — this commit is the
skeleton.

## Layout

```
zeon/
  config.py         ZeonConfig (HF PretrainedConfig)
  modeling_zeon.py  ZeonForCausalLM, ZeonBlock, RecurrentCore
  halting.py        PonderNet halting + geometric prior
  transplant.py     copy weights from any HF causal LM
  losses.py         distillation + ponder loss
scripts/
  transplant_from_hf.py
  train_distill.py
  eval_reasoning.py
tests/
  test_recurrent_shape.py
```

## Quick smoke test

```bash
pip install -e .
pytest tests/ -q
```
