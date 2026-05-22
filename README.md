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

PoC scaffold, production-shaped. Forward + `.generate()` + save/load
roundtrip + transplant from any Llama-style HF causal LM all green.
Innovations (step embeddings + cross-step latent memory) wired in and
covered by tests. Training scripts exist as runnable stubs.
**Not yet trained on a real base model** — this branch is the
architecture-ready scaffold.

## Architectural innovations (vs. plain recurrent transformer)

1. **`StepEmbedding`** — sinusoidal + learned residual, additive into
   the hidden state at the start of each thinking step so the core
   knows its own iteration count.
2. **`CrossStepMemory`** — lightweight gated cross-attention from the
   current step to a sliding window of prior steps' hidden states.
   Initialized as the exact identity so warm-start transplants are
   safe; learns to deviate as training surfaces useful latent shortcuts.
3. **PonderNet halt + entropy bonus** — KL-to-geometric-prior plus a
   subtracted entropy term that prevents the halt distribution from
   collapsing onto step 1.
4. **Recurrent K/V reuse** — within the recurrent core, K and V are
   computed once on step 1 and reused for the remaining latent steps;
   only Q evolves. Cost of latent reasoning stays O(T) per step.
5. **BF16-safe halt head** — sigmoid projection kept in fp32 internally
   so mixed-precision training doesn't crush the halt distribution.

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
