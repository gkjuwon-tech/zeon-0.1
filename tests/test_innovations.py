"""Tests for the architectural innovations that distinguish ZEON from a
plain recurrent transformer: step embeddings and cross-step latent memory.
"""

from __future__ import annotations

import copy

import torch

from zeon import ZeonConfig, ZeonForCausalLM
from zeon.modeling_zeon import CrossStepMemory, StepEmbedding


def _cfg(**over):
    base = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_recurrent_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        max_recurrent_steps=4,
        min_recurrent_steps=1,
        freeze_ffn=False,
        freeze_embed=False,
        freeze_lm_head=False,
    )
    base.update(over)
    return ZeonConfig(**base)


def test_step_embedding_differs_between_steps():
    emb = StepEmbedding(hidden_size=16, max_steps=8)
    a = emb(0)
    b = emb(3)
    assert a.shape == (16,)
    assert not torch.allclose(a, b)


def test_cross_step_memory_starts_as_identity():
    """At init, cross-step gate bias = -4 and o_proj = 0, so the module
    must not perturb the hidden state at all. This protects warm-start
    transplants — the model behaves like a plain recurrent transformer
    until cross-step learns to contribute."""
    csm = CrossStepMemory(hidden_size=16)
    h = torch.randn(2, 4, 16)
    history = torch.randn(2, 4, 3, 16)
    out = csm(h, history)
    assert torch.allclose(out, h, atol=1e-7)


def test_disabling_innovations_yields_smaller_param_count():
    base = ZeonForCausalLM(_cfg(use_step_embedding=False, cross_step_memory=False))
    fancy = ZeonForCausalLM(_cfg(use_step_embedding=True, cross_step_memory=True))
    p_base = sum(p.numel() for p in base.parameters())
    p_fancy = sum(p.numel() for p in fancy.parameters())
    assert p_fancy > p_base


def test_cross_step_module_receives_gradients():
    """A training step must populate gradients on both cross-step weights
    and step-embedding learned residuals — otherwise these modules would
    be dead branches that never actually participate in learning."""
    cfg = _cfg(use_step_embedding=True, cross_step_memory=True,
               max_recurrent_steps=3)
    m = ZeonForCausalLM(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    cross = m.model.recurrent.cross_step
    assert cross.gate.weight.grad is not None and cross.gate.weight.grad.abs().sum() > 0
    assert cross.q_proj.weight.grad is not None and cross.q_proj.weight.grad.abs().sum() > 0
    step_emb = m.model.recurrent.step_embed
    assert step_emb.learned.grad is not None and step_emb.learned.grad.abs().sum() > 0


def test_save_load_roundtrip_with_innovations():
    import tempfile

    from transformers import AutoModelForCausalLM

    cfg = _cfg(use_step_embedding=True, cross_step_memory=True)
    m = ZeonForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        loaded = AutoModelForCausalLM.from_pretrained(d).eval()
    with torch.no_grad():
        a = m(input_ids=ids).logits
        b = loaded(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-5)
