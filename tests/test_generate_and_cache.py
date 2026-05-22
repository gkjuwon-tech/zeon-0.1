"""Tests for the HF-compatible inference path: KV cache, generate, save/load."""

from __future__ import annotations

import tempfile

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from zeon import ZeonConfig, ZeonForCausalLM


def _tiny_cfg(**over):
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
        max_recurrent_steps=2,
        min_recurrent_steps=1,
        freeze_ffn=False,
        freeze_embed=False,
        freeze_lm_head=False,
    )
    base.update(over)
    return ZeonConfig(**base)


def test_forward_returns_past_key_values_when_use_cache():
    cfg = _tiny_cfg()
    model = ZeonForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids=ids, use_cache=True)
    assert out.past_key_values is not None
    assert len(out.past_key_values) == cfg.num_hidden_layers
    for k, v in out.past_key_values:
        assert k.shape == (2, cfg.num_key_value_heads, 8, cfg.head_dim)
        assert v.shape == (2, cfg.num_key_value_heads, 8, cfg.head_dim)


def test_incremental_decode_extends_cache():
    cfg = _tiny_cfg()
    model = ZeonForCausalLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        out1 = model(input_ids=prompt, use_cache=True)
        nxt = out1.logits[:, -1].argmax(-1, keepdim=True)
        out2 = model(input_ids=nxt, past_key_values=out1.past_key_values, use_cache=True)
    assert out2.past_key_values[0][0].size(-2) == 6
    assert torch.isfinite(out2.logits).all()


def test_generate_runs_and_extends_length():
    cfg = _tiny_cfg()
    model = ZeonForCausalLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    gen = model.generate(input_ids=prompt, max_new_tokens=5, do_sample=False, use_cache=True)
    assert gen.shape[1] == 4 + 5


def test_save_load_roundtrip_via_auto_classes():
    cfg = _tiny_cfg()
    model = ZeonForCausalLM(cfg).eval()
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        loaded_cfg = AutoConfig.from_pretrained(d)
        loaded = AutoModelForCausalLM.from_pretrained(d).eval()
    assert isinstance(loaded_cfg, ZeonConfig)
    assert isinstance(loaded, ZeonForCausalLM)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        a = model(input_ids=ids).logits
        b = loaded(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-5)


def test_gradient_checkpointing_smoke():
    cfg = _tiny_cfg(freeze_ffn=False, freeze_embed=False, freeze_lm_head=False)
    model = ZeonForCausalLM(cfg).train()
    model.gradient_checkpointing_enable()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad
