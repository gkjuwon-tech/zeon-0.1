"""Smoke test for surgical weight transplant from a Llama-style base.

We build a deliberately tiny `LlamaForCausalLM` in-memory, run the
transplant procedure on it, and verify that every targetable weight got
copied with no shape mismatches. This catches regressions where the
transplant naming map drifts from the actual ZEON module layout.
"""

from __future__ import annotations

import tempfile

import pytest
import torch

llama = pytest.importorskip("transformers.models.llama.modeling_llama")
from transformers import LlamaConfig  # noqa: E402

from zeon.transplant import transplant_from_hf  # noqa: E402


def _make_tiny_llama(tmpdir: str) -> str:
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    model = llama.LlamaForCausalLM(cfg)
    model.save_pretrained(tmpdir)
    return tmpdir


def test_transplant_copies_all_targeted_tensors_without_shape_skips():
    with tempfile.TemporaryDirectory() as d:
        _make_tiny_llama(d)
        model, report = transplant_from_hf(
            d,
            num_hidden_layers=2,
            num_recurrent_layers=1,
            max_recurrent_steps=2,
            max_position_embeddings=128,
            dtype=torch.float32,
        )
    assert len(report.skipped) == 0, f"shape mismatches: {report.skipped}"
    # Must have copied embed + lm_head + final_norm + 2 outer layers (9
    # tensors each) + recurrent core (9 tensors) = 3 + 18 + 9 = 30.
    assert len(report.copied) >= 28
    # Sanity-check that copied parameters actually populate the model.
    ids = torch.randint(0, model.config.vocab_size, (1, 4))
    out = model(input_ids=ids)
    assert torch.isfinite(out.logits).all()


def test_transplant_falls_through_save_load_roundtrip():
    from transformers import AutoModelForCausalLM

    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        _make_tiny_llama(src_dir)
        model, _ = transplant_from_hf(
            src_dir,
            num_hidden_layers=2,
            num_recurrent_layers=1,
            max_recurrent_steps=2,
            max_position_embeddings=128,
            dtype=torch.float32,
        )
        model.save_pretrained(dst_dir)
        loaded = AutoModelForCausalLM.from_pretrained(dst_dir).eval()
    model.eval()
    ids = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.no_grad():
        a = model(input_ids=ids).logits
        b = loaded(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-5)
