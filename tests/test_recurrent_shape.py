"""Smoke tests: shapes go in, shapes come out, gradients flow where expected."""

import torch

from zeon import ZeonConfig, ZeonForCausalLM
from zeon.halting import ponder_combine, ponder_kl_loss


def _tiny_cfg(**over):
    base = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_recurrent_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        max_recurrent_steps=3,
        min_recurrent_steps=1,
        freeze_ffn=False,
        freeze_embed=False,
        freeze_lm_head=False,
    )
    base.update(over)
    return ZeonConfig(**base)


def test_forward_shape():
    cfg = _tiny_cfg()
    model = ZeonForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids=ids)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.ponder_loss is not None and out.ponder_loss.ndim == 0


def test_loss_and_backward():
    cfg = _tiny_cfg()
    model = ZeonForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids=ids, labels=ids)
    assert out.loss is not None
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_ponder_combine_sums_to_one():
    B, T, D, N = 2, 4, 8, 5
    hs = [torch.randn(B, T, D) for _ in range(N)]
    lams = [torch.sigmoid(torch.randn(B, T)) for _ in range(N)]
    h_out, p, p_rem = ponder_combine(hs, lams)
    assert h_out.shape == (B, T, D)
    assert torch.allclose(p.sum(dim=-1), torch.ones(B, T), atol=1e-5)


def test_ponder_kl_is_finite():
    B, T, N = 2, 4, 5
    lams = [torch.sigmoid(torch.randn(B, T)) for _ in range(N)]
    hs = [torch.randn(B, T, 8) for _ in range(N)]
    _, p, _ = ponder_combine(hs, lams)
    kl = ponder_kl_loss(p, lambda_p=0.2)
    assert torch.isfinite(kl)


def test_freeze_policy_blocks_ffn_gradients():
    cfg = _tiny_cfg(freeze_ffn=True, freeze_embed=True, freeze_lm_head=True)
    model = ZeonForCausalLM(cfg)
    for p in model.model.layers[0].mlp.parameters():
        assert not p.requires_grad
    for p in model.model.embed_tokens.parameters():
        assert not p.requires_grad
    for p in model.lm_head.parameters():
        assert not p.requires_grad
    # halt head + attention + norms should still train
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("halt_head" in n for n in trainable)
    assert any("self_attn" in n or "attn.q_proj" in n for n in trainable)
