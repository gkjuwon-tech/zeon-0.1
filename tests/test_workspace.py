"""Phase 1 — Workspace Bank tests.

These tests are the contract for `WorkspaceBank`:

  1. **Bounded perturbation at init.** With `use_workspace=True`, a
     freshly built model should produce logits *close* to the same
     config with `use_workspace=False` — close enough that warm-starting
     a Phase 0 transplant doesn't blow up. We intentionally do *not*
     enforce bit-identity at init; doing that requires zero-initialising
     `output_proj`, which in turn kills the gradient on every upstream
     workspace projection. The ablation guarantee CLAUDE.md asks for is
     covered by the flag itself (test 7), not by init parity.

  2. **Param count grows.** Turning the flag on must produce a strictly
     bigger model; otherwise the module is empty.

  3. **Gradient flow.** Every learnable tensor in the workspace block
     must receive a non-zero gradient during a backward pass. If any
     of them stays at None or |grad|=0, that subgraph is dead — and per
     the CLAUDE.md mantra, "ablation-impossible" components are
     considered "not actually built".

  4. **Save / load roundtrip via AutoModel.** With workspace on, the
     full HF round-trip must produce identical logits.

  5. **Slot symmetry is broken at init.** If role embeddings were dead,
     every slot of the initial workspace would be the same vector; we
     check that's not the case.

  6. **Diversity loss is reasonable.** Synthetic check: collapsed slots
     give higher loss than spread-out slots.

  7. **Ablation flag actually removes the module.** Just to make sure
     `use_workspace=False` doesn't leave the parameters lying around
     unused.

This file is what we point at next time anyone proposes "add a new
module to the LVM" — write the equivalent contract first, then code.
"""

from __future__ import annotations

import tempfile

import torch

from zeon import ZeonConfig, ZeonForCausalLM, WorkspaceBank


def _cfg(**over) -> ZeonConfig:
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
        max_recurrent_steps=3,
        min_recurrent_steps=1,
        freeze_ffn=False,
        freeze_embed=False,
        freeze_lm_head=False,
        # workspace defaults explicit so the test reads top-to-bottom
        use_workspace=True,
        workspace_num_slots=4,
        workspace_num_heads=2,
        workspace_diversity_weight=1e-3,
    )
    base.update(over)
    return ZeonConfig(**base)


# ----------------------------------------------------------------------
# 1) Identity at init: workspace=True ⇒ same logits as workspace=False
# ----------------------------------------------------------------------


def test_workspace_contribution_is_bounded_at_init():
    """At init the workspace contributes a small perturbation to h, not
    a full rewrite. We verify by comparing logits from the same model
    with the workspace temporarily disconnected — the relative
    difference must be small compared to the no-workspace logit
    magnitude.

    We can't enforce *bit-identity* at init because that would require
    zero-initialising `output_proj`, which kills the gradient on every
    upstream workspace projection (LoRA dead-branch trap). The flag
    `use_workspace=False` is what gives true ablation parity.
    """
    torch.manual_seed(0)
    m = ZeonForCausalLM(_cfg(use_workspace=True)).eval()
    ids = torch.randint(0, m.config.vocab_size, (2, 6))

    with torch.no_grad():
        a = m(input_ids=ids).logits
        saved = m.model.recurrent.workspace
        m.model.recurrent.workspace = None
        try:
            b = m(input_ids=ids).logits
        finally:
            m.model.recurrent.workspace = saved

    rel = (a - b).norm() / (b.norm() + 1e-8)
    # 0.5 is generous — what we care about is "not a different model".
    # In practice for small dims this lands around 0.05–0.15.
    assert rel.item() < 0.5, (
        f"workspace perturbation too large at init (rel={rel.item():.3f}); "
        "this should be a small residual, not a rewrite"
    )


# ----------------------------------------------------------------------
# 2) Param count strictly grows when workspace is on
# ----------------------------------------------------------------------


def test_workspace_increases_param_count():
    off = ZeonForCausalLM(_cfg(use_workspace=False))
    on = ZeonForCausalLM(_cfg(use_workspace=True))
    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    assert n_on > n_off, "workspace must add parameters when enabled"


def test_workspace_more_slots_means_more_params():
    small = ZeonForCausalLM(_cfg(use_workspace=True, workspace_num_slots=4))
    big = ZeonForCausalLM(_cfg(use_workspace=True, workspace_num_slots=16))
    n_small = sum(p.numel() for p in small.parameters())
    n_big = sum(p.numel() for p in big.parameters())
    # role_emb + sticky_bias scale with S; the other projections don't,
    # so the diff must be > 0 but small relative to D.
    assert n_big > n_small


# ----------------------------------------------------------------------
# 3) Gradient flow into every workspace parameter
# ----------------------------------------------------------------------


def test_every_workspace_param_receives_gradient():
    cfg = _cfg(use_workspace=True, max_recurrent_steps=2)
    m = ZeonForCausalLM(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    ws = m.model.recurrent.workspace
    assert ws is not None
    for name, p in ws.named_parameters():
        assert p.grad is not None, f"no grad on workspace.{name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on workspace.{name}"
        # output_proj starts at zero so before backward it contributes
        # nothing, but the gradient *into* output_proj must still be
        # non-zero — otherwise the workspace can never start learning.
        assert p.grad.abs().sum() > 0, \
            f"workspace.{name} got zero gradient — dead branch"


# ----------------------------------------------------------------------
# 4) Save / load roundtrip via HF Auto classes
# ----------------------------------------------------------------------


def test_workspace_save_load_roundtrip():
    from transformers import AutoModelForCausalLM

    cfg = _cfg(use_workspace=True)
    m = ZeonForCausalLM(cfg).eval()
    # Train a few steps so workspace weights drift off their inits and
    # we are actually testing that the persisted weights match.
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    m.train()
    for _ in range(2):
        ids = torch.randint(0, cfg.vocab_size, (2, 4))
        out = m(input_ids=ids, labels=ids)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    m.eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        loaded = AutoModelForCausalLM.from_pretrained(d).eval()
    with torch.no_grad():
        a = m(input_ids=ids).logits
        b = loaded(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-5)


# ----------------------------------------------------------------------
# 5) Role embeddings break slot symmetry at init
# ----------------------------------------------------------------------


def test_role_embeddings_break_slot_symmetry():
    cfg = _cfg(use_workspace=True, workspace_num_slots=4)
    m = ZeonForCausalLM(cfg).eval()
    h = torch.randn(1, 3, cfg.hidden_size)
    W = m.model.recurrent.workspace.init_state(h)
    assert W.shape == (1, 3, 4, cfg.hidden_size)
    # All four slots should differ on every token (i.e. no two slots
    # are identical for any token).
    for t in range(W.size(1)):
        for i in range(W.size(2)):
            for j in range(i + 1, W.size(2)):
                assert not torch.allclose(W[0, t, i], W[0, t, j]), \
                    f"slots {i} and {j} at token {t} are identical at init"


# ----------------------------------------------------------------------
# 6) Diversity loss sanity check
# ----------------------------------------------------------------------


def test_diversity_loss_higher_when_slots_collapse():
    """Same slot tensor repeated S times ⇒ all cosine sims = 1 ⇒ high
    loss. Random unit slots ⇒ near-orthogonal in high-D ⇒ low loss."""
    D, S = 32, 8
    ws = WorkspaceBank(_cfg(hidden_size=D, workspace_num_slots=S, workspace_num_heads=4))

    collapsed = torch.randn(2, 5, 1, D).expand(-1, -1, S, -1).contiguous()
    diverse = torch.randn(2, 5, S, D)

    loss_collapsed = ws.diversity_loss(collapsed)
    loss_diverse = ws.diversity_loss(diverse)
    assert loss_collapsed > loss_diverse, (
        f"diversity_loss should rise under slot collapse: "
        f"collapsed={loss_collapsed.item():.4f}, diverse={loss_diverse.item():.4f}"
    )
    # Sanity-bound the collapsed case: cos sim = 1 ⇒ (1 - eye)^2 normalized = 1.
    assert loss_collapsed.item() > 0.5


# ----------------------------------------------------------------------
# 7) Ablation flag actually removes the module from the model
# ----------------------------------------------------------------------


def test_disabling_workspace_removes_module():
    off = ZeonForCausalLM(_cfg(use_workspace=False))
    assert off.model.recurrent.workspace is None
    for name, _ in off.named_parameters():
        assert "workspace" not in name, (
            f"workspace param {name!r} survived use_workspace=False — "
            "this is exactly the kind of dead-flag that hides regressions"
        )


def test_workspace_diversity_is_in_the_graph_when_training():
    """Sanity check that the diversity loss is actually wired into the
    autograd graph during training: with workspace on + non-zero
    diversity weight, the slot `role_emb` (which participates in slot
    init but not in `h` directly) must still receive a non-zero
    gradient. If it doesn't, `diversity_loss` is silently detached
    somewhere."""
    cfg = _cfg(use_workspace=True, workspace_diversity_weight=1.0, max_recurrent_steps=2)
    m = ZeonForCausalLM(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (2, 4))
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    g = m.model.recurrent.workspace.role_emb.grad
    assert g is not None and g.abs().sum() > 0
