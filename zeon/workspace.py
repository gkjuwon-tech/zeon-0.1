"""Workspace Bank — Phase 1 of the Latent Virtual Machine.

The Phase 0 recurrent core evolves a single hidden vector per token. That
is mathematically an RNN with adaptive depth. Humans don't think with one
number: we juggle a hypothesis here, an intermediate sum there, a
"scratch" buffer for things we're not sure about yet — all updated at
different rates. To do *that* in latent space we need a structured
working memory, not just deeper recurrence.

`WorkspaceBank` gives every token a small bank of S learned "slots" of
dimension D. Each slot carries a *role embedding* (learned, not labelled
by humans) so the slots start as different vectors at init — without
that, gradients can't break the symmetry and they collapse to identical
content within a handful of steps.

Per latent step the block does three things:

1. **Read.** Multi-head cross-attention from the current hidden state
   `h` (one query per token) into the S slots of the same token. This is
   *intra-token* attention; it does not mix information across the
   sequence, so cost stays O(B*T*S*D).
2. **Write.** A learned content vector derived from (h, read) is
   broadcast across slots; each slot then gates the update by a learned
   per-slot "sticky-ness" bias. High sticky bias → slot is a long-term
   memory; low sticky bias → slot is volatile scratch.
3. **Contribute.** An aggregated, projected version of the new workspace
   is added back to `h` as a residual. The projection (`output_proj`)
   uses standard small-normal init so its contribution at step 0 is
   bounded but non-zero — enough that every workspace parameter receives
   a real gradient from the very first backward pass. We deliberately
   do *not* zero-init this projection: that would kill ∂L/∂output_pool
   and every upstream workspace tensor, leaving the bank as a dead
   branch that "trains" but actually doesn't.

The ablation guarantee CLAUDE.md asks for is *flag-level*, not
init-level: with `use_workspace=False` the module is literally absent
from the graph (see `ZeonBlock.__init__`), and the model is bit-for-bit
Phase 0. With it on, the workspace is a small perturbation at init and
a learned component thereafter.

Phase 1 success criteria (per docs/ROADMAP.md):
  * `use_workspace=False` ⇒ identical numerics to Phase 0
  * Every learnable tensor in the workspace receives a non-zero
    gradient during a training step (no dead branches)
  * `diversity_loss` rises when slots collapse, falls when they spread
    out, as verified by a synthetic test
  * Workspace survives an HF save / load roundtrip without drift

Anti-goal: this is **not** "another attention layer hanging off h". The
slots are the unit of thought; `h` is just the projection of the
workspace that the LM head sees. If you find yourself wanting to add
attention across slots of *different* tokens, that's a different feature
(Phase 2 operator routing) and belongs in `zeon/operators.py`, not here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from zeon.config import ZeonConfig


class WorkspaceBank(nn.Module):
    """Per-token bank of S learned slots with gated sticky updates."""

    def __init__(self, cfg: ZeonConfig):
        super().__init__()
        D = cfg.hidden_size
        S = cfg.workspace_num_slots
        H = cfg.workspace_num_heads
        if D % H != 0:
            raise ValueError(
                f"workspace_num_heads={H} must divide hidden_size={D}; "
                "pick a divisor (e.g. 1, 2, 4, 8)."
            )
        self.cfg = cfg
        self.num_slots = S
        self.hidden_size = D
        self.num_heads = H
        self.head_dim = D // H

        # Per-slot role embedding (learned). This is the only thing that
        # *guarantees* distinct slots at init; without it, every slot
        # starts at the same vector and gradients can't break the symmetry
        # in finite training time.
        self.role_emb = nn.Parameter(torch.randn(S, D) * (D ** -0.5))

        # Initialize slot contents from h: each slot t gets a shared
        # projection of h(t), then we add role_emb to specialize.
        self.init_proj = nn.Linear(D, D, bias=False)

        # Read: attention from h (1 query / token) into S workspace slots.
        # Intra-token only — no cross-token mixing here.
        self.read_q = nn.Linear(D, D, bias=False)
        self.read_k = nn.Linear(D, D, bias=False)
        self.read_v = nn.Linear(D, D, bias=False)
        self.read_o = nn.Linear(D, D, bias=False)

        # Write: produce a content vector from (h, read), broadcast across
        # slots; each slot then has its own gate to either accept the new
        # content or stay sticky.
        self.write_in = nn.Linear(D, D, bias=False)
        self.write_mod = nn.Linear(D, D, bias=False)

        # Sticky gate: bias init = +2.0 ⇒ σ(2.0) ≈ 0.88, so slots are
        # *sticky by default*. The model has to actively learn to release
        # a slot. This matches the intuition "workspace is long-term
        # memory unless proven otherwise".
        self.write_gate = nn.Linear(D, D, bias=True)
        self.sticky_bias = nn.Parameter(
            torch.full((S, D), cfg.workspace_sticky_bias_init)
        )

        # Aggregation back into h: pool over slots, project, add to h.
        #
        # We don't try to force "exact identity at init" via zero-init on
        # `output_proj`: doing so would kill the gradient through every
        # upstream workspace projection (∂L/∂output_pool = 0 because the
        # path from output_pool to the loss is multiplied by a zero
        # weight). The point of the workspace is that *it* learns; if
        # half its parameters get no gradient from step 0 we've built
        # the LoRA dead-branch trap. So output_proj uses the standard
        # `initializer_range` (≈ 0.02), which gives a small but non-zero
        # contribution at init and lets every workspace tensor receive a
        # real gradient from the first backward pass.
        #
        # The "you can toggle the workspace off and recover Phase 0
        # numerics" guarantee lives at the `use_workspace=False` config
        # flag — the module is then literally absent from the forward
        # pass (see `ZeonBlock.__init__` and `test_disabling_workspace_
        # removes_module`). That is the ablation semantics CLAUDE.md
        # actually requires; bit-for-bit init parity is not.
        self.output_pool = nn.Linear(D, D, bias=False)
        self.output_proj = nn.Linear(D, D, bias=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_state(self, h: torch.Tensor) -> torch.Tensor:
        """Build the initial workspace tensor from `h`.

        Args:
            h: (B, T, D) — outer-stack hidden state before any thinking step.

        Returns:
            W_0: (B, T, S, D) — per-token slot bank with broken symmetry
                 (each slot starts at `init_proj(h_t) + role_emb_i`).
        """
        base = self.init_proj(h).unsqueeze(2)         # (B, T, 1, D)
        return base + self.role_emb                   # broadcast over S

    def step(
        self,
        h: torch.Tensor,
        W: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One latent VM step.

        Read from W, propose a write, gate-update slots, contribute to h.
        Returns `(h_new, W_new)`.
        """
        read_vec = self._attention_read(h, W)
        update = self._write_update(h, read_vec)

        gate_logits = self.write_gate(W) + self.sticky_bias  # (B,T,S,D)
        gate = torch.sigmoid(gate_logits)
        W_new = gate * W + (1.0 - gate) * update

        pooled = self.output_pool(W_new).mean(dim=2)         # (B,T,D)
        h_contrib = self.output_proj(F.silu(pooled))
        return h + h_contrib, W_new

    def diversity_loss(self, W: torch.Tensor) -> torch.Tensor:
        """Mean squared off-diagonal cosine similarity across slots.

        High value ⇒ slots have collapsed onto similar content (i.e.
        the bank is effectively rank-1). Used as auxiliary loss during
        training to prevent slot collapse.
        """
        W_norm = F.normalize(W, dim=-1)
        sim = torch.einsum("btsd,btud->btsu", W_norm, W_norm)  # (B,T,S,S)
        S = sim.size(-1)
        eye = torch.eye(S, device=sim.device, dtype=sim.dtype)
        off = sim - eye[None, None]
        # Normalize by number of off-diagonal pairs per (B, T) so the
        # scalar magnitude is comparable across configs of different S.
        return (off ** 2).sum(dim=(-2, -1)).mean() / max(S * (S - 1), 1)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _attention_read(self, h: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        B, T, D = h.shape
        S = W.size(2)
        H, d = self.num_heads, self.head_dim
        q = self.read_q(h).view(B, T, H, d)              # (B,T,H,d)
        k = self.read_k(W).view(B, T, S, H, d)           # (B,T,S,H,d)
        v = self.read_v(W).view(B, T, S, H, d)
        attn = torch.einsum("bthd,btshd->bths", q, k) * (d ** -0.5)
        w = F.softmax(attn, dim=-1)                      # (B,T,H,S)
        out = torch.einsum("bths,btshd->bthd", w, v)     # (B,T,H,d)
        return self.read_o(out.reshape(B, T, D))

    def _write_update(self, h: torch.Tensor, read_vec: torch.Tensor) -> torch.Tensor:
        content = self.write_in(h) + self.write_mod(read_vec)  # (B,T,D)
        return content.unsqueeze(2).expand(-1, -1, self.num_slots, -1)
