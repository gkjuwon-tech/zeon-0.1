"""PonderNet-style adaptive halting for recurrent latent reasoning.

We treat each recurrent step n=1..N as offering a "halt now" Bernoulli
gate with probability lambda_n produced by a learned head from the
current hidden state. The probability of actually halting at step n is

    p_n = lambda_n * prod_{i<n}(1 - lambda_i)

and the residual (still-thinking) mass after N steps is

    p_remaining = prod_{i<=N}(1 - lambda_i)

The ponder regularizer is a KL between p and a geometric prior with
parameter `lambda_p`, encouraging halt distributions that aren't
collapsed to step 1 (lazy) or step N (always burn max compute).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaltingHead(nn.Module):
    """Predict per-token halt probability lambda_n in (eps, 1-eps).

    Kept in fp32 internally so that BF16 training doesn't crush the
    sigmoid into all-0/all-1 (which would freeze the halt distribution
    and zero out ponder gradients).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-3):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=True)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, std=1e-3)
        self.eps = eps
        # Force the projection to stay fp32; it's tiny (D -> 1).
        self.proj.to(torch.float32)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.proj(h.float()).squeeze(-1)
        lam = torch.sigmoid(logits)
        return lam.clamp(self.eps, 1.0 - self.eps)


def ponder_combine(
    hidden_states: list[torch.Tensor],
    lambdas: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the PonderNet-weighted hidden state and the halt distribution.

    Args:
        hidden_states: list of N tensors (B, T, D) — one per recurrent step.
        lambdas:       list of N tensors (B, T)    — per-step halt prob.

    Returns:
        h_out:         (B, T, D) expected hidden state under p_n.
        p:             (B, T, N) halt distribution over steps.
        p_remaining:   (B, T)    residual not-yet-halted mass.
    """
    assert len(hidden_states) == len(lambdas) and len(lambdas) >= 1
    N = len(lambdas)
    lam = torch.stack(lambdas, dim=-1)            # (B, T, N)
    one_minus = torch.clamp(1.0 - lam, min=1e-6)
    log_one_minus = torch.log(one_minus)
    # cumulative product of (1 - lambda_i) for i < n
    cum_not_halt = torch.exp(
        torch.cat(
            [
                torch.zeros_like(log_one_minus[..., :1]),
                torch.cumsum(log_one_minus, dim=-1)[..., :-1],
            ],
            dim=-1,
        )
    )
    p = lam * cum_not_halt                         # (B, T, N)
    p_remaining = torch.exp(torch.sum(log_one_minus, dim=-1))  # (B, T)

    # Force-halt the leftover mass on the last step (so probs sum to 1).
    p = p.clone()
    p[..., -1] = p[..., -1] + p_remaining

    H = torch.stack(hidden_states, dim=-2)         # (B, T, N, D)
    h_out = (p.unsqueeze(-1) * H).sum(dim=-2)      # (B, T, D)
    return h_out, p, p_remaining


def ponder_kl_loss(p: torch.Tensor, lambda_p: float) -> torch.Tensor:
    """KL( p || Geometric(lambda_p) ), averaged over batch & time.

    p: (B, T, N).  Geometric prior P(n) = (1 - lambda_p)^{n-1} * lambda_p.
    """
    N = p.size(-1)
    n = torch.arange(1, N + 1, device=p.device, dtype=p.dtype)
    log_prior = (n - 1) * torch.log(torch.tensor(1.0 - lambda_p, device=p.device)) + torch.log(
        torch.tensor(lambda_p, device=p.device)
    )
    log_p = torch.log(p.clamp_min(1e-8))
    kl = (p * (log_p - log_prior)).sum(dim=-1)     # (B, T)
    return kl.mean()


def halt_entropy_bonus(p: torch.Tensor) -> torch.Tensor:
    """Mean entropy of the per-token halt distribution.

    Used as a *bonus* (subtracted from the total loss with a small
    weight) to prevent the halt distribution from collapsing onto a
    single step early in training, which would zero out gradient flow
    through every other recurrent step.
    """
    log_p = torch.log(p.clamp_min(1e-8))
    return -(p * log_p).sum(dim=-1).mean()
