"""Training losses: language-model CE + ponder KL + optional distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def lm_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard shift-by-one causal CE. Pads should be `-100` in `labels`."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def distill_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    ignore_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward-KL on token distributions (teacher || student).

    `ignore_mask` is a 0/1 mask of shape (B, T-1); positions with 0 are
    not counted. Logits are shifted to align with next-token labels.
    """
    s = student_logits[..., :-1, :] / temperature
    t = teacher_logits[..., :-1, :] / temperature
    log_s = F.log_softmax(s, dim=-1)
    log_t = F.log_softmax(t, dim=-1)
    p_t = log_t.exp()
    kl = (p_t * (log_t - log_s)).sum(dim=-1)  # (B, T-1)
    if ignore_mask is not None:
        kl = kl * ignore_mask
        return kl.sum() / ignore_mask.sum().clamp_min(1.0) * (temperature ** 2)
    return kl.mean() * (temperature ** 2)


def combined_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    ponder_loss: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
    *,
    ponder_weight: float = 1e-2,
    distill_weight: float = 1.0,
    distill_temperature: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ce = lm_cross_entropy(student_logits, labels)
    parts: dict[str, torch.Tensor] = {"ce": ce.detach(), "ponder": ponder_loss.detach()}
    total = ce + ponder_weight * ponder_loss
    if teacher_logits is not None:
        ignore = (labels[..., 1:] != -100).to(student_logits.dtype)
        kd = distill_kl(student_logits, teacher_logits, distill_temperature, ignore)
        total = total + distill_weight * kd
        parts["distill"] = kd.detach()
    parts["total"] = total.detach()
    return total, parts
