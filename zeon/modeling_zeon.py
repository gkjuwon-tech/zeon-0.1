"""ZEON model implementation.

A causal LM that interleaves a few standard transformer blocks with a
recurrent "thinking" stack. The recurrent stack applies the same block
(or a small shared bank of blocks) up to K times per forward pass,
allowing latent chain-of-thought without growing the context window.

Designed to be HF-compatible: subclasses PreTrainedModel, registers as
`AutoModelForCausalLM` via `ZeonConfig.model_type == "zeon"`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from zeon.config import ZeonConfig
from zeon.halting import HaltingHead, ponder_combine, ponder_kl_loss


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x.type_as(self.weight) * self.weight)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_pos: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_pos = max_pos

    def forward(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q, k: (B, H, T, D); cos/sin: (T, D)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Attention + FFN
# ---------------------------------------------------------------------------


class GroupedQueryAttention(nn.Module):
    """GQA with RoPE, compatible with Qwen/Llama-style configs."""

    def __init__(self, cfg: ZeonConfig):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
        self.dropout = cfg.attention_dropout

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cached_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if cached_kv is None:
            k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            cos, sin = self.rope(T, x.device, x.dtype)
            q, k = apply_rope(q, k, cos, sin)
            kv_out = (k, v)
        else:
            # In recurrent steps after the first, we re-use the same K, V —
            # only the queries change as the hidden state evolves. This is
            # the cheap trick that keeps recurrent thinking O(T) per step.
            k, v = cached_kv
            cos, sin = self.rope(T, x.device, x.dtype)
            q, _ = apply_rope(q, q.new_zeros(q.shape), cos, sin)
            kv_out = cached_kv

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attention_mask is None,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(attn), kv_out


class SwiGLU(nn.Module):
    def __init__(self, cfg: ZeonConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Recurrent core
# ---------------------------------------------------------------------------


class RecurrentCore(nn.Module):
    """One "thinking step". Applied K times in latent space per forward.

    The K and V projections are computed once on the first step and then
    re-used; only Q evolves as the hidden state thinks. This lets us add
    reasoning depth without re-paying attention cost on long contexts.
    """

    def __init__(self, cfg: ZeonConfig):
        super().__init__()
        self.cfg = cfg
        self.input_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.post_attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg)
        self.halt_head = HaltingHead(cfg.hidden_size)
        if cfg.recurrent_state_mix == "gated":
            self.mix_gate = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
            nn.init.zeros_(self.mix_gate.bias)

    def forward(
        self,
        h: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cached_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, kv_out = self.attn(self.input_norm(h), attention_mask, cached_kv)
        if self.cfg.recurrent_state_mix == "gated":
            g = torch.sigmoid(self.mix_gate(h))
            h = h + g * attn_out
        else:
            h = h + attn_out
        h = h + self.ffn(self.post_attn_norm(h))
        halt_lambda = self.halt_head(h)
        return h, halt_lambda, kv_out


class TransformerBlock(nn.Module):
    """A vanilla pre-norm transformer block (no recurrence, no halt head).

    Used for the `num_hidden_layers` outer stack that feeds into the
    recurrent thinking layers. This is also the shape that `transplant`
    targets when copying weights from a base HF causal LM.
    """

    def __init__(self, cfg: ZeonConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, attention_mask=None):
        h, _ = self.self_attn(self.input_layernorm(x), attention_mask, cached_kv=None)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ---------------------------------------------------------------------------
# ZeonBlock — wraps RecurrentCore with adaptive halting
# ---------------------------------------------------------------------------


class ZeonBlock(nn.Module):
    def __init__(self, cfg: ZeonConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.share_recurrent_weights:
            shared = RecurrentCore(cfg)
            self.cores = nn.ModuleList([shared] * cfg.num_recurrent_layers)
        else:
            self.cores = nn.ModuleList([RecurrentCore(cfg) for _ in range(cfg.num_recurrent_layers)])

    def forward(
        self,
        h: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run up to `max_recurrent_steps` thinking steps with adaptive halt.

        Returns:
            h_out:        (B, T, D) PonderNet-weighted hidden state.
            ponder_loss:  scalar tensor (KL to geometric prior); requires_grad
                          if any halt head parameter does.
        """
        cfg = self.cfg
        # The K, V of the first attention call are cached and reused on
        # subsequent steps. We cache per-layer in the recurrent bank.
        kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.cores)
        hiddens, lambdas = [], []

        for step in range(cfg.max_recurrent_steps):
            current = h
            for li, core in enumerate(self.cores):
                current, lam, kv = core(current, attention_mask, kv_caches[li])
                kv_caches[li] = kv
            h = current
            hiddens.append(h)
            lambdas.append(lam)

            # During inference we can early-exit if every token's
            # cumulative halt mass exceeds threshold. We still keep all
            # collected steps for ponder_combine to weight properly.
            if not self.training and step + 1 >= cfg.min_recurrent_steps:
                with torch.no_grad():
                    lam_stack = torch.stack(lambdas, dim=-1)
                    not_halt = torch.clamp(1.0 - lam_stack, min=1e-6).prod(dim=-1)
                    if (not_halt < (1.0 - cfg.halt_threshold)).all():
                        break

        h_out, p, _ = ponder_combine(hiddens, lambdas)
        ponder_loss = ponder_kl_loss(p, cfg.ponder_lambda_p)
        return h_out, ponder_loss


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class ZeonPreTrainedModel(PreTrainedModel):
    config_class = ZeonConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock", "ZeonBlock"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class ZeonModel(ZeonPreTrainedModel):
    def __init__(self, cfg: ZeonConfig):
        super().__init__(cfg)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.recurrent = ZeonBlock(cfg)
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.embed_tokens(input_ids)
        # `attention_mask` from HF is (B, T) of 1/0; SDPA wants additive mask
        # or None for pure causal. For the skeleton we rely on is_causal=True
        # when no explicit mask is provided.
        sdpa_mask = None
        if attention_mask is not None and (attention_mask == 0).any():
            B, T = attention_mask.shape
            mask = attention_mask[:, None, None, :].to(dtype=h.dtype)
            mask = (1.0 - mask) * torch.finfo(h.dtype).min
            causal = torch.full((T, T), torch.finfo(h.dtype).min, device=h.device, dtype=h.dtype)
            causal = torch.triu(causal, diagonal=1)
            sdpa_mask = mask + causal

        for layer in self.layers:
            h = layer(h, sdpa_mask)
        h, ponder_loss = self.recurrent(h, sdpa_mask)
        h = self.final_norm(h)
        return h, ponder_loss


@dataclass
class ZeonCausalLMOutput(CausalLMOutputWithPast):
    ponder_loss: torch.Tensor | None = None


class ZeonForCausalLM(ZeonPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, cfg: ZeonConfig):
        super().__init__(cfg)
        self.model = ZeonModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.post_init()
        self._apply_freeze_policy()

    def _apply_freeze_policy(self):
        cfg = self.config
        if cfg.freeze_embed:
            for p in self.model.embed_tokens.parameters():
                p.requires_grad = False
        if cfg.freeze_lm_head:
            for p in self.lm_head.parameters():
                p.requires_grad = False
        if cfg.freeze_ffn:
            for layer in self.model.layers:
                for p in layer.mlp.parameters():
                    p.requires_grad = False
            for core in self.model.recurrent.cores:
                for p in core.ffn.parameters():
                    p.requires_grad = False

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
        **_,
    ) -> ZeonCausalLMOutput:
        h, ponder_loss = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = ce + self.config.ponder_loss_weight * ponder_loss

        return ZeonCausalLMOutput(
            loss=loss,
            logits=logits,
            ponder_loss=ponder_loss,
        )
