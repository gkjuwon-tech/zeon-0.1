"""ZEON model implementation.

A causal LM that interleaves a few standard transformer blocks with a
recurrent "thinking" stack. The recurrent stack applies the same block
(or a small shared bank of blocks) up to K times per forward pass,
allowing latent chain-of-thought without growing the context window.

HF-compatible: subclasses `PreTrainedModel`, supports `from_pretrained`
+ `save_pretrained` + `.generate()` via KV cache, gradient checkpointing,
and registers with `AutoConfig` / `AutoModelForCausalLM` (see
`zeon/__init__.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from zeon.config import ZeonConfig
from zeon.halting import HaltingHead, halt_entropy_bonus, ponder_combine, ponder_kl_loss
from zeon.workspace import WorkspaceBank


# Type aliases — outer-layer KV cache is a tuple of N (k, v) per layer.
LayerKV = Tuple[torch.Tensor, torch.Tensor]
PastKV = Tuple[LayerKV, ...]


def _is_valid_kv_entry(entry) -> bool:
    return (
        entry is not None
        and len(entry) >= 2
        and isinstance(entry[0], torch.Tensor)
        and isinstance(entry[1], torch.Tensor)
        and entry[0].numel() > 0
    )


def _to_legacy_cache(pkv) -> Optional[PastKV]:
    """Coerce an HF `Cache` (or tuple, or None) into the legacy tuple format
    we use internally. Returns None for an empty/uninitialized cache."""
    if pkv is None:
        return None
    if isinstance(pkv, tuple):
        if len(pkv) == 0 or not _is_valid_kv_entry(pkv[0]):
            return None
        return pkv
    if isinstance(pkv, Cache):
        to_legacy = getattr(pkv, "to_legacy_cache", None)
        if callable(to_legacy):
            legacy = to_legacy()
            if legacy and len(legacy) > 0 and _is_valid_kv_entry(legacy[0]):
                return tuple(legacy)
        # Fallbacks for various transformers minor versions.
        if hasattr(pkv, "key_cache") and len(pkv.key_cache) > 0 and pkv.key_cache[0] is not None:
            return tuple(zip(pkv.key_cache, pkv.value_cache))
        layers = getattr(pkv, "layers", None)
        if layers:
            extracted = []
            for layer in layers:
                k = getattr(layer, "keys", None)
                if k is None:
                    k = getattr(layer, "key_cache", None)
                v = getattr(layer, "values", None)
                if v is None:
                    v = getattr(layer, "value_cache", None)
                if k is None or v is None or not isinstance(k, torch.Tensor):
                    return None
                extracted.append((k, v))
            return tuple(extracted) if extracted else None
        return None
    raise TypeError(f"Unsupported past_key_values type: {type(pkv).__name__}")


def _to_dynamic_cache(new_pkv: Optional[PastKV]) -> Optional[DynamicCache]:
    """Wrap our internal tuple back into an HF DynamicCache for generate()."""
    if new_pkv is None:
        return None
    from_legacy = getattr(DynamicCache, "from_legacy_cache", None)
    if callable(from_legacy):
        return from_legacy(new_pkv)
    cache = DynamicCache()
    for i, (k, v) in enumerate(new_pkv):
        cache.update(k, v, i)
    return cache


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
        return x.type_as(self.weight) * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE table; supports position_ids-style lookups."""

    def __init__(self, head_dim: int, max_pos: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # Must be persistent so HF's meta-device loader (used by
        # `from_pretrained` with `low_cpu_mem_usage=True`) doesn't leave
        # this buffer uninitialized — otherwise loaded models silently
        # decode with zero RoPE frequencies and diverge from the source.
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.max_pos = max_pos
        self._cache: dict[tuple[torch.device, torch.dtype, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _table(self, seq_len: int, device, dtype):
        key = (device, dtype, seq_len)
        if key not in self._cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cache[key] = (emb.cos().to(dtype), emb.sin().to(dtype))
        return self._cache[key]

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        """Return cos, sin tensors gathered at `position_ids`.

        position_ids: (B, T) int64. Returns cos, sin: (B, T, D).
        """
        max_pos = int(position_ids.max().item()) + 1
        cos, sin = self._table(max(max_pos, 1), position_ids.device, dtype)
        return cos[position_ids], sin[position_ids]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a (B, H, T, D) tensor with (B, T, D) cos/sin."""
    cos = cos.unsqueeze(1)  # (B, 1, T, D)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class OuterAttention(nn.Module):
    """HF-compatible GQA: returns the next-step KV cache for `.generate()`.

    Used in the non-recurrent outer transformer blocks. Accepts an optional
    `past_kv` (k, v) tuple of shape (B, H_kv, T_past, D_head); concatenates
    along the time axis and returns the updated cache when `use_cache=True`.
    """

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
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[LayerKV] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[LayerKV]]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(position_ids, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=-2)
            v = torch.cat([past_kv[1], v], dim=-2)
        new_kv = (k, v) if use_cache else None

        # GQA: repeat KV heads to match query heads.
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_use = k.repeat_interleave(repeat, dim=1)
            v_use = v.repeat_interleave(repeat, dim=1)
        else:
            k_use, v_use = k, v

        is_causal = attention_mask is None and (past_kv is None) and T > 1
        attn = F.scaled_dot_product_attention(
            q, k_use, v_use,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(attn), new_kv


class RecurrentAttention(nn.Module):
    """GQA for the recurrent core.

    The first call computes K, V from the input and caches them for the
    remaining thinking steps; subsequent calls re-use those K, V and only
    refresh Q from the evolved hidden state. This keeps the cost of latent
    reasoning at O(T) per step instead of O(T^2).
    """

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
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cached_kv: Optional[LayerKV] = None,
    ) -> Tuple[torch.Tensor, LayerKV]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(position_ids, x.dtype)
        q = apply_rope(q, cos, sin)

        if cached_kv is None:
            k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            k = apply_rope(k, cos, sin)
            cached_kv = (k, v)
        k, v = cached_kv

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_use = k.repeat_interleave(repeat, dim=1)
            v_use = v.repeat_interleave(repeat, dim=1)
        else:
            k_use, v_use = k, v

        is_causal = attention_mask is None and T > 1
        attn = F.scaled_dot_product_attention(
            q, k_use, v_use,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(attn), cached_kv


class StepEmbedding(nn.Module):
    """Encodes which thinking step the model is currently performing.

    Combines a fixed sinusoidal base (zero-init learned residual on top)
    so that the model knows whether it is in step 1 (just started thinking)
    vs step K (close to halting). Mixed additively into the hidden state at
    the start of each recurrent step.
    """

    def __init__(self, hidden_size: int, max_steps: int):
        super().__init__()
        pe = torch.zeros(max_steps, hidden_size)
        pos = torch.arange(0, max_steps).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Persistent so HF's meta loader doesn't blank it.
        self.register_buffer("sinusoidal", pe, persistent=True)
        self.learned = nn.Parameter(torch.zeros(max_steps, hidden_size))

    def forward(self, step_idx: int) -> torch.Tensor:
        return self.sinusoidal[step_idx] + self.learned[step_idx]


class CrossStepMemory(nn.Module):
    """Lightweight cross-attention from current thinking step to prior steps.

    Lets a token's hidden state at step N look up information it produced
    at steps N-1, N-2, ... within a bounded window. This is what turns the
    recurrent loop from "RNN-style scalar accumulation" into a real
    latent chain-of-thought: each step can reference (and rewrite) the
    intermediate conclusions of previous steps.

    Implemented as a single-head dot-product attention over the per-token
    history of hidden states, gated by a sigmoid so it can be ignored when
    not useful (and so initialization is near-identity).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, hidden_size, bias=True)
        # Zero-init output and bias-shift the gate strongly negative so the
        # module is the identity at init time; weight grows in only if it
        # actually helps the loss go down.
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)
        self.scale = hidden_size ** -0.5

    def forward(self, h: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        # h:       (B, T, D)             — current step's hidden state
        # history: (B, T, S, D)          — last S steps' hidden states
        B, T, S, D = history.shape
        q = self.q_proj(h).unsqueeze(2)                          # (B, T, 1, D)
        k = self.k_proj(history)                                 # (B, T, S, D)
        v = self.v_proj(history)
        attn = (q * k).sum(dim=-1, keepdim=True) * self.scale    # (B, T, S, 1)
        weights = F.softmax(attn, dim=2)
        read = (weights * v).sum(dim=2)                          # (B, T, D)
        gated = torch.sigmoid(self.gate(h)) * self.o_proj(read)
        return h + gated


class SwiGLU(nn.Module):
    def __init__(self, cfg: ZeonConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Vanilla pre-norm decoder block. Transplant target for the outer stack."""

    def __init__(self, cfg: ZeonConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = OuterAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[LayerKV] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[LayerKV]]:
        h, new_kv = self.self_attn(
            self.input_layernorm(x),
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


class RecurrentCore(nn.Module):
    """One latent thinking step. Applied K times per forward pass.

    Each invocation refreshes the queries from the evolved hidden state and
    re-uses cached K, V. Halting probability is computed from the post-FFN
    hidden state for stability.
    """

    def __init__(self, cfg: ZeonConfig):
        super().__init__()
        self.cfg = cfg
        self.input_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = RecurrentAttention(cfg)
        self.post_attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg)
        self.halt_head = HaltingHead(cfg.hidden_size)
        if cfg.recurrent_state_mix == "gated":
            self.mix_gate = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
            nn.init.zeros_(self.mix_gate.bias)

    def forward(
        self,
        h: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cached_kv: Optional[LayerKV] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, LayerKV]:
        attn_out, kv_out = self.attn(self.input_norm(h), position_ids, attention_mask, cached_kv)
        if self.cfg.recurrent_state_mix == "gated":
            g = torch.sigmoid(self.mix_gate(h))
            h = h + g * attn_out
        else:
            h = h + attn_out
        h = h + self.ffn(self.post_attn_norm(h))
        halt_lambda = self.halt_head(h)
        return h, halt_lambda, kv_out


class ZeonBlock(nn.Module):
    """Wraps `RecurrentCore` with adaptive halting and a max-step budget.

    Per thinking step:
      1. Add a step embedding so the core knows which iteration it's on.
      2. Run the core (attention + FFN + halt head).
      3. If `cross_step_memory` is on, let the new hidden state read from
         a sliding window of the W most recent hidden states. This is the
         only place where recurrence becomes more than scalar refinement —
         the model can re-use intermediate latent results from earlier
         steps, which is what makes deep latent chain-of-thought possible.
    """

    def __init__(self, cfg: ZeonConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.share_recurrent_weights:
            shared = RecurrentCore(cfg)
            self.cores = nn.ModuleList([shared] * cfg.num_recurrent_layers)
        else:
            self.cores = nn.ModuleList([RecurrentCore(cfg) for _ in range(cfg.num_recurrent_layers)])
        if cfg.use_step_embedding:
            self.step_embed = StepEmbedding(cfg.hidden_size, cfg.max_recurrent_steps)
        else:
            self.step_embed = None
        if cfg.cross_step_memory:
            self.cross_step = CrossStepMemory(cfg.hidden_size)
        else:
            self.cross_step = None
        if cfg.use_workspace:
            self.workspace = WorkspaceBank(cfg)
        else:
            self.workspace = None
        self.gradient_checkpointing = False

    def forward(
        self,
        h: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        kv_caches: list[Optional[LayerKV]] = [None] * len(self.cores)
        hiddens, lambdas = [], []
        history: list[torch.Tensor] = []  # rolling window of recent hidden states
        workspace = self.workspace.init_state(h) if self.workspace is not None else None
        div_loss_acc = torch.zeros((), device=h.device, dtype=h.dtype)

        for step in range(cfg.max_recurrent_steps):
            current = h
            if self.step_embed is not None:
                current = current + self.step_embed(step)

            for li, core in enumerate(self.cores):
                if self.gradient_checkpointing and self.training:
                    current, lam, kv = torch.utils.checkpoint.checkpoint(
                        core, current, position_ids, attention_mask, kv_caches[li],
                        use_reentrant=False,
                    )
                else:
                    current, lam, kv = core(current, position_ids, attention_mask, kv_caches[li])
                kv_caches[li] = kv

            if self.cross_step is not None and len(history) > 0:
                # Stack the last W steps into a (B, T, S, D) tensor.
                window = history[-cfg.cross_step_memory_window:]
                hist = torch.stack(window, dim=2)
                current = self.cross_step(current, hist)

            if self.workspace is not None:
                current, workspace = self.workspace.step(current, workspace)
                if self.training:
                    div_loss_acc = div_loss_acc + self.workspace.diversity_loss(workspace)

            h = current
            history.append(h)
            # Trim eagerly to bound activation memory under long K.
            if cfg.cross_step_memory and len(history) > cfg.cross_step_memory_window:
                history = history[-cfg.cross_step_memory_window:]

            hiddens.append(h)
            lambdas.append(lam)

            if not self.training and step + 1 >= cfg.min_recurrent_steps:
                with torch.no_grad():
                    lam_stack = torch.stack(lambdas, dim=-1)
                    not_halt = torch.clamp(1.0 - lam_stack, min=1e-6).prod(dim=-1)
                    if (not_halt < (1.0 - cfg.halt_threshold)).all():
                        break

        h_out, p, _ = ponder_combine(hiddens, lambdas)
        ponder_loss = ponder_kl_loss(p, cfg.ponder_lambda_p)
        entropy = halt_entropy_bonus(p)
        ponder_loss = ponder_loss - cfg.halt_entropy_weight * entropy
        if self.workspace is not None and self.training:
            steps_used = max(len(lambdas), 1)
            ponder_loss = ponder_loss + cfg.workspace_diversity_weight * (div_loss_acc / steps_used)
        return h_out, ponder_loss


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def _build_decoder_mask(
    attention_mask: Optional[torch.Tensor],
    T_q: int,
    T_kv: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build an additive (B, 1, T_q, T_kv) mask combining padding + causality.

    Returns `None` when the default SDPA causal behavior is sufficient
    (no padding, prefill or single-token decode).
    """
    if attention_mask is None or not (attention_mask == 0).any():
        return None
    pad = attention_mask[:, None, None, :].to(dtype=dtype)  # (B, 1, 1, T_kv)
    pad = (1.0 - pad) * torch.finfo(dtype).min
    causal = torch.zeros((T_q, T_kv), dtype=dtype, device=device)
    if T_q > 1:
        offset = T_kv - T_q
        ar = torch.arange(T_q, device=device).unsqueeze(1)
        ks = torch.arange(T_kv, device=device).unsqueeze(0)
        causal = torch.where(ks <= offset + ar, causal, torch.full_like(causal, torch.finfo(dtype).min))
    return pad + causal[None, None, :, :]


class ZeonPreTrainedModel(PreTrainedModel):
    config_class = ZeonConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock", "ZeonBlock", "RecurrentCore", "WorkspaceBank"]
    _supports_cache_class = False  # we use the legacy tuple cache format

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
        # `WorkspaceBank` keeps its `output_alpha` scalar at exactly 0
        # so that `use_workspace=True` is a no-op at init — `torch.zeros`
        # in __init__ is correct; `_init_weights` skips Parameters and
        # nothing here overrides it.

    def _set_gradient_checkpointing(self, module, value: bool = False):
        if isinstance(module, ZeonBlock):
            module.gradient_checkpointing = value


class ZeonModel(ZeonPreTrainedModel):
    def __init__(self, cfg: ZeonConfig):
        super().__init__(cfg)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.recurrent = ZeonBlock(cfg)
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[PastKV] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[PastKV]]:
        B, T = input_ids.shape
        past_len = past_key_values[0][0].size(-2) if past_key_values else 0

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0).expand(B, T)

        h = self.embed_tokens(input_ids)
        T_kv = past_len + T
        attn_mask = _build_decoder_mask(attention_mask, T_q=T, T_kv=T_kv, dtype=h.dtype, device=h.device)

        new_pkv: list[LayerKV] = []
        for i, layer in enumerate(self.layers):
            past_i = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                h, new_kv = torch.utils.checkpoint.checkpoint(
                    layer, h, position_ids, attn_mask, past_i, use_cache,
                    use_reentrant=False,
                )
            else:
                h, new_kv = layer(h, position_ids, attn_mask, past_i, use_cache)
            if use_cache:
                new_pkv.append(new_kv)

        # Recurrent thinking: the latent steps operate on positions relative
        # to the current chunk's hidden states only (the outer KV cache
        # already carries the full historical context into `h`).
        recurrent_pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        recurrent_mask = _build_decoder_mask(attention_mask, T_q=T, T_kv=T, dtype=h.dtype, device=h.device) \
            if past_len == 0 else None
        h, ponder_loss = self.recurrent(h, recurrent_pos, recurrent_mask)
        h = self.final_norm(h)
        return h, ponder_loss, (tuple(new_pkv) if use_cache else None)


@dataclass
class ZeonCausalLMOutput(CausalLMOutputWithPast):
    ponder_loss: Optional[torch.Tensor] = None


class ZeonForCausalLM(ZeonPreTrainedModel, GenerationMixin):
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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        legacy = _to_legacy_cache(past_key_values)
        past_len = legacy[0][0].size(-2) if legacy is not None else 0
        if past_len > 0:
            input_ids = input_ids[:, past_len:]
            if input_ids.size(1) == 0:
                input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids")
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        if position_ids is not None and position_ids.size(-1) != input_ids.size(-1):
            position_ids = position_ids[:, -input_ids.size(-1):]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": kwargs.get("use_cache", True),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[PastKV] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: bool = True,
        **_,
    ) -> ZeonCausalLMOutput:
        if use_cache is None:
            use_cache = past_key_values is not None or (not self.training)

        input_was_cache = isinstance(past_key_values, Cache)
        legacy_pkv = _to_legacy_cache(past_key_values)

        h, ponder_loss, new_pkv = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=legacy_pkv,
            use_cache=use_cache,
        )
        if input_was_cache and new_pkv is not None:
            new_pkv = _to_dynamic_cache(new_pkv)
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
            past_key_values=new_pkv,
            ponder_loss=ponder_loss,
        )
