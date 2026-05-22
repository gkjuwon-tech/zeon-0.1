"""Surgical weight transplant: copy from any HF causal LM into Zeon.

The strategy:
  * Read the source `AutoConfig` and derive a matching `ZeonConfig`.
  * Copy `embed_tokens`, `lm_head` and the final norm verbatim.
  * For the outer `num_hidden_layers` blocks, copy attention + FFN +
    norms from the corresponding source layers (the *last* L of them —
    the late layers tend to do the most reasoning, which is what we
    want to keep as the foundation under the recurrent stack).
  * For the recurrent core, initialize from the source's last layer
    (or each recurrent slot from a different source layer if we ever
    decide to unshare weights). Halt head stays at its small init.

Anything we can't copy cleanly is logged and left at the random init
done by `_init_weights`. By default `freeze_*` flags in `ZeonConfig`
will then prevent the copied tensors from drifting during distillation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from zeon.config import ZeonConfig
from zeon.modeling_zeon import ZeonForCausalLM


@dataclass
class TransplantReport:
    copied: list[str]
    skipped: list[str]
    src_layers_used: list[int]

    def summary(self) -> str:
        return (
            f"copied {len(self.copied)} tensors, skipped {len(self.skipped)}; "
            f"used source layers {self.src_layers_used}"
        )


def derive_zeon_config(
    src_cfg,
    *,
    num_hidden_layers: int = 4,
    num_recurrent_layers: int = 2,
    max_recurrent_steps: int = 16,
    max_position_embeddings: int = 16384,
) -> ZeonConfig:
    """Build a ZeonConfig whose dims match `src_cfg` (Qwen/Llama style)."""
    head_dim = getattr(src_cfg, "head_dim", None) or (src_cfg.hidden_size // src_cfg.num_attention_heads)
    return ZeonConfig(
        vocab_size=src_cfg.vocab_size,
        hidden_size=src_cfg.hidden_size,
        intermediate_size=src_cfg.intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_recurrent_layers=num_recurrent_layers,
        num_attention_heads=src_cfg.num_attention_heads,
        num_key_value_heads=getattr(src_cfg, "num_key_value_heads", src_cfg.num_attention_heads),
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        rope_theta=getattr(src_cfg, "rope_theta", 1_000_000.0),
        rms_norm_eps=getattr(src_cfg, "rms_norm_eps", 1e-6),
        max_recurrent_steps=max_recurrent_steps,
        tie_word_embeddings=getattr(src_cfg, "tie_word_embeddings", False),
        bos_token_id=getattr(src_cfg, "bos_token_id", 1),
        eos_token_id=getattr(src_cfg, "eos_token_id", 2),
        pad_token_id=getattr(src_cfg, "pad_token_id", None),
        base_model_name_or_path=getattr(src_cfg, "_name_or_path", None),
    )


def _copy(dst: nn.Parameter, src: torch.Tensor, name: str, report: TransplantReport):
    if dst.shape == src.shape:
        with torch.no_grad():
            dst.copy_(src.to(dst.dtype))
        report.copied.append(name)
    else:
        report.skipped.append(f"{name} (shape {tuple(dst.shape)} vs {tuple(src.shape)})")


def _layer_state(model, idx: int) -> dict[str, torch.Tensor]:
    """Pull a single layer's weights out of an HF causal LM, normalized
    to the names ZEON uses internally. Works for Llama/Qwen/Mistral-style
    `.model.layers[i]`."""
    base = getattr(model, "model", model)
    layer = base.layers[idx]
    return {
        "input_layernorm.weight": layer.input_layernorm.weight.data,
        "self_attn.q_proj.weight": layer.self_attn.q_proj.weight.data,
        "self_attn.k_proj.weight": layer.self_attn.k_proj.weight.data,
        "self_attn.v_proj.weight": layer.self_attn.v_proj.weight.data,
        "self_attn.o_proj.weight": layer.self_attn.o_proj.weight.data,
        "post_attention_layernorm.weight": layer.post_attention_layernorm.weight.data,
        "mlp.gate_proj.weight": layer.mlp.gate_proj.weight.data,
        "mlp.up_proj.weight": layer.mlp.up_proj.weight.data,
        "mlp.down_proj.weight": layer.mlp.down_proj.weight.data,
    }


def transplant_from_hf(
    src_model_name_or_path: str,
    *,
    num_hidden_layers: int = 4,
    num_recurrent_layers: int = 2,
    max_recurrent_steps: int = 16,
    max_position_embeddings: int = 16384,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = None,
) -> tuple[ZeonForCausalLM, TransplantReport]:
    src_cfg = AutoConfig.from_pretrained(src_model_name_or_path, trust_remote_code=True)
    src = AutoModelForCausalLM.from_pretrained(
        src_model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    src.eval()

    cfg = derive_zeon_config(
        src_cfg,
        num_hidden_layers=num_hidden_layers,
        num_recurrent_layers=num_recurrent_layers,
        max_recurrent_steps=max_recurrent_steps,
        max_position_embeddings=max_position_embeddings,
    )
    model = ZeonForCausalLM(cfg).to(dtype=dtype)

    report = TransplantReport(copied=[], skipped=[], src_layers_used=[])
    src_base = getattr(src, "model", src)

    # Embeddings + lm_head + final norm
    _copy(model.model.embed_tokens.weight, src_base.embed_tokens.weight.data,
          "embed_tokens", report)
    if hasattr(src, "lm_head"):
        _copy(model.lm_head.weight, src.lm_head.weight.data, "lm_head", report)
    if hasattr(src_base, "norm"):
        _copy(model.model.final_norm.weight, src_base.norm.weight.data, "final_norm", report)

    # Outer transformer stack: take the LAST `num_hidden_layers` source layers.
    src_depth = len(src_base.layers)
    outer_idx = list(range(src_depth - num_hidden_layers, src_depth))
    report.src_layers_used.extend(outer_idx)

    for dst_i, src_i in enumerate(outer_idx):
        st = _layer_state(src, src_i)
        dst = model.model.layers[dst_i]
        _copy(dst.input_layernorm.weight, st["input_layernorm.weight"],
              f"layers.{dst_i}.input_layernorm", report)
        _copy(dst.self_attn.q_proj.weight, st["self_attn.q_proj.weight"],
              f"layers.{dst_i}.self_attn.q_proj", report)
        _copy(dst.self_attn.k_proj.weight, st["self_attn.k_proj.weight"],
              f"layers.{dst_i}.self_attn.k_proj", report)
        _copy(dst.self_attn.v_proj.weight, st["self_attn.v_proj.weight"],
              f"layers.{dst_i}.self_attn.v_proj", report)
        _copy(dst.self_attn.o_proj.weight, st["self_attn.o_proj.weight"],
              f"layers.{dst_i}.self_attn.o_proj", report)
        _copy(dst.post_attention_layernorm.weight, st["post_attention_layernorm.weight"],
              f"layers.{dst_i}.post_attention_layernorm", report)
        _copy(dst.mlp.gate_proj.weight, st["mlp.gate_proj.weight"],
              f"layers.{dst_i}.mlp.gate_proj", report)
        _copy(dst.mlp.up_proj.weight, st["mlp.up_proj.weight"],
              f"layers.{dst_i}.mlp.up_proj", report)
        _copy(dst.mlp.down_proj.weight, st["mlp.down_proj.weight"],
              f"layers.{dst_i}.mlp.down_proj", report)

    # Recurrent core: seed from the very last source layer so reasoning
    # starts close to the teacher's last-layer behavior.
    last = _layer_state(src, src_depth - 1)
    report.src_layers_used.append(src_depth - 1)
    unique_cores: Iterable[nn.Module]
    if cfg.share_recurrent_weights:
        unique_cores = [model.model.recurrent.cores[0]]
    else:
        unique_cores = list(model.model.recurrent.cores)

    for ci, core in enumerate(unique_cores):
        _copy(core.input_norm.weight, last["input_layernorm.weight"],
              f"recurrent.cores.{ci}.input_norm", report)
        _copy(core.attn.q_proj.weight, last["self_attn.q_proj.weight"],
              f"recurrent.cores.{ci}.attn.q_proj", report)
        _copy(core.attn.k_proj.weight, last["self_attn.k_proj.weight"],
              f"recurrent.cores.{ci}.attn.k_proj", report)
        _copy(core.attn.v_proj.weight, last["self_attn.v_proj.weight"],
              f"recurrent.cores.{ci}.attn.v_proj", report)
        _copy(core.attn.o_proj.weight, last["self_attn.o_proj.weight"],
              f"recurrent.cores.{ci}.attn.o_proj", report)
        _copy(core.post_attn_norm.weight, last["post_attention_layernorm.weight"],
              f"recurrent.cores.{ci}.post_attn_norm", report)
        _copy(core.ffn.gate_proj.weight, last["mlp.gate_proj.weight"],
              f"recurrent.cores.{ci}.ffn.gate_proj", report)
        _copy(core.ffn.up_proj.weight, last["mlp.up_proj.weight"],
              f"recurrent.cores.{ci}.ffn.up_proj", report)
        _copy(core.ffn.down_proj.weight, last["mlp.down_proj.weight"],
              f"recurrent.cores.{ci}.ffn.down_proj", report)

    model._apply_freeze_policy()
    return model, report
