"""Transplant a base HF causal LM into a ZeonForCausalLM and save it.

Example:
    python scripts/transplant_from_hf.py \
        --src Qwen/Qwen2.5-1.5B \
        --out checkpoints/zeon-from-qwen2_5-1p5b \
        --outer-layers 4 --recurrent-layers 2 --max-steps 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from zeon.transplant import transplant_from_hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF repo id or local path of the base causal LM")
    ap.add_argument("--out", required=True, help="output directory for the Zeon checkpoint")
    ap.add_argument("--outer-layers", type=int, default=4)
    ap.add_argument("--recurrent-layers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--max-pos", type=int, default=16384)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default=None)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model, report = transplant_from_hf(
        args.src,
        num_hidden_layers=args.outer_layers,
        num_recurrent_layers=args.recurrent_layers,
        max_recurrent_steps=args.max_steps,
        max_position_embeddings=args.max_pos,
        dtype=dtype,
        device_map=args.device_map,
    )
    print("[transplant]", report.summary())
    if report.skipped:
        print("[transplant] skipped:")
        for s in report.skipped:
            print("  -", s)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    print(f"[transplant] wrote checkpoint to {out}")


if __name__ == "__main__":
    main()
