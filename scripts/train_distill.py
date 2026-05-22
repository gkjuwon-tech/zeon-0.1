"""Distillation training loop for ZEON.

This is a deliberately small, runnable stub — single-GPU, no DeepSpeed,
no checkpointing — so the architecture work can iterate quickly. Swap
in `accelerate launch` + sharded optimizers once shapes stabilize.

Trains the student (ZeonForCausalLM) against a frozen teacher (the base
HF causal LM that was transplanted), on a streaming text dataset.
"""

from __future__ import annotations

import argparse
from itertools import islice

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from zeon import ZeonForCausalLM
from zeon.losses import combined_loss


def collate(batch, tokenizer, max_len):
    texts = [ex["text"] for ex in batch if ex.get("text")]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100
    return enc.input_ids, enc.attention_mask, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="path to Zeon checkpoint (from transplant)")
    ap.add_argument("--teacher", required=True, help="HF repo id of the teacher (same as transplant src)")
    ap.add_argument("--tokenizer", default=None, help="defaults to --teacher")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu", help="HF dataset id")
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--dataset-config", default="sample-10BT")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--ponder-weight", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.teacher, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    student = ZeonForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16).to(args.device)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    ds = load_dataset(args.dataset, name=args.dataset_config, split=args.dataset_split, streaming=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        collate_fn=lambda b: collate(b, tok, args.max_len),
    )

    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[train] trainable params: {n_trainable/1e6:.2f}M")

    student.train()
    for step, (input_ids, attn, labels) in enumerate(islice(loader, args.steps)):
        input_ids = input_ids.to(args.device)
        attn = attn.to(args.device)
        labels = labels.to(args.device)

        out = student(input_ids=input_ids, attention_mask=attn, labels=labels)
        with torch.no_grad():
            t_out = teacher(input_ids=input_ids, attention_mask=attn)
        loss, parts = combined_loss(
            out.logits,
            labels,
            ponder_loss=out.ponder_loss,
            teacher_logits=t_out.logits,
            ponder_weight=args.ponder_weight,
            distill_weight=args.distill_weight,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % args.log_every == 0:
            msg = " ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
            print(f"step {step:>6d}  {msg}")

    student.save_pretrained(args.student + "-distilled")
    print(f"[train] saved -> {args.student}-distilled")


if __name__ == "__main__":
    main()
