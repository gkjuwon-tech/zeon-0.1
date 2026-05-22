"""Quick eval harness for reasoning tasks.

Runs greedy generation on a small set of math/logic prompts and reports
pass rates against gold answers. Real evals (GSM8K, MATH, ARC-AGI) plug
in via `lm-eval-harness`; this stub exists so we can sanity-check
*before* spending GPU hours.
"""

from __future__ import annotations

import argparse
import re

import torch
from transformers import AutoTokenizer

from zeon import ZeonForCausalLM


PROBES = [
    ("If Alice has 17 apples and gives 5 to Bob, then eats 3, how many does she have?", "9"),
    ("12 * 13 = ?", "156"),
    ("If today is Wednesday, what day will it be 100 days from now?", "Friday"),
    ("A train leaves at 9:15 and arrives at 11:42. How long was the trip in minutes?", "147"),
]


def extract_answer(text: str) -> str:
    m = re.search(r"(-?\d+(?:\.\d+)?|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", text)
    return m.group(1) if m else text.strip().splitlines()[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    model = ZeonForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    model.eval()

    correct = 0
    for q, gold in PROBES:
        enc = tok(q, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_new,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen_text = tok.decode(out_ids[0, enc.input_ids.size(1):], skip_special_tokens=True)
        pred = extract_answer(gen_text)
        ok = str(gold).lower() in pred.lower()
        correct += int(ok)
        print(("OK " if ok else "X  "), q, "->", pred, "(gold:", gold, ")")

    print(f"\nscore: {correct}/{len(PROBES)}")


if __name__ == "__main__":
    main()
