# ZEON

> **We are not growing the model. We are changing the way it thinks.**

ZEON is an architecture research project. We are trying to make an LLM
run a small **learned virtual machine inside its own latent space** —
instead of buying intelligence with more context tokens, we buy it with
*deeper thinking* inside a short context window.

It's a long shot. We say so up front, several places below. We also say
why we think it's a long shot worth running.

---

## What we are *not* building

If your first instinct on reading the next list is *"but with one more
attention head bolted on…"*, you're in the wrong repo. We will close
that PR.

- **Run the transformer K times.** Universal Transformer (Dehghani '18).
  Already exists. Done.
- **Swap out attention.** Mamba / RWKV / linear-attn variants. Those
  buy you cheap *long context*. Our hypothesis is that long context is
  the wrong axis to scale on.
- **Output-token Chain-of-Thought.** Wastes context, every CoT token
  pays full compute, and you can't selectively spend more compute on
  the hard parts of the problem. We want reasoning that happens
  *outside the output token stream*.
- **Bigger model + more data.** Scaling-only is somebody else's job —
  one with way more GPUs than us. Not our angle.

If any change you propose collapses into one of the bullets above with
a small twist on top, it isn't ZEON. We'd rather merge nothing than
merge another Universal-Transformer-with-a-hat.

## What we *actually* build

A model that performs inference inside a **Latent Virtual Machine
(LVM)** — a small, learned interpreter that operates over a structured
latent state. Pieces:

| Component | What it does | Inspiration |
|---|---|---|
| **Workspace Bank** | 16 structured slots of working memory per token. Slots specialise into roles (hypothesis / evidence / scratch / conclusion / …) — *not by labelling, by gradient*. | Baddeley's working memory; Slot Attention (Locatello '20) |
| **Operator Library** | 32–64 small "mini-expert" modules. A learned router picks top-k per step. | MoE; neural ISA / programmable nets |
| **Verifier Head** | A smaller auxiliary model that scores every intermediate workspace state. Gives dense gradient through the latent steps instead of one CE signal at the end. | Process reward models (Lightman '23) |
| **Halting Critic** | RL-trained stop policy. Reward = correctness − step cost. Replaces PonderNet's heuristic geometric prior. | PonderNet (Banino '21) + REINFORCE / PPO |
| **Energy Head** | At inference, runs K rollouts with different latent noise and picks the lowest-energy one. Self-Consistency, but in latent space. | EBMs; Self-Consistency (Wang '23) |

All inside **one HF-compatible model**. All with **per-component
ablation flags** (otherwise the component "exists" in the way a feature
flag named `enable_world_peace` exists). All initialised by
**transplant**: we copy embedding + late layers + LM head from a real
base model (Qwen / Llama / DeepSeek) and freeze them, then graft the
LVM machinery on top.

---

## One-line hypothesis

Today's LLMs are **big dictionary + shallow inference engine.**
ZEON is **same dictionary + much deeper inference engine.**

We freeze the dictionary (transplant + freeze) and learn the engine.

---

## Why we think this is feasible (and where it can die)

We are not allergic to admitting that this is risky. Specifically:

* **Universal Transformer + PonderNet exists and underperforms its
  promise.** That's exactly why we want to go past it. Adaptive depth
  alone is mathematically still an RNN; that's not the bet. Phase 0 in
  this repo is essentially UT+PonderNet, on purpose, as the baseline we
  have to beat.
* **MoE works at scale**, so we know top-k routing over many small
  experts is at least *trainable* at moderate sizes. Our operators are
  smaller (Phase 2) but the failure modes (collapse, dead experts, load
  imbalance) are documented and have known mitigations.
* **Process reward models work** for step-level supervision in
  reasoning traces (OpenAI '23, DeepSeek's process rewards). Our
  Verifier Head (Phase 3) reuses that recipe over latent states instead
  of output tokens.
* **RL for adaptive computation** is the hairy phase. We have an
  "if-this-fails-fall-back-to-expert-iteration" escape hatch in
  `docs/ROADMAP.md`. We expect to use it.

Where this can die:
* Slot collapse in the Workspace Bank (all 16 slots learn the same
  vector). Mitigation: diversity loss, distinct role embeddings,
  optional slot-competition (Slot Attention).
* Dead operators / routing collapse. Mitigation: load-balancing loss,
  Gumbel-softmax temperature scheduling, warm-up forced rotation.
* Inference cost > 5× base. Hard limit; if we hit it the project is
  not worth shipping over just-use-a-bigger-model.

We *will* keep score honestly. Every phase ends with a benchmark
number, written down in `docs/PHASE_NOTES/`. No "looks promising,
let's move on" without numbers.

---

## Current status

**Phase 1 / 7** — Workspace Bank in main, behind `use_workspace=True`.

What's actually merged:
- ✅ Phase 0 — HF-compatible base, KV cache, `generate()`, transplant
  from Llama/Qwen, PonderNet halting, step embeddings, cross-step
  memory, CI, smoke tests.
- ✅ Phase 1 — Workspace Bank: per-token 16-slot bank with role
  embeddings, multi-head intra-token read attention, gated sticky
  writes, diversity loss. Toggleable via `use_workspace` flag.
- [ ] Phase 2 — Operator Library
- [ ] Phase 3 — Verifier Head
- [ ] Phase 4 — Halting Critic (RL)
- [ ] Phase 5 — Parallel Rollouts + Energy Head
- [ ] Phase 6 — Self-Distillation in latent space
- [ ] Phase 7 — Benchmarks + release

The full plan is in [`docs/ROADMAP.md`](docs/ROADMAP.md), with the
"if this fails …" section for every phase (because at least one of
them will).

Target architecture, end-of-year: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Phase write-ups go in [`docs/PHASE_NOTES/`](docs/PHASE_NOTES/) — read
phase 1's note before opening a PR against Phase 2.

---

## Quickstart

```bash
pip install -e ".[dev]"
pytest tests/ -q          # 26/26 green at time of writing
ruff check zeon scripts tests
```

Transplant Phase 0 / 1 from a base model:

```bash
python scripts/transplant_from_hf.py \
    --src Qwen/Qwen2.5-1.5B \
    --out checkpoints/zeon-p1
```

Load the result like any HF model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import zeon  # registers AutoConfig / AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("checkpoints/zeon-p1")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
out = model.generate(**tok("Solve: 23*47=", return_tensors="pt"),
                     max_new_tokens=32, do_sample=False)
print(tok.decode(out[0]))
```

(The Phase-0/1 checkpoint will not actually be smart yet. That's
not the point — the point is the latent-VM machinery loads, the KV
cache works, `.generate()` works, save/load round-trips, ablation
flags do what they say. The smarts are Phases 3–6.)

---

## What we want to measure, end of year

Versus the vanilla base model's inference:

1. **Same average FLOPs**: +20% absolute on reasoning benchmarks
   (GSM8K, MATH, ARC-AGI, BBH).
2. **5× fewer output tokens** (because the reasoning happens in
   latent space, not in the output stream).
3. **More latent steps on harder problems** — the Halting Critic
   should auto-allocate compute by difficulty. Step-count ↔
   difficulty Spearman > 0.6.
4. **Each component shows clear ablation impact** — turning off
   Workspace / Operator / Verifier / Halt-Critic / Energy individually
   must visibly cost score.

If we hit fewer than 2 of those four, we write it up, admit it, and
the next ZEON-shaped project starts somewhere better-informed.

---

## Contributing

This is research code. We want collaborators, and we want collaborators
who are willing to push back on the design.

* Read [`docs/ROADMAP.md`](docs/ROADMAP.md) and
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) **before** opening
  feature PRs. Drive-by "what if we added more X" PRs without a
  hypothesis attached will be politely declined.
* Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for PR / branch / test
  conventions and the three-question sanity check we use before
  starting any new architectural piece.
* Be nice. See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Sharp
  technical disagreement is *expected*; personal attacks are not.

Good first issues: anything labelled
[`good first issue`](https://github.com/gkjuwon-tech/zeon-0.1/issues?q=is%3Aissue+label%3A%22good+first+issue%22)
or
[`help wanted`](https://github.com/gkjuwon-tech/zeon-0.1/issues?q=is%3Aissue+label%3A%22help+wanted%22).

For research-shaped contributions (new component, new training
recipe, new ablation), open a `research proposal` issue first —
template included — so we can argue about the design *before* the code
exists.

---

## License

Apache-2.0. Use it. Fork it. Just don't claim you trained the base
model from scratch on three RTX 4090s.
