# ZEON Architecture (target, end-of-year)

> This is the **destination drawing** — what ZEON should look like at
> the end of year one. Today's code lands you at Phase 0 (baseline)
> and, with this PR, Phase 1 (workspace bank). Everything else here
> is the target we're aiming for, including the parts that might
> blow up. We expect at least one block in this document to die in
> contact with real GPUs; the failure protocols live in
> `docs/ROADMAP.md`.

---

## 1. Core hypothesis

**Today's LLM**: big dictionary (embedding + FFN) + a single forward
pass of attention reasoning.
**ZEON**: same-size dictionary (transplanted, frozen) + a *learned
virtual machine* running in latent space on top of it.

In compact pseudocode:

```
LLM:    y = LLM(x)
      = Decoder(Embed(x))                # one forward pass

ZEON:  h_0 = OuterEncoder(Embed(x))      # transplanted, frozen
       W_0 = InitWorkspace(h_0)
       for t in 1..K:
           op  = Router(W_{t-1}, h_0)            # which operator
           W_t = ApplyOperator(op, W_{t-1}, h_0) # update workspace
           v_t = Verifier(W_t, x)                # step-level reward
           if HaltCritic(W_t) > τ: break         # adaptive halt (RL)
       y = LMHead(Project(W_K))          # transplanted, frozen
```

`K_max` is small (16–32). Context length is also small (16k target).
The loop *is* the model. Every step the network is, in some real
sense, executing a learned program.

---

## 2. System block diagram

```
                       ┌─────────────────────────────────────┐
                       │           Input tokens (x)          │
                       └─────────────────┬───────────────────┘
                                         │
                                ┌────────▼────────┐
                                │  Token Embed    │ ← transplanted, frozen
                                └────────┬────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │   Outer Transformer Stack   │ ← transplanted, mostly frozen
                          │   (~4 layers, no halt)      │
                          └──────────────┬──────────────┘
                                         │ h_0 ∈ (B, T, D)
                                         │
                  ┌──────────────────────▼──────────────────────┐
                  │             Workspace Init                  │ ← Phase 1
                  │   W_0 ∈ (B, T, S, D), S = 16                │
                  │   role_emb learned per slot                 │
                  └──────────────────────┬──────────────────────┘
                                         │
                       ╔═════════════════▼═════════════════╗
                       ║   LATENT VIRTUAL MACHINE LOOP     ║
                       ║   for t in 1..K_max:              ║
                       ║                                   ║
                       ║   ┌──── ROUTER ─────────────────┐ ║ ← Phase 2
                       ║   │ W_{t-1}, h_0 → op_idx top-k │ ║
                       ║   └────────────┬────────────────┘ ║
                       ║                │                  ║
                       ║   ┌────────────▼────────────────┐ ║ ← Phase 1
                       ║   │      READ HEADS (M)         │ ║
                       ║   │ Pull from h_0 + W_{t-1}     │ ║
                       ║   └────────────┬────────────────┘ ║
                       ║                │                  ║
                       ║   ┌────────────▼────────────────┐ ║ ← Phase 2
                       ║   │   APPLY OPERATOR (sparse)   │ ║
                       ║   │ FFN_{op} + Attn_{op}        │ ║
                       ║   └────────────┬────────────────┘ ║
                       ║                │                  ║
                       ║   ┌────────────▼────────────────┐ ║ ← Phase 1
                       ║   │      WRITE HEADS (M)        │ ║
                       ║   │ Gated update of W slots     │ ║
                       ║   └────────────┬────────────────┘ ║
                       ║                │                  ║
                       ║                │ W_t              ║
                       ║                ├──→ VERIFIER ──→ score_t (Phase 3, train-only)
                       ║                │                  ║
                       ║   ┌────────────▼────────────────┐ ║ ← Phase 4
                       ║   │      HALT CRITIC            │ ║
                       ║   │ W_t → P(halt) → break?      │ ║
                       ║   └────────────┬────────────────┘ ║
                       ╚════════════════╪══════════════════╝
                                        │ W_K
                                        │
                          ┌─────────────▼─────────────┐
                          │    Workspace → Hidden     │
                          │    Project(W_K) ∈ (T, D)  │
                          └─────────────┬─────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │       LM Head (frozen)    │ ← transplanted
                          └─────────────┬─────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │ logits / sampling │
                              └───────────────────┘

At inference, Phase 5 wraps this whole thing in a K_rollout=8 loop
with different latent noise injections, and the Energy Head picks
the winner.
```

---

## 3. Per-component specs

### 3.1 Embedding + Outer Stack + LM Head — Phase 0, shipped

**Role**: knowledge storage. Transplanted from base, frozen.

**Build**:
- `embed_tokens`: copied verbatim from the HF base model
- 4 outer transformer blocks: the *last* 4 layers of base (most
  abstract representations live there)
- `lm_head`: copied verbatim

**Why freeze**: we are rebuilding *reasoning*, not knowledge. Knowledge
training was already done by someone with trillions of tokens. We
take the dictionary, then build a smarter consumer of that dictionary
on top.

---

### 3.2 Workspace Bank — Phase 1, this PR

**Role**: structured working memory. The *materials* of thought.

**State tensor**:
```python
W ∈ (B, T, S, D)
  B = batch
  T = sequence length (per-token reasoning)
  S = number of slots (default 16)
  D = hidden dim
```

Every token position carries its own per-token workspace. The `T`
dimension looks excessive at first; without it we lose token-level
locality (Token-i's reasoning state shouldn't be confused with
Token-j's). At training time we recover compute via sequence
compression or packed batches.

**Role embeddings**: `role_emb[i] ∈ R^D` is learned per slot. Slot
identity is *not* labelled by humans — it emerges from gradient. The
init `W_0[t, i] = init_proj(h_t) + role_emb[i]` guarantees that all
S slots start as distinct vectors so the symmetry can break in
finite training time.

**Diversity loss**: squared off-diagonal cosine similarity across
slots, weighted by `cfg.workspace_diversity_weight`. Prevents slot
collapse.

**On exact identity at init**: we deliberately *don't* zero-initialise
the workspace's output projection. That trick (sometimes called
"residual identity init") kills the gradient on every upstream
workspace projection from step 0 — a LoRA dead-branch trap. The
ablation guarantee CLAUDE.md asks for lives at the flag level:
`use_workspace=False` ⇒ module not present in the graph at all,
forward pass is bit-for-bit Phase 0. See comments in
`zeon/workspace.py` for the full rationale.

---

### 3.3 Operator Library — Phase 2

**Role**: the *toolbox* of thought. A learned instruction set.

**One operator**:
```python
class Operator(nn.Module):
    norm     : RMSNorm
    sub_attn : SmallAttention    # attention across workspace slots
    sub_ffn  : SwiGLU(D, d_inter_small)
    type_emb : Parameter(D,)     # operator identity embedding
```

**Library**: N = 32 (or 64) operators, each small. Total operator
params ≈ one outer-stack layer.

**Top-k routing** (k = 2–4):
```python
def Router(W, h_0):
    pooled = W.mean(dim=slots)              # (B, T, D)
    logits = pooled @ all_op_type_emb       # (B, T, N)
    return top_k(softmax(logits), k=2)
```

Only the selected operators forward — sparse compute.

---

### 3.4 Read / Write heads — Phases 1–2

**Read heads (M = 4)** — phase 1, shipped:
- Inputs: `(W_{t-1}, h_0, op_type)`
- One head = one learned query projection over (workspace slots + h)

**Write heads (M = 4)** — phase 1, shipped (in simplified form):
- Inputs: `(operator_output, W_{t-1})`
- Per-slot sticky-ness gate is learned
- Update rule:
  ```
  W_t[i] = gate_i * W_{t-1}[i] + (1 - gate_i) * update_i
  ```
- Sticky bias init = +2.0 → σ(2.0) ≈ 0.88 → default behaviour is
  "remember"; the model has to actively learn to release a slot.

---

### 3.5 Verifier Head — Phase 3

**Role**: per-step "is the workspace heading toward the right answer?"
A process reward model in latent space.

**Build**:
- Smaller transformer, ~1/4 to 1/10 the body
- Inputs: question `x` + current workspace `W_t`
- Output: `score_t ∈ [0, 1]`

**Training data**:
- Stage 3a: teacher solution traces + ground truth → supervised
- Stage 3b: body's high-confidence answers → self-training

**Use**:
- Train: `aux_loss = -mean(log score_t)` for trajectories that ended
  on a correct answer
- Inference: optional auxiliary signal for early exit

---

### 3.6 Halt Critic — Phase 4

**Role**: per-step decide "halt or continue", trained with RL.

**Build**:
```python
class HaltCritic(nn.Module):
    policy : SmallNet     # W_t → logit_halt
    value  : SmallNet     # W_t → V(state)
```

**Reward**:
- `+1` if final answer is correct, else `0`
- `−c * step_count`, c small (≈ 0.01)
- optional: `+α * verifier_score` as dense shaping reward

**Training**:
- Warm-up: supervised classifier ("halt iff verifier score ≥ τ")
- Main: REINFORCE → PPO
- PonderNet KL stays in as a baseline, weight annealed 1.0 → 0.5 → 0.1

---

### 3.7 Energy Head — Phase 5

**Role**: among K rollouts, which answer is the most trustworthy.

**Build**:
```python
class EnergyHead(nn.Module):
    encoder : SmallNet    # final workspace → energy logits
    score   : Linear(D, 1)
```

**Training (contrastive)**:
- Same input, K rollouts
- positive: final workspace from a correct-answer trajectory
- negative: final workspace from an incorrect-answer trajectory
- margin loss; hard-negative mine on "confidently wrong" cases

**Inference**:
- K = 8 parallel rollouts
- Output the answer from the rollout with the **lowest** energy score

---

## 4. Training objective (cumulative)

By Phase 5 the total loss looks like this:

```
L_total = L_lm                            # next-token CE                (Phase 0+)
        + λ_ponder    * L_ponder          # PonderNet KL                 (Phase 0+, annealed)
        + λ_distill   * L_distill         # base-model distillation      (Phase 0+)
        + λ_diversity * L_slot_diversity  # workspace anti-collapse      (Phase 1+)
        + λ_balance   * L_op_load_balance # operator load balance        (Phase 2+)
        + λ_verifier  * L_verifier        # per-step verifier signal     (Phase 3+)
        + λ_rl        * L_halt_rl         # halt critic RL               (Phase 4+)
        + λ_energy    * L_energy_contrast # energy head contrastive      (Phase 5+)
        + λ_compress  * L_self_distill    # latent compression           (Phase 6+)
```

Coefficients λ are introduced / removed by phase. We *never* try to
turn all of these on at once — that's how you get loss landscapes
that look like static.

### Data mix
- Pre-distillation: base model outputs on general corpora (FineWeb-edu
  etc.)
- Reasoning fine-tune: GSM8K, MATH, BBH, Open-Math-Instruct
- RL phase: problems with binary correctness labels (the source of
  the +1 reward)

---

## 5. Interface (HF compat is non-negotiable)

ZEON must **always** load via:
```python
from transformers import AutoModelForCausalLM
import zeon  # registers AutoConfig / AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("path/to/zeon-checkpoint")
out = model.generate(input_ids=..., max_new_tokens=128)
```

Internals can do anything; external API is identical. If this breaks,
the whole training/eval pipeline downstream breaks with it. End-of-
phase checks:

```bash
pytest tests/test_generate_and_cache.py
pytest tests/test_transplant.py
pytest tests/test_workspace.py
```

---

## 6. Directory layout (target)

```
zeon/
├── __init__.py             # AutoModel registration
├── config.py               # ZeonConfig (every phase's flags live here)
├── modeling_zeon.py        # ZeonForCausalLM (top-level)
├── workspace.py            # WorkspaceBank, read/write heads        (Phase 1, shipped)
├── operators.py            # OperatorLibrary, Router                 (Phase 2)
├── verifier.py             # VerifierHead                            (Phase 3)
├── halting.py              # PonderNet (P0) + HaltCritic (P4)
├── energy.py               # EnergyHead, multi-rollout               (Phase 5)
├── distill.py              # Self-distillation                        (Phase 6)
├── transplant.py           # base model → ZEON weight surgery
└── losses.py               # combined-loss assembly

scripts/
├── transplant_from_hf.py
├── train_phase{1,2,3,4,5,6}.py
└── eval_reasoning.py

tests/
├── test_recurrent_shape.py     # Phase 0
├── test_generate_and_cache.py  # Phase 0
├── test_transplant.py          # Phase 0
├── test_innovations.py         # Phase 0
├── test_workspace.py           # Phase 1, shipped
├── test_operators.py           # Phase 2
├── test_verifier.py            # Phase 3
└── ...

docs/
├── ROADMAP.md
├── ARCHITECTURE.md             # this file
└── PHASE_NOTES/                # per-phase retrospectives
    ├── phase1_workspace.md
    ├── phase2_operators.md
    └── ...
```

---

## 7. Principles we will not bend

1. **Every component has an ablation flag.** No flag, no merge.
2. **HF compat is non-negotiable.** If `AutoModelForCausalLM.from_pretrained()`
   breaks, the phase fails.
3. **Every phase ends with a benchmark number.** Vibes are not a metric.
4. **Retrospectives are written, not remembered.** Thought that lives
   only in heads dies within a month.
5. **Inference cost ≤ 5× base.** If we exceed that, just use a bigger
   base model — the LVM thesis stops paying for itself.

---

## 8. One line

> **ZEON is not a transformer. ZEON is a latent-space virtual machine
> that *uses* a transformer as its dictionary.**

Before adding code, ask:
> *"Is this one more thing bolted onto a transformer?"*

If yes → don't.
If no → ship it.
