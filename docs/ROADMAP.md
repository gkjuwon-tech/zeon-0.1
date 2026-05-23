# ZEON — 12-month chart

> **Destination**: a learned virtual machine that runs inside latent space.
> **Departure**: an HF-compatible transformer baseline.
> **Fuel**: a small pile of GPUs and a stubborn refusal to fork yet another LLaMA.
> **Captain**: human.
> **First mate**: an agent.

This document is the **plan**, not the README. Read it before opening
a feature PR, before arguing about an attention head, before adding
"just one more" anything. We've fooled ourselves enough times.

---

## Standing orders (read out loud, every phase)

1. **Seasoning a transformer ≠ innovation.**
   The instant your proposed fix is "add another attention" / "another
   gate" / "another embedding head", *redesign the phase*. Universal
   Transformer + PonderNet + Recurrent-Depth Transformer already exist.
   If we land inside that triangle we lost.

2. **You can't dial yourself out of a wrong design.**
   `hidden_size`, `num_layers`, `K_max`, `learning_rate` — these knobs
   do not fix a missing component. If your instinct is "let's just
   tune", check Standing Order 1 again.

3. **Ablation-impossible = not built.**
   Every component must have an `enable_X` flag in `ZeonConfig`. Flag
   off → score must measurably drop. If it doesn't drop, the component
   is decorative and we delete it.

4. **No benchmark = guess.**
   Every phase ends with at least one reasoning benchmark number,
   written into `docs/PHASE_NOTES/phase{N}_*.md`. No numbers, no
   "phase complete" tag. Vibes are not a metric.

5. **Failure is a measurement.**
   When phase 3 falls over we either build phase 3' or we delete it and
   move on. We do not spend a year defending our self-image. Read the
   *"if it falls over"* block at the end of every phase, every time
   we get stuck.

---

## Phases at a glance

```
M0    M1    M2    M3    M4    M5    M6    M7    M8    M9    M10   M11   M12
 │     │           │           │           │           │           │
[P0]──[P1: Workspace]──[P2: Operators]──[P3: Verifier]──[P4: RL Halt]──[P5: Rollouts]──[P6: Self-Distill]──[P7: Bench]
seasoning workbench       toolbox          inspector         time-cop      parallel-minds       recall            stage
```

Risk markers:  🔥 = "should be fine", 🔥🔥🔥 = "expect to bleed",
🔥🔥🔥🔥🔥 = "this is the phase that might kill the project".

---

## Phase 0 — Seasoning  ✓ shipped

**Codename**: seasoning
**Status**: shipped (M0, 1 week)
**Risk**: 🔥

### What landed
- HF-compatible base (`ZeonForCausalLM`, registered on `AutoModelForCausalLM`)
- KV cache + `.generate()` + `position_ids`
- Transplant from Llama / Qwen via `zeon.transplant`
- PonderNet halting (per-token sigmoid + KL prior + entropy bonus)
- `StepEmbedding` + `CrossStepMemory` (UT + RWKV-flavoured)
- 17 unit tests / `ruff` clean / CI workflow

### Honest self-assessment
Phase 0 is "deep transformer + light seasoning". It sits inside the
Universal Transformer (Dehghani '18) ∩ PonderNet (Banino '21) ∩
Recurrent Depth Transformer (Geiping '25) intersection on purpose.
**It is the baseline we have to beat.** We do not expect Phase 0 to be
smart; we expect it to *load*, *generate*, *transplant*, and survive
save/load round-trips.

### Why we still shipped it
1. Transplant infra is a prerequisite for every later phase.
2. HF-compat gives us training/eval scaffolding for free.
3. We need a *concrete* baseline to compare every later number against.

Phase 0 is **building materials**, not the building. The building
starts at Phase 1.

---

## Phase 1 — The Workbench  (in progress → landed in this PR)

**Codename**: workbench
**Duration**: M1–M3 (8 weeks budgeted)
**Depends on**: Phase 0
**Risk**: 🔥🔥🔥

### Why
Running one hidden vector through K recurrent steps is mathematically
an RNN with adaptive depth. Humans don't think with one number — we
keep multiple things in mind at once (hypothesis, evidence, scratch
buffer, conclusion-so-far) and update them at *different rates*.

So we need a **structured working memory**: not a single vector but a
*bank of slots*, each with a *role*, each with its own *update rate*
(some sticky, some volatile).

### What we built
```python
class WorkspaceBank(nn.Module):
    role_emb        : Parameter(S, D)       # learned per-slot identity
    init_proj       : Linear(D, D)          # h_t → slot init
    read_{q,k,v,o}  : Linear(D, D)          # multi-head intra-token attn
    write_in        : Linear(D, D)          # h → content
    write_mod       : Linear(D, D)          # read → content modulator
    write_gate      : Linear(D, D)          # gating logits per slot
    sticky_bias     : Parameter(S, D)       # +2.0 default ⇒ slots persist
    output_pool     : Linear(D, D)          # slot pool → projected h-residual
    output_proj     : Linear(D, D)
```

Per latent step, inside `ZeonBlock.forward`:
1. **Read.** Multi-head attention from `h_t` (one query/token) into the
   S slots of that same token. Intra-token only — no cross-token mix.
2. **Write.** Content = `write_in(h) + write_mod(read_vec)`, broadcast
   across slots, then gated with per-slot sticky bias.
3. **Contribute.** Pool over slots → silu → `output_proj` → residual
   addition to `h`. Workspace is *part of* `h`'s update, not a separate
   side-channel.

State carries across steps within one forward pass.

### Training recipe (when we get to it on real data)
- **Cross-entropy** on the LM head, as always.
- **Diversity loss** — squared off-diagonal cosine similarity across
  slots — multiplied by `cfg.workspace_diversity_weight`. Prevents
  slot collapse (the failure mode where all S slots learn the same
  vector and we've spent compute on nothing).
- **Distillation aux signal** (optional, P2 onward): teacher's late
  hidden state ≈ workspace "conclusion-ish" slot. The slot doesn't get
  a human label; the gradient picks which slot is "conclusion".

### Risks
| Failure | Symptom | Mitigation |
|---|---|---|
| Slot collapse | every slot = same vector | diversity loss, distinct role embeddings, possibly Slot-Attention-style competition |
| Sticky learning fails | all slots overwritten every step | sticky bias init = +2.0; gate sparsity bonus |
| Dead branch | gradient never reaches workspace | small (not zero) `output_proj` init; flag-level ablation instead of zero-init ablation (see `workspace.py` comments) |
| Param explosion | (B,T,S,D) blows VRAM | S=16 default; clear escape hatch to S=8 if needed |

### What "phase 1 done" looks like
- ✅ `use_workspace=False` ⇒ bit-for-bit identical to Phase 0 numerics
- ✅ Every workspace parameter receives a non-zero gradient
- ✅ `diversity_loss` passes a synthetic monotonicity test
- ✅ HF save/load round-trip preserves logits
- [ ] On a small GSM8K subset (warm-started from Qwen2.5-1.5B + light
      fine-tune), `use_workspace=True` shows **+2–5% absolute** over
      the matched `use_workspace=False` baseline
- [ ] Per-slot activation cluster visualisation shows visible
      separation (i.e. the slots are *doing something different*)

The last two checks land in `docs/PHASE_NOTES/phase1_workspace.md`
once we run them on actual hardware. Until then, what's in main is
"the wiring is correct, the gradient flows, the flag works".

### If it falls over
If slot structure refuses to emerge → introduce **slot competition**
(Locatello '20) so slots actively compete for input attention mass.
Still nothing → declare workspace a no-op and go straight to Phase 2;
Operators (P2) might force a natural division of labour on their own.

---

## Phase 2 — The Toolbox  (Operator Library)

**Codename**: toolbox
**Duration**: M3–M5 (8 weeks)
**Depends on**: Phase 1
**Risk**: 🔥🔥🔥🔥

### Why
Right now every thinking step runs the *same* FFN. That implicitly
claims "all reasoning has the same shape". It doesn't. `23 * 47 = ?`
and `why did this character lie?` are different ops. The model should
be able to pick which op to run, per step.

What we're building is, frankly, a **learned ISA**.

### What we'll build
```python
class OperatorLibrary(nn.Module):
    operators : nn.ModuleList[OperatorModule]   # N = 32–64
    router    : RouterHead                      # state → top-k op pick
    type_emb  : Parameter(N, D)                 # per-op meta-embedding
```

Each operator: a small FFN + small attention sub-block + small norm.
Inputs are workspace slots; outputs are workspace slots. Operators are
small *on purpose*; specialisation requires capacity scarcity.

Router: workspace state + op type embedding → top-k logits. Gumbel-
softmax during training, hard top-k at inference. k = 2 – 4.

### Training recipe
- **Load-balancing loss**, MoE-style (Shazeer '17, GShard).
- **Type distillation**: force operator group A to mimic the teacher's
  early layers, group B to mimic middle, group C to mimic late. This
  gives the router a *natural prior* to learn over.
- **Sparsity entropy**: keep router from collapsing to one expert or
  smearing across all of them.

### Risks
| Failure | Symptom | Mitigation |
|---|---|---|
| Routing collapse | router picks the same op forever | load balancing + Gumbel-softmax temperature schedule |
| Operator interference | one op handles contradictory tasks | top-k diversity bonus |
| Dead operators | some ops never called | force-rotate during warm-up |
| Inference blow-up | N ops all in VRAM at once | shared base + low-rank op-specific adapters |

### Success criteria
- Different problem types ⇒ visibly different operator-usage histograms
- `ablate(op_i)` breaks *only one class of problems* (specialisation)
- MATH small subset: **+3–7% absolute** over Phase 1

### If it falls over
If routing refuses to separate, label operators *explicitly* by task
type (math / logic / commonsense / …) and try supervised routing. If
that also fails, fall back to multi-task heads — Phase 2 thesis is
weakened, but the infra survives.

---

## Phase 3 — The Inspector  (Verifier Head)

**Codename**: inspector
**Duration**: M5–M7 (8 weeks)
**Depends on**: P1 + P2
**Risk**: 🔥🔥

### Why
Today the training signal is one CE loss at the *end* of K thinking
steps. Among those K steps, *we don't know which one helped*. Gradient
gets diluted across the recurrence.

Fix: a second, smaller model that scores every intermediate workspace.
A **process reward model** in latent space. It gives us dense,
step-resolved gradient.

### What we'll build
```python
class VerifierHead(nn.Module):
    encoder    : SmallTransformer    # ~1/4 the body size
    score_head : Linear(D, 1)        # 0 (away from answer) → 1 (toward)
```

Inputs: question context + current workspace + ground-truth (during
training) or teacher output (during self-supervision).
Outputs: per-step "are we getting closer?" score.

### Training recipe
**Stage 3a (supervised, 4 weeks)**
- Data: GSM8K, MATH (correct answer + solution trace).
- Forward the body on N varied inputs; cache per-step workspaces.
- Train the verifier on "is this workspace closer to a correct answer?"
- Trained *separately* from the body. No interference yet.

**Stage 3b (joint, 4 weeks)**
- Body's loss += `λ * verifier_step_score`. Schedule λ: 0 → 0.1 → 0.3.
- Verifier keeps fine-tuning against current body outputs.

### Risks
| Failure | Symptom | Mitigation |
|---|---|---|
| Verifier outsmarts body | verifier 100% / body 50% | cap verifier size, label smoothing |
| Verifier wrong | scores junk-state highly | ablation set, independent eval |
| Joint training diverges | wallpaper-pattern loss | warm-up, gradient clipping |

### Success criteria
- Verifier per-step score ↔ actual final correctness, **Pearson > 0.7**
- Removing verifier signal: same-step-count convergence becomes
  **~1.5× slower**
- ZEON+Verifier at same wall-clock: **+3–5%** over Phase 2

### If it falls over
Drop to **outcome-only reward** (+1 only on final correctness) and
fold into Phase 4 RL. Less efficient, but it works.

---

## Phase 4 — The Time Cop  (Halting Critic + RL)

**Codename**: time-cop
**Duration**: M7–M9 (8 weeks)
**Depends on**: P3
**Risk**: 🔥🔥🔥🔥🔥

### Why
PonderNet's KL-to-geometric prior is a *heuristic*. "Roughly halt like
a geometric distribution" is unrelated to the actual difficulty
distribution of the data we care about.

The real halting criterion is **"will one more thinking step raise my
probability of being right?"** — that's RL. Action = halt / continue.
Reward = correctness − step cost.

### What we'll build
```python
class HaltCritic(nn.Module):
    policy_head : SmallNet    # workspace → P(halt)
    value_head  : SmallNet    # workspace → V(state)
```

Reward shape:
- `+1` on correct answer
- `-c` per step (c small, e.g. 0.01)
- Verifier delta as auxiliary shaping reward

Training:
- **REINFORCE + baseline** (use the existing PonderNet KL as a
  baseline) to get going.
- Switch to **PPO** with the value head when stable.
- Anneal KL prior weight 1.0 → 0.5 → 0.1.

### Risks
| Failure | Symptom | Mitigation |
|---|---|---|
| RL variance explodes | training loss flapping | reward normalization, GAE |
| Halt collapses to step-1 | model panics out instantly | shrink step cost, entropy bonus |
| Halt collapses to step-K | model never halts | grow step cost |
| Cold start | early-training rewards mostly negative | warm-start from a supervised halt classifier trained on Phase 3 verifier |

### Success criteria
- Easy problem (arithmetic): mean step count **1–3**
- Hard problem (MATH level 5): mean step count **8–16**
- At matched mean step count, **+5% absolute** over PonderNet-only
- Step count ↔ problem difficulty, **Spearman > 0.6**

### If it falls over
Fall back to **expert iteration**: supervise a greedy halt policy on
verifier scores, regenerate data, repeat. Weaker than PPO but stable.

---

## Phase 5 — Parallel Minds  (Rollouts + Energy)

**Codename**: parallel-minds
**Duration**: M9–M11 (8 weeks)
**Depends on**: P4
**Risk**: 🔥🔥🔥

### Why
One latent trajectory → if it's wrong, you're done. Humans go "hmm,
let me think about it differently" and retry. Self-Consistency (Wang
'23) does this in the token space (N samples, majority vote).

ZEON does it in the **latent** space — K different noise seeds → K
different latent trajectories. Output tokens may coincide, but the
*reasoning* is different.

### What we'll build
```python
class EnergyHead(nn.Module):
    encoder : SmallNet      # final workspace → trust score
    score   : Linear(D, 1)
```

At inference:
1. Run K = 8 rollouts on the same input with different latent noise.
2. Score each rollout's final workspace via Energy Head.
3. Pick lowest-energy (= highest-trust) answer.

Training: contrastive on (workspace_correct_traj) vs
(workspace_incorrect_traj). Hard-negative-mine on cases where the
model is *confidently wrong*.

### Risks
| Failure | Symptom | Mitigation |
|---|---|---|
| Rollout diversity 0 | K runs give 1 answer | forced latent noise injection |
| Energy uncorrelated with correctness | Pearson ≈ 0 | better contrastive design, temperature tuning |
| K× inference cost | 8× slower at K=8 | parallel rollouts + early-exit on dominant vote |

### Success criteria
- K=8 vs K=1: **+8–12% pass@1**
- Energy ↔ correctness, **Pearson > 0.6** (calibrated confidence)
- Wall-clock K=8 / K=1 < **3×** (parallelism win)

### If it falls over
Fall back to **token-space majority vote** (= plain Self-Consistency).
Less distinctive but proven.

---

## Phase 6 — Recall  (Self-Distillation in Latent Space)

**Codename**: recall
**Duration**: M11–M12 (4 weeks)
**Depends on**: P5
**Risk**: 🔥🔥

### Why
Humans start by computing 9 + 9 + 9 + 9 + 9, and end by *just knowing*
9 × 5 = 45. Compression of thought. A reasoning system that never
compresses is just memorising harder.

ZEON should do the same: a frequently-seen pattern should require
fewer thinking steps over time.

### What we'll build
Two modes of the **same** model:
- **Teacher mode**: `K_max = 16`, takes its time
- **Student mode**: `K_max = 4`, fast path

Training:
- Same input through both modes simultaneously
- KL distil student logits ← teacher logits
- MSE on final-workspace between student and avg of teacher's last 4
- Student keeps the right to *grow K* via the Halt Critic on hard
  inputs — incompressible problems get more compute automatically.

### Risks
| Failure | Symptom | Mitigation |
|---|---|---|
| Student trails teacher | -10% accuracy | tune distil weight, anneal K slowly |
| Student always hits K_max | compression failed | tune step cost |

### Success criteria
- Easy problems: student (K=4) ≈ teacher (K=16) on accuracy
- Hard problems: student auto-grows K via Halt Critic
- Mean inference cost: **−40 to −60%** vs teacher
- Accuracy loss: **−2% or less**

### If it falls over
Ship teacher only; accept the inference cost.

---

## Phase 7 — Stage  (Benchmarks + release)

**Codename**: stage
**Duration**: M12 (4 weeks)
**Depends on**: whichever of P1–P6 survived
**Risk**: 🔥

### What we measure
**Reasoning benchmarks**:
- GSM8K (grade-school math)
- MATH (competition math)
- BBH (BIG-Bench Hard, 23 hard tasks)
- ARC-AGI-1 / ARC-AGI-2 (visual abstraction)
- MMLU-Pro (academic knowledge + reasoning)
- HumanEval / MBPP (code)

**Against**:
- The same base model, vanilla inference
- The same base model + CoT
- The same base model + Self-Consistency (majority vote)
- Other reasoning systems (DeepSeek-R1 et al), parameter-matched

**Controls**:
- Same average FLOPs
- Same number of output tokens (latent reasoning vs token CoT)
- Same average wall-clock

### Deliverables
- Eval report (quant + qual; failure cases included on purpose)
- Verified checkpoint, pushed to HF Hub
- Training code + a reproduction recipe
- A paper, *iff* the numbers are real
- A public post-mortem, iff the numbers are not

### Concrete targets (these aren't aspirations, they're tripwires)
1. **GSM8K** — beat base+Self-Consistency by **+5% absolute** with
   **3× fewer output tokens**.
2. **MATH** — beat base+CoT by **+10% absolute** with **5× fewer
   output tokens**.
3. **BBH** — beat base+CoT by **+5% absolute on average**, with at
   least one reasoning-heavy subtask at **+15%**.
4. **ARC-AGI-1** — beat base+CoT by **+8% absolute**. (If we don't
   win *here*, the latent-VM thesis is in trouble.)

Hit fewer than 2/4 → we say so, in print, and the next ZEON-shaped
project starts somewhere smarter.

---

## End-of-phase ritual

```
[ ] Component's enable flag works
[ ] Flag-off measurably hurts score (otherwise: we didn't build it)
[ ] One reasoning benchmark number recorded
[ ] ≥ 5 failure cases analysed by hand
[ ] Tech-debt sweep before next phase
[ ] PR + retrospective in docs/PHASE_NOTES/ + next-phase decision
```

The retrospective is **mandatory and written**. Insight that lives
only in our heads dies within a month.

---

## Reality check

This chart is **a map drawn on parchment**. Storms happen. Plausible
shifts:

- Phase 1 closes in 4 weeks not 8 → we get a month back.
- Phase 2 (Operators) refuses to train → +2 months, P3 compresses.
- Phase 4 (RL) eats a whole quarter → fold P5 and P6 together.
- OpenAI / Anthropic / DeepSeek publishes "ZEON, by us, with more
  GPUs" mid-project → reposition (specialise on a niche we still own),
  do *not* quit.

We do a **1-week plan review** at the gate of every phase. The chart
is alive.

---

## One-line summary

> **We are not adding more seasoning. We are building a small virtual
> machine in latent space.**

Before every commit, ask:
> *"Is this just one more thing bolted onto a transformer?"*

If yes → redesign the phase. If no → ship it.
