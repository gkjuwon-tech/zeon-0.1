# Phase 1 — Workbench retrospective

> Status: **shipped to main, not yet validated.**
> Code lands in this PR. Benchmark validation is the *next* thing.
> Per CLAUDE.md, this file is mandatory before Phase 2 entry.

## TL;DR

We landed the WorkspaceBank infrastructure (Phase 1 of 7).
`ZeonConfig.use_workspace` defaults to `True` and the bank is exercised
by every recurrent step. On a smoke level (unit tests, save/load,
gradient flow) it works. **The required GSM8K +2–5% over Phase 0
hasn't been measured yet** — that lives in this same retro, to be
filled in when we run it.

## What we built

`zeon/workspace.py` — `WorkspaceBank`:
- Per-token state tensor `W ∈ (B, T, S, D)`, default `S=16`
- Learned per-slot `role_emb` so slots start asymmetric (without it
  gradient can't break the symmetry in finite training time)
- `init_proj(h_t) + role_emb` builds `W_0` from the outer-stack hidden
  state
- Multi-head intra-token cross-attention (`read_{q,k,v,o}`); reads
  scale `O(B*T*S*D)`, no cross-token mixing
- Content write proposal = `write_in(h) + write_mod(read_vec)`,
  broadcast across slots
- Per-slot sticky gate: `sigmoid(write_gate(W) + sticky_bias)`, with
  `sticky_bias` init = +2.0 (default sticky)
- Aggregation back to `h`: `output_proj(silu(output_pool(W).mean(dim=2)))`
- `diversity_loss(W)` = squared off-diagonal cosine similarity

Integration in `zeon/modeling_zeon.py`:
- `ZeonBlock.__init__` builds the workspace iff `cfg.use_workspace`,
  else `self.workspace = None`
- `ZeonBlock.forward`:
  - `W_0 = self.workspace.init_state(h)` if enabled, else nothing
  - Each recurrent step: `current, W = self.workspace.step(current, W)`
  - During training, accumulate `diversity_loss(W)` per step and add
    `(div_acc / steps) * cfg.workspace_diversity_weight` to ponder_loss
- HF compat: `WorkspaceBank` is registered in `_no_split_modules` for
  HF's auto sharding; save/load round-trips via `AutoModelForCausalLM`

Config (`zeon/config.py`):
- `use_workspace: bool = True`
- `workspace_num_slots: int = 16`
- `workspace_num_heads: int = 4`
- `workspace_diversity_weight: float = 1e-3`
- `workspace_sticky_bias_init: float = 2.0`

Tests (`tests/test_workspace.py`, 9 new):
1. `test_workspace_contribution_is_bounded_at_init` — relative
   perturbation < 0.5 when flag is on
2. `test_workspace_increases_param_count` — flag on adds params
3. `test_workspace_more_slots_means_more_params`
4. `test_every_workspace_param_receives_gradient` — no dead branches
5. `test_workspace_save_load_roundtrip` — HF Auto* compat
6. `test_role_embeddings_break_slot_symmetry`
7. `test_diversity_loss_higher_when_slots_collapse`
8. `test_disabling_workspace_removes_module` — flag-level ablation
   really removes the params
9. `test_workspace_diversity_is_in_the_graph_when_training`

Total tests at the time of this writeup: **26 passing**.

## Design choices that surprised us

**The "identity at init" pattern is a trap.** Our first cut had
`output_proj.weight` zero-initialised — the textbook residual-gate
trick. This gives bit-identical numerics at init (`use_workspace=True`
behaves exactly like `use_workspace=False` until trained). But it also
zeros every upstream workspace projection's gradient (`∂L/∂output_pool`
is multiplied by the zero `output_proj`, which is zero), so the
workspace is a **dead branch** until some other path lifts the
projection off zero. That other path doesn't exist. We caught this
because the gradient-flow test failed.

Replaced with: standard small-normal init on `output_proj`. The
workspace contributes a *small* perturbation at init (test 1: relative
norm < 0.5, in practice 0.05–0.15 for the test config), and every
workspace parameter gets a real gradient from step 1. The
ablation guarantee CLAUDE.md asks for is now provided by the *flag*
itself: `use_workspace=False` ⇒ module literally not in the graph.
This is also a better honest definition of "ablation".

Side effect: we discovered the existing `CrossStepMemory` was making
the same "identity at init" claim in comments, but `PreTrainedModel.
post_init` was overwriting its zero inits with normal_ anyway, so the
claim was already broken in main. Not fixing that in this PR — its
gradient tests pass for the same accidental reason the workspace's
naive version *almost* did.

**Workspace must live in the `(B, T, S, D)` tensor**, not `(B, S, D)`.
A first instinct is "one workspace per sample" because that's what
human working memory looks like. But losing the `T` dimension throws
away per-token locality — token `i`'s reasoning state shouldn't get
mixed with token `j`'s before attention even runs. Cost is real
(16× h memory when `S=16, D=hidden`), but for the small test configs
it's < 1MB.

**Sticky bias init = +2.0.** σ(2.0) ≈ 0.88 means at init each slot
keeps ~88% of its previous content. We default to "remember"; the
network has to actively learn to release a slot. The alternative
(`+0.0`, no prior) gave volatile slots in early experiments and didn't
learn structure. (Smoke-level observation; will revisit when we have
real training runs.)

## Open / unresolved

- **GSM8K small-subset score not yet measured.** This is the actual
  validation. Until we have it, this phase is "in code, not yet
  validated". This needs a small Qwen2.5-1.5B transplant + ~1 GPU-day
  of warm-up + GSM8K-200 eval. Tracking issue: TBD.
- **Slot interpretability not yet measured.** ROADMAP success criterion
  asks for "visible clustering of per-slot activation patterns". Needs
  the same training run + a visualization script. Will land in this
  retro when done.
- **Failure-case analysis not yet done.** ROADMAP requires 5 manual
  failure cases. Will land here after validation.

## What we'd do differently if we restarted

- Write the gradient-flow test *first*, then the design. We'd have
  caught the "identity at init kills gradients" mistake in 5 minutes
  instead of after the first commit.
- Don't claim "identity at init" in component docstrings unless you
  also *prove it inside the full model*. Component-isolated tests
  don't catch what `post_init` does.

## Next phase decision

Once GSM8K validation is in:

- **If +2–5% over Phase 0 on GSM8K-200**: proceed to Phase 2
  (operator library) on the ROADMAP schedule.
- **If 0% to +2%**: re-examine slot collapse / sticky bias /
  diversity weight before declaring P1 done. Likely 2–3 weeks extra.
- **If negative**: read the *"if it falls over"* block in the
  ROADMAP Phase 1 section. Probably introduce slot competition
  (Locatello '20) and re-evaluate.
