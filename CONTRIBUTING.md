# Contributing to ZEON

We want collaborators. We want collaborators who *push back* on the
design. Pull requests that look like ML résumé-bait — "let's add one
more head" / "what if we used GELU instead of SiLU" — will get a
polite no. Pull requests that argue with the roadmap, propose a new
ablation, kill a dead branch, or land a benchmark number we didn't
have before will get a fast yes.

This document is the operating manual: how to set things up, how to
think about a PR before you write it, what the review bar is.

---

## 0. Before you write any code

Read these, in order. Each one is short:

1. [`README.md`](README.md) — what we're building (and what we're not)
2. [`docs/ROADMAP.md`](docs/ROADMAP.md) — the 12-month phase plan,
   with explicit "if this fails …" exits for every phase
3. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the target system
   at year end
4. [`CLAUDE.md`](CLAUDE.md) — the five standing orders and the
   three-question self-check we use *on ourselves*

You don't need to agree with everything. You do need to know what
the priors are, because that's the rubric your PR will be reviewed
against.

---

## 1. The three questions (we ask before *every* feature)

Apply these to whatever you want to ship. If you can't answer any
of them, the PR isn't ready yet — open an issue instead.

1. **"Is this just one more thing bolted onto a transformer?"**
   If yes → the design needs to change before code is written.
   "Yet another small attention head" is not a feature.

2. **"If I turn this off, does score drop?"**
   If you don't know → add the ablation flag *first*, then write
   the feature, then prove the score drops without it. No proof,
   no merge.

3. **"Which benchmark / which metric measures this?"**
   If you can't name one → pick one before the module exists. This
   doesn't have to be Tens-of-GPU expensive; a 200-example subset
   on a single GPU is enough to gate the merge.

This rubric applies to architecture PRs. It does *not* apply to
docs / tests / CI / typo fixes — go ahead, those don't need a
benchmark.

---

## 2. Dev setup

```bash
git clone https://github.com/gkjuwon-tech/zeon-0.1.git
cd zeon-0.1
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Sanity:
pytest tests/ -q          # 26 tests at time of writing
ruff check zeon scripts tests
```

If `pytest -q` is red on a clean checkout, **that's the first issue**
to file. Don't paper over it.

CPU-only is fine for development and the included unit tests. Real
training / eval needs a GPU; we use HF transformers and PyTorch so
any normal CUDA / ROCm setup should work.

---

## 3. Branches and commits

**Branch names**: `<author>/<short-topic>` is fine. We don't
prescribe a deep taxonomy. Some good examples:

```
park/phase1-workspace-tests
alice/fix-rope-meta-init
bob/op-router-load-balance
```

**Commit messages**: no template, but two unwritten rules:

* The first line should let a hurried reviewer understand the change
  without opening the diff. "fix bug" doesn't.
* Imperative tense. `add diversity loss` not `added diversity loss`.

**Squashing**: we usually squash on merge so your branch history
doesn't have to be pristine — that said, a clean history makes
review faster.

---

## 4. PRs

Open one with [`fetch_pr_template`-friendly markdown that fills out
the template](.github/pull_request_template.md). The template is
short on purpose. The important fields:

- **Phase**: which roadmap phase does this touch? (Use `infra` /
  `docs` / `chore` if none.)
- **What this does**: 2–4 sentences.
- **Ablation**: if you added or changed a component, *which flag
  toggles it*? What does the score look like with the flag off?
- **Benchmark number**: for any non-trivial architecture change,
  at least one number. A small one is fine. No number = block.
- **Risks**: the one paragraph that matters most. What could break.
  Where you'd look first if CI is green but the model gets worse.

Drive-by PRs that don't fill the template will be closed with a
link back to the template.

### Size

There is no hard line limit, but:

- **< 400 lines diff**: usually merges fast
- **400–1000**: merges, with extra review
- **> 1000**: split it. Almost always possible.

The Phase 1 PR is itself ~1.6k lines because it's the entire
WorkspaceBank + tests + docs rewrite + governance, and that's the
exception not the rule.

### CI

Three things will run:
1. `pytest tests/ -q` — the unit suite. Must be green.
2. `ruff check zeon scripts tests` — must be clean.
3. (Optional) the smoke transplant from a small base model. Slow,
   only kicks in on label `wants-transplant-smoke`.

A red CI is a red merge button. If CI is red on `main`, that's a
*separate* issue that gets fixed first; don't sneak past it.

---

## 5. Code style (the unwritten parts)

We try to write code that the *next contributor* can read on a Friday
afternoon. Things we care about:

- **Comments explain *why*, not *what*.** The `for` loop knows what
  it's doing. The reader wants to know why it exists.
- **Avoid one-letter variables**, except `i, j, k, t, x` and standard
  ML names (`B, T, D, S, H, L`). `q, k, v` and `pos_ids` are fine.
- **Names match the math when there is math.** `W` (workspace) is
  `(B, T, S, D)` because the spec says so. Don't rename it to
  `workspace_state_tensor`.
- **Type hints on public functions.** Optional on private helpers
  if the body is short enough that the types are obvious.
- **No `Any`.** If you reach for `Any`, your types aren't right.
- **No `getattr` games on objects you own.** If you're doing that to
  poke at a `nn.Module`, refactor the API.
- **No `from x import *`.**
- **Use `dataclasses` or named tuples** for small structs in tests /
  scripts. We use HF-style `Config` classes for model config because
  that's required by HF.

`ruff` enforces a subset of the above mechanically. The rest is taste,
which is what code review is for.

---

## 6. Adding a new component

If your PR adds a new `nn.Module` that participates in the recurrent
loop, the checklist is:

- [ ] Module lives in its own file in `zeon/` (or extends an existing
      one for *very* small additions)
- [ ] `ZeonConfig` gains an `enable_<feature>` flag, default chosen
      deliberately (and called out in the PR description)
- [ ] `ZeonBlock` reads the flag and either instantiates the module
      or sets it to `None`; the `None` path must produce bit-for-bit
      Phase 0 numerics
- [ ] At least one test asserts the flag-off path matches the
      flag-off-by-removing-the-attribute path (i.e. the flag is real,
      not decorative)
- [ ] At least one test exercises the new module's forward pass with
      realistic shapes
- [ ] At least one test asserts every learnable parameter receives a
      non-zero gradient on a backward pass (no dead branches)
- [ ] Save / load round-trip via `AutoModelForCausalLM.from_pretrained`
      preserves logits
- [ ] `zeon/transplant.py` updated (even if the update is "this
      component does not transplant, here's why")
- [ ] If diversity / load-balance / regularization losses are added,
      they're routed through `zeon/losses.py`
- [ ] `docs/ARCHITECTURE.md` updated with the new spec
- [ ] `docs/PHASE_NOTES/phase{N}_*.md` retrospective started

That's a lot. It's all small individually, and skipping any of it
*will* be flagged in review.

---

## 7. Research-shaped contributions

For changes that are research, not engineering — a new training
objective, a new architectural component, a new evaluation methodology
— open a **`research proposal` issue first**, using the issue template.
We argue about the design *before* the code exists. That saves
everyone time.

A good research proposal:

- One paragraph on what you're proposing
- One paragraph on *why this is not seasoning* (specifically: which
  of the "Universal Transformer / PonderNet / Recurrent-Depth"
  failure modes does it avoid?)
- A pointer to a paper / blog / preprint, if you have one
- A back-of-envelope on what failure mode could kill it

You don't have to be right. You just have to be falsifiable.

---

## 8. Disagreement is welcome. Disrespect is not.

We have strong design opinions. We are also occasionally wrong about
them. If you have a substantive disagreement, *bring it*. Sharp
technical criticism is welcomed and rewarded.

[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) covers behavior; the very
short version is: argue with the *idea*, not the *person*.

---

## 9. Help, who do I ask?

- Architecture / design questions: open an issue tagged `question`.
- Bug reports: open an issue with the `bug` template; include the
  exact `pytest` invocation that reproduces it.
- "Is this a good first issue for me": look for the
  [`good first issue`](https://github.com/gkjuwon-tech/zeon-0.1/issues?q=is%3Aissue+label%3A%22good+first+issue%22)
  label.
- "Is X already being worked on": search issues; if not, claim it
  in a comment.

---

## 10. The vibe

We are trying to do something hard, that might not work, with not a
lot of GPUs. We are pretty sure most of the value will come from
honest negative results — ablations that show what *doesn't* work,
written up in the open. If you're here to put another shipped
transformer on your résumé, this is the wrong place. If you're here
to argue about whether the LVM thesis can be made real, welcome.

Don't be the person adding one more spoon of seasoning.

Be the person building a virtual machine in latent space.
