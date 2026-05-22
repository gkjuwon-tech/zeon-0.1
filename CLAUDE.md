# ZEON — Project Memory

> **Agent: read this file first, every session, before any other work.**
> Yes, even if you've read it before. Habits drift; memory doesn't.

This is the in-repo memory document for AI coding agents (Claude
Code, Devin, Cursor, Aider, whatever). It's the agent-facing twin of
`README.md`. If you're a human, this is *also* useful — it's the
checklist we use on ourselves before opening a PR.

There is an identical-content `AGENTS.md` for tools that look for
that filename. Both files redirect to the same rules.

---

## 0. One-line identity

**ZEON is not a project that adds seasoning to a transformer.**
**ZEON is a project that builds a Latent Virtual Machine which *uses*
the transformer as its dictionary.**

If you find yourself confused about which one we're doing, *stop and
re-read this sentence*. Then go.

---

## 1. The five standing orders (read before every commit)

1. **Seasoning a transformer is not innovation.**
   The instant your proposed fix is "add another attention head" /
   "another gate" / "another small embedding head", *redesign the
   work*. Universal Transformer + PonderNet + Recurrent-Depth
   Transformer already exists; sitting inside that triangle is a
   defeat.

2. **You cannot dial yourself out of a wrong design.**
   Adjusting `hidden_size`, `num_layers`, `K_max`, `learning_rate`, or
   weights is *not progress*. It's seasoning mode trying to come back
   in. Change the *structure*.

3. **Ablation-impossible = not built.**
   Every component needs a toggle flag. Flag off ⇒ score must drop.
   If it doesn't drop, the component is decorative — delete it.

4. **No benchmark = guess.**
   Every phase ends with at least one reasoning benchmark number.
   Opinion-based "phase complete" declarations are forbidden.

5. **Failure is a measurement, not an embarrassment.**
   When Phase 3 falls over, build Phase 3' or skip to Phase 4. Don't
   spend a year defending your self-image. The `if it falls over`
   blocks in `docs/ROADMAP.md` exist for a reason — re-read them when
   stuck.

---

## 2. The three-question self-check (run before any code)

1. **"Is this just one more thing bolted onto a transformer?"**
   → Yes → *redesign the work first*. Don't write code yet.

2. **"If I turn this off, does the score drop?"**
   → Don't know → *put in the ablation flag before anything else*.

3. **"Can I measure this? Which benchmark, which metric?"**
   → No answer → *pick the benchmark before writing the module*.

Code that fails any of the three above does not get committed.

---

## 3. Where we are / where we're going

**Now**: Phase 1 (workbench) is *in code*. With this PR, `WorkspaceBank`
exists, all flags work, 26 tests green. **Phase 1 is not yet
*validated*** — the benchmark sub-tasks (GSM8K small-subset
+2–5% over Phase 0) still need to run on real hardware. That's not
"phase done"; that's "phase ready to evaluate".

**Next**: validate Phase 1 → write the retrospective into
`docs/PHASE_NOTES/phase1_workspace.md` → only then enter Phase 2
(operators).

Full plan: [`docs/ROADMAP.md`](docs/ROADMAP.md).
Target system: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Per-phase retros: [`docs/PHASE_NOTES/`](docs/PHASE_NOTES/).

End-of-phase ritual (no shortcuts):
- [ ] Component's enable flag works
- [ ] Flag-off measurably hurts score (otherwise: didn't build it)
- [ ] One reasoning benchmark number recorded
- [ ] ≥ 5 failure cases analyzed by hand
- [ ] PR + retrospective + next-phase decision

---

## 4. Technical non-negotiables

For every change, all of these must remain true:

- **HF compat**: `AutoModelForCausalLM.from_pretrained()` must work.
  If it breaks, the phase fails.
- **Tests green**: `pytest tests/ -q` always green; `ruff` always
  clean. No "I'll fix the test later" commits.
- **Transplantable**: when adding a new component, update
  `zeon/transplant.py` accordingly. If it cannot be migrated from a
  base model, redesign it.
- **Inference cost ≤ 5× base.** If we cross that, the whole LVM thesis
  stops paying for itself and we should just use a bigger model.
- **Context length assumption: 16k.** We are not chasing 2M context.
  The bet is "short context + deep latent thinking".

---

## 5. Documentation obligations

- At the end of every phase, write a retrospective at
  `docs/PHASE_NOTES/phase{N}_*.md`. Contents: what we built,
  measured numbers, ≥ 5 failure cases, next-phase decision.
- Insight kept only in heads dies within a month. **Write it down.**
- No retro = no entry to the next phase.

---

## 6. Reporting style (human preference)

When messaging the human collaborator at runtime:

- Korean is fine; *keep the tension*. Short, direct.
- No vague abstractions ("optimization", "improvement"). Speak in
  numbers.
- Frequent short progress updates. No "let me batch up the report".
- Failures / blockers / suspicions: surface *immediately*. Don't bury.

(This applies to chat messages. Code, docs, and PRs are in English.)

---

## 7. Documents map

- `README.md` — public face of the project. Short, sharp.
- `docs/ROADMAP.md` — 12-month plan. When stuck, re-read the "if it
  falls over" section of the current phase.
- `docs/ARCHITECTURE.md` — target architecture at year end. Update
  every time a phase closes.
- `docs/PHASE_NOTES/` — per-phase retrospectives. **Mandatory** before
  entering the next phase.
- `CONTRIBUTING.md` — branch / PR / review process for human contributors.
- `CODE_OF_CONDUCT.md` — community rules.

---

## 8. The last line

> **Don't be the person adding one more spoon of seasoning.**
> **Be the person building a virtual machine in latent space.**

Re-read that sentence once before writing code today.

Go.
