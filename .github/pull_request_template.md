<!--
  Please fill out this template. Drive-by PRs that delete it will be
  closed with a link back to here.

  Short version of the rules:
  - Architecture changes need an ablation flag and at least one
    benchmark number (no matter how small). See CONTRIBUTING.md §1.
  - Docs / tests / CI / typo PRs do NOT need a benchmark — just say
    "docs only" in the Phase field below.
-->

### Phase

<!-- One of: phase-0 / phase-1 / phase-2 / ... / infra / docs / chore -->

### What this does

<!-- 2–4 sentences. Imagine the reviewer is in a hurry. -->

### Why this is not seasoning

<!--
  For architecture PRs only.
  If your change is a new module / new training objective / new
  inference trick, answer: *why is this not "transformer + one more
  attention head"?* If it is — redesign before opening the PR.
-->

### Ablation

<!--
  For architecture PRs only.
  - Which `ZeonConfig` flag toggles this off?
  - What happens to the relevant metric when the flag is off?
  - Confirm: the flag-off path must produce bit-for-bit identical
    numerics to the previous-phase baseline.
-->

### Benchmark

<!--
  For architecture PRs only.
  At minimum one number. Even a 200-example subset is acceptable to
  unblock review; full benchmarks live in `docs/PHASE_NOTES/`.
-->

### Tests

<!-- What you added / changed. Output of `pytest tests/ -q`. -->

### Risks / what could break

<!--
  The most important section. What's the failure mode if CI is green
  but the model gets worse? What would you look at first?
-->

### Checklist

- [ ] `pytest tests/ -q` is green locally
- [ ] `ruff check zeon scripts tests` is clean
- [ ] HF compat still works (`AutoModelForCausalLM.from_pretrained` round-trip)
- [ ] If a new component: ablation flag added; flag-off identical to baseline
- [ ] If a new component: at least one gradient-flow test (no dead branches)
- [ ] If a new component: `zeon/transplant.py` updated (or PR explains why N/A)
- [ ] `docs/ARCHITECTURE.md` updated if the diagram changed
- [ ] `docs/PHASE_NOTES/phase{N}_*.md` updated if this closes a phase
