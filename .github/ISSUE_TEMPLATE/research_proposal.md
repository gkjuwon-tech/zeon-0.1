---
name: Research proposal
about: Propose a new component, training objective, or eval methodology.
title: "[research] "
labels: research, needs-discussion
assignees: ''
---

<!--
  Before writing code for a non-trivial architectural change, open
  one of these. We argue about the design first. This saves
  everyone, including you, a *lot* of time.

  Required reading before filling this out:
   - README.md
   - docs/ROADMAP.md
   - docs/ARCHITECTURE.md
   - CLAUDE.md (the three-question self-check is the rubric)

  You don't have to be right. You just have to be falsifiable.
-->

### One-paragraph pitch

<!-- What are you proposing, in plain English. -->

### Why this is not seasoning

<!--
  Required.

  Specifically: which of {Universal Transformer, PonderNet,
  Recurrent-Depth Transformer, MoE, RWKV, Mamba, Self-Consistency}
  does this *not* reduce to? Why?

  If you can't answer this clearly, the proposal is probably not
  ready yet. That's OK — open it anyway, mark it as `early-draft`,
  and we'll help you sharpen it.
-->

### How it fits the existing pipeline

<!--
  Which phase does this belong to? Which module does it live in?
  Which existing flag would gate it (or what new flag would you add)?
-->

### Falsifiable predictions

<!--
  What would make this *fail*? Concretely.

  Examples:
  - "If load-balance loss collapses to a single expert in the first
    5k steps, this is broken."
  - "If verifier ↔ correctness Pearson < 0.4 after Stage 3a, the
    process reward thesis doesn't hold and we drop to outcome-only."

  No falsifiable prediction = vibes. Vibes are not enough.
-->

### Prior art

<!--
  Papers, blog posts, preprints. Even informal "I saw someone on
  Twitter try this" is useful — link the tweet.
-->

### Cost estimate

<!--
  Rough back-of-envelope.

  - Compute: GPU-hours to validate?
  - Engineering: weeks of work?
  - Risk: 🔥 ... 🔥🔥🔥🔥🔥
-->

### What you'd like from this issue

<!--
  - "Just a sanity check on the design"
  - "Looking for someone to pair on the implementation"
  - "Want to know if this is already in someone's queue"
  - other …
-->
