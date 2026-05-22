# AGENTS.md → see CLAUDE.md

This repo's canonical agent-rules document is [`CLAUDE.md`](CLAUDE.md)
(historical reasons — that's what we started with, and Claude Code
still looks for that filename).

`AGENTS.md` exists so other agent tooling (Codex, Aider, Cursor,
Devin) that searches for it doesn't get confused. The actual rules
live there. **Open `CLAUDE.md` and read it before doing anything in
this repository.**

Short version of what's inside:

1. ZEON is not "transformer with seasoning". It's a latent-VM that
   *uses* a transformer as its dictionary.
2. Every component needs an ablation flag. If turning it off doesn't
   hurt the score, you didn't build it.
3. Every phase ends with a benchmark number, written to
   `docs/PHASE_NOTES/`. No numbers, no "phase complete".
4. HF compat (`AutoModelForCausalLM.from_pretrained`) is
   non-negotiable.
5. If you're tuning `hidden_size` to fix a design problem, you're not
   fixing anything. Change the structure.

Stop reading this file. Open `CLAUDE.md`.
