# Overview

This project studies myth not as a static text but as a **reading process**, and measures how differently human and LLM cognition performs that reading. When several annotators — a human and frontier LLMs — segment the same fixed narrative into functional states and causal transitions, they disagree. **The disagreement is the object of study.** The pipeline decomposes it into text-driven variance (the passage genuinely admits several readings), cognition-driven variance (a reader is unstable where the text is in fact fixed), and artifact (a damaged or indeterminate passage no reader can resolve).

## What is analyzed

Three Sumerian corpora, each analyzed independently (different editions, different lineation, never cross-compared):

- *Inana and Enki* — ETCSL 1.3.1 (block-level)
- *Inanna's Descent* — CDLI composite P468903 (line-level)
- *Gudea Cylinder A* — RIME 3/1.01.07, CDLI P431881 (line-level)

Each corpus is read by three model families — Claude Opus, Gemini Pro, and Qwen (thinking mode) — at **10 fresh-session runs each**, plus a human annotator at **2 spaced runs** (a blind 3rd is the project's top priority). Every reader segments the text and assigns each segment a **function** (the *state* — what is happening) and a **transition_to** (the *direction* — where it leads).

## The central reframing: state vs. direction

Earlier versions measured only the state layer (function) via transition matrices and entropy. The current core finding is that **state and direction behave differently**: cognitions converge on *what* happens and diverge on *where it leads*. Direction is freer than state in every corpus, and the human diverges from the models most in direction. Direction is the layer carrying the signal, and the sharpest instrument — **conditional transition agreement** (agreement on direction *restricted to lines where two readers already agree on state*) — isolates it with state held equal.

## Why not exact-match comparison

Sources segment at very different resolutions (Qwen cuts finest, Gemini coarsest), so index-wise comparison (segment 1 vs segment 1) is meaningless. Everything is **projected to lines**: a coarse and a fine reading of the same span contribute the same per-line labels, so the label/agreement/freedom metrics are invariant to how finely each source sliced.

## Metric families (see `04_how_to_read_results.md`)

- **Cross-source freedom map** (+ JSD decomposition) — how much a line's disagreement is genuine text-openness vs. shared wavering.
- **Conditional transition agreement** — direction reading with state held equal, within- and between-family.
- **Self-consistency & behavioral** — run-to-run stability of one source, and whether its instability sits where the text is ambiguous.
- **Boundary vs. label** — separates *where* a source cuts from *how* it labels.
- **Forks & lacunae** — per-fork text-vs-model verdict; damaged/indeterminate zones flagged as artifacts, not findings.
- **Transition matrices / entropy / L1 / JS** — the structural transition layer (`analysis.py`), retained from the original design.

## Why it matters

It compares human and machine reading at the level of cognitive *process* — how a mind connects an event to what it produces — not textual resemblance. The method generalizes beyond myth (ritual, legal, historical texts; model-vs-model comparison). The **coarse ordering is robust**: models are far more self-consistent than the human and cluster together on state, the human is the outlier, and direction is freer than state. The **fine geometry** of who-reads-like-whom in direction is text-conditioned — itself a result: myth is a discriminating stimulus.

## Status & limits

- All results rest on **2 human runs per corpus** — provisional until a blind 3rd; numbers and some conclusions may shift then.
- **Qwen's finer granularity** is a per-source confound: controlled for the label/agreement/freedom metrics by line projection, but not for boundary-based metrics.
