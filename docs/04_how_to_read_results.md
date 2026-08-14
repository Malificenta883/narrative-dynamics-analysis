# How to Read the Results

This guide explains how to interpret the outputs the pipeline writes to `figures/`. The headline layer is **state vs. direction** and the **decomposition of disagreement**; the transition-matrix layer (§7) is retained from the earlier version. Worked numbers below are real, mostly from *Inana and Enki* (N = 10 Opus + 10 Gemini + 10 Qwen + 2 human).

## 0. The one idea

Every reader assigns each segment a `function` (**state** — what is happening) and a `transition_to` (**direction** — where it leads). Disagreement is **decomposed, never averaged into a "correct" answer.** Read every number as: *how much*, and *of what kind*, is the disagreement here — text-driven, cognition-driven, or artifact.

---

## 1. Freedom map — `freedom_variants_*.txt`, `freedom_transition_*.txt`

Per-line normalized entropy of the assigned label across sources, summarized four ways:

- **(A) per-file** — every run is one vote (sources with more runs dominate; diagnostic only).
- **(B) per-source mixture** — each source weighted equally (the honest "freedom of the text").
- **(C) JSD between sources** — genuine between-source divergence, each source's own jitter subtracted out.
- **(D) within-source wavering** — everyone waffling alike.

Identity: `H(mixture) = D (mean-within) + C (JSD)` — two orthogonal halves of the same freedom.

**Read it:** high **B** = the text admits several readings here. **C > D** → sources genuinely read differently (real divergence); **D > C** → shared ambiguity (all waver, incl. the human). Run with `--field function` for state, `--field transition_to` for direction.

**Example (Enki, B / C / D):** function `0.32 / 0.21 / 0.17`; direction `0.52 / 0.32 / 0.30`. Direction is freer, and on Enki C≈D (genuine divergence). On Descent and Gudea direction-freedom is **D-dominated** (shared wavering) — so "direction-freedom is real between-source divergence" is an Enki property, not universal.

---

## 2. Conditional transition agreement — `direction_analysis_*.txt`

Three agreements per pair, each **within-family** and **between-family**:

- **function** — do they agree on the state?
- **raw transition** — do they agree on direction? (confounded — direction is a less stable field)
- **conditional transition** — agreement on direction *only on lines where function already agrees*. This isolates pure direction-reading with state held equal, and is the sharpest metric.

**Read the conditional block:** within-family high + between-family lower = a per-architecture direction regime (**"family signature"**). The human's pairs are the lowest = the human reads causal direction unlike the models.

**Example (Enki conditional):** within-Gemini 0.74, within-Qwen 0.68, within-Opus 0.54; Opus↔human 0.20 (lowest). **Caveat:** the signature is a *tendency with exceptions* — Opus is loose on Enki (within 0.54 < Opus↔Qwen 0.58), Gemini is loose on Gudea; it is cleanest on Descent.

---

## 3. Self-consistency & behavioral — `onemodel_*.txt`

For a single source run N times in fresh sessions:

- **pairwise line self-agreement** — how often two runs of the same source match. Models **61–87%**, human **27–34%**: models are a different class of run-to-run stability.
- **RAW self-consistency** (entropy-based) — less informative at n = 2; for the human, prefer the pairwise number.
- **behavioral** (needs `--freedom-from <corpus folder>`) — stability *inside the text-free zones*. The RAW → behavioral drop says whether a source's instability concentrates where the text is genuinely ambiguous.

**Example (Enki):** Opus 0.92 → 0.82 (instability concentrates in the free zones); Gemini 0.85 → 0.84 (holds its invariant even where the text is open).

---

## 4. Boundary vs. label — `onemodel_*.txt`

Splits self-agreement into two orthogonal axes: **boundary F1** (*where* it cuts) and **label agreement** (*how* it names), with `gap = boundary − label`.

**Read it:** gap ≈ 0 → cutting and naming equally (un)stable — a single operation (the models). Large **positive** gap → **stable structure, unstable naming**: the human on Enki/Descent (boundary 0.82, label 0.30, gap **+0.52**) re-cuts in the same places but re-labels the blocks. A plain agreement number (human ≈ 0.30) hides this; the split shows the human's variance is *structured*.

**Note:** boundary F1 is **granularity-sensitive** — Qwen's high boundary agreement partly reflects simply having more boundaries to match (Qwen cuts finest). Label/freedom metrics are line-normalized and do not carry this.

---

## 5. Forks & lacunae — `onemodel_*.txt`

- **Fork** = a function-pair a source is bistable between. The mean cross-source freedom over those lines gives a verdict: **high → text-driven** ambiguity (the passage is legitimately open); **low → model jitter** (the others agree; only this source wavered).
- **Lacuna registry** (`*_lacunae.txt`) distinguishes *missing* text (harmless — the myth reads around it) from *indeterminate* text (no invariant exists to read it). Indeterminate zones showing high freedom are flagged as **artifacts** and excluded from claims about interpretive freedom (e.g. Enki 311–331, the frog / halub-tree passage).

---

## 6. Cross-corpus roll-up — `comparison_summary.*`, `comparison_by_model.*`

State/direction agreement and freedom across all three corpora — models averaged (`summary`) or per source (`by_model`). Use these to see **what is robust vs. text-conditioned**: the models-cluster/human-apart pattern holds on state everywhere; on direction it holds on Enki and Gudea but not Descent (there the human sits close to Opus).

---

## 7. Transition matrices, entropy, L1, JS — `analysis.py` (structural layer)

Retained from the original design. For each source: `P(to_state | from_state)`; Shannon entropy per state (**H = 0** deterministic single outgoing edge; **higher = branching**); **L1** and **Jensen–Shannon** distance between sources' matrices; directed transition graphs (edge weight = probability). These describe the *shape* of narrative flow independent of segmentation. They were the project's original core; the state/direction and freedom metrics above are the current headline, but this layer still runs and is useful for a structural overview.

---

## 8. What these results do NOT claim

- No model is ranked "better." There is **no single correct segmentation**, and **no segment-by-segment matching**.
- **Human rows are provisional** (2 runs per corpus) — numbers and some conclusions may shift after a blind 3rd run.
- **Granularity differs by source** (Qwen finest, Gemini coarsest); label/freedom metrics are line-normalized, boundary metrics are not.

**What they do show:** even with identical prompts and a controlled vocabulary, human and model cognition instantiate distinct reading dynamics — converging on state, diverging on direction, with the human the outlier and each model (mostly) its own direction regime.
