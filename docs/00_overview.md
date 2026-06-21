# Overview: Transition-Based Analysis of Mythological Dynamics

This project operates on a fundamental paradigm shift regarding the nature of ancient texts. **Myth can be defined as a neurosensorial reconstruction of experience. A condensate of perception engraved into the shape of storytelling. It’s not a way to describe the world. It is a language of varied consciousness.**

Consequently, this project studies myth not as a static text, but as an active cognitive **process**. Instead of comparing surface-level text similarities or semantic embeddings, the analysis focuses on the "narrative dynamics"—how consciousness **moves** from one functional state to another. The core object of study is the sequence of transitions that structures the myth[cite: 4].

---

## What Is Analyzed

The dataset consists of parallel annotations of the same mythological corpus (the Sumerian Gudea cycle), produced by:
* **A human annotator** (acting as the biological cognitive baseline).
* **Large Language Models** (Claude, GPT, Gemini) attempting to reconstruct this logic.

Each system segments the narrative and assigns to each unit:
1. A **function** (e.g., preparation, contact, exchange, disruption).
2. A **transition** routing the narrative from a previous function to the next.

The goal is not to force identical segmentation, but to compare the resulting transition dynamics.

---

## The Methodological Shift: Why Transitions?

Because human and machine systems segment narratives differently, index-wise comparisons (segment 1 vs. segment 1) are conceptually misleading and statistically unstable. 

To capture the "language of varied consciousness," this pipeline utilizes representations that are **invariant to segmentation**. These metrics capture *how the narrative behaves*, rather than how it is sliced:

* **Transition Matrices:** Building empirical probability distributions where $P(s_{j} \mid s_{i})$ represents the likelihood of transitioning between states.
* **Transition Entropy ($H$):** Measuring whether a narrative move is strictly deterministic or highly variable via Shannon entropy.
* **Matrix Comparison:** Quantifying systemic differences using L1 distance (absolute structural difference) and Jensen–Shannon divergence (distributional divergence).
* **Transition Graphs:** Visualizing dominant probability-weighted macro-structures to allow qualitative inspection.

---

## Empirical Outcomes

The analysis reveals that while LLMs often agree with humans on *which functions exist* within a text, they fundamentally differ in **how transitions are distributed**. 

* **Topological Divergence:** Some models enforce deterministic narrative collapse, while others generate branching, high-entropy structures. 
* **Invisible Discrepancies:** These structural differences persist even when surface-level textual similarity appears high. These differences cannot be captured by embedding clustering alone.

---

## Broader Implications

By treating myth as a structured cognitive process, this framework bridges the gap between the humanities and rigorous data analysis. It enables the formal evaluation of human vs. machine interpretations at the level of hidden **narrative dynamics**, rather than textual resemblance.

The core architecture is domain-agnostic and can be applied to:
* Other myths and ritual narratives.
* Historical or legal texts.
* Model comparison and LLM reasoning evaluation beyond mythological content.
