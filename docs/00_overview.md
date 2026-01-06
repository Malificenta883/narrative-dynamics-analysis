# Overview: Transition-Based Analysis of Myth Structure

This project studies myth not as a static text, but as a **process**.

Instead of comparing surface-level similarity between texts or embeddings,
the analysis focuses on how a narrative **moves** from one functional state to another.
The core object of study is the *sequence of transitions* that structures the myth.

## What is analyzed

The dataset consists of multiple annotations of the same myth
(Gudea Cycle, Sumerian tradition), produced by:

- a human annotator (baseline)
- several large language models (Claude, GPT, Gemini)

Each annotation segments the myth into units and assigns to each unit:
- a **function** (e.g. preparation, contact, exchange, disruption, etc.)
- a **transition** from a previous function to a next one

The goal is not to force identical segmentation,
but to compare the **resulting transition dynamics**.

## Why transitions, not exact matches

Different systems segment narratives differently.
Therefore, index-wise comparison (segment 1 vs segment 1) is unstable
and conceptually misleading.

This project instead uses representations that are **invariant to segmentation**:
- transition matrices
- entropy of outgoing transitions
- probability-weighted transition graphs

These representations capture *how the narrative behaves*, not how it is sliced.

## Core methods

1. **Transition matrices**  
   For each source, a matrix is built where  
   `matrix[from_state][to_state] = P(transition)`.

2. **Transition entropy**  
   For each functional state, Shannon entropy is computed over its outgoing transitions.
   This measures how deterministic or variable a narrative move is.

3. **Matrix comparison**  
   Differences between sources are quantified using:
   - L1 distance (absolute structural difference)
   - Jensen–Shannon divergence (distributional divergence)

4. **Transition graphs**  
   Directed graphs visualize dominant transitions,
   allowing qualitative inspection of narrative structure.

## What the results show

The analysis reveals that:
- models often agree on *which functions exist*,
  but differ in **how transitions are distributed**
- some models exhibit more deterministic narrative behavior,
  others produce more branching structures
- differences persist even when surface similarity appears high

These differences cannot be captured by embedding clustering alone.

## Why this matters

This approach treats myth as a structured cognitive process.
It allows comparison between human and machine interpretations
at the level of **narrative dynamics**, not textual resemblance.

The same framework can be applied to:
- other myths
- ritual narratives
- historical or legal texts
- model comparison beyond mythological content
