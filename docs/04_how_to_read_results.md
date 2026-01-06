# How to Read Transition Results

This document explains how to interpret the **transition-based results** of the analysis.
All interpretations refer directly to the numerical outputs produced by the code
(transition matrices, entropy values, L1 distance, and Jensen–Shannon divergence).

The focus is exclusively on **transition_from → transition_to** dynamics.

---

## 1. What Is Being Compared

Each source (human annotation and language models) produces a sequence of narrative segments.
For every segment, two fields are extracted:

- `transition_from`
- `transition_to`

All transitions operate within a fixed set of narrative states:

- preparation
- contact
- exchange
- disruption
- negotiation
- stabilization
- return

From these annotations, **transition matrices** are constructed:

P(to_state | from_state)

Each row represents a probability distribution over possible next states.

---

## 2. How to Read a Transition Matrix

Each row answers the question:

> If the narrative is currently in state X, where does it go next?

Example (normalized matrix row):

FROM: disruption  
→ negotiation: 100%

Interpretation:
Whenever the system assigns the state `disruption`, it always transitions to `negotiation`.

This indicates **deterministic narrative logic**, not an error.

---

### What to Look For

- Rows with a single 100% transition  
  → fully deterministic behavior.

- Rows with several non-zero transitions  
  → branching or interpretive uncertainty.

- Self-loops (state → same state)  
  → persistence or stabilization of narrative function.

- Rows with all zeros  
  → the state never appears as a source state in that system.

---

## 3. Transition Entropy

Transition entropy is computed per source state:

H(from) = − Σ p(to) log₂ p(to)

---

### Interpreting Entropy Values

- H = 0  
  → fully deterministic (only one possible transition).

- Higher H  
  → greater uncertainty or branching.

---

### Overall Mean Entropy (from results)

| Source  | Mean Entropy |
|--------|--------------|
| mine   | 1.4234       |
| claude | 0.9345       |
| gpt    | 1.1468       |
| gemini | 1.1684       |

Interpretation:

- Human annotation exhibits the **highest average entropy**.
- Claude is the **most deterministic** system.
- GPT and Gemini occupy intermediate positions.

This reflects different assumptions about narrative openness rather than correctness.

---

### Example: `disruption`

| Source  | Entropy |
|--------|---------|
| mine   | 1.0000  |
| claude | 0.0000  |
| gpt    | 0.0000  |
| gemini | 0.0000  |

Interpretation:
All models treat `disruption` as a single-outcome state.
The human annotation allows multiple continuations.

This indicates a **structural difference in narrative modeling**.

---

### Example: `return` (Human Annotation)

Entropy = 0.0000

Meaning:
Whenever `return` appears, it either always leads to the same next state
or functions as narrative closure.

Zero entropy here signals **closure**, not lack of data.

---

## 4. L1 Distance Between Transition Matrices

L1 distance is computed as:

L1 = Σ |P_human − P_model|

---

### What L1 Measures

- The total difference in transition probabilities.
- Sensitivity to where probability mass is allocated.

Higher L1 values indicate greater structural divergence
between narrative transition logics.

L1 answers the question:

> How different are the transition systems overall?

---

## 5. Jensen–Shannon Divergence

Jensen–Shannon (JS) divergence is computed row-wise
between transition distributions and then averaged.

---

### What JS Captures

- Differences in uncertainty structure.
- Whether two systems distribute probabilities similarly,
  even if the dominant transitions differ.

JS answers the question:

> Do the systems resolve narrative uncertainty in the same way?

---

## 6. Transition Graphs

Transition graphs visualize the same matrices:

- Nodes represent narrative states.
- Directed edges represent transitions above a probability threshold.
- Edge thickness corresponds to transition probability.

---

### How to Read the Graphs

- Few thick edges  
  → rigid, deterministic narrative flow.

- Multiple medium-weight edges  
  → branching interpretation.

- Cycles  
  → recursive or sustained narrative states.

Graphs provide an intuitive view of structural differences
without relying on segment alignment.

---

## 7. What These Results Do Not Claim

- They do not rank models by quality.
- They do not assume a single correct narrative structure.
- They do not rely on segment-by-segment matching.

They demonstrate that:

Even with identical prompts and controlled vocabularies,
different systems instantiate **distinct narrative transition logics**.

---

## 8. Why Transition Analysis Is Central

Exact-match comparisons are unstable due to differing segmentation strategies.

Transition-based analysis:
- survives re-segmentation,
- captures narrative process rather than labels,
- aligns with structural approaches to myth analysis.

For this reason, transition dynamics form the core analytical layer of this study.
