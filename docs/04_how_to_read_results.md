# How to Read the Analytical Metrics

This document explains how to interpret the mathematical and topological outputs produced by the `analysis.py` pipeline. The analysis evaluates how humans and LLMs model narrative state transitions across the Sumerian corpus.

---

## 1. Transition Probability Matrices

For every narrative segment, the system extracts a `transition_from` and `transition_to` state based on a fixed controlled vocabulary (e.g., preparation, contact, return)[cite: 3]. 

From these sequences, we construct discrete Markov transition matrices:

$$P(s_{j} \mid s_{i})$$

Each row represents a probability distribution over the possible next states, answering the question: *If the narrative is currently in state $s_{i}$, where does it go next?*[cite: 3].

* **Deterministic Logic:** A row with a single 100% transition indicates absolute structural rigidity (not an error)[cite: 3].
* **Interpretive Branching:** Rows with dispersed probabilities indicate narrative uncertainty[cite: 3].

---

## 2. Information Theoretic Metrics

### Shannon Entropy ($H$)
Transition entropy is computed per source state to measure local determinism[cite: 3]:

$$H(s_{i}) = - \sum P(s_{j} \mid s_{i}) \log_2 P(s_{j} \mid s_{i})$$

* $H = 0$: Absolute determinism (collapse of uncertainty)[cite: 3].
* $H > 0$: High entropy indicates dynamic branching or over-segmentation.

### Flattened Jensen–Shannon Divergence (JSD)
To compare how two different systems resolve narrative uncertainty[cite: 3], we flatten the transition matrices and compute the symmetric JS Divergence. It captures whether two systems distribute probability mass similarly, independent of local alignment[cite: 3].

---

## 3. Structural and Sequence Distances

### L1 Norm Distance
Computes the absolute macroscopic difference in transition probabilities between human and machine matrices[cite: 3]:

$$L_1 = \sum | P_{human} - P_{model} |$$

Higher $L_1$ values indicate a fundamental divergence in narrative logic[cite: 3].

### Graph Edit Distance (GED)
Transition matrices are instantiated as directed graphs[cite: 3]. GED calculates the minimum number of graph operations (edge insertions/deletions/substitutions) required to transform an LLM's narrative topology into the human baseline. It reveals structural distortions like Markov loops or disconnected subgraphs.

### Dynamic Time Warping (DTW)
Because exact-match segment comparisons are unstable due to differing tokenization and chunking strategies[cite: 3], we use DTW. This algorithm finds the optimal non-linear alignment between two chronologically ordered sequences of cognitive states, allowing us to compare the "narrative pulse" regardless of varying segment lengths.

---

## 4. Visualizing Transition Topology

The pipeline outputs circular layout graphs to visualize the transition space:
* **Nodes:** Discrete narrative states[cite: 3].
* **Edges:** Transitions above a statistical threshold[cite: 3].
* **Thickness:** Corresponds to the transition probability weight[cite: 3].

Thick, singular edges reflect linear optimization, while cycles indicate recursive cognitive regimes (e.g., human reading cycles)[cite: 3].
