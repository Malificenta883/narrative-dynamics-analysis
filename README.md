# Narrative Dynamics Analysis: Human vs. AI Semantic Topologies

Comparative meta-analysis of narrative segmentation patterns between human scholars and frontier large language models (Claude 4.5, 4.6 Sonnet, Gemini 3, 3.1 Pro, GPT-5.2, 5.5) applied to ancient Sumerian mythological corpora. 

## Overview

This project explores the cognitive gap between human macro-narrative comprehension and AI local statistical parsing. By converting the Sumerian mythological texts (Gudea Cylinder A, Inanna's Descent, Inanna and Enki) into directed Markov chains, this pipeline mathematically proves how LLMs fail to retain macro-narrative topology in highly entropic texts.

The analysis utilizes:
- **Transition Matrices** to model the physical flow between functional narrative states.
- **Shannon Entropy ($H$)** to measure cognitive determinism vs. branching uncertainty.
- **Dynamic Time Warping (DTW)** for sequence alignment and handling index shifts.
- **Edge-Graph Edit Distance (Edge-GED)** to quantify topological divergence.
- **Universal Human Baseline Aggregation** to extract the core human reading algorithm across multiple texts.

## Architecture & Data Structure

The repository has been scaled from a single-text script into a hierarchical meta-pipeline. Data is stored in `data/` categorized by myth corpus.

```text
narrative-dynamics-analysis/
├── data/
│   ├── gudea/
│   │   ├── mine.json (Human Anchor)
│   │   ├── claude.json
│   │   ├── gpt.json
│   │   └── gemini.json
│   ├── inanna_descent/
│   └── inanna_enki/
└── src/
    └── analysis.py (Meta-Cycle Execution)

Each JSON segment maps the narrative state, transition logic, cognitive frames, and anomaly types.

Installation & Usage

# Clone the repository
git clone [https://github.com/malificenta883/narrative-dynamics-analysis.git](https://github.com/malificenta883/narrative-dynamics-analysis.git)
cd narrative-dynamics-analysis

# Install dependencies (requires networkx, matplotlib)
pip install -r requirements.txt

# Run the complete meta-pipeline (generates Universal Baseline)
python src/analysis.py --data-dir data

# Run without generating matplotlib graphs
python src/analysis.py --data-dir data --no-graphs

## 📊 Key Findings (Empirical Results)

The aggregation of the Sumerian corpus reveals fundamental architectural differences in text processing:

> **The Human Pulse (Cyclical Topology)** > Human annotation exhibits dynamic entropy—maintaining high uncertainty ($H > 2.0$) during narrative 'contact' and 'exchange' phases, but strictly collapsing to absolute determinism ($H = 0.0$) at the 'return' phase, seamlessly looping into new 'disruptions'. Humans read in cycles.

> **GPT's Markov Loops** > GPT models exhibit severe absorbing state bugs. In highly entropic resolutions, GPT falls into a `return -> return` loop (50% probability), losing the macro-context of the myth.

> **Claude's Fractal Noise** > Claude engages in micro-parsing, over-segmenting the text and artificially inflating entropy across all nodes (mean $H \approx 1.89$). It reacts to local syntax (e.g., the word "give") rather than global semantics.

> **Gemini's Entropy Collapse** > Gemini acts as a strict optimizer. It flattens the narrative topology, eliminating both zones of high uncertainty and rigid determinism, forcing the myth into a linear, flat computational pipe.

---

## ⚙️ Narrative State Model

The framework uses seven functional states based on ritual and mythological theory:

* **Preparation** — Setup and anticipation
* **Contact** — Initial divine-human encounter
* **Exchange** — Bidirectional transfer
* **Disruption** — Breakdown or confusion
* **Negotiation** — Mediation and clarification
* **Stabilization** — Resolution and integration
* **Return** — Completion and closure


Citation
If you use this methodology or dataset, please cite:

@software{sumerian_narrative_dynamics,
  author = {Koshel Marharyta},
  title = {Narrative Dynamics Analysis: Human vs. AI Semantic Topologies},
  year = {2026},
  url = {[https://github.com/malificenta883/narrative-dynamics-analysis](https://github.com/malificenta883/narrative-dynamics-analysis)}
}

## Contact

Koshel Marharyta 
- Email: eleovora882@gmail.com
- ORCID: [https://orcid.org/0000-xxxx-xxxx-xxxx]
