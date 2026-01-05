# narrative-dynamics-analysis
This repository explores differences in narrative dynamics between human annotation and frontier language models using transition matrices, entropy, and graph-based analysis.
# Narrative Dynamics Analysis

Comparative analysis of narrative segmentation patterns between human annotation and frontier language models (Claude, Gemini, GPT) using transition matrices, entropy measures, and graph-based visualization.

## Overview

This project analyzes the Gudea Cylinder A text, comparing how human scholars and AI models segment and interpret ancient Sumerian narrative structure. The analysis uses:

- **Transition matrices** to model narrative flow between functional states
- **Shannon entropy** to measure determinism vs. branching in transitions
- **Weighted directed graphs** to visualize narrative pathways
- **UMAP/HDBSCAN clustering** (optional) for exploratory analysis

## Installation
```bash
# Clone the repository
git clone https://github.com/malificenta883/narrative-dynamics-analysis.git
cd narrative-dynamics-analysis

# Install dependencies
pip install -r requirements.txt
```
Run core analysis (no plots)
python src/analysis.py --data-dir data --no-graphs

Run with plots (optional)
python src/analysis.py --data-dir data


## Data Structure

Annotations are stored in `data/` as JSON files:
- `gudea_segments_mine.json` - Human expert annotation (baseline)
- `gudea_segments_claude4.5sonnet.json` - Claude 4.5 Sonnet
- `gudea_segments_gemini3PRO.json` - Gemini 3 Pro
- `gudea_segments_gpt5.2.json` - GPT-5.2

Each segment contains:
```json
{
  "segment_id": "1",
  "text_en": "Description",
  "function": "preparation|contact|exchange|disruption|negotiation|stabilization|return",
  "transition_from": "previous_state",
  "transition_to": "next_state",
  "cognitive_frame": ["authority_invocation", "reciprocity", ...],
  "markers": ["key", "terms", ...],
  "anomaly_type": "spatial|temporal|sensory|normative|status",
  "risk_mode": "none|overload|coercion|deception|seduction|distortion",
  "outcome_tag": "channel_opened|partial_transfer|stabilization|...",
  "confidence": 0.95
}
```

## Usage

### Basic Analysis
```bash
python src/analysis.py


This runs the core analysis pipeline:
1. Transition entropy report (per-state determinism)
2. Transition matrices (normalized probabilities)
3. Weighted directed graphs (visual representation)

### With Exploratory Clustering
```bash
python src/analysis.py --exploratory-umap


Includes UMAP/HDBSCAN clustering on markers and cognitive frames.

## Key Metrics

### Transition Entropy
Measures uncertainty in state transitions:
- **H = 0**: Deterministic (single outgoing edge)
- **H > 2**: High branching/uncertainty

### L1 Distance
Sum of absolute differences between probability matrices:
```
L1 = Σ|P_human(i→j) - P_model(i→j)|
```

### Jensen-Shannon Divergence
Symmetric KL-divergence for comparing distributions (0 = identical, higher = more different).

## Example Output
```
🌪️ TRANSITION ENTROPY REPORT (bits)
Source     | Mean H   | Weighted H
----------------------------------------
mine       |   1.2345 |     1.2000
claude     |   1.4567 |     1.4200
gpt        |   1.6789 |     1.6500
gemini     |   1.3456 |     1.3100
```

## Project Structure
```
narrative-dynamics-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── gudea_segments_mine.json
│   ├── gudea_segments_claude4.5sonnet.json
│   ├── gudea_segments_gemini3PRO.json
│   └── gudea_segments_gpt5.2.json
└── src/
    └── analysis.py
```

## Methodology

### Narrative State Model
Seven functional states based on ritual and mythological theory:
1. **Preparation** - Setup and anticipation
2. **Contact** - Initial divine-human encounter
3. **Exchange** - Bidirectional transfer
4. **Disruption** - Breakdown or confusion
5. **Negotiation** - Mediation and clarification
6. **Stabilization** - Resolution and integration
7. **Return** - Completion and closure

### Analysis Pipeline
1. Load segmented data from all annotators
2. Build normalized transition matrices (row-stochastic)
3. Calculate Shannon entropy per state
4. Compute distance metrics (L1, JS divergence)
5. Generate directed graphs with edge weights

## Key Findings

(Add your research findings here after running analysis)

## Citation

If you use this work, please cite:
```bibtex
@software{gudea_narrative_dynamics,
  author = {Koshel Marharyta},
  title = {Narrative Dynamics Analysis: Human vs. AI Annotation},
  year = {2025},
  url = {https://github.com/malificenta883/narrative-dynamics-analysis}
}
```

## License

MIT License - see LICENSE file

## Contact

Koshel Marharyta - eleovora882@gmail.com

## Acknowledgments

- Gudea Cylinder A translation based on [https://cdli.earth/artifacts/431881]
