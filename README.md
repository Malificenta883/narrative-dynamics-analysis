# narrative-dynamics-analysis

**A computational method for separating text-driven from model-driven variance in narrative segmentation — and for measuring how human vs. LLM cognition resolves narrative ambiguity.**

When several annotators — humans and frontier LLMs — segment the same narrative into functional states, they disagree. **The disagreement is the data.** This repository provides metrics that decompose it: how much variance is a property of the *text* (genuine underdetermination — multiple readings are legitimate), how much is a property of the *reading cognition* (instability where the text is in fact fixed), and how much is an artifact of a *damaged source* (no reading is recoverable). What the pipeline measures is not annotation quality but refraction — how a human vs. an LLM mind structures the same ambiguous narrative cell, how stably, and how far they converge. Ancient Sumerian myth (*Inana and Enki*, ETCSL 1.3.1; *Inanna's Descent*, CDLI composite P468903) is the controlled stimulus: a fixed, richly structured, non-transparent text held constant across all readers. The myth is not the object of study — it is a clean laboratory for measuring how cognition resolves narrative ambiguity, and whether it resolves it stably and in alignment across human and machine.

---

## What this measures

The pipeline operates on line-projected segmentations, so results are independent of how finely each source segmented (granularity is normalized away).

- **Cross-source freedom map** — per-line normalized entropy of the assigned function across *different* sources (human + models). High = the text admits multiple readings there; low = the text dictates. Requires ≥3 sources.
- **Self-consistency (RAW) & behavioral** — for a single source run N times in fresh sessions: RAW = raw run-to-run stability; behavioral = stability *inside the text-free zones* (weighted by the external cross-source freedom map). The gap between them tells you whether a source's instability is spread evenly or concentrated where the text is ambiguous.
- **Boundary vs. label decomposition** — splits self-agreement into two orthogonal axes: *where* a source cuts (segment boundaries) vs. *how* it labels (function on shared lines). A source can cut identically yet label differently — invisible in a single agreement number.
- **Fork analysis** — for every function-pair a source is bistable between, the mean cross-source freedom over those lines → a per-fork verdict: text-driven vs. model-driven.
- **Lacuna / indeterminacy registry** — high freedom on physically damaged or culturally unreadable passages is flagged as an *artifact*, not a finding, and excluded from claims about interpretive freedom. Distinguishes *missing* text (harmless; the myth reads around it) from *indeterminate* text (no invariant exists to read it; no reader, human or model, can assign a function).
- **Transition-level metrics** — transition matrices, Shannon entropy per state, L1 distance, Jensen–Shannon divergence.

---

## Key results — Inana and Enki (ETCSL 1.3.1, block-level, N=10 Opus + 10 Gemini + 2 human)

**Self-consistency and behavioral (freedom-weighted):**

| source | RAW self-consistency | behavioral (in free zones) |
|--------|---------------------:|---------------------------:|
| Opus   | 0.915 | 0.656 |
| Gemini | 0.853 | 0.783 |
| human  | 0.757 | 0.667 |

Opus is the most self-consistent overall but its instability *concentrates in the free zones* (0.92 → 0.66). Gemini is rougher overall but *holds its invariant better where the text is genuinely ambiguous* (0.85 → 0.78). Instability is both model- and text-specific — there is no single "reliability" number for a model.

**Boundary vs. label decomposition** (where each source cuts vs. how it labels):

| source | boundary agreement | label agreement | gap |
|--------|-------------------:|----------------:|----:|
| Opus   | 0.83 | 0.87 | −0.04 |
| Gemini | 0.66 | 0.78 | −0.12 |
| human  | 0.82 | 0.30 | **+0.52** |

The models' instability is *homogeneous* — neither axis holds firmer (gap ≈ 0). The human's is *localized*: boundaries as stable as Opus, but labels far more variable. Segmentation ("where are the episodes") and labeling ("what are they") behave as **separate operations with different stability for the human, and as a single operation for the models.** A plain line-projected agreement number (human ≈ 0.30) hides this; the decomposition shows the human's variance is structured, sitting in labeling, not segmentation.

*(The human runs are 9 months apart; part of the low label agreement is genuine temporal drift, tracked as a separate variable, not conflated with instantaneous inconsistency.)*

**Fork structure.** The top real fork shared by both models is `return ↔ stabilization` on the myth's **finale** — a text-specific ambiguity (the ending is legitimately open between the two), not a model quirk. The single highest-freedom zone (311–331, the frog / halub-tree passage, freedom ~0.62) is correctly flagged as **lacuna/indeterminate** and excluded — it abuts a ~10–15 line gap and has no cultural invariant to read it.

### Second corpus — Inanna's Descent (CDLI P468903, line-level)

On Descent, `negotiation` is the structural instability hub (it conflicts with most other functions). On Enki it is nearly absent. **The instability hub is text-specific, not a schema defect** — the seven-state schema, derived endogenously from the me-transfer myth rather than imported from external narratology, does not collapse at the same categories regardless of material. This is evidence that the states function as a stable measurement grid across myths.

---

## Install & run

```bash
git clone https://github.com/malificenta883/narrative-dynamics-analysis.git
cd narrative-dynamics-analysis
pip install -r requirements.txt
```

**Cross-source comparison** (human vs. models):
```bash
python src/analysis.py --data-dir data/inanna_enki --no-graphs
```

**Single-source dispersion** (one source vs. itself over N runs) + freedom + boundary/label + forks:
```bash
python src/onemodel.py --data-dir data --source opus \
  --freedom-from data/inanna_enki \
  --lacunae data/inanna_enki/inanna_enki_lacunae.txt
```

**Prepare raw model output** — models return a bare `segments` array; wrap attaches the canonical run header (source/date/model version/prompt id) at save time, so metadata is never model-generated:
```bash
python src/fix_json.py  data/inanna_enki            # sanitize (quotes, commas, fences)
python src/wrap_runs.py data/inanna_enki --corpus inanna_enki \
  --text-variant ETCSL_1.3.1_EN --prompt-id seg_v5 --in-place
```

---

## Data, schema & provenance

Annotations live in `data/<myth>/` as JSON, one file per run: `{source}_run{N}.json`. The loader accepts a bare segment array or a header-object with `segments`. Myths are kept in separate folders; each carries its own `text_variant` (Enki = ETCSL, Descent = CDLI — different editions, different lineation, never cross-compared).

Provenance is closed end-to-end:
- `prompt_id` in each header (e.g. `seg_v5`) resolves to a file in `prompts/`.
- Line numbering resolves to the numbered source texts and map files in `docs/` (`*_numbered.txt`, `*_map.tsv`), which record how each line number maps to the original ETCSL/CDLI reference — including how sub-lines were merged and how lacunae were reserved.

Only `function`, `transition_from/to`, and `line_start/end` drive the current metrics; other fields (`cognitive_frame`, `markers`, `anomaly_type`, `evidence`, …) are carried for future analysis and are not compared across prompt versions.

---

## Method notes (honest limits)

- **Freedom needs ≥3 sources.** With two it is near-binary. Behavioral-in-free-zones is meaningful only against an *external* cross-source freedom map — never one derived from the same runs (circular).
- **Effort tier held constant.** All sources compared at each model's strongest available reasoning mode. Cross-vendor tiers are not calibrated to a common unit — stated as a known confound, not hidden.
- **Access channel.** Sources collected via anonymous UI sessions (fresh session per run, treated as independent draws); temperature/seed not settable. Model version + date logged to guard against silent checkpoint changes.
- **Cross-myth granularity differs by edition.** Descent is line-level (CDLI); Enki is block-level (ETCSL gives block ranges, not lines). Each myth is analyzed within itself; no cross-myth line comparison is made.
- **Lacunae are registered explicitly** (`*_lacunae.txt`) and only *indeterminate* zones are flagged as artifacts; *missing*-but-readable gaps are not.
- **Human temporal drift.** The two human runs are 9 months apart; between-run label change is tracked as drift, not treated as simultaneous uncertainty.

---

## Repository layout

```
narrative-dynamics-analysis/
├── README.md
├── requirements.txt
├── src/
│   ├── analysis.py        # cross-source: freedom, transitions, entropy, JSD
│   ├── onemodel.py        # single-source: self-consistency, behavioral, clustering,
│   │                      #   forks, boundary/label decomposition, lacuna flagging
│   ├── line_alignment.py  # line projection & agreement
│   ├── wrap_runs.py       # attach canonical run headers (with consistency guards)
│   ├── fix_json.py        # sanitize raw model JSON (quotes, commas, fences, wrapping)
│   ├── extract_en.py      # parse CDLI bilingual text -> numbered lines (Descent)
│   ├── extract_enki.py    # parse ETCSL block text -> numbered blocks (Enki)
│   └── fix_text_variant.py
├── data/
│   ├── inanna_descent/    # opus_run1..10, gemini_run*, mine.json  (CDLI, line-level)
│   └── inanna_enki/       # opus_run1..10, gemini_run1..10, mine_run1..2,
│                          #   inanna_enki_lacunae.txt              (ETCSL, block-level)
├── prompts/               # seg_v4.md, seg_v5.md, ...  (referenced by prompt_id)
├── docs/                  # numbered source texts + map files (provenance)
└── figures/
```

---

## Citation

```bibtex
@software{narrative_dynamics,
  author = {Koshel, Marharyta},
  title  = {Narrative Dynamics Analysis: separating text-driven from model-driven
            variance in LLM narrative segmentation},
  year   = {2026},
  url    = {https://github.com/malificenta883/narrative-dynamics-analysis}
}
```

Source texts: *Inana and Enki* — ETCSL 1.3.1 · *Inanna's Descent* — CDLI composite P468903.
License: MIT
