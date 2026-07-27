# narrative-dynamics-analysis

**A computational method for measuring how human vs. LLM cognition reads narrative — separating agreement on *what happens* (state) from divergence on *where it leads* (causal direction), and text-driven from model-driven variance.**

When several annotators — humans and frontier LLMs — segment the same narrative into functional states, they disagree. **The disagreement is the data.** This repository provides metrics that decompose it: how much variance is a property of the *text* (genuine underdetermination — multiple readings are legitimate), how much is a property of the *reading cognition* (instability where the text is in fact fixed), and how much is an artifact of a *damaged source* (no reading is recoverable). What the pipeline measures is not annotation quality but refraction — how a human vs. an LLM mind structures the same ambiguous narrative cell, how stably, and how far they converge. Ancient Sumerian myth (*Inana and Enki*, ETCSL 1.3.1; *Inanna's Descent*, CDLI composite P468903) is the controlled stimulus: a fixed, richly structured, non-transparent text held constant across all readers. The myth is not the object of study — it is a clean laboratory for measuring how cognition resolves narrative ambiguity, and whether it resolves it stably and in alignment across human and machine.

---

## What this measures

The pipeline operates on line-projected segmentations, so results are independent of how finely each source segmented (granularity is normalized away).

- **Cross-source freedom map** — per-line normalized entropy of the assigned function across *different* sources (human + models). High = the text admits multiple readings there; low = the text dictates. Requires ≥3 sources.
- **Self-consistency (RAW) & behavioral** — for a single source run N times in fresh sessions: RAW = raw run-to-run stability; behavioral = stability *inside the text-free zones* (weighted by the external cross-source freedom map). The gap between them tells you whether a source's instability is spread evenly or concentrated where the text is ambiguous.
- **Boundary vs. label decomposition** — splits self-agreement into two orthogonal axes: *where* a source cuts (segment boundaries) vs. *how* it labels (function on shared lines). A source can cut identically yet label differently — invisible in a single agreement number.
- **Fork analysis** — for every function-pair a source is bistable between, the mean cross-source freedom over those lines → a per-fork verdict: text-driven vs. model-driven.
- **Lacuna / indeterminacy registry** — high freedom on physically damaged or culturally unreadable passages is flagged as an *artifact*, not a finding, and excluded from claims about interpretive freedom. Distinguishes *missing* text (harmless; the myth reads around it) from *indeterminate* text (no invariant exists to read it; no reader, human or model, can assign a function).

### State vs. direction — the central axis

Each segment carries two fields: `function` (the **state** — what is happening) and `transition_to` (the **direction** — where it leads). These are measured separately, and the key finding of the project is that they behave very differently:

- **Direction carries roughly twice the signal of state.** Cross-source freedom on `transition_to` is ~2× that on `function` (Enki: 0.53 vs 0.29), and a JSD decomposition shows this is genuine between-source divergence, not shared wavering. Cognitions converge on *what* happens and diverge on *where it leads*.
- **Conditional transition agreement** — the sharpest instrument: agreement on `transition_to` *restricted to lines where two sources already agree on* `function`. This isolates pure direction-reading with state held equal — given that both call the state X, do they agree on where X leads? Computed within-source, within-family, and between-family.
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

*(The two Enki human runs are 9 months apart; part of the low label agreement is genuine temporal drift, tracked as a separate variable, not conflated with instantaneous inconsistency. Descent uses the standard ~1-month spacing.)*

**Fork structure.** The top real fork shared by both models is `return ↔ stabilization` on the myth's **finale** — a text-specific ambiguity (the ending is legitimately open between the two), not a model quirk. The single highest-freedom zone (311–331, the frog / halub-tree passage, freedom ~0.62) is correctly flagged as **lacuna/indeterminate** and excluded — it abuts a ~10–15 line gap and has no cultural invariant to read it.

**Direction vs. state — the signal is in direction.** `transition_to` equals the next segment's `function` only ~70% of the time — it is an independent judgement about causal flow, not derivable from the state labels. `[SOLID]`

| field         | mean freedom | JSD (between) | within-wavering |
|---------------|-------------:|--------------:|----------------:|
| function      | 0.289        | 0.190         | 0.182           |
| transition_to | 0.514        | 0.372         | 0.306           |

Direction disagreement is ~2× state and JSD-dominated → real between-source divergence, clustered at narrative hinge points. `[SOLID]`

**Direction reading is a family signature** (conditional transition agreement — direction given state already agrees): `[SOLID for models]`

| pair            | conditional agreement |
|-----------------|----------------------:|
| within-Gemini   | 0.736 |
| within-Opus     | 0.537 |
| Opus ↔ Gemini   | 0.458 |
| Gemini ↔ human  | 0.378 |
| Opus ↔ human    | 0.202 |

Each model agrees with itself on direction more than with the other → direction is a per-architecture regime. The human is the farthest point from both — cognitions converge on state, diverge on direction, and the human diverges most. *(Human column `[PROVISIONAL]` — rests on 2 runs.)*

### Second corpus — Inanna's Descent (CDLI P468903, line-level, N=10 Opus + 10 Gemini + 2 human)

**Schema validity.** On Descent, `negotiation` is the structural instability hub (it conflicts with most other functions). On Enki it is nearly absent. **The instability hub is text-specific, not a schema defect** — the seven-state schema, derived endogenously from the me-transfer myth rather than imported from external narratology, does not collapse at the same categories regardless of material. Evidence that the states function as a stable measurement grid across myths. `[SOLID]`

**Descent is more divisive than Enki, even on state** — freedom 0.49 vs 0.29, JSD 0.43 vs 0.19. The amount and location of divergence is text-driven; the myth dictates how far cognitions split. `[SOLID]`

**What replicated, what didn't** (conditional transition agreement, Descent vs Enki):

| pair            | Descent | Enki |
|-----------------|--------:|-----:|
| within-Opus     | 0.556   | 0.537 |
| within-Gemini   | 0.541   | 0.736 |
| Opus ↔ Gemini   | 0.396   | 0.458 |
| Opus ↔ human    | 0.479   | 0.202 |
| Gemini ↔ human  | 0.358   | 0.378 |

- **Replicated:** the state/direction split (converge on *what*, diverge on *where*); the human distant from both models; granularity orthogonal to reading (Opus cuts 29–48 segments, Gemini 8–14, human 18–19 — same pattern regardless). `[SOLID for models]`
- **Falsified on the second corpus (a useful catch):** "the human is specifically farthest from Opus in direction" was Enki-specific — on Descent the human is *closest* to Opus. The surviving, sharper claim: **human↔Gemini direction-distance is stable across myths (~0.37), while human↔Opus swings (0.20–0.48).** Gemini reads causal direction in a consistently non-human regime; Opus's proximity to the human reading is text-dependent. `[PROVISIONAL — 2 human runs]`
- **"Lighthouse vs. interlocutor"** `[HYPOTHESIS, n=2]` — a centroid probe finds both human runs sit at the same distance from Gemini's reading (spread 0.02) but wildly different distances from Opus's (spread 0.32). Gemini behaves as an invariant target; Opus as a responsive one, resonating with whichever human reading occurred — the swing localises to ~15 lines in two edge zones (opening, finale). Falsifiable: a 3rd human run should again land ~0.4 from Gemini and anywhere from Opus.

*Both corpora currently rest on 2 human runs each; the human axis is `[PROVISIONAL]` until a blind 3rd annotation. This is stated as the main limit, not hidden.*

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

**Direction analysis** (state vs. direction agreement — the central metrics) — three agreements per pair (function / raw transition / conditional transition), within-family and between-family. Files must be named `{family}_run{N}.json`:
```bash
python src/direction_analysis.py data/inanna_enki --lines-from docs/inanna_enki_numbered.txt
```

**Freedom + JSD decomposition on either field** (`--field function` for state, `--field transition_to` for direction):
```bash
python src/freedom_variants.py data/inanna_enki --lines-from docs/inanna_enki_numbered.txt --field transition_to
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
- **Human runs are spaced in time by design.** The interval erases episodic memory of the prior run, so re-annotation reflects cognitive structure rather than recall of previous labels — the interval is a method, not an accident. The prescribed spacing is ≥1 month (used for Descent, and the standard for all runs going forward). The two Enki runs happen to be 9 months apart — a larger, incidental gap; between-run label change there is tracked as drift, not treated as simultaneous uncertainty.
- **Two script tiers.** `analysis.py` / `onemodel.py` are the stable overview (state, self-consistency, forks, lacunae). `direction_analysis.py` / `freedom_variants.py` are the lighter, all-runs stats for the direction findings and the paper. Both operate on the same data and provenance.

---

## Repository layout

```
narrative-dynamics-analysis/
├── README.md
├── requirements.txt
├── src/
│   ├── analysis.py        # cross-source overview: freedom, transitions, entropy, JSD,
│   │                      #   line projection & agreement
│   ├── direction_analysis.py  # state vs direction: function / raw-transition /
│   │                      #   CONDITIONAL-transition agreement, within & between family
│   ├── freedom_variants.py    # freedom + JSD decomposition, --field function|transition_to
│   ├── onemodel.py        # single-source: self-consistency, behavioral, clustering,
│   │                      #   forks, boundary/label decomposition, lacuna flagging
│   ├── wrap_runs.py       # attach canonical run headers (with consistency guards)
│   ├── fix_json.py        # sanitize raw model JSON (quotes, commas, fences, wrapping)
│   ├── fix_text_variant.py  # correct the text_variant metadata field across runs
│   ├── extract_en.py      # parse CDLI bilingual text -> numbered lines (Descent)
│   └── extract_enki.py    # parse ETCSL block text -> numbered blocks (Enki)
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
