# narrative-dynamics-analysis

**A computational method for measuring how human vs. LLM cognition reads narrative — separating agreement on *what happens* (state) from divergence on *where it leads* (causal direction), and text-driven from model-driven variance.**

When several annotators — humans and frontier LLMs — segment the same narrative into functional states, they disagree. **The disagreement is the data.** This repository provides metrics that decompose it: how much variance is a property of the *text* (genuine underdetermination — multiple readings are legitimate), how much is a property of the *reading cognition* (instability where the text is in fact fixed), and how much is an artifact of a *damaged source* (no reading is recoverable). What the pipeline measures is not annotation quality but refraction — how a human vs. an LLM mind structures the same ambiguous narrative cell, how stably, and how far they converge. Ancient Sumerian literature is the controlled stimulus — a fixed, richly structured, non-transparent text held constant across all readers. Three corpora are analyzed independently: *Inana and Enki* (ETCSL 1.3.1, block-level), *Inanna's Descent* (CDLI composite P468903, line-level), and the *Gudea Cylinder A* (RIME 3/1.01.07, CDLI P431881, line-level). The myth is not the object of study — it is a clean laboratory for measuring how cognition resolves narrative ambiguity, and whether it resolves it stably and in alignment across human and machine.

**Sources.** Each corpus is read by three frontier model families — Claude Opus (`claude_opus_4.8`), Gemini Pro, and Qwen (`qwen_3.8_max_thinking`) — at 10 fresh-session runs each, plus a human annotator at 2 spaced runs. Adding a third model family (Qwen) is what lets the "family signature" claims below be tested across architectures rather than asserted from two.

---

## What this measures

The pipeline operates on line-projected segmentations, so results are independent of how finely each source segmented (granularity is normalized away).

- **Cross-source freedom map** — per-line normalized entropy of the assigned function across *different* sources (human + models). High = the text admits multiple readings there; low = the text dictates. Requires ≥3 sources (now four: three model families + human).
- **Self-consistency (RAW) & behavioral** — for a single source run N times in fresh sessions: RAW = raw run-to-run stability; behavioral = stability *inside the text-free zones* (weighted by the external cross-source freedom map). The gap between them tells you whether a source's instability is spread evenly or concentrated where the text is ambiguous.
- **Boundary vs. label decomposition** — splits self-agreement into two orthogonal axes: *where* a source cuts (segment boundaries) vs. *how* it labels (function on shared lines). A source can cut identically yet label differently — invisible in a single agreement number.
- **Fork analysis** — for every function-pair a source is bistable between, the mean cross-source freedom over those lines → a per-fork verdict: text-driven vs. model-driven.
- **Lacuna / indeterminacy registry** — high freedom on physically damaged or culturally unreadable passages is flagged as an *artifact*, not a finding, and excluded from claims about interpretive freedom. Distinguishes *missing* text (harmless; the myth reads around it) from *indeterminate* text (no invariant exists to read it; no reader, human or model, can assign a function).

### State vs. direction — the central axis

Each segment carries two fields: `function` (the **state** — what is happening) and `transition_to` (the **direction** — where it leads). These are measured separately, and the key finding of the project is that they behave very differently:

- **Direction is freer than state in every corpus** — but the size of the gap is text-dependent, not a constant. Cross-source freedom on `transition_to` vs. `function`: Enki 0.52 vs 0.32 (~1.6×), Descent 0.59 vs 0.48, Gudea 0.51 vs 0.43. On Enki the gap is largest and JSD-dominated (genuine between-source divergence); on Descent and Gudea the within-source share is larger. *(With two model families the Enki ratio was ~2×; adding Qwen — which disagrees more on state — compressed it. The direction > state ordering is robust across corpora and model sets; its magnitude is not.)* Cognitions converge on *what* happens and diverge on *where it leads*.
- **Conditional transition agreement** — the sharpest instrument: agreement on `transition_to` *restricted to lines where two sources already agree on* `function`. This isolates pure direction-reading with state held equal — given that both call the state X, do they agree on where X leads? Computed within-source, within-family, and between-family.
- **Transition-level metrics** — transition matrices, Shannon entropy per state, L1 distance, Jensen–Shannon divergence.

---

## Key results — *Inana and Enki* (ETCSL 1.3.1, block-level; N = 10 Opus + 10 Gemini + 10 Qwen + 2 human)

**Self-consistency and behavioral (freedom-weighted):**

| source | RAW self-consistency | behavioral (in free zones) |
|--------|---------------------:|---------------------------:|
| Opus   | 0.915 | 0.821 |
| Gemini | 0.853 | 0.844 |
| Qwen   | 0.870 | 0.772 |
| human  | 0.757 | 0.591 |

Opus is the most self-consistent overall but its instability *concentrates in the free zones* (0.92 → 0.82). Gemini is rougher overall yet *holds its invariant best where the text is genuinely ambiguous* (0.85 → 0.84 — almost no drop). Qwen sits between. Instability is both model- and text-specific — there is no single "reliability" number for a model.

**Boundary vs. label decomposition** (where each source cuts vs. how it labels):

| source | boundary agreement | label agreement | gap |
|--------|-------------------:|----------------:|----:|
| Opus   | 0.83 | 0.87 | −0.04 |
| Gemini | 0.66 | 0.78 | −0.12 |
| Qwen   | 0.95 | 0.84 | +0.11 |
| human  | 0.82 | 0.30 | **+0.52** |

The models' gaps are small — segmentation and labeling hold at similar stability (a single connected operation). The human's is extreme: boundaries as stable as Opus, but labels far more variable. Segmentation ("where are the episodes") and labeling ("what are they") behave as **separate operations with different stability for the human, and as a single operation for the models.** A plain line-projected agreement number (human ≈ 0.30) hides this; the decomposition shows the human's variance is *structured*, sitting in labeling, not segmentation.

*(The two Enki human runs are 9 months apart; part of the low label agreement is genuine temporal drift, tracked as a separate variable, not conflated with instantaneous inconsistency. Descent and Gudea use the standard ~1-month spacing.)*

**Direction vs. state — the signal is in direction.** `transition_to` equals the next segment's `function` only ~70% of the time — an independent judgement about causal flow, not derivable from the state labels. `[SOLID]`

| field         | mean freedom | JSD (between) | within-wavering |
|---------------|-------------:|--------------:|----------------:|
| function      | 0.318        | 0.210         | 0.169           |
| transition_to | 0.520        | 0.318         | 0.297           |

Direction disagreement exceeds state and, on Enki, is JSD-dominated → real between-source divergence, clustered at narrative hinge points. `[SOLID]`

**Direction reading is a family signature** (conditional transition agreement — direction given state already agrees): `[SOLID for models, with one exception]`

| pair            | conditional agreement |
|-----------------|----------------------:|
| within-Gemini   | 0.736 |
| within-Qwen     | 0.678 |
| within-Opus     | 0.537 |
| Opus ↔ Qwen     | 0.576 |
| Gemini ↔ Qwen   | 0.492 |
| Opus ↔ Gemini   | 0.458 |
| Gemini ↔ human  | 0.378 |
| Qwen ↔ human    | 0.220 |
| Opus ↔ human    | 0.202 |

Gemini and Qwen each agree with themselves on direction more than with anyone else → a per-architecture regime. Opus is the loose one (within-Opus 0.537, *below* its agreement with Qwen 0.576): Opus does not hold a tight direction-signature and reads causal flow much like Qwen here. The human is the farthest point from all three models (0.20–0.38) — cognitions converge on state, diverge on direction, and the human diverges most. *(Human rows `[PROVISIONAL]` — rest on 2 runs.)*

**Fork structure.** The top real fork shared by the models is `return ↔ stabilization` on the myth's **finale** — a text-specific ambiguity (the ending is legitimately open between the two), not a model quirk. The single highest-freedom zone (311–331, the frog / halub-tree passage) is correctly flagged as **lacuna/indeterminate** and excluded — it abuts a ~10–15 line gap and has no cultural invariant to read it.

---

## Second corpus — *Inanna's Descent* (CDLI P468903, line-level; N = 10 Opus + 10 Gemini + 10 Qwen + 2 human)

**Schema validity.** On Descent, `negotiation` is the structural instability hub (it conflicts with most other functions). On Enki it is nearly absent. **The instability hub is text-specific, not a schema defect** — the seven-state schema, derived endogenously from the me-transfer myth rather than imported from external narratology, does not collapse at the same categories regardless of material. Evidence that the states function as a stable measurement grid across myths. `[SOLID]`

**Descent is more divisive than Enki, even on state** — freedom 0.48 vs 0.32, JSD 0.34 vs 0.21. The amount and location of divergence is text-driven; the myth dictates how far cognitions split. `[SOLID]`

**Family signature holds cleanly here** — every model agrees with itself on direction more than with any other (within-Opus 0.556 > 0.51/0.40; within-Gemini 0.541 > 0.40/0.36; within-Qwen 0.655 > 0.51/0.36). The Opus-looseness seen on Enki does not appear; the signature is corpus-conditioned. `[SOLID for models]`

**What replicated, what didn't** (conditional transition agreement):

| pair            | Descent | Enki | Gudea |
|-----------------|--------:|-----:|------:|
| within-Opus     | 0.556   | 0.537 | 0.697 |
| within-Gemini   | 0.541   | 0.736 | 0.543 |
| within-Qwen     | 0.655   | 0.678 | 0.643 |
| Opus ↔ Gemini   | 0.396   | 0.458 | 0.633 |
| Opus ↔ human    | 0.479   | 0.202 | 0.100 |
| Gemini ↔ human  | 0.358   | 0.378 | 0.163 |
| Qwen ↔ human    | 0.383   | 0.220 | 0.143 |

- **Replicated across all three corpora:** the state/direction split (converge on *what*, diverge on *where*); granularity orthogonal to reading (Opus cuts finely, Gemini coarsely, same pattern regardless).
- **Corpus-dependent (a useful catch):** "the human is specifically farthest from Opus in direction" was Enki- and Gudea-true but **falsified on Descent**, where the human is *closest* to Opus (0.479). The human↔Gemini distance is stable on Enki and Descent (~0.37) but collapses on Gudea (0.163) — so even that hedge does not survive the third corpus. The honest surviving claim is weaker and cleaner: **the human reads causal direction differently from the models, but *which* model it lands nearest is text-dependent, not a fixed property.** `[PROVISIONAL — 2 human runs]`
- **"Lighthouse vs. interlocutor"** `[HYPOTHESIS, n=2]` — on Descent a centroid probe finds both human runs sit at the same distance from Gemini's reading (spread 0.02) but wildly different distances from Opus's (spread 0.32): Gemini behaves as an invariant target, Opus as a responsive one. Falsifiable with a blind 3rd human run.

---

## Third corpus — *Gudea Cylinder A* (RIME 3/1.01.07, CDLI P431881, line-level; N = 10 Opus + 10 Gemini + 10 Qwen + 2 human)

Gudea is the longest and structurally hardest text (814 lines) and serves as a contrasting third case — a royal building hymn rather than a me-transfer myth.

**Freedom** — function 0.43, direction 0.51; here the direction-freedom is **within-source-dominated** (within 0.33 > JSD 0.26), i.e. the high direction-freedom is mostly *everyone wavering alike*, not sources reading differently. This is the opposite decomposition to Enki, and it means the JSD-dominated "genuine divergence" reading of direction-freedom is an Enki property, not a universal one. `[SOLID]`

**Family signature is weakest here** — within-Gemini (0.543) is *below* Gemini's agreement with Opus (0.633) and Qwen (0.606); Gemini does not hold a direction-signature on Gudea. Opus and Qwen still do (within 0.697, 0.643). So the "each model its own regime" claim is a tendency with model- and corpus-specific exceptions (Opus loose on Enki, Gemini loose on Gudea), not a law. `[PROVISIONAL]`

**Human separation is strongest here** — the human is very far from every model in direction (0.10–0.16), the most extreme of the three corpora. But the boundary/label decomposition also shifts: the human's boundaries are *unstable* on Gudea (boundary 0.48, vs 0.82 on Enki), so on this text the human re-segments differently too, not only re-labels. The "stable structure, unstable naming" pattern is Enki/Descent-strong and Gudea-weak. `[PROVISIONAL]`

**Self-consistency** — RAW: Opus 0.78, Qwen 0.76, human 0.77, Gemini 0.70; behavioral all 0.64–0.74. Everyone is least self-consistent on Gudea and most on Enki, tracking Enki being the most "dictating" text.

---

## Cross-corpus synthesis — what is robust vs. text-dependent

| claim | status |
|-------|--------|
| Cognitions converge on state, diverge on direction | robust across all 3 corpora `[SOLID for models]` |
| Direction freer than state | robust (ordering), magnitude text-dependent (1.2–1.6×) |
| Direction-freedom is genuine between-source divergence (JSD-dominated) | **Enki only**; Descent & Gudea are within-source-dominated |
| Each model reads direction in its own regime (family signature) | tendency with exceptions (Opus loose on Enki, Gemini loose on Gudea); cleanest on Descent |
| The human is the outlier in direction | robust on Enki & Gudea; on Descent the human sits *close to Opus* |
| Human variance is in labeling, not segmentation | strong on Enki & Descent; weak on Gudea (boundaries also drift there) |
| Seven-state schema is a stable grid (instability hub is text-specific) | robust `[SOLID]` |

**What survives the granularity confound and the small human N** is the coarse clustering: the models are far more self-consistent run-to-run than the human (61–87% vs 27–34% line agreement) and agree with each other more than with the human **on state, in every corpus** (between-model 0.53–0.64 vs model–human 0.31–0.48). The models cluster; the human sits apart. It is only the finer question of *direction* — and *which* model the human lands nearest — that is text-conditioned (on Descent the human even closes on Opus). Because the label/agreement/freedom metrics are line-projected, this clustering is not an artifact of Qwen cutting finer; the granularity confound bites the boundary metric, not the state-agreement one.

The consistent meta-finding: **the coarse ordering is robust, the fine geometry is text-conditioned.** What generalizes is the *method* (state/direction separation, the freedom/JSD decomposition, boundary-vs-label, the real-line and lacuna guards), the models-cluster/human-apart tendency on state, and the ordering direction > state. The precise geometry of who-reads-like-whom in *direction* is a property of each text, not a fixed fact about the models — which is itself the result: myth is a discriminating stimulus.

*All three corpora rest on 2 human runs each; every human row is `[PROVISIONAL]` until a blind 3rd annotation. This is stated as the main limit, not hidden.*

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

Swap `inanna_enki` for `inanna_descent` or `gudea` (and the matching `--text-variant` / `docs/*_numbered.txt`) to run the other corpora.

---

## Data, schema & provenance

Annotations live in `data/<myth>/` as JSON, one file per run: `{source}_run{N}.json`. The loader accepts a bare segment array or a header-object with `segments`. Myths are kept in separate folders; each carries its own `text_variant` (Enki = ETCSL; Descent & Gudea = CDLI — different editions, different lineation, never cross-compared).

Provenance is closed end-to-end:
- `prompt_id` in each header (`seg_v3` … `seg_v5`) resolves to a file in `prompts/`.
- Line numbering resolves to the numbered source texts and map files in `docs/` (`*_numbered.txt`, `*_map.tsv`), which record how each line number maps to the original ETCSL/CDLI reference — including how sub-lines were merged and how lacunae were reserved.

Only `function`, `transition_from/to`, and `line_start/end` drive the current metrics; other fields (`cognitive_frame`, `markers`, `anomaly_type`, `evidence`, …) are carried for future analysis and are not compared across prompt versions.

---

## Method notes (honest limits)

- **Human axis is provisional (n = 2 retests).** All three corpora rest on 2 human runs each; the intra-annotator claims stand only until a blind 3rd annotation. Numbers and even some conclusions may shift once the 3rd retest lands — the human rows are indicative, not settled. This is the project's top priority, not a footnote.
- **Adding a model family changes the freedom numbers.** Cross-source freedom rises with more sources; the Enki direction/state ratio dropped from ~2× (2 families) to ~1.6× (3 families). Reported metrics always state which sources are in the pool.
- **Granularity is a per-source confound, partly controlled by design.** Sources segment at very different resolutions — median segments/run: Qwen ≈ 28–42, Opus ≈ 21–34, Gemini ≈ 8–24, human ≈ 16–26. **Qwen in thinking mode cuts the finest — up to ~2× the others.** The pipeline projects every segmentation to lines precisely to neutralize this for the *label / agreement / freedom* metrics: a coarse and a fine reading of the same span contribute the same per-line labels, so those metrics are granularity-normalized. **Boundary-based metrics** (boundary F1) and any segment-count view remain granularity-sensitive by construction and should be read with Qwen's finer cutting in mind — e.g. Qwen's high boundary agreement partly reflects simply having more boundaries to match.
- **Freedom needs ≥3 sources.** With two it is near-binary. Behavioral-in-free-zones is meaningful only against an *external* cross-source freedom map — never one derived from the same runs (circular).
- **Effort tier held constant.** All sources compared at each model's strongest available reasoning mode. Cross-vendor tiers are not calibrated to a common unit — a known confound, not hidden.
- **Access channel.** Sources collected via anonymous/temporal UI sessions (fresh session per run, treated as independent draws); temperature/seed not settable. Model version + date logged to guard against silent checkpoint changes.
- **Cross-myth granularity differs by edition.** Descent and Gudea are line-level (CDLI); Enki is block-level (ETCSL gives block ranges, not lines). Each myth is analyzed within itself; no cross-myth line comparison is made.
- **Lacunae are registered explicitly** (`*_lacunae.txt`) and only *indeterminate* zones are flagged as artifacts; *missing*-but-readable gaps are not.
- **Human runs are spaced in time by design.** The interval erases episodic memory of the prior run, so re-annotation reflects cognitive structure rather than recall. The prescribed spacing is ≥1 month (Descent, Gudea, and the standard going forward). The two Enki runs happen to be 9 months apart — a larger, incidental gap; between-run label change there is tracked as drift, not simultaneous uncertainty.
- **Two script tiers.** `analysis.py` / `onemodel.py` are the stable overview (state, self-consistency, forks, lacunae). `direction_analysis.py` / `freedom_variants.py` are the lighter, all-runs stats for the direction findings and the paper. Both operate on the same data and provenance.

---

## Repository layout

```
narrative-dynamics-analysis/
├── README.md
├── requirements.txt
├── src/
│   ├── analysis.py            # cross-source overview: freedom, transitions, entropy, JSD,
│   │                          #   line projection & agreement
│   ├── direction_analysis.py  # state vs direction: function / raw-transition /
│   │                          #   CONDITIONAL-transition agreement, within & between family
│   ├── freedom_variants.py    # freedom + JSD decomposition, --field function|transition_to
│   ├── onemodel.py            # single-source: self-consistency, behavioral, clustering,
│   │                          #   forks, boundary/label decomposition, lacuna flagging
│   ├── wrap_runs.py           # attach canonical run headers (with consistency guards)
│   ├── fix_json.py            # sanitize raw model JSON (quotes, commas, fences, wrapping)
│   ├── fix_text_variant.py    # correct the text_variant metadata field across runs
│   ├── outlier_distance.py    # per-run distance-to-centroid outlier probe
│   ├── outlier_by_zone.py     # localise a source's divergence to line zones
│   ├── extract_en.py          # parse CDLI bilingual text -> numbered lines (Descent)
│   └── extract_enki.py        # parse ETCSL block text -> numbered blocks (Enki)
├── data/
│   ├── inanna_enki/     # opus/gemini/qwen _run1..10, mine_run1..2, inanna_enki_lacunae.txt  (ETCSL, block-level)
│   ├── inanna_descent/  # opus/gemini/qwen _run1..10, mine_run1..2                            (CDLI, line-level)
│   └── gudea/           # opus/gemini/qwen _run1..10, mine_run1..2                            (CDLI/RIME, line-level)
├── prompts/                   # seg_v3.md, seg_v4.md, seg_v5.md  (referenced by prompt_id)
├── docs/                      # numbered source texts + map files (provenance)
└── figures/                   # generated tables, freedom maps, run-clustering dendrograms
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

Source texts: *Inana and Enki* — ETCSL 1.3.1 · *Inanna's Descent* — CDLI composite P468903 · *Gudea Cylinder A* — RIME 3/1.01.07, CDLI P431881.
License: MIT
