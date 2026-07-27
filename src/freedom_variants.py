#!/usr/bin/env python3
"""
freedom_variants.py — compare three ways of computing the freedom map.

  (A) per-file      : entropy over ALL run files, each file = one vote.
                      This is what build_external_freedom() currently does.
                      Sources with more runs dominate (10 Opus + 10 Gemini + 2 human
                      => the map is 91% model-made). Conflates text-freedom with
                      model jitter.

  (B) per-source mix: each source contributes a DISTRIBUTION over functions
                      (share of its own runs), the distributions are averaged with
                      equal weight, entropy is taken of the mixture.
                      Equal source weights; no run is discarded; ties (your 1:1
                      human runs) enter honestly as 0.5/0.5.

  (C) JSD           : Jensen-Shannon divergence between the per-source distributions
                      = H(mixture) - mean(H(source)).
                      This is the BETWEEN-source disagreement with each source's own
                      jitter subtracted out. Zero when sources read alike on average
                      (even if each wavers); maximal when each source is confident but
                      on a different reading. This is "the text admits several readings"
                      in the pure sense.

TWO GUARDS (both matter):

  --lines-from <numbered.txt>
      Only count line numbers that ACTUALLY EXIST in the source text. The block
      numbering reserves gaps for lacunae (e.g. Enki: 34-54, 165-242, 594-599 ...),
      and project_to_lines() expands range(start, end+1) blindly — so a segment
      spanning a gap assigns labels to numbers with no text behind them. On Enki
      that was 111 phantom lines out of 701 (16%), mostly agreeing trivially and
      diluting every average downward.

  --min-obs N   (default 2)
      A source needs >= N observations on a line to contribute. With one run a
      source's distribution has H=0 — it looks maximally confident — which inflates
      JSD. Lines where a source has a coverage hole would otherwise masquerade as
      genuine between-source divergence.

Usage:
  python freedom_variants.py <folder-with-run-jsons> [--lines-from FILE] [--min-obs N]
  python freedom_variants.py ../data/inanna_enki --lines-from ../docs/inanna_enki_numbered.txt
"""

import sys, json, math, re
from pathlib import Path
from collections import Counter, defaultdict


def load_real_lines(path):
    """Read a numbered text file ('7-16<TAB>text' or '7<TAB>text') and return the set
    of line numbers that actually carry text. Reserved lacuna gaps are absent."""
    real = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r'^\s*(\d+)(?:\s*-\s*(\d+))?\s', line)
        if m:
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a
            real |= set(range(a, b + 1))
    return real

FUNCTIONS = ["preparation", "contact", "exchange", "disruption",
             "negotiation", "stabilization", "return"]
RUN_RE = re.compile(r"^([a-zA-Z0-9]+)_run(\d+)\.json$")


def load_segments(path):
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("segments", [])


def project(segs, field="function", real_lines=None):
    """segment -> {line: label}, where label is taken from `field`
    ('function' for state, 'transition_to' for direction).
    If real_lines is given, lines in lacuna gaps are dropped."""
    out = {}
    for s in segs:
        try:
            a, b = int(s["line_start"]), int(s["line_end"])
        except (KeyError, TypeError, ValueError):
            continue
        lab = str(s.get(field, "")).strip().lower()
        if lab in ("", "-"):
            continue
        for ln in range(a, b + 1):
            if real_lines is None or ln in real_lines:
                out[ln] = lab
    return out


def entropy(dist):
    """dist: {label: probability}. Returns Shannon entropy in bits."""
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def main(folder, real_lines=None, min_obs=2, field="function"):
    folder = Path(folder)
    # source -> list of projections (one per run)
    by_source = defaultdict(list)
    for fp in sorted(folder.glob("*.json")):
        m = RUN_RE.match(fp.name)
        if not m:
            continue
        try:
            segs = load_segments(fp)
        except Exception as e:
            print(f"skip {fp.name}: {e}")
            continue
        if segs:
            by_source[m.group(1)].append(project(segs, field=field, real_lines=real_lines))

    if len(by_source) < 2:
        print(f"need >=2 sources, found: {list(by_source)}")
        sys.exit(1)

    print(f"field: {field}")
    print("sources:", {s: len(v) for s, v in by_source.items()})
    n_src = len(by_source)
    norm7 = math.log2(len(FUNCTIONS))     # ceiling for (A) and (B)
    normS = math.log2(n_src)              # ceiling for (C): JSD <= log2(n_sources)

    all_lines = set()
    for runs in by_source.values():
        for p in runs:
            all_lines |= set(p)

    # GUARD 1: drop phantom lines (reserved lacuna numbers with no text behind them)
    if real_lines is not None:
        phantom = all_lines - real_lines
        all_lines &= real_lines
        print(f"real-line filter: kept {len(all_lines)}, dropped {len(phantom)} phantom "
              f"({100*len(phantom)/(len(all_lines)+len(phantom)):.0f}%) — numbers reserved "
              f"for lacunae that segments spanned over")

    freedom_A, freedom_B, freedom_C, freedom_D = {}, {}, {}, {}
    human_H = {}
    skipped_thin = 0

    # which source is the human? (for the model-jitter test)
    human_key = next((s for s in by_source if s in ("mine", "human")), None)

    for ln in sorted(all_lines):
        # ---- per-source distributions -------------------------------------
        per_source_dist = {}
        for src, runs in by_source.items():
            labels = [p[ln] for p in runs if ln in p]
            # GUARD 2: a source with <min_obs observations has H=0 and looks
            # falsely confident, which inflates JSD. Exclude it from this line.
            if len(labels) < min_obs:
                if labels:
                    skipped_thin += 1
                continue
            c = Counter(labels)
            tot = sum(c.values())
            per_source_dist[src] = {f: n / tot for f, n in c.items()}

        if len(per_source_dist) < 2:
            continue  # need >=2 sources present on this line

        # ---- (A) per-file entropy (current behaviour) ----------------------
        flat = [p[ln] for runs in by_source.values() for p in runs if ln in p]
        c = Counter(flat); tot = sum(c.values())
        freedom_A[ln] = entropy({f: n / tot for f, n in c.items()}) / norm7

        # ---- (B) mixture of per-source distributions -----------------------
        k = len(per_source_dist)
        mix = defaultdict(float)
        for d in per_source_dist.values():
            for f, p in d.items():
                mix[f] += p / k
        H_mix = entropy(mix)
        freedom_B[ln] = H_mix / norm7

        # ---- (C) JSD = H(mix) - mean H(source) -----------------------------
        mean_H = sum(entropy(d) for d in per_source_dist.values()) / k
        jsd = H_mix - mean_H
        freedom_C[ln] = max(0.0, jsd) / math.log2(k)   # normalise by actual k present

        # ---- (D) mean within-source entropy: "everyone wavers, alike" ------
        # H(mix) = mean_H + JSD  — two orthogonal halves of the same quantity.
        # high mean_H + low JSD  = SHARED AMBIGUITY (text admits two readings and
        #                          every cognition, human included, oscillates)
        # low mean_H + high JSD  = DIVERGENT READINGS (each is confident, on different things)
        # high mean_H for models but low for the human = MODEL JITTER
        freedom_D[ln] = mean_H / norm7
        human_H[ln] = (entropy(per_source_dist[human_key]) / norm7
                       if human_key in per_source_dist else None)

    # ---------------- summary ----------------
    def stats(d, name):
        if not d:
            print(f"{name}: no data"); return
        vals = list(d.values())
        hi = [ln for ln, v in d.items() if v >= 0.5]
        print(f"{name:<28} mean={sum(vals)/len(vals):.3f}  "
              f"max={max(vals):.3f}  lines>0={sum(1 for v in vals if v>0):>4}  "
              f"lines>=0.5={len(hi):>4}")

    print()
    print("=" * 76)
    stats(freedom_A, "(A) per-file  [current]")
    stats(freedom_B, "(B) per-source mixture")
    stats(freedom_C, "(C) JSD  between-source divergence")
    stats(freedom_D, "(D) mean within-source H (shared wavering)")
    print("=" * 76)
    if skipped_thin:
        print(f"note: {skipped_thin} source-line cells skipped for <{min_obs} observations")

    # --- three-regime classifier -------------------------------------------
    # H(mix) splits into  mean_H (D, everyone wavers) + JSD (C, they differ).
    #   D high, C low  -> SHARED AMBIGUITY  (text admits >1 reading; all oscillate)
    #   D low,  C high -> DIVERGENT READINGS (each confident, on different labels)
    #   models' H high but human's H low -> MODEL JITTER (human is steady, models aren't)
    HI, LO = 0.30, 0.20

    def regime(ln):
        c = freedom_C.get(ln, 0.0)      # between-source
        d = freedom_D.get(ln, 0.0)      # within-source (shared)
        h = human_H.get(ln)             # human's own entropy (None if absent)
        if c >= HI and d < LO:
            return "DIVERGENT readings (each sure, on different labels)"
        if d >= HI and c < LO:
            return "SHARED ambiguity (all waver alike, incl. human)"
        if h is not None and h < LO and d >= HI:
            return "MODEL jitter (human steady, models waver)"
        if c < LO and d < LO:
            return "text dictates (all agree)"
        return "mixed"

    # collapse consecutive lines with the same regime into spans
    def spans_by_regime():
        rows = []
        cur = None
        for ln in sorted(freedom_C):
            r = regime(ln)
            if cur and cur[2] == r and ln == cur[1] + 1:
                cur[1] = ln
            else:
                if cur:
                    rows.append(cur)
                cur = [ln, ln, r]
        if cur:
            rows.append(cur)
        return rows

    print("\nZones by regime (C=between-source, D=shared wavering, h=human's own H):")
    print(f"{'lines':>12}{'C':>7}{'D':>7}{'h':>7}   regime")
    for a, b, r in spans_by_regime():
        if r == "text dictates (all agree)":
            continue  # skip the boring majority
        mid = (a + b) // 2
        c = freedom_C.get(mid, 0); d = freedom_D.get(mid, 0)
        h = human_H.get(mid)
        hs = f"{h:.2f}" if h is not None else "  —"
        span = f"{a}" if a == b else f"{a}-{b}"
        print(f"{span:>12}{c:>7.2f}{d:>7.2f}{hs:>7}   {r}")

    # dump for downstream use
    out = folder / "freedom_variants.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_file": freedom_A, "per_source_mixture": freedom_B,
                   "jsd_between_sources": freedom_C,
                   "within_source_H": freedom_D, "human_H": human_H}, f)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    # positional args are non-flag tokens that aren't the VALUE of a flag
    FLAGS_WITH_VALUE = {"--lines-from", "--min-obs", "--field"}
    args, skip = [], False
    for i, a in enumerate(sys.argv[1:], start=1):
        if skip:
            skip = False
            continue
        if a in FLAGS_WITH_VALUE:
            skip = True
            continue
        if a.startswith("--"):
            continue
        args.append(a)
    if not args:
        print(__doc__); sys.exit(1)

    real = None
    if "--lines-from" in sys.argv:
        i = sys.argv.index("--lines-from")
        real = load_real_lines(sys.argv[i + 1])
        print(f"loaded {len(real)} real line numbers from {sys.argv[i + 1]}")

    mo = 2
    if "--min-obs" in sys.argv:
        mo = int(sys.argv[sys.argv.index("--min-obs") + 1])

    fld = "function"
    if "--field" in sys.argv:
        fld = sys.argv[sys.argv.index("--field") + 1].strip().lower()
        if fld not in ("function", "transition_to"):
            print(f"--field must be 'function' or 'transition_to', got {fld!r}")
            sys.exit(1)

    main(args[0], real_lines=real, min_obs=mo, field=fld)
