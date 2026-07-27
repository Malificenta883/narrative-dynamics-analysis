#!/usr/bin/env python3
"""
direction_analysis.py — analyse how sources read the DIRECTION of the narrative
(transition_to), as opposed to the STATE (function).

Key finding this operationalises: cognitions largely agree on WHAT a passage is
(function) but diverge on WHERE it leads (transition_to); the human diverges from
every model more than the models diverge from each other, specifically in direction.

Three metrics, each computed within-source, within-family, and between-family:

  1. function agreement          — do they agree on the state?
  2. transition agreement (raw)  — do they agree on the direction? (confounded by
                                   the fact that direction is a less stable field)
  3. CONDITIONAL transition       — on the lines where they ALREADY agree on function,
     agreement (the key metric)    do they agree on direction? This isolates pure
                                   causal-direction reading with state held equal.
                                   This is the sharpest discriminator.

Usage:
  python direction_analysis.py <folder> --lines-from inanna_enki_numbered.txt
"""
import sys, json, re
from pathlib import Path
from collections import defaultdict
import statistics as st

RUN_RE = re.compile(r"^([a-zA-Z0-9]+)_run(\d+)\.json$")


def load_real_lines(path):
    real = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r'^\s*(\d+)(?:\s*-\s*(\d+))?\s', line)
        if m:
            a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
            real |= set(range(a, b + 1))
    return real


def project_both(path, real):
    """line -> (function, transition_to)"""
    segs = json.load(open(path, encoding="utf-8-sig"))
    segs = segs if isinstance(segs, list) else segs.get("segments", [])
    out = {}
    for s in segs:
        try:
            a, b = int(s["line_start"]), int(s["line_end"])
        except (KeyError, TypeError, ValueError):
            continue
        fn = str(s.get("function", "")).strip().lower()
        tt = str(s.get("transition_to", "")).strip().lower()
        for ln in range(a, b + 1):
            if real is None or ln in real:
                out[ln] = (fn, tt)
    return out


def func_agree(pa, pb):
    common = set(pa) & set(pb)
    if not common:
        return None
    return sum(1 for ln in common if pa[ln][0] == pb[ln][0] and pa[ln][0]) / len(common)


def trans_agree(pa, pb):
    common = set(pa) & set(pb)
    valid = [ln for ln in common if pa[ln][1] not in ("", "-") and pb[ln][1] not in ("", "-")]
    if not valid:
        return None
    return sum(1 for ln in valid if pa[ln][1] == pb[ln][1]) / len(valid)


def cond_trans_agree(pa, pb):
    """Agreement on transition_to, restricted to lines where function already agrees.
    Pure direction reading, state held equal."""
    common = set(pa) & set(pb)
    same_func = [ln for ln in common if pa[ln][0] == pb[ln][0] and pa[ln][0]]
    if not same_func:
        return None
    ok = sum(1 for ln in same_func
             if pa[ln][1] == pb[ln][1] and pa[ln][1] not in ("", "-"))
    return ok / len(same_func)


def group_pairs(g1, g2, metric, same_group=False):
    vals = []
    for i, a in enumerate(g1):
        for j, b in enumerate(g2):
            if same_group and j <= i:
                continue
            v = metric(a, b)
            if v is not None:
                vals.append(v)
    return vals


def main(folder, lines_from):
    folder = Path(folder)
    real = load_real_lines(lines_from) if lines_from else None

    fams = defaultdict(list)
    for fp in sorted(folder.glob("*.json")):
        m = RUN_RE.match(fp.name)
        if not m:
            continue
        try:
            fams[m.group(1)].append(project_both(fp, real))
        except Exception as e:
            print(f"skip {fp.name}: {e}")

    families = list(fams)
    print(f"families: {[(f, len(fams[f])) for f in families]}\n")

    for label, metric in [("FUNCTION (state)", func_agree),
                          ("TRANSITION raw (direction)", trans_agree),
                          ("TRANSITION | function agrees  (pure direction)", cond_trans_agree)]:
        print(f"=== {label} ===")
        print("  within-family:")
        for f in families:
            v = group_pairs(fams[f], fams[f], metric, same_group=True)
            if v:
                print(f"    {f:<10} {st.mean(v):.3f}  (n={len(v)})")
        print("  between-family:")
        for i in range(len(families)):
            for j in range(i + 1, len(families)):
                a, b = families[i], families[j]
                v = group_pairs(fams[a], fams[b], metric)
                if v:
                    print(f"    {a}<->{b:<10} {st.mean(v):.3f}  (n={len(v)})")
        print()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__); sys.exit(1)
    lf = sys.argv[sys.argv.index("--lines-from") + 1] if "--lines-from" in sys.argv else None
    main(args[0], lf)
