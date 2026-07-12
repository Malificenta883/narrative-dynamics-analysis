#!/usr/bin/env python3
"""
Prepare 'Inana and Enki' (ETCSL block edition) for AI segmentation.

Decisions (per user):
  * BLOCK is the unit (ETCSL gives block-level ranges like '1-10', not line-level).
  * LACUNAE are RESERVED: "about N lines missing" / "N lines fragmentary" advance
    the running line counter by N, leaving a gap in the numbering (position kept).
  * CONTINUOUS numbering across the whole myth: SEGMENT A/B/C resets in the source
    are ignored; we assign our own running line numbers 1..N.
  * Each myth is self-contained; original ETCSL ranges are kept in the map file
    for provenance only.

Output line = one ETCSL block. We assign it a running [start,end] where the span
equals the number of original lines the block covered (from its 'a-b' label), so
block widths and lacuna gaps together reproduce the text's positional structure.

Outputs:
  inanna_enki_numbered.txt  -> "start-end<TAB>text"   (feed to the model)
  inanna_enki_map.tsv       -> running range, ETCSL label, segment, text (provenance)
"""

import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/user-data/uploads/Inanna_and_Enki.txt")
OUT_TXT = Path("/mnt/user-data/outputs/inanna_enki_numbered.txt")
OUT_MAP = Path("/mnt/user-data/outputs/inanna_enki_map.tsv")

# a block: "1-10Text..." or "129-130 (Inana speaks:) ..." or "131-142" (empty)
BLOCK_RE = re.compile(r'^\s*(\d+)-(\d+)\s*(.*)$')
# a single-line block sometimes appears as "N Text" (rare) — allow "NText"
SINGLE_RE = re.compile(r'^\s*(\d+)\s+(\S.*)$')
# lacuna notes: "about 6 lines missing", "6 lines missing", "2 lines fragmentary", "1 line fragmentary"
LACUNA_RE = re.compile(r'^\s*(?:about\s+)?(\d+)\s+lines?\s+(?:missing|fragmentary)\s*$', re.I)
# segment headers / title
HEADER_RE = re.compile(r'^\s*(SEGMENT\b.*|Inana and Enki:.*)$', re.I)


def main():
    raw = SRC.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [ln.rstrip("\r") for ln in raw]

    rows = []          # (run_start, run_end, etcsl_label, segment, text)
    counter = 1        # running line number
    current_seg = None
    pending_text = None  # for a block whose text is on following lines (e.g. 131-142)

    def flush_pending():
        # if a block had no inline text but text followed on later lines
        nonlocal pending_text
        if pending_text is not None:
            r = rows[-1]
            rows[-1] = (r[0], r[1], r[2], r[3], (r[4] + " " + pending_text).strip())
            pending_text = None

    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue

        hm = HEADER_RE.match(ln)
        if hm:
            flush_pending()
            m2 = re.match(r'\s*(SEGMENT\s+\S+)', ln, re.I)
            if m2:
                current_seg = m2.group(1)
            i += 1
            continue

        lm = LACUNA_RE.match(ln)
        if lm:
            flush_pending()
            gap = int(lm.group(1))
            counter += gap   # reserve numbers for the missing lines
            i += 1
            continue

        bm = BLOCK_RE.match(ln)
        if bm:
            flush_pending()
            a, b = int(bm.group(1)), int(bm.group(2))
            width = b - a + 1
            text = bm.group(3).strip()
            run_start = counter
            run_end = counter + width - 1
            counter = run_end + 1
            etcsl_label = f"{a}-{b}"
            if text:
                rows.append((run_start, run_end, etcsl_label, current_seg, text))
            else:
                # block header with text on following line(s)
                rows.append((run_start, run_end, etcsl_label, current_seg, ""))
                pending_text = ""  # will collect following non-structural lines
            i += 1
            continue

        # a line that is not header/lacuna/block: it's continuation text
        # (e.g. text following a bare '131-142' block, or '(A third deity speaks:)')
        if rows:
            if pending_text is not None:
                pending_text = (pending_text + " " + ln).strip()
            else:
                # append to last row's text (rare continuation)
                r = rows[-1]
                rows[-1] = (r[0], r[1], r[2], r[3], (r[4] + " " + ln).strip())
        i += 1

    flush_pending()

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TXT.open("w", encoding="utf-8") as ftxt, OUT_MAP.open("w", encoding="utf-8") as fmap:
        fmap.write("run_start\trun_end\tetcsl_label\tsegment\ttext\n")
        for rs, re_, lab, seg, text in rows:
            span = f"{rs}" if rs == re_ else f"{rs}-{re_}"
            ftxt.write(f"{span}\t{text}\n")
            fmap.write(f"{rs}\t{re_}\t{lab}\t{seg or ''}\t{text}\n")

    total_span = rows[-1][1] if rows else 0
    print(f"Parsed {len(rows)} blocks.")
    print(f"  running line span: 1 .. {total_span} (includes reserved lacuna gaps)")
    empty = sum(1 for r in rows if not r[4].strip())
    if empty:
        print(f"  ⚠️ {empty} blocks ended up with empty text — check these.")
    print(f"  first block: {rows[0][0]}-{rows[0][1]} [{rows[0][3]}] {rows[0][4][:50]}")
    print(f"  last  block: {rows[-1][0]}-{rows[-1][1]} [{rows[-1][3]}] {rows[-1][4][:50]}")
    print(f"\nWrote:\n  {OUT_TXT}\n  {OUT_MAP}")


if __name__ == "__main__":
    main()
