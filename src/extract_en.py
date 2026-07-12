#!/usr/bin/env python3
"""
Extract the English translation from an ETCSL-style bilingual text and align it
to clean integer line numbers.

Input format (mixed, handled automatically):
  1. <sumerian ...>
   en: <english for line 1>
  2. <sumerian ...>
   en: <english for line 2>
  ...
  13.a. <sumerian ...>
   en: <english for sub-line 13a>
  ...
  398. <english directly, no sumerian, no 'en:' prefix>   <- late fragmentary lines

Rules:
  * Section headers (e.g. 'surface a') and blank lines are skipped.
  * A numbered line whose text is Sumerian is paired with the following ' en:' line.
  * A numbered line whose text is already English (no following ' en:') is taken as-is.
  * Sub-lines like 13.a are kept as separate rows; ETCSL label preserved separately.
  * Output is renumbered to clean consecutive integers 1..N, with a mapping table,
    because the annotation pipeline needs contiguous integer line_start/line_end.

Outputs:
  inanna_en_numbered.txt   ->  "N<TAB>english"   (feed this to the model)
  inanna_line_map.tsv      ->  "new_int<TAB>etcsl_label<TAB>english"  (provenance)
"""

import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/user-data/uploads/Inanna_descent2.txt")
OUT_TXT = Path("/mnt/user-data/outputs/inanna_en_numbered.txt")
OUT_MAP = Path("/mnt/user-data/outputs/inanna_line_map.tsv")

# a numbered line marker at the start: "1.", "13.", "13.a.", "13.a" etc.
# REQUIRES a dot after the number (real line markers are "N." not "N ms.")
NUM_RE = re.compile(r'^\s*(\d+(?:\.[a-z])?)\.\s+(.*\S)\s*$')
# manuscript-variant notes like "1 ms. adds ..." / "2 mss. add ..." — NOT line numbers
MS_VARIANT_RE = re.compile(r'^\s*\d+\s+mss?\.', re.I)
EN_RE = re.compile(r'^\s*en:\s*(.*\S)\s*$')
# heuristic: is a string "mostly English" vs Sumerian transliteration?
# Sumerian transliteration is full of digits-in-words, hyphens, {d}, sz, e2, ki, etc.
SUMERO_HINT = re.compile(r'(\{[a-z]\}|[a-z]2\b|[a-z]3\b|sz|kur-ra|mu-un|ba-|-ke4|-ta\b|\bme\b)')

def looks_english(s: str) -> bool:
    # English lines have normal words and spaces, few hyphenated clusters, no {d}
    if '{' in s:
        return False
    # count "wordy" tokens vs transliteration tokens
    hyphen_clusters = len(re.findall(r'\b\w+-\w+(?:-\w+)+\b', s))
    if hyphen_clusters >= 2:
        return False
    # if it has typical English function words, call it English
    if re.search(r'\b(the|she|her|and|to|of|you|will|in|on|with|from)\b', s, re.I):
        return True
    return hyphen_clusters == 0

def main():
    raw = SRC.read_text(encoding="utf-8", errors="replace").splitlines()
    # normalize: strip trailing \r
    lines = [ln.rstrip("\r") for ln in raw]

    rows = []  # (parent_int_label, english)
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if MS_VARIANT_RE.match(ln):
            i += 1
            continue  # manuscript variant note, not a numbered line
        m = NUM_RE.match(ln)
        if not m:
            i += 1
            continue  # header / blank / stray
        label, rest = m.group(1), m.group(2)

        english = None
        if i + 1 < n:
            em = EN_RE.match(lines[i + 1])
            if em:
                english = em.group(1)
                i += 2
        if english is None:
            if looks_english(rest):
                english = rest
            else:
                english = "..."
            i += 1

        # collapse sub-lines like "13.a" into their parent integer line "13"
        parent = label.split(".")[0]

        if rows and rows[-1][0] == parent:
            prev_label, prev_en = rows[-1]
            if prev_en == "...":
                merged = english
            elif english == "...":
                merged = prev_en
            else:
                merged = prev_en + " " + english
            rows[-1] = (prev_label, merged)
        else:
            rows.append((parent, english))

    # use the ETCSL integer label directly as the line number (gaps preserved,
    # sub-lines already merged into their parent) -> matches the existing 412 scheme
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TXT.open("w", encoding="utf-8") as ftxt, OUT_MAP.open("w", encoding="utf-8") as fmap:
        fmap.write("line\tenglish\n")
        for label, en in rows:
            ftxt.write(f"{label}\t{en}\n")
            fmap.write(f"{label}\t{en}\n")

    nums = [int(lab) for lab, _ in rows]
    print(f"Parsed {len(rows)} content lines.")
    print(f"  line range: {min(nums)} .. {max(nums)}")
    gaps = sorted(set(range(min(nums), max(nums) + 1)) - set(nums))
    if gaps:
        print(f"  gaps (missing line numbers, as in ETCSL): {gaps}")
    placeholders = sum(1 for _, en in rows if en == "...")
    if placeholders:
        ph = [lab for lab, en in rows if en == "..."]
        print(f"  ⚠️ {placeholders} lines with no English (kept as '...'): {ph}")
    print(f"\nWrote:\n  {OUT_TXT}\n  {OUT_MAP}")

if __name__ == "__main__":
    main()
