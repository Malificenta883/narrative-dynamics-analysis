#!/usr/bin/env python3
"""
fix_json.py — проверить и почистить JSON-файлы прогонов разметки в папке.

Ловит и чинит типовые поломки:
  - висячие запятые перед ] или }        (,]  ,})
  - двойные запятые                       (,,)
  - CRLF / одиночные CR внутри файла
  - обёртка ```json ... ```  фенсы
  - преамбула до массива ("Here is the JSON:")
  - отсутствие обрамляющих [ ]  (голая последовательность объектов)
  - двойное вложение массива  [[ {...} ]]  ->  [ {...} ]

НЕ чинит (показывает место, правишь руками):
  - незакрытые/перепутанные кавычки внутри строк
  - неэкранированные внутренние кавычки
  - обрезанный на лимите вывод (последний объект недописан)

Использование:
  python fix_json.py <файл-или-папка>            # сухой прогон: только отчёт
  python fix_json.py <файл-или-папка> --write      # применить починку
  python fix_json.py <папка> --write --backup      # + сохранить .bak копии
"""

import re, sys, json
from pathlib import Path


def try_load(text):
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, e


def auto_fix(raw: str):
    actions = []
    t = raw
    if "\r" in t:
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        actions.append("normalized CRLF->LF")
    fence = re.match(r'^\s*```(?:json)?\s*\n(.*?)\n```\s*$', t, re.DOTALL)
    if fence:
        t = fence.group(1)
        actions.append("removed ``` code fence")
    m = re.search(r'[\[{]', t)
    if m and m.start() > 0:
        head = t[:m.start()].strip()
        if head and not head.startswith(("[", "{")):
            t = t[m.start():]
            actions.append(f"removed preamble ({head[:30]!r})")
    s = t.strip()
    if not s.startswith("["):
        s = "[\n" + s + "\n]"
        actions.append("wrapped in [ ]")
    t = s
    if re.search(r',\s*,', t):
        t = re.sub(r',\s*,', ',', t)
        actions.append("collapsed double commas")
    if re.search(r',(\s*[\]}])', t):
        t = re.sub(r',(\s*[\]}])', r'\1', t)
        actions.append("removed trailing commas")
    data, err = try_load(t)
    if err is None and isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        data = data[0]
        t = json.dumps(data, ensure_ascii=False, indent=2)
        actions.append("un-nested double array [[...]] -> [...]")

    # if still broken, try escaping stray inner double-quotes line by line.
    # only applied where the fixed line parses as a valid JSON string -> safe.
    data, err = try_load(t)
    if err is not None:
        lines = t.split("\n")
        changed_any = False
        for i, line in enumerate(lines):
            m = re.match(r'^(\s*)"(.*)"(\s*,?\s*)$', line)
            if not m:
                continue
            indent, body, tail = m.groups()
            if '"' not in body:
                continue
            # skip key:value lines like  "function": "contact"  — those aren't
            # array-element strings; their quotes are structural, not stray.
            if re.match(r'^[^"]*"\s*:\s*', line.strip()):
                continue
            if '":' in body or '": ' in body:
                continue
            candidate_body = re.sub(r'(?<!\\)"', r'\\"', body)
            candidate = f'{indent}"{candidate_body}"{tail}'
            # validate the repaired element in isolation (strip trailing comma)
            test = candidate.strip().rstrip(",")
            try:
                json.loads(test)
                lines[i] = candidate
                changed_any = True
            except Exception:
                pass  # don't touch lines we can't safely repair
        if changed_any:
            t = "\n".join(lines)
            actions.append("escaped inner quotes")

    return t, actions


def process(path: Path, write: bool, backup: bool):
    raw = path.read_text(encoding="utf-8-sig")
    data, err = try_load(raw)
    if err is None:
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
            fixed = json.dumps(data[0], ensure_ascii=False, indent=2)
            if write:
                if backup: path.with_suffix(path.suffix + ".bak").write_text(raw, encoding="utf-8")
                path.write_text(fixed, encoding="utf-8")
            return f"FIXED {path.name}: un-nested [[...]] ({len(data[0])} segs)"
        n = len(data) if isinstance(data, list) else "?"
        return f"ok    {path.name}: valid ({n} segs)"
    fixed_text, actions = auto_fix(raw)
    data2, err2 = try_load(fixed_text)
    if err2 is None:
        n = len(data2) if isinstance(data2, list) else "?"
        if write:
            if backup: path.with_suffix(path.suffix + ".bak").write_text(raw, encoding="utf-8")
            path.write_text(json.dumps(data2, ensure_ascii=False, indent=2), encoding="utf-8")
            return f"FIXED {path.name}: {', '.join(actions)} -> {n} segs"
        return f"would fix {path.name}: {', '.join(actions)} -> {n} segs"
    ls = fixed_text.split("\n")
    ctx = []
    for i in range(max(0, err2.lineno - 2), min(len(ls), err2.lineno + 1)):
        mark = " >>" if i + 1 == err2.lineno else "   "
        ctx.append(f"{mark}{i+1}: {ls[i].strip()[:90]}")
    did = f" (applied: {', '.join(actions)})" if actions else ""
    return (f"MANUAL {path.name}: {err2.msg} at line {err2.lineno}{did}\n" + "\n".join(ctx))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv
    backup = "--backup" in sys.argv
    if not args:
        print("usage: python fix_json.py <file-or-folder> [--write] [--backup]")
        sys.exit(1)
    target = Path(args[0])
    files = sorted(target.glob("*.json")) if target.is_dir() else [target]
    files = [f for f in files if not f.name.endswith(".bak")]
    if not files:
        print(f"no .json files in {target}")
        sys.exit(1)
    manual = 0
    for f in files:
        out = process(f, write, backup)
        print(out)
        if out.startswith("MANUAL"):
            manual += 1
    print("\n" + "-" * 40)
    if manual:
        print(f"WARNING: {manual} file(s) need MANUAL fixing (shown above).")
    if not write:
        print("(dry run - nothing written. add --write to apply, --backup for .bak copies.)")
    else:
        print("done.")


if __name__ == "__main__":
    main()
