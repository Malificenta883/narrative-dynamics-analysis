#!/usr/bin/env python3
"""
Точечная замена ТОЛЬКО значения text_variant во всех JSON-файлах прогонов.
Меняет "ETCSL_1.7.4_EN" -> "CDLI_Literary_000343_Inanna_composite_P468903".
Ничего больше не трогает: сегменты, prompt_id, даты, model-блок остаются как есть.

Использование:
  # сухой прогон (только показать, что изменится, файлы не трогаются):
  python fix_text_variant.py ..\\data\\inanna_descent

  # реально записать:
  python fix_text_variant.py ..\\data\\inanna_descent --write
"""

import json
import sys
from pathlib import Path

OLD_PREFIX = "ETCSL"   # ловим ЛЮБОЕ значение, начинающееся с ETCSL (1.4.1, 1.7.4, ...)
NEW = "CDLI_Literary_000343_Inanna_composite_P468903"


def process(path: Path, write: bool) -> str:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as e:
        return f"ERR   {path.name}: не читается ({e})"

    if not isinstance(data, dict):
        return f"skip  {path.name}: не объект с шапкой (голый массив?)"

    tv = data.get("text_variant")
    if tv is None:
        return f"skip  {path.name}: нет поля text_variant"
    if tv == NEW:
        return f"ok    {path.name}: уже CDLI"
    if not str(tv).startswith(OLD_PREFIX):
        return f"skip  {path.name}: text_variant = {tv!r} (не ETCSL*, не трогаю)"

    old_val = tv
    data["text_variant"] = NEW
    if write:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return f"FIXED {path.name}: {old_val} -> {NEW}"
    else:
        return f"would fix {path.name}: {old_val} -> {NEW}"


def main():
    if len(sys.argv) < 2:
        print("укажи файл или папку. напр.: python fix_text_variant.py ..\\data\\inanna_descent [--write]")
        sys.exit(1)
    write = "--write" in sys.argv
    target = Path([a for a in sys.argv[1:] if not a.startswith("--")][0])

    if target.is_dir():
        files = sorted(target.glob("*.json"))
        if not files:
            print(f"В {target} нет .json"); sys.exit(1)
        for fp in files:
            print(process(fp, write))
    elif target.is_file():
        print(process(target, write))
    else:
        print(f"Не найдено: {target}"); sys.exit(1)

    if not write:
        print("\n(сухой прогон — ничего не записано. Добавь --write, чтобы применить.)")


if __name__ == "__main__":
    main()
