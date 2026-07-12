#!/usr/bin/env python3
"""
wrap_runs.py — надевает каноническую шапку на сырые ответы моделей.

Модель по новому промпту возвращает ГОЛЫЙ массив сегментов [ {...}, {...} ].
Этот скрипт оборачивает его в полную шапку (substrate-aware) так, чтобы
source/run_index брались из ИМЕНИ ФАЙЛА, а различающиеся поля — из аргументов.
Модель к шапке не прикасается => три класса ошибок (перепутанный source,
незаполненная версия, забытая дата) невозможны by design.

Имя входного файла: {source}_run{N}.json   (например opus_run1.json, gemini_run3.json)
  - source     <- префикс до _run
  - run_index  <- число после _run

Ветки шапки:
  - human  (source == --human-source, по умолчанию "mine"): поле "blind", без "model"
  - model  (все прочие): блок "model" с версией/temperature/prompt_id/sample_index,
           поле "blind" опускается (его смысл несёт fresh_session + anonymous UI)

Использование:
  # обернуть все сырые файлы моделей в папке, единым батчем:
  python wrap_runs.py data/inanna_descent \
      --corpus inanna_descent --text-variant ETCSL_1.7.4_EN \
      --model-version "gemini_ultra (anonymous UI, 2026-07-01)" \
      --date 2026-07-01 --prompt-id seg_v4

  # человеческий прогон:
  python wrap_runs.py data/inanna_descent/mine_run1.json \
      --corpus inanna_descent --text-variant ETCSL_1.7.4_EN \
      --human-source mine --date 2026-06-15 --blind false --prompt-id seg_v3

Скрипт НЕ трогает файлы, у которых top-level уже объект с "segments"
(значит шапка уже надета — например твои готовые Opus). Только голые массивы.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

RUN_FILE_RE = re.compile(r"^([a-zA-Z0-9]+)_run(\d+)\.json$")


def build_header(source, run_index, args):
    """Собирает шапку по substrate-ветке. Порядок ключей — как в твоём образце."""
    is_human = (source == args.human_source)
    run_id = f"{source}_{args.corpus}_{args.date}_r{run_index}"  # уникален на прогон
    header = {
        "run_id": run_id,
        "source": source,
        "corpus": args.corpus,
        "run_index": run_index,
        "annotated_date": args.date,
    }
    if is_human:
        # human-ветка: blind осмыслен
        if args.blind is not None:
            header["blind"] = (args.blind.lower() == "true")
        header["prior_run_id"] = None
        header["text_variant"] = args.text_variant
        if args.text_id:
            header["text_id"] = args.text_id
        if args.composite_id:
            header["composite_id"] = args.composite_id
        header["prompt_id"] = args.prompt_id
    else:
        # model-ветка: blind опускаем, несём fresh_session + блок model
        header["fresh_session"] = True
        header["prior_run_id"] = None
        header["text_variant"] = args.text_variant
        if args.text_id:
            header["text_id"] = args.text_id
        if args.composite_id:
            header["composite_id"] = args.composite_id
        header["model"] = {
            "model_version": args.model_version or "UNKNOWN (set --model-version)",
            "temperature": None,               # anonymous UI: не задаётся
            "prompt_id": args.prompt_id,
            "sample_index": run_index,
        }
    return header


def wrap_one(path: Path, args) -> str:
    m = RUN_FILE_RE.match(path.name)
    if not m:
        return f"skip  {path.name}: имя не в формате {{source}}_run{{N}}.json"
    source, run_index = m.group(1), int(m.group(2))

    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    # already-wrapped file (header object with segments)
    if isinstance(data, dict) and "segments" in data:
        if not args.fix_header:
            return f"skip  {path.name}: уже с шапкой (top-level объект с segments)"
        # FIX MODE: rebuild the header with corrected metadata, keep segments as-is.
        segments = data["segments"]
        old = data
        header = build_header(source, run_index, args)
        # preserve fields that legitimately came from the old header, if present
        for keep in ("blind",):
            if keep in old and keep not in header:
                header[keep] = old[keep]
        header["segments"] = segments
        out = path if args.in_place else path.with_name(f"{path.stem}_fixed.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(header, f, ensure_ascii=False, indent=2)
        old_tv = old.get("text_variant", "(none)")
        return (f"fixed {path.name}: text_variant {old_tv} -> {args.text_variant}"
                + ("" if args.in_place else f"  -> {out.name}"))

    if not isinstance(data, list):
        return f"ERR   {path.name}: ожидался голый массив сегментов, получен {type(data).__name__}"

    header = build_header(source, run_index, args)
    # normalize line_start/line_end to int where possible (models sometimes emit "7")
    coerced = 0
    for s in data:
        for k in ("line_start", "line_end"):
            v = s.get(k)
            if isinstance(v, str) and v.strip().lstrip("-").isdigit():
                s[k] = int(v.strip())
                coerced += 1
    header["segments"] = data

    def _as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    covered = 0
    for s in data:
        a, b = _as_int(s.get("line_start")), _as_int(s.get("line_end"))
        if a is not None and b is not None:
            covered += b - a + 1

    out = path if args.in_place else path.with_name(f"{path.stem}_wrapped.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(header, f, ensure_ascii=False, indent=2)
    return f"ok    {path.name} -> {out.name}  [{len(data)} сегм., ~{covered} строк покрыто]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target", help="файл {source}_run{N}.json или папка с ними")
    p.add_argument("--corpus", required=True)
    p.add_argument("--text-variant", default="CDLI_Literary_000343_Inanna_composite_P468903",
                   help="идентификатор текста-варианта. По умолчанию CDLI-композит Inanna's Descent "
                        "(P468903). НЕ используй ETCSL для этого мифа — линиация не совпадает.")
    p.add_argument("--text-id", default="",
                   help="стабильный идентификатор текста (напр. CDLI P-номер P468903 для Descent). "
                        "Пусто по умолчанию — для ETCSL-текстов (Enki) обычно не нужен.")
    p.add_argument("--composite-id", default="",
                   help="composite/Literary номер (напр. Q000343 для Descent). Пусто по умолчанию.")
    p.add_argument("--date", required=True, help="annotated_date, напр. 2026-07-01")
    p.add_argument("--prompt-id", required=True, help="напр. seg_v4 (новый промпт) или seg_v3")
    p.add_argument("--model-version", default=None,
                   help='для моделей, напр. "gemini_ultra (anonymous UI, 2026-07-01)"')
    p.add_argument("--human-source", default="mine",
                   help="какой source считать человеческим (ветка с blind). По умолчанию mine")
    p.add_argument("--blind", default=None, help="для human-прогона: true/false")
    p.add_argument("--in-place", action="store_true",
                   help="перезаписать исходный файл (иначе пишет *_wrapped.json)")
    p.add_argument("--fix-header", action="store_true",
                   help="ПЕРЕОБЕРНУТЬ уже обёрнутые файлы: пересобрать шапку с новыми "
                        "метаданными (напр. исправить text_variant), сегменты не трогая. "
                        "Иначе такие файлы пропускаются.")
    args = p.parse_args()

    tgt = Path(args.target)

    # --- guard 1: folder name vs --corpus mismatch ---
    # if the target folder is named after a myth and it differs from --corpus, warn.
    folder = tgt.name if tgt.is_dir() else tgt.parent.name
    if folder and folder != args.corpus and not folder.startswith("."):
        # only trip if the folder name looks like a corpus label (letters/underscores)
        if re.fullmatch(r'[a-z0-9_]+', folder or ""):
            print(f"⚠️  Папка называется '{folder}', но --corpus '{args.corpus}'. "
                  f"Похоже, ты в папке одного мифа, а метку ставишь другого.")
            resp = input("    Продолжить всё равно? [y/N] ").strip().lower()
            if resp not in ("y", "yes", "д", "да"):
                print("    Отменено. Проверь --corpus или папку."); sys.exit(1)

    # --- guard 2: corpus vs text_variant sanity (descent=CDLI, enki=ETCSL) ---
    tv = (args.text_variant or "").upper()
    corpus_l = args.corpus.lower()
    suspicious = (
        ("descent" in corpus_l and "ETCSL" in tv) or
        ("enki" in corpus_l and "CDLI" in tv)
    )
    if suspicious:
        print(f"⚠️  --corpus '{args.corpus}' с --text-variant '{args.text_variant}' выглядит противоречиво.")
        print("    Напоминание: Inanna's Descent = CDLI (построчно); Inanna and Enki = ETCSL (блочно).")
        resp = input("    Точно так и надо? [y/N] ").strip().lower()
        if resp not in ("y", "yes", "д", "да"):
            print("    Отменено. Проверь --text-variant."); sys.exit(1)

    if tgt.is_dir():
        files = sorted(tgt.glob("*_run*.json"))
        if not files:
            print(f"В {tgt} нет файлов вида *_run*.json"); sys.exit(1)
        for fp in files:
            print(wrap_one(fp, args))
    elif tgt.is_file():
        print(wrap_one(tgt, args))
    else:
        print(f"Не найдено: {tgt}"); sys.exit(1)


if __name__ == "__main__":
    main()
