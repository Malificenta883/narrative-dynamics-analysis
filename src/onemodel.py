#!/usr/bin/env python3
"""
onemodel.py — Narrative Dynamics Analysis, single-source variant.

Тот же пайплайн, что в analysis.py, но СРАВНИВАЕТ ОДИН ИСТОЧНИК С САМИМ СОБОЙ
поверх его N прогонов (run_log.py: sample_index / слепой ретест), а не разные
источники друг с другом. Это и есть behavioral axis в чистом виде — без
помехи от cross-source сравнения.

Ожидаемые файлы:
  data/<myth>/{source}_run1.json, {source}_run2.json, ...
Имя источника НЕ фиксировано (в отличие от SOURCE_ORDER в analysis.py) —
может быть claude, gpt, mine, gemini или что угодно ещё. Скрипт сам находит
все группы "{source}_run{N}.json" в папке мифа и анализирует каждую группу
независимо (если в одной папке лежат прогоны нескольких источников —
обработает их по очереди, не сравнивая источники между собой).

Использование:
  python onemodel.py --data-dir data
  python onemodel.py --data-dir data --source claude   # ограничиться одним источником
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from analysis import (
    FUNCTIONS_ORDER,
    project_to_lines,
    line_agreement,
    divergence_map,
    freedom_map,
    behavioral_subjectivity,
    load_run_file,
    _norm,
)
import json

RUN_FILE_RE = re.compile(r"^([a-zA-Z0-9]+)_run(\d+)\.json$")
# для --freedom-from: принимаем и {src}.json (run 1), и {src}_run{N}.json
RUN_FILE_RE_ANY = re.compile(r"^([a-zA-Z0-9]+)(?:_run(\d+))?\.json$")

# необязательный фокус-диапазон для построчной freedom (ставится из --free-lines)
FREE_LINES_FOCUS = None

# необязательный реестр лакун/неопределимости (ставится из --lacunae)
LACUNAE = []


def load_lacunae(path):
    """Читает файл реестра лакун: строки 'START-END  TYPE  note'.
    Возвращает список (start, end, type, note). # — комментарии."""
    zones = []
    try:
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 2)
            if not parts:
                continue
            rng = parts[0]
            typ = parts[1].lower() if len(parts) > 1 else "missing"
            note = parts[2] if len(parts) > 2 else ""
            a, b = (rng.split("-", 1) + [rng])[:2] if "-" in rng else (rng, rng)
            try:
                zones.append((int(a), int(b), typ, note))
            except ValueError:
                continue
    except FileNotFoundError:
        print(f"⚠️ --lacunae {path} не найден, пометка лакун пропущена.")
    return zones


def _indeterminate_overlap(a, b, adjacent=1):
    """Пересекает ли [a,b] indeterminate-зону (или примыкает в пределах `adjacent`)?
    Возвращает note зоны или None. 'missing'-зоны игнорируются по решению исследователя."""
    for (s, e, typ, note) in LACUNAE:
        if typ != "indeterminate":
            continue
        if a <= e + adjacent and b >= s - adjacent:
            return note or f"{s}-{e}"
    return None


def find_source_runs(myth_dir: Path) -> Dict[str, List[Tuple[int, List[Dict[str, Any]]]]]:
    """Группирует все {source}_run{N}.json в папке мифа по имени источника.
    Run 1 ДОЛЖЕН называться {source}_run1.json (суффикс _run обязателен для glob)."""
    runs_by_source: Dict[str, List[Tuple[int, List[Dict[str, Any]]]]] = defaultdict(list)
    for file_path in sorted(myth_dir.glob("*_run*.json")):
        m = RUN_FILE_RE.match(file_path.name)
        if not m:
            continue
        src, run_idx = m.group(1), int(m.group(2))
        try:
            meta, segs = load_run_file(file_path)   # понимает голый список И объект-с-шапкой
        except Exception as e:
            print(f"  [!!!] БИТЫЙ ФАЙЛ: {file_path}. Причина: {e}")
            continue
        if not isinstance(segs, list) or not segs:
            continue
        runs_by_source[src].append((run_idx, segs, meta))
    for src in runs_by_source:
        runs_by_source[src].sort(key=lambda x: x[0])
    return runs_by_source


def build_external_freedom(freedom_dir: Path) -> Tuple[Dict[int, float], int]:
    """Cross-source freedom-карта из мульти-источниковой папки (mine/claude/gpt/gemini,
    с прогонами или без). Каждый файл = отдельная метка в позиции; freedom = энтропия
    распределения функций по строке поверх ВСЕХ источников и прогонов вместе.
    Возвращает (freedom_map, число_различимых_источников)."""
    inputs: Dict[str, Dict[int, str]] = {}
    distinct_sources = set()
    for file_path in sorted(freedom_dir.glob("*.json")):
        m = RUN_FILE_RE_ANY.match(file_path.name)
        if not m:
            continue
        try:
            _, segs = load_run_file(file_path)
        except Exception:
            continue
        if not segs:
            continue
        inputs[file_path.stem] = project_to_lines(segs)
        distinct_sources.add(m.group(1))
    return freedom_map(inputs), len(distinct_sources)


def run_pairwise_line_agreement(source: str, runs: List[Tuple[int, List[Dict[str, Any]]]]) -> Dict[int, Dict[int, str]]:
    """Печатает попарное построчное согласие run_i vs run_j одного источника."""
    projections = {idx: project_to_lines(segs) for idx, segs, _ in runs}
    run_ids = [idx for idx, _, _ in runs]

    print("\n" + "=" * 80)
    print(f"🧭 LINE-ALIGNED SELF-AGREEMENT — source: {source.upper()} ({len(runs)} прогонов)")
    print("=" * 80)
    for idx, segs, _ in runs:
        print(f"  run{idx:<3}: {len(segs):>3} сегментов -> {len(projections[idx]):>4} строк покрыто")

    print(f"\nПопарное согласие по строкам (run x run):")
    header = " " * 8 + "".join(f"run{i:<7}" for i in run_ids)
    print(header)
    for i in run_ids:
        row = f"run{i:<5}"
        for j in run_ids:
            if i == j:
                row += f"{'—':>10}"
            else:
                frac, _, _ = line_agreement(projections[i], projections[j])
                row += f"{100*frac:>9.1f}%"
        print(row)

    anchor = run_ids[0]
    # среднее попарное согласие (внедиагональ) — компактный headline-скаляр
    pair_vals = []
    for i in run_ids:
        for j in run_ids:
            if i < j:
                frac, _, _ = line_agreement(projections[i], projections[j])
                pair_vals.append(frac)
    if pair_vals:
        print(f"\nСреднее попарное согласие (внедиагональ): {100*sum(pair_vals)/len(pair_vals):.1f}%")

    for j in run_ids[1:]:
        dm = divergence_map(projections[anchor], projections[j])
        if not dm:
            continue
        print(f"\nКарта расхождений run{anchor} -> run{j} (top-8):")
        for (fa, fb), k in dm.most_common(8):
            print(f"  {k:>3} строк | run{anchor}: {fa:<13} -> run{j}: {fb}")

    return projections


def segment_boundaries(segs):
    """Множество позиций границ сегментов (line_start каждого сегмента + финальный line_end).
    Позволяет сравнивать, ГДЕ источник режет, отдельно от того, КАК называет."""
    bounds = set()
    for s in segs:
        a = s.get("line_start")
        b = s.get("line_end")
        try:
            a = int(a); b = int(b)
        except (TypeError, ValueError):
            continue
        bounds.add(a)       # начало сегмента = граница
    # добавим конец последнего как замыкающую границу
    ends = []
    for s in segs:
        try:
            ends.append(int(s.get("line_end")))
        except (TypeError, ValueError):
            pass
    if ends:
        bounds.add(max(ends) + 1)   # позиция сразу за последней строкой
    return bounds


def boundary_vs_label_report(source, runs, tol=2):
    """Разделяет self-agreement на ДВЕ ортогональные оси:
      1. Boundary agreement — насколько совпадают ГРАНИЦЫ сегментов (где резал),
         с допуском tol строк (граница считается общей, если у другого прогона
         есть граница в пределах ±tol).
      2. Label agreement — на строках, покрытых ОБОИМИ, доля где совпал ярлык
         (это и есть то, что построчный RAW меряет, но здесь мы его отделяем
         от вопроса про границы).
    Мотивация: человек может стабильно резать (границы совпадают), но по-разному
    называть крупные блоки -> низкий построчный RAW при высоком согласии о структуре.
    Это различие теряется в едином RAW-числе."""
    run_ids = [idx for idx, _, _ in runs]
    if len(run_ids) < 2:
        return
    segs_by = {idx: segs for idx, segs, _ in runs}
    bounds_by = {idx: segment_boundaries(segs) for idx, segs, _ in runs}
    proj_by = {idx: project_to_lines(segs) for idx, segs, _ in runs}

    def boundary_f1(ba, bb, tol):
        """F1 совпадения границ с допуском ±tol строк."""
        if not ba and not bb:
            return 1.0
        matched_a = sum(1 for x in ba if any(abs(x - y) <= tol for y in bb))
        matched_b = sum(1 for y in bb if any(abs(x - y) <= tol for x in ba))
        prec = matched_b / len(bb) if bb else 0.0
        rec = matched_a / len(ba) if ba else 0.0
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    def label_agreement(pa, pb):
        """Доля совпадения ярлыков на общих строках."""
        common = set(pa) & set(pb)
        if not common:
            return None
        same = sum(1 for ln in common if pa[ln] == pb[ln])
        return same / len(common)

    b_vals, l_vals = [], []
    for i in run_ids:
        for j in run_ids:
            if i < j:
                b_vals.append(boundary_f1(bounds_by[i], bounds_by[j], tol))
                la = label_agreement(proj_by[i], proj_by[j])
                if la is not None:
                    l_vals.append(la)

    if not b_vals:
        return
    mean_b = sum(b_vals) / len(b_vals)
    mean_l = sum(l_vals) / len(l_vals) if l_vals else float("nan")

    print("\n" + "-" * 80)
    print("📐 ГРАНИЦЫ vs ЯРЛЫКИ (разложение self-agreement)")
    print("-" * 80)
    print(f"  Boundary agreement (где резал, ±{tol} стр.):  {mean_b:.3f}")
    print(f"  Label agreement    (как назвал, на общих стр.): {mean_l:.3f}")
    gap = mean_b - mean_l
    if gap >= 0.20:
        print(f"  -> РАЗРЫВ {gap:+.2f}: режет стабильно, но по-разному НАЗЫВАЕТ.")
        print(f"     Нестабильность в означивании, НЕ в сегментации. Построчный RAW")
        print(f"     занижен длиной переименованных блоков — смотри на эти два числа, не на RAW.")
    elif gap <= -0.20:
        print(f"  -> РАЗРЫВ {gap:+.2f}: называет согласованно, но режет по-разному.")
        print(f"     Нестабильность в сегментации (границы пляшут), не в ярлыках.")
    else:
        print(f"  -> границы и ярлыки расходятся примерно одинаково ({gap:+.2f}).")


def run_self_subjectivity_report(source: str,
                                 projections: Dict[int, Dict[int, str]],
                                 external_freedom: Dict[int, float] = None) -> None:
    """RAW self-consistency источника поверх его прогонов (всегда считается),
    + freedom-взвешенный behavioral — ТОЛЬКО если передана внешняя cross-source freedom.

    Почему так: 'свобода текста' = недоопределённость, видимая в разбросе по РАЗНЫМ
    источникам. Из прогонов ОДНОГО источника её построить нельзя: разброс между его
    прогонами — это его собственная дрожь, не свойство текста. Если взвесить
    самосогласие на freedom, посчитанную по тем же прогонам, фильтр (где прогоны
    расходятся) и величина (где прогоны согласны) взаимоисключающи -> метрика = 0
    тождественно. Поэтому здесь freedom приходит ИЗВНЕ или behavioral помечается n/a."""
    print("\n" + "=" * 80)
    print(f"🧠 SELF-CONSISTENCY / BEHAVIORAL — source: {source.upper()}")
    print("=" * 80)

    run_ids = sorted(projections)
    if len(run_ids) < 2:
        print("  [ЗАГЛУШКА] нужно >=2 прогона одного источника (слепой ретест / N сэмплов).")
        return

    runs_list = [projections[idx] for idx in run_ids]

    # ---- RAW: карта дрожи между прогонами + её дополнение ----
    instab = freedom_map({f"run{idx}": projections[idx] for idx in run_ids})
    if instab:
        n_waver = sum(1 for w in instab.values() if w > 0)
        mean_instab = sum(instab.values()) / len(instab)
        raw_consistency = 1.0 - mean_instab
        print(f"Карта дрожи (между прогонами {source}): {n_waver}/{len(instab)} строк, "
              f"где прогоны расходятся (это дрожь модели, НЕ свобода текста).")
        print(f"RAW self-consistency = {raw_consistency:.3f}  "
              f"(1 - средняя нормир. энтропия меток по строкам; "
              f"1 = модель повторяет себя дословно, ниже = вихляет).")
    else:
        print("  Нет покрытых строк — проверь формат файлов и line_start/line_end.")
        return

    # ---- FREEDOM-WEIGHTED behavioral: только с внешней cross-source freedom ----
    print("\nBehavioral в текст-свободных зонах (нужна внешняя cross-source freedom):")
    if external_freedom is None:
        print("  [n/a] freedom не передана. Из одного источника её не построить (см. docstring).")
        print("        Передай --freedom-from <папка с mine/claude/gpt/gemini того же мифа>.")
    else:
        v = behavioral_subjectivity(runs_list, external_freedom)
        if v is None:
            print("  [n/a] во внешней freedom нет зон >= порога, пересекающихся с этими прогонами.")
        else:
            print(f"  behavioral = {v:.3f}  (само-согласие источника в зонах, "
                  f"свободных ПО ДРУГИМ источникам; высоко = несёт устойчивый инвариант там, "
                  f"где текст реально допускает варианты).")
        report_per_line_freedom(external_freedom, focus_range=FREE_LINES_FOCUS)
        report_pair_instability_vs_freedom(runs_list, external_freedom)


def report_per_line_freedom(freedom, top_n=20, focus_range=None):
    """Печатает freedom ПО СТРОКАМ: топ самых свободных строк (свёрнутый в диапазоны)
    и, если задан focus_range=(a,b), среднюю/поточечную freedom на этом отрезке.
    freedom[ln] в [0..1]: 0 = все источники согласны (текст диктует), 1 = максимум разброса."""
    if not freedom:
        print("\n  (freedom пуста — построчную карту не построить)")
        return

    # --- топ свободных строк, свёрнутый в непрерывные диапазоны ---
    free_lines = sorted(ln for ln, w in freedom.items() if w > 0)
    print(f"\nFreedom по строкам: {len(free_lines)} строк со свободой >0 "
          f"из {len(freedom)} покрытых.")
    if free_lines:
        # группируем подряд идущие строки с близкой freedom в диапазоны
        hi = sorted((ln for ln, w in freedom.items() if w >= 0.5), )
        if hi:
            ranges = []
            start = prev = hi[0]
            for ln in hi[1:]:
                if ln == prev + 1:
                    prev = ln
                else:
                    ranges.append((start, prev)); start = prev = ln
            ranges.append((start, prev))
            print(f"  Высоко-свободные зоны (freedom >= 0.5), {len(ranges)} диапазон(ов):")
            for a, b in ranges[:top_n]:
                avg = sum(freedom[ln] for ln in range(a, b + 1)) / (b - a + 1)
                span = f"{a}" if a == b else f"{a}-{b}"
                flag = _indeterminate_overlap(a, b)
                tag = f"  [LACUNA/indeterminate: {flag} — артефакт, НЕ свобода текста]" if flag else ""
                print(f"    строки {span:<11} ({b-a+1:>3} стр.)  freedom~{avg:.2f}{tag}")
        else:
            print("  (нет строк с freedom >= 0.5 — свобода везде частичная/низкая)")

    # --- фокус на заданном диапазоне (напр. те самые 68 строк negotiation) ---
    if focus_range:
        a, b = focus_range
        vals = [freedom[ln] for ln in range(a, b + 1) if ln in freedom]
        if not vals:
            print(f"\n  [focus {a}-{b}] нет покрытых строк в этом диапазоне.")
            return
        mean = sum(vals) / len(vals)
        n_free = sum(1 for v in vals if v > 0)
        n_hi = sum(1 for v in vals if v >= 0.5)
        print(f"\n  [focus {a}-{b}] средняя freedom={mean:.2f}; "
              f"{n_free}/{len(vals)} строк со свободой >0, {n_hi}/{len(vals)} с freedom>=0.5.")
        if mean >= 0.4:
            print(f"    -> эти строки СВОБОДНЫ по другим источникам: дрожь Opus здесь = отклик на свободу текста.")
        elif mean <= 0.15:
            print(f"    -> здесь другие источники СОГЛАСНЫ: текст диктует, а дрожал только Opus = его нестабильность.")
        else:
            print(f"    -> промежуточно: частичная свобода, однозначно не отнести ни к тексту, ни к модели.")


def report_pair_instability_vs_freedom(runs_list, freedom, min_lines=5):
    """Для КАЖДОЙ пары функций, между которыми источник бистабилен, считает
    среднюю freedom по строкам этой пары -> развод 'свобода текста' vs 'дрожь модели'.

    Строка засчитывается в пару {X,Y} (вариант «б»): два самых частых ярлыка среди
    прогонов на этой строке — X и Y (доминирующая ось разброса), при наличии разброса.
    """
    if not freedom:
        return
    all_lines = set().union(*[set(r) for r in runs_list])
    pair_lines = defaultdict(list)
    for ln in all_lines:
        labels = [r[ln] for r in runs_list if ln in r]
        if len(labels) < 2:
            continue
        c = Counter(labels)
        if len(c) < 2:                      # все прогоны согласны -> строка не бистабильна
            continue
        (f1, _), (f2, _) = c.most_common(2)
        pair_lines[frozenset((f1, f2))].append(ln)

    rows = []
    for pair, lines in pair_lines.items():
        if len(lines) < min_lines:
            continue
        fr = [freedom.get(ln, 0.0) for ln in lines]
        rows.append((len(lines), sum(fr) / len(fr), tuple(sorted(pair)), sorted(lines)))
    rows.sort(reverse=True)

    if not rows:
        print(f"\nРазвилки: нет пар с >= {min_lines} строк (разброс слишком точечный).")
        return

    print(f"\nРАЗВИЛКИ Opus x свобода текста (пары с >= {min_lines} строк):")
    print(f"  {'пара функций':<34}{'строк':>7}{'freedom':>10}   вердикт")
    for n, mean_fr, pair, lines in rows:
        name = f"{pair[0]} <-> {pair[1]}"
        # check if this fork's lines sit on an indeterminate zone
        lac = None
        if lines:
            lac = _indeterminate_overlap(min(lines), max(lines))
        if lac:
            verdict = f"LACUNA/indeterminate ({lac}) — НЕ text_ambiguous, артефакт неопределимости"
        elif mean_fr >= 0.4:
            verdict = "свобода текста (и другие расходятся)"
        elif mean_fr <= 0.15:
            verdict = "дрожь модели (другие согласны)"
        else:
            verdict = "промежуточно"
        print(f"  {name:<34}{n:>7}{mean_fr:>10.2f}   {verdict}")
        print(f"      строки: {_collapse_ranges(lines)}")
    print("  Читать: высокая freedom -> развилка на недоопределённости ТЕКСТА;")
    print("          низкая -> на этих строках источники согласны, дрожал только Opus.")
    print("  Для ручного разбора: открой текст на этих строках и тегируй причину —")
    print("  schema_underdiff / text_ambiguous / model_worldview_error (пример: осёл уже роет = стройка, не переговоры).")


def _collapse_ranges(lines):
    """[1,2,3,7,8,20] -> '1-3, 7-8, 20' для читаемого списка номеров строк."""
    if not lines:
        return "-"
    out, start, prev = [], lines[0], lines[0]
    for ln in lines[1:]:
        if ln == prev + 1:
            prev = ln
        else:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = ln
    out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(out)


def compare_function_sequences_dtw_self(source: str, runs: List[Tuple[int, List[Dict[str, Any]]]]) -> None:
    """DTW по сегментам между прогонами одного источника — granularity-sensitive, для справки."""
    if len(runs) < 2:
        return
    anchor_idx, anchor_segs = runs[0][0], runs[0][1]
    anchor_funcs = [_norm(x.get("function")) for x in anchor_segs]

    print("\n" + "=" * 80)
    print(f"🧬 SHAPE / DTW между прогонами — source: {source.upper()} (granularity-sensitive)")
    print(f"Anchor: run{anchor_idx} (длина: {len(anchor_funcs)})")
    print("=" * 80)

    for idx, segs, _ in runs[1:]:
        target_funcs = [_norm(x.get("function")) for x in segs]
        n, m = len(anchor_funcs), len(target_funcs)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = i
        for j in range(1, m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if anchor_funcs[i - 1] == target_funcs[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        sim = (1 - dp[n][m] / max(n, m)) * 100 if max(n, m) > 0 else 0
        print(f"run{idx:<5} | Длина: {m:>3} | Edit Dist: {dp[n][m]:>3} | Сходство: {sim:>5.1f}%")


# =====================================================================
# Кластеризация прогонов по матрице согласия (моды интерпретации)
# =====================================================================
def _agreement_distance_matrix(run_ids, projections):
    """D[i][j] = 1 - построчное согласие. Та же мера, что в таблице выше."""
    n = len(run_ids)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            frac, _, _ = line_agreement(projections[run_ids[i]], projections[run_ids[j]])
            D[i][j] = D[j][i] = 1.0 - frac
    return D


def _average_linkage(D):
    """Чистый Python: агломеративная кластеризация, average linkage.
    Возвращает (merges, cut_snapshots): merges — порядок слияний с высотой,
    cut_snapshots[k] — членство при разрезе на k кластеров."""
    n = len(D)
    clusters = {i: [i] for i in range(n)}

    def cdist(a, b):
        s = sum(D[x][y] for x in clusters[a] for y in clusters[b])
        c = len(clusters[a]) * len(clusters[b])
        return s / c if c else 0.0

    merges = []
    next_id = n
    cut_snapshots = {len(clusters): {cid: list(m) for cid, m in clusters.items()}}
    while len(clusters) > 1:
        ids = list(clusters)
        best = None
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dd = cdist(ids[i], ids[j])
                if best is None or dd < best[0]:
                    best = (dd, ids[i], ids[j])
        dd, a, b = best
        clusters[next_id] = clusters[a] + clusters[b]
        del clusters[a]
        del clusters[b]
        merges.append((a, b, dd, next_id))
        next_id += 1
        cut_snapshots[len(clusters)] = {cid: list(m) for cid, m in clusters.items()}
    return merges, cut_snapshots


def _labels_at_k(cut_snapshots, k, n):
    snap = cut_snapshots.get(k)
    if snap is None:
        return None
    labels = [0] * n
    for c, (_cid, members) in enumerate(sorted(snap.items())):
        for m in members:
            labels[m] = c
    return labels


def _silhouette(D, labels):
    """Silhouette из готовой матрицы расстояний (без sklearn)."""
    n = len(D)
    members = defaultdict(list)
    for i, lab in enumerate(labels):
        members[lab].append(i)
    if len(members) < 2:
        return None
    sils = []
    for i in range(n):
        own = members[labels[i]]
        if len(own) <= 1:
            sils.append(0.0)
            continue
        a = sum(D[i][j] for j in own if j != i) / (len(own) - 1)
        b = min(sum(D[i][j] for j in members[l]) / len(members[l])
                for l in members if l != labels[i])
        sils.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return sum(sils) / len(sils)


def _print_text_dendrogram(merges, run_ids, n):
    label = {i: f"run{run_ids[i]}" for i in range(n)}
    for a, b, dd, new_id in merges:
        la, lb = label.get(a, f"[{a}]"), label.get(b, f"[{b}]")
        label[new_id] = f"({la}+{lb})"
        print(f"  d={dd:.3f}  {la}  +  {lb}")


def _maybe_png_dendrogram(D, run_ids, source):
    try:
        import numpy as np
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (scipy/matplotlib недоступны — png-дерево пропущено, текстового достаточно)")
        return
    Z = linkage(squareform(np.array(D), checks=False), method="average")
    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(Z, labels=[f"run{r}" for r in run_ids], ax=ax)
    ax.set_title(f"Run clustering — {source} (dist = 1 - line agreement)")
    ax.set_ylabel("distance")
    out = Path(f"dendrogram_{source}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  [png] дендрограмма сохранена: {out}")


def cluster_runs(source, runs, projections):
    """Группирует прогомы по взаимному согласию и выводит ТРИ оси сразу
    (содержание / длина / дата), чтобы отделить моды интерпретации от
    эффекта гранулярности и от даты."""
    run_ids = sorted(projections)
    n = len(run_ids)
    if n < 3:
        return
    print("\n" + "=" * 80)
    print(f"🌳 КЛАСТЕРИЗАЦИЯ ПРОГОНОВ — source: {source.upper()} (по матрице согласия)")
    print("=" * 80)

    D = _agreement_distance_matrix(run_ids, projections)
    merges, cut_snapshots = _average_linkage(D)

    best_k, best_s = None, None
    print("Silhouette по числу кластеров k (ориентир — где максимум):")
    for k in range(2, min(5, n - 1) + 1):
        labels = _labels_at_k(cut_snapshots, k, n)
        s = _silhouette(D, labels)
        if s is None:
            continue
        print(f"  k={k}: silhouette={s:+.3f}")
        if best_s is None or s > best_s:
            best_s, best_k = s, k
    print(f"  -> рекомендованный k={best_k} (silhouette={best_s:+.3f}); "
          f"на N={n} это ОРИЕНТИР, не истина (silhouette шумноват при малом N).")

    seg_count = {idx: len(segs) for idx, segs, _ in runs}
    date = {idx: meta.get("annotated_date", "?") for idx, _, meta in runs}
    labels = _labels_at_k(cut_snapshots, best_k, n)

    print(f"\nТри оси разом при k={best_k}:")
    print(f"  {'run':<7}{'кластер':<10}{'сегментов':<12}{'дата'}")
    for i in sorted(range(n), key=lambda i: (labels[i], run_ids[i])):
        rid = run_ids[i]
        print(f"  run{rid:<4}{labels[i]:<10}{seg_count.get(rid, '?'):<12}{date.get(rid, '?')}")

    print("\nЧитать так:")
    print("  • кластеры по СОДЕРЖАНИЮ (длина и дата перемешаны внутри) -> моды интерпретации (то, что искали).")
    print("  • кластеры совпадают с ДЛИНОЙ (короткие vs длинные) -> кластеризуется нарезка, не смысл.")
    print("  • кластеры совпадают с ДАТОЙ (29 vs 30) -> артефакт; чекпойнт не менялся, так что НЕ ожидается.")

    print("\nДендрограмма (текст; малое d рано = тесно, большое d поздно = далеко):")
    _print_text_dendrogram(merges, run_ids, n)
    _maybe_png_dendrogram(D, run_ids, source)


def main(data_dir: Path, only_source: str = None, freedom_from: str = None) -> None:
    if not data_dir.exists():
        print(f"Директория {data_dir} не найдена.")
        return
    myth_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not myth_dirs:
        print(f"В папке {data_dir} нет подпапок с мифами.")
        return

    external_freedom = None
    if freedom_from:
        fdir = Path(freedom_from)
        if not fdir.exists():
            print(f"⚠️ --freedom-from {fdir} не найдена, behavioral пойдёт как n/a.")
        else:
            external_freedom, n_src = build_external_freedom(fdir)
            note = "" if n_src >= 3 else "  (ВНИМАНИЕ: <3 источников — freedom недооценена)"
            print(f"[freedom] cross-source freedom из {fdir.name}: "
                  f"{n_src} источник(ов){note}")
            print(f"[freedom] ВНИМАНИЕ: ось строк должна быть тем же мифом/text_variant, "
                  f"что и прогоны ниже — иначе номера строк не совпадут.")

    for myth_dir in myth_dirs:
        runs_by_source = find_source_runs(myth_dir)
        if only_source:
            runs_by_source = {s: r for s, r in runs_by_source.items() if s == only_source}
        if not runs_by_source:
            continue

        print("\n" + "░" * 80)
        print(f"🚀 SINGLE-SOURCE АНАЛИЗ: {myth_dir.name.upper()}")
        print("░" * 80)

        for source, runs in runs_by_source.items():
            if len(runs) < 2:
                print(f"\n⚠️ {source}: найден только 1 прогон ({len(runs)}), behavioral axis недоступна.")
                continue
            projections = run_pairwise_line_agreement(source, runs)
            boundary_vs_label_report(source, runs)
            run_self_subjectivity_report(source, projections, external_freedom=external_freedom)
            cluster_runs(source, runs, projections)
            compare_function_sequences_dtw_self(source, runs)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=str(Path(__file__).parent / "data"))
    p.add_argument("--source", type=str, default=None, help="ограничиться одним источником (напр. claude)")
    p.add_argument("--freedom-from", type=str, default=None,
                   help="папка с мульти-источниковой разметкой того же мифа для cross-source freedom")
    p.add_argument("--free-lines", type=str, default=None,
                   help="диапазон строк для точечной проверки freedom, напр. 380-412")
    p.add_argument("--lacunae", type=str, default=None,
                   help="файл реестра лакун/неопределимости (START-END TYPE note); "
                        "зоны type=indeterminate помечаются как артефакт, не text_ambiguous")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.free_lines:
        try:
            a, b = args.free_lines.split("-")
            FREE_LINES_FOCUS = (int(a), int(b))
        except Exception:
            print(f"⚠️ --free-lines '{args.free_lines}' не разобран, ожидается формат A-B (напр. 380-412)")
    if args.lacunae:
        LACUNAE = load_lacunae(args.lacunae)
        n_ind = sum(1 for z in LACUNAE if z[2] == "indeterminate")
        print(f"[lacunae] загружено {len(LACUNAE)} зон ({n_ind} indeterminate) из {args.lacunae}")
    main(Path(args.data_dir), args.source, args.freedom_from)
