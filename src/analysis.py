#!/usr/bin/env python3
"""
Narrative Dynamics Analysis — Meta-Pipeline (v2)
Прогоняет анализ по всем мифам (папкам) внутри директории data/.

ЧТО ИЗМЕНИЛОСЬ ОТ v1 (важно):
  1. ПОЧИНЕНО сравнение функций. Старый compare_function_and_transitions
     сравнивал anchor[i] с other[i] ПО ИНДЕКСУ СЕГМЕНТА. Но у источников разное
     число сегментов с разными границами (mine=19, claude=46), поэтому сегмент i
     у разных источников — это РАЗНЫЕ строки текста (mine seg6 = строки 116-164,
     claude seg6 = строки 68-77, ноль общих). Индексное сравнение мерило шум.
     Теперь функции проецируются на ОБЩУЮ ОСЬ СТРОК (каждая строка наследует
     функцию своего сегмента), и сравнение идёт построчно. Это убирает и
     granularity-конфаунд: длинный/короткий источник больше не штрафуется.
  2. DTW по сегментам ОСТАВЛЕН, но честно помечен как "shape, granularity-sensitive".
     Построчное согласие — это granularity-free измерение, смотри на него.
  3. ДОБАВЛЕНЫ оси субъектности: freedom_map (cross-source, осмысленна при >=3 источниках),
     imputed_subjectivity (отклонение от консенсуса; считается при >=3 источниках) и
     behavioral_subjectivity (устойчивость инварианта поверх N прогонов; N-устойчивая,
     доля по моде, взвешенная по CROSS-SOURCE freedom).
  4. Лоадер толерантен к формату: голый список ИЛИ объект-с-шапкой {run_id,...,segments};
     номера строк читаются из line_start/line_end (фолбэк — числа в text_en).
     Single-source вариант (один источник vs сам себя поверх N прогонов) — в onemodel.py.

Ожидаемая структура:
  data/<myth>/{mine,claude,gpt,gemini}.json  (+ опц. {source}_run{N}.json для N прогонов)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message=".*n_neighbors is larger than the dataset size.*")
warnings.filterwarnings("ignore", message=".*All labels are -1.*")

HAS_PLOT = False
plt = None
try:
    import matplotlib.pyplot as _plt  # type: ignore
    plt = _plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False
    plt = None

try:
    import networkx as nx  # type: ignore
except Exception as e:
    raise ImportError("networkx is required for this script.") from e

# -----------------------------
# Configuration
# -----------------------------
FUNCTIONS_ORDER = [
    "preparation", "contact", "exchange", "disruption",
    "negotiation", "stabilization", "return"
]
SOURCE_ORDER = ["mine", "claude", "gpt", "gemini"]


# -----------------------------
# IO & Meta-Cycle Loader
# -----------------------------
@dataclass(frozen=True)
class DatasetBundle:
    texts: List[Dict[str, Any]]
    sources: List[str]
    ids: List[str]
    per_source: Dict[str, List[Dict[str, Any]]]


RUN_FILE_RE = re.compile(r"^([a-zA-Z]+)(?:_run(\d+))?\.json$")


def load_run_file(path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Толерантный лоадер одного файла разметки. Возвращает (meta, segments).
    Понимает ОБА формата:
      (a) голый список   [ {segment}, ... ]                      (старый mine.json)
      (b) объект-с-шапкой { "run_id":..., "segments":[...] }     (новый формат с run_log)
    Это та же логика, что в line_alignment.load_run — держим скрипты на одном лоадере.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"source": Path(path).stem, "run_index": 1}, data
    if isinstance(data, dict):
        meta = {k: v for k, v in data.items() if k != "segments"}
        return meta, data.get("segments", [])
    raise ValueError(f"{path}: неожиданный JSON top-level type ({type(data).__name__})")


def load_myth_datasets(myth_dir: Path) -> Tuple[DatasetBundle, Dict[str, List[List[Dict[str, Any]]]]]:
    """
    Поддерживает N прогонов на источник (run_log.py: sample_index / blind retest).
    Файлы: {source}.json (run 1, он же anchor для остального пайплайна) и
    {source}_run{N}.json (доп. прогоны того же источника — нужны для behavioral axis).
    Возвращает (bundle с anchor-прогоном на источник, all_runs[source] = список прогонов).
    """
    runs_by_source: Dict[str, List[Tuple[int, List[Dict[str, Any]]]]] = defaultdict(list)

    for file_path in sorted(myth_dir.glob("*.json")):
        m = RUN_FILE_RE.match(file_path.name)
        if not m:
            continue
        src, run_idx_str = m.group(1), m.group(2)
        if src not in SOURCE_ORDER:
            continue
        try:
            meta, segs = load_run_file(file_path)
        except Exception as e:
            print(f"  [!!!] БИТЫЙ ФАЙЛ: {file_path}. Причина: {e}")
            continue
        if not isinstance(segs, list) or not segs:
            continue
        # Индекс прогона: имя файла главнее (файлы зовутся _run{N}), затем шапка, затем 1.
        if run_idx_str:
            run_idx = int(run_idx_str)
        else:
            ri = meta.get("run_index")
            run_idx = int(ri) if ri is not None else 1
        runs_by_source[src].append((run_idx, segs))

    per_source: Dict[str, List[Dict[str, Any]]] = {}
    for src, runs in runs_by_source.items():
        runs.sort(key=lambda x: x[0])
        per_source[src] = runs[0][1]  # anchor run (run_idx==1) — используется во всём остальном пайплайне

    texts, sources, ids = [], [], []
    for src in SOURCE_ORDER:
        if src not in per_source:
            continue
        segs = per_source[src]
        texts.extend(segs)
        sources.extend([src] * len(segs))
        ids.extend([f"{src}_{i+1}" for i in range(len(segs))])

    bundle = DatasetBundle(texts=texts, sources=sources, ids=ids, per_source=per_source)
    all_runs = {src: [segs for _, segs in runs] for src, runs in runs_by_source.items()}
    return bundle, all_runs


def _norm(x: Any) -> str:
    return "" if x is None else str(x).strip().lower()


# =====================================================================
# NEW: Line-axis projection (the fix)
# =====================================================================
def parse_line_range(text_en: str) -> Optional[Tuple[int, int]]:
    """Вытаскивает (start, end) из хвоста text_en. Понимает '-' и '–', суффиксы '13h','305c'."""
    m = re.search(r"(\d+)\s*[-\u2013]\s*(\d+)", str(text_en))
    return (int(m.group(1)), int(m.group(2))) if m else None


def segment_range(seg: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Канон — явные line_start/line_end; фолбэк — парсинг чисел из text_en."""
    ls, le = seg.get("line_start"), seg.get("line_end")
    if ls is not None and le is not None:
        return int(ls), int(le)
    return parse_line_range(seg.get("text_en", ""))


def project_to_lines(segments: List[Dict[str, Any]]) -> Dict[int, str]:
    """Список сегментов -> {номер_строки: function}. При перекрытии побеждает поздний сегмент."""
    line2fn: Dict[int, str] = {}
    for s in segments:
        r = segment_range(s)
        if not r:
            continue
        fn = _norm(s.get("function"))
        a, b = (r[0], r[1]) if r[0] <= r[1] else (r[1], r[0])
        for ln in range(a, b + 1):
            line2fn[ln] = fn
    return line2fn


def line_agreement(proj_a: Dict[int, str], proj_b: Dict[int, str]) -> Tuple[float, int, int]:
    """Доля общих строк, где функции совпали."""
    common = sorted(set(proj_a) & set(proj_b))
    if not common:
        return 0.0, 0, 0
    agree = sum(1 for ln in common if proj_a[ln] == proj_b[ln])
    return agree / len(common), agree, len(common)


def divergence_map(proj_a: Dict[int, str], proj_b: Dict[int, str]) -> Counter:
    """Counter (func_a, func_b) по строкам, где источники разошлись."""
    common = set(proj_a) & set(proj_b)
    return Counter((proj_a[ln], proj_b[ln]) for ln in common if proj_a[ln] != proj_b[ln])


def freedom_map(projections: Dict[str, Dict[int, str]]) -> Dict[int, float]:
    """
    Податливость текста по строкам. projections = {source: {line: function}}.
    freedom[line] = нормированная энтропия функций, присвоенных этой строке разными
    источниками. 0 = все согласны (текст диктует), 1 = максимальный разброс.
    ОСМЫСЛЕННО при >=3 источниках; с 2 это просто согласны/нет.
    """
    if not projections:
        return {}
    all_lines = set().union(*[set(p) for p in projections.values()])
    norm = math.log2(len(FUNCTIONS_ORDER))
    out: Dict[int, float] = {}
    for ln in all_lines:
        labels = [p[ln] for p in projections.values() if ln in p]
        if len(labels) < 2:
            out[ln] = 0.0
            continue
        counts = Counter(labels)
        total = sum(counts.values())
        H = -sum((n / total) * math.log2(n / total) for n in counts.values())
        out[ln] = H / norm if norm > 0 else 0.0
    return out


def imputed_subjectivity(source: str,
                         projections: Dict[str, Dict[int, str]],
                         freedom: Dict[int, float],
                         freedom_threshold: float = 0.3) -> Optional[float]:
    """
    ЗАГЛУШКА до появления >=3 источников.
    Идея: насколько `source` отклоняется от консенсуса ОСТАЛЬНЫХ источников
    в зонах свободы текста (взвешенно по freedom). Возвращает долю free-зон,
    где source != мода остальных.
    Пока источников < 3 — консенсус не из чего строить, возвращаем None.
    """
    others = {s: p for s, p in projections.items() if s != source}
    if len(others) < 2:
        return None  # STUB: нужно >=3 источника всего (source + 2 других)

    me = projections[source]
    num = den = 0.0
    for ln, fn in me.items():
        others_here = [p[ln] for p in others.values() if ln in p]
        if not others_here:
            continue
        w = freedom.get(ln, 0.0)
        if w < freedom_threshold:
            continue
        consensus = Counter(others_here).most_common(1)[0][0]
        den += w
        if fn != consensus:
            num += w
    return num / den if den > 0 else 0.0


def behavioral_subjectivity(runs: List[Dict[int, str]],
                            freedom: Dict[int, float],
                            freedom_threshold: float = 0.3,
                            min_runs: int = 2) -> Optional[float]:
    """
    Поведенческая ось: устойчивость инварианта источника поверх N его прогонов
    (для человека — слепой ретест), В ЗОНАХ СВОБОДЫ текста, взвешенно по freedom.

    ВАЖНО: `freedom` обязана приходить из CROSS-SOURCE пула (разброс по РАЗНЫМ
    источникам/прогонам вместе). Если подать freedom, посчитанную по прогонам
    того же одного источника, метрика вырождается (вес = измеряемая величина).
    Для одного источника без других — см. onemodel.py: там считается raw
    self-consistency, а сюда freedom передаётся ИЗВНЕ.

    На каждой свободной строке self-consistency = доля прогонов, совпавших с модой
    (1.0 = единогласие). N-устойчиво: один шальной прогон не обнуляет строку.
    Строка учитывается, если присутствует хотя бы в min_runs прогонах.
    Возвращает None, если ни одной квалифицирующей строки нет (а не ложный 0.0).
    """
    if len(runs) < 2:
        return None
    all_lines = set().union(*[set(r) for r in runs])
    num = den = 0.0
    for ln in all_lines:
        labels = [r[ln] for r in runs if ln in r]
        if len(labels) < min_runs:
            continue
        w = freedom.get(ln, 0.0)
        if w < freedom_threshold:
            continue
        modal = Counter(labels).most_common(1)[0][1]
        den += w
        num += w * (modal / len(labels))
    return num / den if den > 0 else None


# -----------------------------
# NEW: Line-aligned reports
# -----------------------------
def run_line_alignment_report(per_source: Dict[str, List[Dict[str, Any]]], anchor: str = "mine") -> Dict[str, Dict[int, str]]:
    """Строит проекции на строки, печатает согласие по строкам и карту расхождений vs anchor."""
    projections = {src: project_to_lines(per_source[src]) for src in SOURCE_ORDER if src in per_source}

    print("\n" + "=" * 80)
    print("🧭 LINE-ALIGNED COMPARISON (общая ось строк, granularity-free)")
    print("Сравнение функций по строкам, а не по индексу сегмента.")
    print("=" * 80)
    for src in SOURCE_ORDER:
        if src in projections:
            print(f"  {src:<8}: {len(per_source[src]):>3} сегментов -> {len(projections[src]):>4} строк покрыто")

    if anchor not in projections:
        return projections

    print(f"\nСогласие функций по строкам vs {anchor.upper()}:")
    print(f"{'SRC':<8} | {'agree%':>7} | {'lines':>10}")
    print("-" * 32)
    for src in SOURCE_ORDER:
        if src not in projections or src == anchor:
            continue
        frac, ag, tot = line_agreement(projections[anchor], projections[src])
        print(f"{src:<8} | {100*frac:>6.1f}% | {ag:>4}/{tot:<5}")

    for src in SOURCE_ORDER:
        if src not in projections or src == anchor:
            continue
        dm = divergence_map(projections[anchor], projections[src])
        if not dm:
            continue
        print(f"\nКарта расхождений {anchor} -> {src} (по строкам, top-8):")
        for (fa, fb), k in dm.most_common(8):
            print(f"  {k:>3} строк | {anchor}: {fa:<13} -> {src}: {fb}")

    return projections


def run_subjectivity_axes_report(projections: Dict[str, Dict[int, str]],
                                 runs_by_source: Optional[Dict[str, List[Dict[int, str]]]] = None) -> None:
    """Печатает оси субъектности. freedom считается по ВСЕМ прогонам ВСЕХ источников
    (run_log.py: каждый run_id/sample_index — отдельная метка в позиции), а не только
    по одному anchor-прогону на источник — иначе свобода текста недооценивается там,
    где разброс виден только между прогонами одного источника."""
    print("\n" + "=" * 80)
    print("🧠 SUBJECTIVITY AXES (freedom / imputed / behavioral)")
    print("=" * 80)

    n_src = len(projections)

    freedom_inputs: Dict[str, Dict[int, str]] = dict(projections)
    if runs_by_source:
        for src, runs in runs_by_source.items():
            for i, run_proj in enumerate(runs):
                freedom_inputs[f"{src}__run{i+1}"] = run_proj
    free = freedom_map(freedom_inputs)

    if n_src >= 2:
        free_lines = [ln for ln, w in free.items() if w > 0]
        if free_lines:
            mean_free = sum(free.values()) / len(free)
            print(f"Freedom map: {len(free_lines)} строк с ненулевой свободой, "
                  f"средняя свобода = {mean_free:.3f} (0=текст диктует, 1=макс. разброс)")
        else:
            print("Freedom map: разброса нет (мало источников).")

    print("\nImputed subjectivity (отклонение от консенсуса в зонах свободы):")
    if n_src < 3:
        print("  [ЗАГЛУШКА] нужно >=3 источника. Добавь gpt.json, gemini.json.")
    else:
        for src in SOURCE_ORDER:
            if src not in projections:
                continue
            v = imputed_subjectivity(src, projections, free)
            print(f"  {src:<8}: {v:.3f}" if v is not None else f"  {src:<8}: n/a")

    print("\nBehavioral subjectivity (устойчивость инварианта поверх N прогонов):")
    if not runs_by_source or all(len(v) < 2 for v in runs_by_source.values()):
        print("  [ЗАГЛУШКА] нужен слепой ретест / N прогонов одного источника (>=2 разметки).")
    else:
        for src, runs in runs_by_source.items():
            v = behavioral_subjectivity(runs, free)
            print(f"  {src:<8}: {v:.3f}" if v is not None else f"  {src:<8}: n/a (нужно >=2 прогона)")


# -----------------------------
# Sequences & Shape (DTW) — granularity-sensitive, kept for reference
# -----------------------------
def compare_function_sequences_dtw(per_source: Dict[str, List[Dict[str, Any]]], anchor_source: str = "mine") -> None:
    if anchor_source not in per_source:
        return
    anchor_funcs = [_norm(x.get("function")) for x in per_source[anchor_source]]

    print("\n" + "=" * 80)
    print("🧬 SHAPE / DTW по СЕГМЕНТАМ (granularity-sensitive — для справки)")
    print("Внимание: чувствительно к числу сегментов. Granularity-free метрика — выше (line-aligned).")
    print(f"Anchor: {anchor_source} (длина: {len(anchor_funcs)})")
    print("=" * 80)

    for src in SOURCE_ORDER:
        if src not in per_source or src == anchor_source:
            continue
        target_funcs = [_norm(x.get("function")) for x in per_source[src]]
        n, m = len(anchor_funcs), len(target_funcs)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = i
        for j in range(1, m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if anchor_funcs[i-1] == target_funcs[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
        sim = (1 - dp[n][m] / max(n, m)) * 100 if max(n, m) > 0 else 0
        print(f"{src.upper():<8} | Длина: {m:>3} | Edit Dist: {dp[n][m]:>3} | Сходство: {sim:>5.1f}%")


# -----------------------------
# Matrices & Entropy & JSD  (без изменений в логике)
# -----------------------------
Matrix = Dict[str, Dict[str, float]]
Matrices = Dict[str, Matrix]


def build_transition_matrices(texts: List[Dict[str, Any]], sources: List[str], normalize: bool = True) -> Tuple[Matrices, List[str]]:
    matrices: Matrices = {src: {fr: {to: 0.0 for to in FUNCTIONS_ORDER} for fr in FUNCTIONS_ORDER} for src in set(sources)}
    for item, src in zip(texts, sources):
        fr, to = _norm(item.get("transition_from")), _norm(item.get("transition_to"))
        if fr in FUNCTIONS_ORDER and to in FUNCTIONS_ORDER:
            matrices[src][fr][to] += 1.0
    if normalize:
        for src, mat in matrices.items():
            for fr in FUNCTIONS_ORDER:
                row_sum = sum(mat[fr].values())
                if row_sum > 0:
                    for to in FUNCTIONS_ORDER:
                        mat[fr][to] /= row_sum
    return matrices, FUNCTIONS_ORDER


def row_entropy(row_probs: Dict[str, float]) -> float:
    H = 0.0
    for p in row_probs.values():
        p = float(p)
        if p <= 0:
            continue
        H -= p * math.log(p, 2)
    return max(H, 0.0)


def transition_entropy(matrix: Matrix, states: List[str], ignore_empty_rows: bool = True) -> Tuple[Dict[str, float], float, float]:
    entropy_by_state: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    for fr in states:
        row = matrix[fr]
        row_sum = sum(float(v) for v in row.values())
        if row_sum <= 0:
            entropy_by_state[fr] = 0.0
            weights[fr] = 0.0
            continue
        entropy_by_state[fr] = row_entropy(row)
        weights[fr] = row_sum
    vals = [entropy_by_state[fr] for fr in states if not (ignore_empty_rows and weights[fr] <= 0)]
    mean_entropy = sum(vals) / len(vals) if vals else 0.0
    total_w = sum(weights.values())
    weighted_entropy = (sum(entropy_by_state[fr] * weights[fr] for fr in states) / total_w) if total_w > 0 else 0.0
    return entropy_by_state, mean_entropy, weighted_entropy


def run_transition_entropy_report(texts: List[Dict[str, Any]], sources: List[str]) -> None:
    matrices, states = build_transition_matrices(texts, sources, normalize=True)
    order = [s for s in SOURCE_ORDER if s in matrices]
    print("\n" + "=" * 80)
    print("🌪️ TRANSITION ENTROPY REPORT (bits)")
    print("H=0 deterministic; выше — больше ветвления. ВНИМАНИЕ: H=0 часто = состояние")
    print("посещено 1 раз (разреженность), а не настоящий детерминизм.")
    print("=" * 80)
    print("\nOverall entropy (mean over non-empty rows):")
    print(f"{'Source':<10} | {'Mean H':>8} | {'Weighted H':>10}")
    print("-" * 36)
    per_source: Dict[str, Tuple[Dict[str, float], float, float]] = {}
    for src in order:
        ent_by_state, mean_H, weighted_H = transition_entropy(matrices[src], states, ignore_empty_rows=True)
        per_source[src] = (ent_by_state, mean_H, weighted_H)
        print(f"{src:<10} | {mean_H:8.4f} | {weighted_H:10.4f}")
    print("\nEntropy by FROM-state (bits):")
    header = "FROM\\SRC".ljust(14) + "".join(s.rjust(10) for s in order)
    print(header + "\n" + "-" * len(header))
    for fr in states:
        row = fr.ljust(14)
        for src in order:
            row += f"{per_source[src][0].get(fr, 0.0):10.4f}"
        print(row)


def run_transition_matrix_report(texts: List[Dict[str, Any]], sources: List[str]) -> None:
    matrices, states = build_transition_matrices(texts, sources, normalize=True)
    print("\n" + "=" * 80)
    print("🔁 TRANSITION MATRICES (Probabilities)")
    print("=" * 80)
    for src in SOURCE_ORDER:
        if src not in matrices:
            continue
        print(f"\nSource: {src}\n" + "-" * 40)
        for fr in states:
            if sum(matrices[src][fr].values()) > 0:
                row_str = " ".join([f"{matrices[src][fr][to]*100:5.1f}%" for to in states])
                print(f"{fr[:4].upper()} | {row_str}")


def _js_divergence_flattened(mat_p: Matrix, mat_q: Matrix, states: List[str], eps: float = 1e-12) -> float:
    p_flat = [mat_p[fr][to] for fr in states for to in states]
    q_flat = [mat_q[fr][to] for fr in states for to in states]
    sp, sq = sum(p_flat), sum(q_flat)
    if sp <= 0 or sq <= 0:
        return 0.0
    p_flat = [x / sp for x in p_flat]
    q_flat = [x / sq for x in q_flat]
    m_flat = [0.5 * (p + q) for p, q in zip(p_flat, q_flat)]

    def kl(a, b):
        return sum(av * math.log2(av / max(bv, eps)) for av, bv in zip(a, b) if av > 0)

    return 0.5 * kl(p_flat, m_flat) + 0.5 * kl(q_flat, m_flat)


def compare_transition_matrices(matrices: Matrices, states: List[str], anchor: str = "mine") -> None:
    if anchor not in matrices:
        return
    ref = matrices[anchor]
    print("\n" + "=" * 80)
    print(f"📊 СРАВНЕНИЕ МАТРИЦ ПЕРЕХОДОВ vs {anchor.upper()} (L1 & JSD)")
    print("=" * 80)
    for src in SOURCE_ORDER:
        if src not in matrices or src == anchor:
            continue
        mat = matrices[src]
        js = _js_divergence_flattened(ref, mat, states)
        l1, diffs = 0.0, []
        for fr in states:
            for to in states:
                d = abs(ref[fr][to] - mat[fr][to])
                l1 += d
                if d > 0:
                    diffs.append((d, fr, to, ref[fr][to], mat[fr][to]))
        diffs.sort(reverse=True, key=lambda x: x[0])
        print(f"\n{src.upper()}:\n  L1 distance: {l1:.4f}\n  Flattened JSD: {js:.4f}")
        for d, fr, to, a, b in diffs[:5]:
            print(f"    {fr[:4]}->{to[:4]}: {anchor}={a*100:4.0f}% | {src}={b*100:4.0f}% | Δ={d*100:4.0f}%")


# -----------------------------
# Edge-Edit Distance (GED)  (без изменений)
# -----------------------------
def run_edge_ged_report(texts: List[Dict[str, Any]], sources: List[str], anchor: str = "mine", min_weight: float = 0.15) -> None:
    matrices, states = build_transition_matrices(texts, sources, normalize=True)
    order = [s for s in SOURCE_ORDER if s in matrices]
    print("\n" + "=" * 80)
    print(f"🧩 EDGE-EDIT DISTANCE (GED-like) min_weight>={min_weight}")
    print("=" * 80)
    dist = {a: {} for a in order}
    for a in order:
        for b in order:
            d = 0.0
            for fr in states:
                for to in states:
                    w1, w2 = matrices[a][fr][to], matrices[b][fr][to]
                    if w1 >= min_weight and w2 >= min_weight:
                        d += abs(w1 - w2)
                    elif w1 >= min_weight:
                        d += w1
                    elif w2 >= min_weight:
                        d += w2
            dist[a][b] = d
    header = " " * 10 + "".join(f"{s:>10}" for s in order)
    print(header + "\n" + "-" * len(header))
    for a in order:
        print(f"{a:<10}" + "".join(f"{dist[a][b]:10.4f}" for b in order))


# -----------------------------
# Graphs  (без изменений)
# -----------------------------
def plot_all_transition_graphs(texts: List[Dict[str, Any]], sources: List[str], title_suffix: str) -> None:
    if not HAS_PLOT:
        return
    matrices, states = build_transition_matrices(texts, sources)
    for src in SOURCE_ORDER:
        if src not in matrices:
            continue
        mat = matrices[src]
        G = nx.DiGraph()
        G.add_nodes_from(states)
        for fr in states:
            for to in states:
                if mat[fr][to] >= 0.15:
                    G.add_edge(fr, to, weight=mat[fr][to])
        if G.number_of_edges() == 0:
            continue
        pos = nx.circular_layout(G)
        plt.figure(figsize=(10, 8))
        plt.title(f"Graph: {src.upper()} - {title_suffix}", pad=20, fontsize=14, fontweight="bold")
        nx.draw_networkx_nodes(G, pos, node_color="#E6E6FA", node_size=3500, edgecolors="white")
        nx.draw_networkx_labels(G, pos, font_size=9)
        nx.draw_networkx_edges(G, pos, edge_color="#555555", arrows=True, arrowsize=18,
                               connectionstyle="arc3,rad=0.15", min_target_margin=15)
        edge_labels = {(u, v): f"{d['weight']*100:.0f}%" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=9,
                                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
        plt.axis("off")
        plt.tight_layout()
        plt.show()


# -----------------------------
# Main Loop
# -----------------------------
def main(data_dir: Path, no_graphs: bool = False) -> None:
    if not data_dir.exists():
        print(f"Директория {data_dir} не найдена.")
        return
    myth_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not myth_dirs:
        print(f"В папке {data_dir} нет подпапок с мифами.")
        return

    universal_texts, universal_sources = [], []

    for myth_dir in myth_dirs:
        print("\n" + "░" * 80)
        print(f"🚀 АНАЛИЗ КОРПУСА: {myth_dir.name.upper()}")
        print("░" * 80)

        bundle, runs_by_source_raw = load_myth_datasets(myth_dir)
        if not bundle.texts:
            print(f"⚠️ Нет валидных .json в {myth_dir.name}, пропускаем.")
            continue

        universal_texts.extend(bundle.texts)
        universal_sources.extend(bundle.sources)

        for src, runs in runs_by_source_raw.items():
            if len(runs) > 1:
                print(f"  [run_log] {src}: {len(runs)} прогонов найдено -> behavioral axis активна")

        # Транзишн-метрики (по transition_from/to)
        run_transition_entropy_report(bundle.texts, bundle.sources)

        # ПОЧИНЕННОЕ сравнение по строкам + оси субъектности
        projections = run_line_alignment_report(bundle.per_source, anchor="mine")
        runs_by_source_proj = {
            src: [project_to_lines(segs) for segs in segs_list]
            for src, segs_list in runs_by_source_raw.items()
        }
        run_subjectivity_axes_report(projections, runs_by_source=runs_by_source_proj)

        # Shape/DTW по сегментам — для справки, помечен как granularity-sensitive
        compare_function_sequences_dtw(bundle.per_source)

        matrices, states = build_transition_matrices(bundle.texts, bundle.sources)
        compare_transition_matrices(matrices, states)
        run_edge_ged_report(bundle.texts, bundle.sources)

        if not no_graphs:
            plot_all_transition_graphs(bundle.texts, bundle.sources, title_suffix=myth_dir.name)

    # Универсальный уровень: матрицы агрегируются, НО line-проекция — нет
    # (номера строк у разных мифов свои; общую ось строк через мифы строить нельзя).
    if universal_texts:
        print("\n\n" + "█" * 80)
        print("🌍 UNIVERSAL BASELINE (агрегация матриц переходов по всем мифам)")
        print("ПРИМЕЧАНИЕ: line-aligned сравнение здесь не считается — оси строк у мифов разные.")
        print("█" * 80)
        run_transition_entropy_report(universal_texts, universal_sources)
        run_transition_matrix_report(universal_texts, universal_sources)
        uni_matrices, uni_states = build_transition_matrices(universal_texts, universal_sources)
        compare_transition_matrices(uni_matrices, uni_states)
        run_edge_ged_report(universal_texts, universal_sources)
        if not no_graphs:
            plot_all_transition_graphs(universal_texts, universal_sources, title_suffix="UNIVERSAL")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=str(Path(__file__).parent.parent / "data"))
    p.add_argument("--no-graphs", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(Path(args.data_dir), args.no_graphs)
