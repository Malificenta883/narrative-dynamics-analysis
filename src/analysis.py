#!/usr/bin/env python3
"""
Narrative Dynamics Analysis — Meta-Pipeline
Прогоняет анализ по всем мифам (папкам) внутри директории data/.
Ожидаемая структура:
  data/
    gudea/
      mine.json, claude.json, gpt.json, gemini.json
    inanna_descent/
      mine.json, claude.json, gpt.json, gemini.json
"""

from __future__ import annotations

import argparse
import json
import math
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

try:
    import umap  # type: ignore
    import hdbscan  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    umap = None
    hdbscan = None
    SentenceTransformer = None

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

def load_myth_datasets(myth_dir: Path) -> DatasetBundle:
    """Загружает стандартные JSON-файлы (mine.json, gpt.json...) из папки мифа."""
    per_source: Dict[str, List[Dict[str, Any]]] = {}

    # ИСПРАВЛЕНИЕ 1: восстановлен отступ — if внутри цикла for
    for src in SOURCE_ORDER:
        file_path = myth_dir / f"{src}.json"
        if file_path.exists():
            # Меняем кодировку на utf-8-sig — это заставит Питон игнорировать невидимый мусор от Windows
            with open(file_path, "r", encoding="utf-8-sig") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        per_source[src] = data
                except Exception as e:
                    # Оставляем ровно один принт
                    print(f"  [!!!] БИТЫЙ ФАЙЛ: {file_path}. Причина: {e}")

    # ИСПРАВЛЕНИЕ 2: инициализация списков до их использования
    texts: List[Dict[str, Any]] = []
    sources: List[str] = []
    ids: List[str] = []

    for src in SOURCE_ORDER:
        if src not in per_source:
            continue
        segs = per_source[src]
        texts.extend(segs)
        sources.extend([src] * len(segs))
        ids.extend([f"{src}_{i+1}" for i in range(len(segs))])

    return DatasetBundle(texts=texts, sources=sources, ids=ids, per_source=per_source)

def _norm(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()

# -----------------------------
# Sequences & Exact Match (DTW)
# -----------------------------
def compare_function_and_transitions(texts: List[Dict[str, Any]], sources: List[str], anchor_source: str = "mine") -> None:
    by_src = defaultdict(list)
    for item, src in zip(texts, sources):
        by_src[src].append(item)

    if anchor_source not in by_src or not by_src[anchor_source]:
        return

    anchor = by_src[anchor_source]
    other_sources = [s for s in by_src.keys() if s != anchor_source]

    print("\n" + "=" * 80)
    print("🔎 СРАВНЕНИЕ: function + transition_from/to (exact match)")
    print(f"Anchor: {anchor_source}")
    print("=" * 80)
    print(f"{'SRC':<8} | {'N':>3} | {'func_match':>10} | {'trans_match':>11} | {'both_match':>10}")
    print("-" * 60)

    for src in sorted(other_sources):
        lst = by_src[src]
        n = min(len(anchor), len(lst))
        if n == 0:
            continue

        func_ok, trans_ok, both_ok = 0, 0, 0
        for i in range(n):
            a, b = anchor[i], lst[i]
            a_func, b_func = _norm(a.get("function")), _norm(b.get("function"))
            a_tr = (_norm(a.get("transition_from")), _norm(a.get("transition_to")))
            b_tr = (_norm(b.get("transition_from")), _norm(b.get("transition_to")))

            fmatch = (a_func == b_func) and a_func != ""
            tmatch = (a_tr == b_tr) and (a_tr != ("", ""))

            if fmatch: func_ok += 1
            if tmatch: trans_ok += 1
            if fmatch and tmatch: both_ok += 1

        print(f"{src:<8} | {n:>3} | {100.0*func_ok/n:>9.1f}% | {100.0*trans_ok/n:>10.1f}% | {100.0*both_ok/n:>9.1f}%")

def compare_function_sequences_dtw(texts: List[Dict[str, Any]], sources: List[str], anchor_source: str = "mine") -> None:
    by_src = defaultdict(list)
    for item, src in zip(texts, sources):
        by_src[src].append(item)

    if anchor_source not in by_src:
        return

    anchor_funcs = [_norm(x.get("function")) for x in by_src[anchor_source]]

    print("\n" + "=" * 80)
    print(f"🧬 СРАВНЕНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ (Sequence Alignment/DTW)")
    print(f"Anchor: {anchor_source} (длина: {len(anchor_funcs)})")
    print("=" * 80)

    for src in sorted(by_src.keys()):
        if src == anchor_source: continue
        target_funcs = [_norm(x.get("function")) for x in by_src[src]]
        
        n, m = len(anchor_funcs), len(target_funcs)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1): dp[i][0] = i
        for j in range(1, m + 1): dp[0][j] = j
            
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if anchor_funcs[i-1] == target_funcs[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

        sim = (1 - dp[n][m] / max(n, m)) * 100 if max(n, m) > 0 else 0
        print(f"{src.upper():<8} | Длина: {m:>3} | Edit Dist: {dp[n][m]:>3} | Сходство: {sim:>5.1f}%")


# -----------------------------
# Matrices & Entropy & JSD
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
    """Shannon entropy in bits for one distribution row."""
    H = 0.0
    for p in row_probs.values():
        p = float(p)
        if p <= 0:
            continue
        H -= p * math.log(p, 2)
    return max(H, 0.0)


def transition_entropy(
    matrix: Matrix,
    states: List[str],
    ignore_empty_rows: bool = True,
) -> Tuple[Dict[str, float], float, float]:
    """
    Returns:
      entropy_by_state: H(from_state)
      mean_entropy: mean over (non-empty) rows
      weighted_entropy: weighted by outgoing mass
    """
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
        weights[fr] = row_sum  # ~1.0 if normalized
    vals: List[float] = []
    for fr in states:
        if ignore_empty_rows and weights[fr] <= 0:
            continue
        vals.append(entropy_by_state[fr])
    mean_entropy = sum(vals) / len(vals) if vals else 0.0
    total_w = sum(weights.values())
    weighted_entropy = (
        sum(entropy_by_state[fr] * weights[fr] for fr in states) / total_w
        if total_w > 0 else 0.0
    )
    return entropy_by_state, mean_entropy, weighted_entropy


def run_transition_entropy_report(texts: List[Dict[str, Any]], sources: List[str]) -> None:
    matrices, states = build_transition_matrices(texts, sources, normalize=True)
    order = [s for s in SOURCE_ORDER if s in matrices]
    print("\n" + "=" * 80)
    print("🌪️ TRANSITION ENTROPY REPORT (bits)")
    print("H=0 means deterministic; higher H means more branching/uncertainty.")
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
    print(header)
    print("-" * len(header))
    for fr in states:
        row = fr.ljust(14)
        for src in order:
            ent_by_state = per_source[src][0]
            row += f"{ent_by_state.get(fr, 0.0):10.4f}"
        print(row)
    print("\nMost deterministic states per source (lowest H, excluding empty rows):")
    for src in order:
        ent_by_state = per_source[src][0]
        non_empty = []
        for fr in states:
            if sum(matrices[src][fr].values()) > 0:
                non_empty.append((ent_by_state[fr], fr))
        non_empty.sort()
        top = non_empty[:3] if non_empty else []
        print(f"  {src:<8}: " + ", ".join(f"{fr} (H={H:.3f})" for H, fr in top))


def run_transition_matrix_report(texts: List[Dict[str, Any]], sources: List[str]) -> None:
    matrices, states = build_transition_matrices(texts, sources, normalize=True)
    print("\n" + "=" * 80)
    print("🔁 TRANSITION MATRICES (Probabilities)")
    print("=" * 80)
    
    for src in SOURCE_ORDER:
        if src not in matrices: continue
        print(f"\nSource: {src}\n" + "-"*40)
        for fr in states:
            row_str = " ".join([f"{matrices[src][fr][to]*100:5.1f}%" for to in states])
            if sum(matrices[src][fr].values()) > 0:
                print(f"{fr[:4].upper()} | {row_str}")

def _js_divergence_flattened(mat_p: Matrix, mat_q: Matrix, states: List[str], eps: float = 1e-12) -> float:
    p_flat = [mat_p[fr][to] for fr in states for to in states]
    q_flat = [mat_q[fr][to] for fr in states for to in states]
    sp, sq = sum(p_flat), sum(q_flat)
    if sp <= 0 or sq <= 0: return 0.0
    
    p_flat = [x / sp for x in p_flat]
    q_flat = [x / sq for x in q_flat]
    m_flat = [0.5 * (p + q) for p, q in zip(p_flat, q_flat)]

    def kl(a: List[float], b: List[float]) -> float:
        return sum(av * math.log2(av / max(bv, eps)) for av, bv in zip(a, b) if av > 0)

    return 0.5 * kl(p_flat, m_flat) + 0.5 * kl(q_flat, m_flat)

def compare_transition_matrices(matrices: Matrices, states: List[str], anchor: str = "mine") -> None:
    if anchor not in matrices: return
    ref = matrices[anchor]
    
    print("\n" + "=" * 80)
    print(f"📊 СРАВНЕНИЕ МАТРИЦ ПЕРЕХОДОВ vs {anchor.upper()} (L1 & JSD)")
    print("=" * 80)

    for src in SOURCE_ORDER:
        if src not in matrices or src == anchor: continue
        mat = matrices[src]
        js = _js_divergence_flattened(ref, mat, states)
        
        l1, diffs = 0.0, []
        for fr in states:
            for to in states:
                d = abs(ref[fr][to] - mat[fr][to])
                l1 += d
                if d > 0: diffs.append((d, fr, to, ref[fr][to], mat[fr][to]))

        diffs.sort(reverse=True, key=lambda x: x[0])
        print(f"\n{src.upper()}:\n  L1 distance: {l1:.4f}\n  Flattened JSD: {js:.4f}")
        for d, fr, to, a, b in diffs[:5]:
            print(f"    {fr[:4]}->{to[:4]}: {anchor}={a*100:4.0f}% | {src}={b*100:4.0f}% | Δ={d*100:4.0f}%")

# -----------------------------
# Edge-Edit Distance (GED)
# -----------------------------
def run_edge_ged_report(texts: List[Dict[str, Any]], sources: List[str], anchor: str = "mine", min_weight: float = 0.15) -> None:
    matrices, states = build_transition_matrices(texts, sources, normalize=True)
    order = [s for s in SOURCE_ORDER if s in matrices]
    
    print("\n" + "="*80)
    print(f"🧩 EDGE-EDIT DISTANCE (GED-like) min_weight>={min_weight}")
    print("="*80)

    dist = {a: {} for a in order}
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            d = 0.0
            for fr in states:
                for to in states:
                    w1, w2 = matrices[a][fr][to], matrices[b][fr][to]
                    if w1 >= min_weight and w2 >= min_weight: d += abs(w1 - w2)
                    elif w1 >= min_weight: d += w1
                    elif w2 >= min_weight: d += w2
            dist[a][b] = d

    header = " " * 10 + "".join(f"{s:>10}" for s in order)
    print(header + "\n" + "-" * len(header))
    for a in order:
        print(f"{a:<10}" + "".join(f"{dist[a][b]:10.4f}" for b in order))

# -----------------------------
# Graphs & Clusters (Wrappers)
# -----------------------------
def plot_all_transition_graphs(texts: List[Dict[str, Any]], sources: List[str], title_suffix: str) -> None:
    if not HAS_PLOT: return
    matrices, states = build_transition_matrices(texts, sources)
    
    for src in SOURCE_ORDER:
        if src not in matrices: continue
        mat = matrices[src]
        
        G = nx.DiGraph()
        G.add_nodes_from(states)
        for fr in states:
            for to in states:
                # Порог отсечения энтропийного шума (оставляем только сильные паттерны)
                if mat[fr][to] >= 0.15: 
                    G.add_edge(fr, to, weight=mat[fr][to])
        
        if G.number_of_edges() == 0: continue
        
        # 1. Жесткая круговая топология. Все узлы прибиты к своим местам.
        # Это позволяет визуально сравнивать графы разных моделей (отсутствующие ребра сразу видны как пустоты).
        pos = nx.circular_layout(G)
        
        plt.figure(figsize=(10, 8))
        plt.title(f"Graph: {src.upper()} - {title_suffix}", pad=20, fontsize=14, fontweight="bold")
        
        # 2. Отрисовка узлов
        nx.draw_networkx_nodes(G, pos, node_color="#E6E6FA", node_size=3500, edgecolors="white")
        nx.draw_networkx_labels(G, pos, font_size=9)
        
        # 3. Отрисовка ребер с изгибом (arc3), чтобы встречные вероятности (A->B и B->A) не сливались в одну линию
        nx.draw_networkx_edges(
            G, pos, 
            edge_color="#555555", 
            arrows=True, 
            arrowsize=18, 
            connectionstyle="arc3,rad=0.15",
            min_target_margin=15
        )
        
        # 4. Отрисовка текста с белой подложкой (bbox) и смещением (label_pos) от центра к началу вектора
        edge_labels = {(u, v): f"{d['weight']*100:.0f}%" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos, 
            edge_labels=edge_labels, 
            label_pos=0.3, 
            font_size=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1)
        )
        
        plt.axis("off")
        plt.tight_layout()
        plt.show()
# -----------------------------
# Main Loop (Meta-Analysis)
# -----------------------------
def main(data_dir: Path, no_graphs: bool = False) -> None:
    if not data_dir.exists():
        print(f"Директория {data_dir} не найдена.")
        return

    myth_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not myth_dirs:
        print(f"В папке {data_dir} нет подпапок с мифами.")
        return

    # Контейнеры для Universal Matrix
    universal_texts = []
    universal_sources = []

    # 1. ЛОКАЛЬНЫЙ АНАЛИЗ (по каждому мифу)
    for myth_dir in myth_dirs:
        print("\n" + "░"*80)
        print(f"🚀 АНАЛИЗ КОРПУСА: {myth_dir.name.upper()}")
        print("░"*80)
        
        bundle = load_myth_datasets(myth_dir)
        if not bundle.texts:
            print(f"⚠️ Нет валидных .json файлов в {myth_dir.name}, пропускаем.")
            continue

        # Собираем данные для глобального анализа
        universal_texts.extend(bundle.texts)
        universal_sources.extend(bundle.sources)

        run_transition_entropy_report(bundle.texts, bundle.sources)
        compare_function_and_transitions(bundle.texts, bundle.sources)
        compare_function_sequences_dtw(bundle.texts, bundle.sources)
        matrices, states = build_transition_matrices(bundle.texts, bundle.sources)
        compare_transition_matrices(matrices, states)
        run_edge_ged_report(bundle.texts, bundle.sources)
        
        if not no_graphs:
            plot_all_transition_graphs(bundle.texts, bundle.sources, title_suffix=myth_dir.name)

    # 2. ГЛОБАЛЬНЫЙ АНАЛИЗ (Universal Baseline)
    if universal_texts:
        print("\n\n" + "█"*80)
        print("🌍 UNIVERSAL HUMAN BASELINE (Агрегация всех мифов)")
        print("█"*80)
        
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
