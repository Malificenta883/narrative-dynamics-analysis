#!/usr/bin/env python3
"""
Narrative Dynamics Analysis — Gudea (multi-source segmentation)

Core (publication-ready):
  1) Transition entropy report
  2) Transition matrices per source (+ top transitions)
  3) Matrix comparison vs human baseline (L1 + JS + top diffs)
  4) Weighted directed transition graphs (optional plotting)

Optional (exploratory; off by default):
  - UMAP/HDBSCAN clustering on arbitrary label fields using sentence embeddings.

Expected data layout:
  repo_root/
    analysis.py
    data/
      gudea_segments_mine.json
      gudea_segments_claude4.5sonnet.json
      gudea_segments_gpt5.2.json
      gudea_segments_gemini3PRO.json
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- Warnings: keep your original intent, but scoped ---
warnings.filterwarnings("ignore", message=".*n_neighbors is larger than the dataset size.*")
warnings.filterwarnings("ignore", message=".*All labels are -1.*")

# --- Optional deps (plotting / exploratory) ---
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
    raise ImportError("networkx is required for this script (graphs + helpers).") from e

# UMAP/HDBSCAN + sentence embeddings only needed if you enable exploratory clustering
try:
    import umap  # type: ignore
except Exception:
    umap = None

try:
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None


# -----------------------------
# Configuration / Controlled Vocab
# -----------------------------

FUNCTIONS_ORDER = [
    "preparation",
    "contact",
    "exchange",
    "disruption",
    "negotiation",
    "stabilization",
    "return",
]

DEFAULT_FILES = {
    "mine": "gudea_segments_mine.json",
    "claude": "gudea_segments_claude4.5sonnet.json",
    "gpt": "gudea_segments_gpt5.2.json",
    "gemini": "gudea_segments_gemini3PRO.json",
}

SOURCE_ORDER = ["mine", "claude", "gpt", "gemini"]


# -----------------------------
# IO
# -----------------------------

def load_segments(data_dir: Path, filename: str) -> List[Dict[str, Any]]:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    return data


@dataclass(frozen=True)
class DatasetBundle:
    texts: List[Dict[str, Any]]
    sources: List[str]
    ids: List[str]
    per_source: Dict[str, List[Dict[str, Any]]]


def load_all_datasets(data_dir: Path, files: Dict[str, str]) -> DatasetBundle:
    per_source: Dict[str, List[Dict[str, Any]]] = {}
    for src, fname in files.items():
        per_source[src] = load_segments(data_dir, fname)

    # Maintain stable concatenation order
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

    if len(texts) != len(sources) or len(texts) != len(ids):
        raise RuntimeError("Internal mismatch: texts/sources/ids lengths diverged.")

    return DatasetBundle(texts=texts, sources=sources, ids=ids, per_source=per_source)


# -----------------------------
# Transition Matrices
# -----------------------------

Matrix = Dict[str, Dict[str, float]]  # matrix[from][to] -> value
Matrices = Dict[str, Matrix]          # source -> matrix


def _init_matrix(states: List[str]) -> Matrix:
    return {fr: {to: 0.0 for to in states} for fr in states}


def build_transition_matrices(
    texts: List[Dict[str, Any]],
    sources: List[str],
    allowed_states: Optional[List[str]] = None,
    normalize: bool = True,
) -> Tuple[Matrices, List[str]]:
    """
    Build per-source transition matrices from segment fields:
      transition_from, transition_to

    If normalize=True: rows become probability distributions (row-normalized).
    """
    if allowed_states is None:
        states = sorted(set(
            (t.get("transition_from") for t in texts if t.get("transition_from")) |
            (t.get("transition_to") for t in texts if t.get("transition_to"))
        ))
    else:
        states = list(allowed_states)

    matrices: Matrices = {}
    for src in set(sources):
        matrices[src] = _init_matrix(states)

    # Count transitions
    for item, src in zip(texts, sources):
        fr = item.get("transition_from")
        to = item.get("transition_to")
        if fr not in states or to not in states:
            # ignore garbage states / None
            continue
        matrices[src][fr][to] += 1.0

    # Normalize rows
    if normalize:
        for src, mat in matrices.items():
            for fr in states:
                row_sum = sum(mat[fr].values())
                if row_sum <= 0:
                    continue
                for to in states:
                    mat[fr][to] = mat[fr][to] / row_sum

    return matrices, states


def print_transition_matrix(
    mat: Matrix,
    states: List[str],
    title: str,
    as_percent: bool = True,
) -> None:
    print("\n" + title)
    header = "FROM\\TO".ljust(16) + "".join(s.rjust(14) for s in states)
    print(header)
    print("-" * len(header))

    for fr in states:
        row = fr.ljust(16)
        for to in states:
            v = float(mat[fr][to])
            if as_percent:
                row += f"{(v*100):13.1f}%"
            else:
                row += f"{v:14.2f}"
        print(row)


def run_transition_matrix_report(texts: List[Dict[str, Any]], sources: List[str], normalize: bool = True) -> None:
    matrices, states = build_transition_matrices(
        texts, sources,
        allowed_states=FUNCTIONS_ORDER,
        normalize=normalize,
    )

    print("\n" + "=" * 80)
    print("🔁 TRANSITION MATRICES (transition_from → transition_to)")
    print(f"normalize = {normalize} (True=probabilities, False=counts)")
    print("=" * 80)

    for src in SOURCE_ORDER:
        if src not in matrices:
            continue
        mat = matrices[src]
        print_transition_matrix(mat, states, title=f"Source: {src}", as_percent=normalize)

        # Top transitions
        triples: List[Tuple[float, str, str]] = []
        for fr in states:
            for to in states:
                v = float(mat[fr][to])
                if v > 0:
                    triples.append((v, fr, to))
        triples.sort(reverse=True, key=lambda x: x[0])

        print("\nTop transitions:")
        for v, fr, to in triples[:10]:
            if normalize:
                print(f"  {fr} → {to}: {v*100:.1f}%")
            else:
                print(f"  {fr} → {to}: {int(v)}")

        print("\n" + "-" * 80)

    # Comparison vs baseline
    compare_transition_matrices(matrices, states, anchor="mine", top_k=12)


# -----------------------------
# Entropy
# -----------------------------

def row_entropy(row_probs: Dict[str, float], eps: float = 1e-12) -> float:
    """Shannon entropy in bits for one distribution row."""
    H = 0.0
    for p in row_probs.values():
        p = float(p)
        if p <= 0:
            continue
        H -= p * math.log(p + eps, 2)
    return H


def transition_entropy(
    matrix: Matrix,
    states: List[str],
    ignore_empty_rows: bool = True,
) -> Tuple[Dict[str, float], float, float]:
    """
    Returns:
      entropy_by_state: H(from_state)
      mean_entropy: mean over (non-empty) rows
      weighted_entropy: weighted by outgoing mass (kept for clarity)
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
    matrices, states = build_transition_matrices(
        texts, sources,
        allowed_states=FUNCTIONS_ORDER,
        normalize=True,
    )

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


# -----------------------------
# Matrix comparison (L1 + JS)
# -----------------------------

def _safe_log2(x: float) -> float:
    return math.log(x, 2)


def _js_divergence_row(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence for two distributions (one row)."""
    sp = sum(p.values())
    sq = sum(q.values())
    if sp <= 0 or sq <= 0:
        return 0.0
    p = {k: v / sp for k, v in p.items()}
    q = {k: v / sq for k, v in q.items()}

    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        out = 0.0
        for k, av in a.items():
            if av <= 0:
                continue
            bv = b.get(k, 0.0)
            if bv <= 0:
                bv = eps
            out += av * (_safe_log2(av) - _safe_log2(bv))
        return out

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compare_transition_matrices(
    matrices: Matrices,
    states: List[str],
    anchor: str = "mine",
    top_k: int = 10,
) -> None:
    """
    Compare all sources to anchor:
      - L1 distance over all cells
      - mean JS divergence over rows
      - top-k absolute probability diffs
    """
    if anchor not in matrices:
        print(f"⚠️ No matrix for anchor='{anchor}'")
        return

    ref = matrices[anchor]

    print("\n" + "=" * 80)
    print(f"📌 TRANSITION MATRIX COMPARISON vs '{anchor}'")
    print("=" * 80)

    for src in SOURCE_ORDER:
        if src not in matrices or src == anchor:
            continue

        mat = matrices[src]

        # L1 over all cells
        l1 = 0.0
        diffs: List[Tuple[float, str, str, float, float]] = []
        for fr in states:
            for to in states:
                a = float(ref[fr][to])
                b = float(mat[fr][to])
                d = abs(a - b)
                l1 += d
                if d > 0:
                    diffs.append((d, fr, to, a, b))
        diffs.sort(reverse=True, key=lambda x: x[0])

        # JS mean over rows
        js_vals = []
        for fr in states:
            js_vals.append(_js_divergence_row(ref[fr], mat[fr]))
        js_mean = sum(js_vals) / len(js_vals) if js_vals else 0.0

        print(f"\nSource: {src}")
        print(f"  L1 distance: {l1:.6f}")
        print(f"  Mean JS divergence (rows): {js_mean:.6f}")

        print(f"  Top {top_k} cell diffs (abs):")
        for d, fr, to, a, b in diffs[:top_k]:
            print(f"    {fr:>12} → {to:<12} | {anchor}={a*100:6.1f}%  {src}={b*100:6.1f}%  Δ={d*100:6.1f}%")

    print("\n" + "-" * 80)


# -----------------------------
# Graph visualization (optional)
# -----------------------------

def plot_transition_graph(
    mat: Matrix,
    states: List[str],
    title: str,
    min_weight: float = 0.15,
    layout_seed: int = 42,
) -> None:
    if not HAS_PLOT:
        print("(Skipping plot, matplotlib not found)")
        return

    G = nx.DiGraph()
    for s in states:
        G.add_node(s)

    for fr in states:
        for to in states:
            w = float(mat[fr][to])
            if w >= min_weight:
                G.add_edge(fr, to, weight=w)

    if G.number_of_edges() == 0:
        print(f"(No edges above min_weight={min_weight} for {title})")
        return

    pos = nx.spring_layout(G, seed=layout_seed)

    plt.figure(figsize=(10, 8))
    plt.title(title)

    nx.draw_networkx_nodes(G, pos, node_size=2200, edgecolors="black")
    edges = G.edges(data=True)
    widths = [6 * d["weight"] for (_, _, d) in edges]

    nx.draw_networkx_edges(
        G, pos,
        arrowstyle="->",
        arrowsize=20,
        width=widths,
        connectionstyle="arc3,rad=0.15"
    )

    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    edge_labels = {(u, v): f"{d['weight']*100:.0f}%" for (u, v, d) in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_all_transition_graphs(texts: List[Dict[str, Any]], sources: List[str], min_weight: float = 0.15) -> None:
    matrices, states = build_transition_matrices(
        texts, sources,
        allowed_states=FUNCTIONS_ORDER,
        normalize=True,
    )

    for src in SOURCE_ORDER:
        if src not in matrices:
            continue
        plot_transition_graph(
            matrices[src],
            states,
            title=f"Transition Graph — {src.upper()}",
            min_weight=min_weight,
        )


# -----------------------------
# Exploratory: embeddings + UMAP + HDBSCAN
# -----------------------------

def _require_exploratory_deps() -> None:
    missing = []
    if SentenceTransformer is None:
        missing.append("sentence-transformers")
    if umap is None:
        missing.append("umap-learn")
    if hdbscan is None:
        missing.append("hdbscan")
    if missing:
        raise ImportError(
            "Exploratory clustering requires: " + ", ".join(missing) +
            "\nInstall via pip, e.g.: pip install " + " ".join(missing)
        )


def plot_clusters(X_2d, labels, sources, title: str) -> None:
    if not HAS_PLOT:
        print("(Skipping plot, matplotlib not found)")
        return

    # Markers by source (kept from your intent; feel free to adjust)
    markers_style = {"mine": "o", "claude": "s", "gemini": "^", "gpt": "D"}

    plt.figure(figsize=(10, 8))

    # Noise first
    noise_idx = [i for i, lab in enumerate(labels) if lab == -1]
    if noise_idx:
        plt.scatter(X_2d[noise_idx, 0], X_2d[noise_idx, 1], marker="x", alpha=0.3, label="Noise (-1)")

    for src in sorted(set(sources), key=lambda x: SOURCE_ORDER.index(x) if x in SOURCE_ORDER else 999):
        idx = [i for i, s in enumerate(sources) if s == src and labels[i] != -1]
        if not idx:
            continue
        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            marker=markers_style.get(src, "o"),
            alpha=0.7,
            s=60,
            label=src,
            edgecolors="white",
            linewidth=0.5,
        )

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_cluster_analysis_with_embeddings(
    texts: List[Dict[str, Any]],
    sources: List[str],
    ids: List[str],
    target_field: str,
    plot_title: str = "Cluster Analysis",
    emb_model_name: str = "all-MiniLM-L6-v2",
) -> None:
    _require_exploratory_deps()

    print("\n" + "=" * 80)
    print(f"🚀 EMBEDDING ANALYSIS: {plot_title}")
    print(f"   Field used for analysis: '{target_field}'")
    print("=" * 80)

    processed_texts: List[str] = []
    for item in texts:
        data = item.get(target_field)

        if isinstance(data, list):
            text_val = " ".join(map(str, data))
        elif data is None:
            text_val = ""
        else:
            text_val = str(data)

        processed_texts.append(text_val)

    if all(not s.strip() for s in processed_texts):
        print(f"⚠️ Warning: field '{target_field}' is empty for all elements.")
        return

    # Embeddings
    model = SentenceTransformer(emb_model_name)
    print("Generating embeddings...")
    embeddings = model.encode(processed_texts, convert_to_numpy=True, normalize_embeddings=True)

    n_samples = embeddings.shape[0]
    if n_samples < 2:
        print("Too few items for clustering.")
        return

    # UMAP
    n_neighbors = min(10, n_samples - 1)
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
    )
    X_2d = reducer.fit_transform(embeddings)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=2,
        metric="euclidean",
        allow_single_cluster=False,
    )
    labels = clusterer.fit_predict(X_2d)

    # Examples
    print(f"\n--- RESULTS (Field: {target_field}) ---")
    for lbl in sorted(set(labels)):
        print("\nNoise (not clustered):" if lbl == -1 else f"\nCluster {lbl}:")
        cluster_indices = [i for i, x in enumerate(labels) if x == lbl]
        for idx in cluster_indices[:3]:
            print(f"   [{ids[idx]} | {sources[idx]}]: {processed_texts[idx]}")
        if len(cluster_indices) > 3:
            print(f"   ... and {len(cluster_indices) - 3} more elements")

    # Cross-statistics
    print("\n--- CLUSTER STATISTICS ---")
    print(f"{'Cluster':<10} | {'Total':<6} | {'Mine':<5} | {'Claude':<6} | {'Gemini':<6} | {'GPT':<5}")
    print("-" * 55)

    cluster_sources: Dict[int, List[str]] = defaultdict(list)
    for src, label in zip(sources, labels):
        cluster_sources[int(label)].append(src)

    for label, src_list in sorted(cluster_sources.items(), key=lambda x: x[0]):
        c = Counter(src_list)
        cluster_name = f"{label}" if label != -1 else "Noise"
        print(
            f"{cluster_name:<10} | "
            f"{len(src_list):<6} | "
            f"{c['mine']:<5} | "
            f"{c['claude']:<6} | "
            f"{c['gemini']:<6} | "
            f"{c['gpt']:<5}"
        )

    plot_clusters(X_2d, labels, sources, plot_title)


# -----------------------------
# Main / CLI
# -----------------------------

def main(
    data_dir: Path,
    run_exploratory_umap: bool = False,
    plot_graphs: bool = True,
    graph_min_weight: float = 0.15,
) -> None:
    bundle = load_all_datasets(data_dir, DEFAULT_FILES)

    texts = bundle.texts
    sources = bundle.sources
    ids = bundle.ids

    # 1) Transition entropy
    run_transition_entropy_report(texts, sources)

    # 2) Transition matrices + comparison vs baseline inside
    run_transition_matrix_report(texts, sources, normalize=True)

    # 3) Graphs (optional)
    if plot_graphs:
        plot_all_transition_graphs(texts, sources, min_weight=graph_min_weight)

    # 4) Exploratory
    if run_exploratory_umap:
        run_cluster_analysis_with_embeddings(
            texts, sources, ids,
            target_field="markers",
            plot_title="Clustering by Markers (exploratory)",
        )
        run_cluster_analysis_with_embeddings(
            texts, sources, ids,
            target_field="cognitive_frame",
            plot_title="Clustering by Cognitive Frame (exploratory)",
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent / "data"),
        help="Path to data directory with gudea_segments_*.json files.",
    )
    p.add_argument(
        "--exploratory-umap",
        action="store_true",
        help="Run optional embedding+UMAP+HDBSCAN clustering (needs extra deps).",
    )
    p.add_argument(
        "--no-graphs",
        action="store_true",
        help="Disable plotting transition graphs (still computes matrices).",
    )
    p.add_argument(
        "--min-weight",
        type=float,
        default=0.15,
        help="Minimum edge weight for transition graph visualization.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        data_dir=Path(args.data_dir),
        run_exploratory_umap=bool(args.exploratory_umap),
        plot_graphs=not bool(args.no_graphs),
        graph_min_weight=float(args.min_weight),
    )
