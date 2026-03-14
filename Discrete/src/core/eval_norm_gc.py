"""
core/eval_norm_gc.py
====================
Normalized evaluation budget for Graph Coloring algorithms — makes evals_cost
comparable across algorithms that have different per-eval computational costs.

Background
----------
All GC solvers increment evals_cost once per conflict-counting call, but the
work done per increment differs:

  HC_GC   →  delta_recolor(v)      →  O(deg(v)) neighbor scan
  SA_GC   →  delta_recolor(v)      →  O(deg(v)) neighbor scan
  GA_GC   →  problem.evaluate(x)   →  O(|E|) edge scan (full evaluation)
  ACO_GC  →  problem.evaluate(x)   →  O(|E|) edge scan (full evaluation)
  DFS_GC  →  problem.evaluate(tmp) →  O(|E|) edge scan (checkpoint only)

Reference unit
--------------
1 gc_eval = one call to problem.evaluate(x)
          = O(|E|) conflict count (iterate over all edges once)

This is the natural reference for GC: it is the minimum unit of information
gain (you learn the conflict count of exactly one complete coloring).

Work factors
------------
For an average-degree graph with mean degree d_avg = 2|E|/n:

  HC_GC / SA_GC  →  delta_recolor costs O(deg(v)) ≈ O(d_avg)
                     Full evaluate() costs O(|E|) = O(n * d_avg / 2)
                     So delta is ~2/n as expensive as full eval
                     work_factor = 2/n  (n-dependent, like TSP HC/SA)

  GA_GC / ACO_GC →  full evaluate() per logged eval
                     work_factor = 1.0  (reference)

  DFS_GC         →  evaluate() only at checkpoints, not per backtrack step
                     evals_cost is not a useful budget proxy for DFS
                     work_factor = 1.0  (checkpoints use full evaluate)

Note: work_factor for HC_GC/SA_GC depends on n (number of nodes), matching
the same structure as the TSP normalizer where HC/SA use O(1) delta vs O(n).

Formula
-------
    normalized_evals = raw_evals_cost × work_factor(algo, n)

    HC_GC / SA_GC:  factor = 2/n   (delta ≈ 2 adj checks vs n*d_avg/2 for full)
    GA_GC / ACO_GC: factor = 1.0
    DFS_GC:         factor = 1.0
"""

from __future__ import annotations

from typing import Union
import numpy as np

# Complexity class per algo: mirrors TSP eval_norm structure.
#   k=0 → O(1) / O(deg)  delta move (HC/SA)
#   k=1 → O(|E|)         full evaluate  (GA, ACO, DFS)
# work_factor = n^(k - 1)  where ref_k = 1
#
# For GC:
#   HC/SA delta_recolor scans deg(v) neighbors ≈ mean degree d = 2|E|/n
#   Full evaluate scans all |E| edges
#   Ratio: d / |E| = (2|E|/n) / |E| = 2/n  →  same n^(0-1) = 1/n 

_COMPLEXITY_GC: dict[str, int] = {
    "HC_GC":  0,   # O(deg(v)) delta — treated as O(1) per node, ref = O(n)
    "SA_GC":  0,   # same delta move
    "GA_GC":  1,   # O(|E|) full evaluate
    "ACO_GC": 1,   # O(|E|) full evaluate per ant
    "DFS_GC": 1,   # O(|E|) evaluate at checkpoints
}

_REF_K: int = 1   # reference = full evaluate(), same as TSP


def work_factor_gc(algo: str, n: int) -> float:
    """
    Return the multiplier that converts raw evals_cost → normalized evals
    for a Graph Coloring algorithm, relative to one O(|E|) evaluate() call.

    Parameters
    ----------
    algo : str
        Algorithm name: "HC_GC", "SA_GC", "GA_GC", "ACO_GC", or "DFS_GC".
        Case-insensitive.
    n : int
        Number of nodes in the graph.

    Returns
    -------
    float
        Multiplier: normalized = raw_evals * work_factor_gc(algo, n)

    Examples
    --------
    >>> work_factor_gc("HC_GC",  30)   # 1/30 ≈ 0.0333
    >>> work_factor_gc("GA_GC",  30)   # 1.0
    >>> work_factor_gc("ACO_GC", 30)   # 1.0
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    key = algo.upper()
    if key not in _COMPLEXITY_GC:
        raise ValueError(
            f"Unknown GC algorithm '{algo}'. Known: {list(_COMPLEXITY_GC)}"
        )
    k = _COMPLEXITY_GC[key]
    return float(n ** (k - _REF_K))


def normalize_evals_gc(
    evals_cost: Union[int, float, np.ndarray],
    algo: str,
    n: int,
) -> Union[float, np.ndarray]:
    """
    Convert raw evals_cost → normalized evals (evaluate()-equivalent units)
    for a Graph Coloring algorithm.

    Parameters
    ----------
    evals_cost : int | float | np.ndarray
        Raw evaluation count(s) from RunResult or trace.
    algo : str
        Algorithm name (e.g. "HC_GC", "GA_GC").
    n : int
        Number of nodes.

    Returns
    -------
    float or np.ndarray

    Examples
    --------
    >>> normalize_evals_gc(50_000, "HC_GC",  30)  # →  1_666.7  (delta is cheap)
    >>> normalize_evals_gc(50_000, "GA_GC",  30)  # → 50_000.0  (reference)
    >>> normalize_evals_gc(50_000, "ACO_GC", 30)  # → 50_000.0  (reference)
    """
    factor = work_factor_gc(algo, n)
    if isinstance(evals_cost, np.ndarray):
        return evals_cost.astype(float) * factor
    return float(evals_cost) * factor


def budget_for_algo_gc(target_normalized: int, algo: str, n: int) -> int:
    """
    Given a target normalized budget, return the raw evals_cost limit to
    configure a GC solver so it spends exactly that much normalized budget.

    Parameters
    ----------
    target_normalized : int
        Desired normalized eval budget (in evaluate()-equivalent units).
    algo : str
        Algorithm name.
    n : int
        Number of nodes.

    Returns
    -------
    int
        Raw evals_cost cap to use when constructing the solver.

    Examples
    --------
    >>> budget_for_algo_gc(50_000, "HC_GC",  30)  # → 1_500_000  (delta cheap)
    >>> budget_for_algo_gc(50_000, "GA_GC",  30)  # →    50_000  (reference)
    >>> budget_for_algo_gc(50_000, "ACO_GC", 30)  # →    50_000  (reference)
    """
    factor = work_factor_gc(algo, n)
    if factor <= 0:
        raise ValueError("work_factor must be positive")
    return max(1, int(round(target_normalized / factor)))


def add_normalized_evals_gc(df, n_col: str = "n_cities") -> "pd.DataFrame":
    """
    Add a 'normalized_evals' column to a pandas DataFrame of GC run/trace rows.

    Parameters
    ----------
    df : pd.DataFrame
        Run log or trace log loaded from CSV.
    n_col : str
        Column name holding the node count (default: "n_cities", which GC solvers
        reuse for n_nodes).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with 'normalized_evals' (and optionally
        'normalized_evals_best_found') columns added.
    """
    import pandas as pd  # local import — pandas is optional

    df = df.copy()

    def _norm(row: "pd.Series") -> float:
        try:
            return float(
                normalize_evals_gc(
                    float(row["evals_cost"]),
                    str(row["algorithm"]),
                    int(row[n_col]),
                )
            )
        except Exception:
            return float("nan")

    df["normalized_evals"] = df.apply(_norm, axis=1)

    if "evals_best_found" in df.columns:
        def _norm_best(row: "pd.Series") -> float:
            try:
                ebf = row["evals_best_found"]
                if str(ebf).strip() == "":
                    return float("nan")
                return float(
                    normalize_evals_gc(float(ebf), str(row["algorithm"]), int(row[n_col]))
                )
            except Exception:
                return float("nan")
        df["normalized_evals_best_found"] = df.apply(_norm_best, axis=1)

    return df


def print_budget_table_gc(target_normalized: int, n: int) -> None:
    """
    Print a human-readable budget comparison table for all GC algorithms.

    Example (target=50_000, n=30):

    Budget table (Graph Coloring, n=30) — target_normalized=50,000
    ────────────────────────────────────────────────────────────────
    Algo       work_factor   raw_budget   norm_budget_check
    ────────────────────────────────────────────────────────────────
    HC_GC           0.0333    1,500,000            50,000.0
    SA_GC           0.0333    1,500,000            50,000.0
    GA_GC           1.0000       50,000            50,000.0
    ACO_GC          1.0000       50,000            50,000.0
    DFS_GC          1.0000       50,000            50,000.0
    """
    algos = ["HC_GC", "SA_GC", "GA_GC", "ACO_GC", "DFS_GC"]
    print(f"\nBudget table (Graph Coloring, n={n}) — target_normalized={target_normalized:,}\n")
    print(f"{'Algo':<10} {'work_factor':>12} {'raw_budget':>12} {'norm_budget_check':>18}")
    print("─" * 58)
    for algo in algos:
        wf  = work_factor_gc(algo, n)
        raw = budget_for_algo_gc(target_normalized, algo, n)
        print(
            f"{algo:<10} {wf:>12.4f} "
            f"{raw:>12,} "
            f"{raw * wf:>18,.1f}"
        )
    print()


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== work_factor_gc (n=30) ===")
    for name in ["HC_GC", "SA_GC", "GA_GC", "ACO_GC", "DFS_GC"]:
        print(f"  {name:<10}  factor={work_factor_gc(name, 30):.4f}")

    print()
    print("=== normalize_evals_gc (raw=50_000, n=30) ===")
    for name in ["HC_GC", "SA_GC", "GA_GC", "ACO_GC", "DFS_GC"]:
        print(f"  {name:<10}  normalized={normalize_evals_gc(50_000, name, 30):>12,.1f}")

    print()
    print_budget_table_gc(50_000, n=30)
