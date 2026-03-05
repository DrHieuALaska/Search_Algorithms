"""
core/eval_norm.py
=================
Normalized evaluation budget — makes evals_cost comparable across algorithms
that have fundamentally different per-eval computational costs.

Background
----------
Raw evals_cost counts "how many times was the objective queried", but each
query is NOT equal work:

  HC / SA  →  delta_2opt()        →  O(1)   — 4 array lookups, 2 subtractions
  GA       →  full tour sum       →  O(n)   — n edge lookups + sum
  TLBO     →  full tour sum       →  O(n)   — same as GA
  ACO      →  construct + score   →  O(n²)  — n city steps × scan unvisited

Reference unit: O(n) full tour evaluation (GA / TLBO baseline).
This keeps GA and TLBO evals_cost unchanged, scales HC/SA down, and ACO up.

Formula
-------
    work_factor(algo, n) = cost_per_eval / n          (ref = 1 full tour eval)

    normalized_evals = raw_evals_cost × work_factor

Work factors:
    HC   →  1/n   (O(1) delta, n times cheaper than a full eval)
    SA   →  1/n   (same delta_2opt)
    GA   →  1.0   (O(n) — reference)
    TLBO →  1.0   (O(n) — same as GA)
    ACO  →  n     (O(n²) construction, n times more expensive than GA)

Usage
-----
    from core.eval_norm import work_factor, normalize_evals, add_normalized_evals

    # Single value
    nev = normalize_evals(evals_cost=50_000, algo="HC", n=30)   # → 1666.7

    # Pandas DataFrame of trace rows
    df = add_normalized_evals(df, n_col="n_cities")
"""

from __future__ import annotations

from typing import Union
import numpy as np

# ── Known algorithms and their complexity class ──────────────────────────────

#: Maps algorithm name → complexity exponent k where cost ∝ n^k
#: HC/SA: k=0  (O(1) delta)
#: GA/TLBO: k=1  (O(n) full eval)
#: ACO: k=2  (O(n²) construction)
_COMPLEXITY: dict[str, int] = {
    "HC":   0,
    "SA":   0,
    "GA":   1,
    "TLBO": 1,
    "ACO":  2,
}

# Reference complexity = 1 (O(n) full tour eval — GA/TLBO baseline)
_REF_K: int = 1


def work_factor(algo: str, n: int) -> float:
    """
    Return the multiplier that converts raw evals_cost → normalized evals
    relative to one O(n) full tour evaluation.

    Parameters
    ----------
    algo : str
        Algorithm name: "HC", "SA", "GA", "TLBO", or "ACO".
    n : int
        Problem size (number of cities).

    Returns
    -------
    float
        Multiplier such that:  normalized = raw_evals * work_factor(algo, n)

    Examples
    --------
    >>> work_factor("HC", 30)    # 1/30 ≈ 0.0333
    >>> work_factor("GA", 30)    # 1.0
    >>> work_factor("ACO", 30)   # 30.0
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    algo_upper = algo.upper()
    if algo_upper not in _COMPLEXITY:
        raise ValueError(
            f"Unknown algorithm '{algo}'. Known: {list(_COMPLEXITY)}"
        )
    k = _COMPLEXITY[algo_upper]
    # factor = n^(k - ref_k) = n^(k - 1)
    exponent = k - _REF_K
    return float(n ** exponent)


def normalize_evals(
    evals_cost: Union[int, float, np.ndarray],
    algo: str,
    n: int,
) -> Union[float, np.ndarray]:
    """
    Convert raw evals_cost → normalized evals (O(n)-equivalent units).

    Parameters
    ----------
    evals_cost : int | float | np.ndarray
        Raw evaluation count(s) from RunResult or trace array.
    algo : str
        Algorithm name.
    n : int
        Problem size (number of cities).

    Returns
    -------
    float or np.ndarray
        Normalized evaluation count(s).

    Examples
    --------
    >>> normalize_evals(50_000, "HC",  30)   # →  1_666.7  (HC is cheap)
    >>> normalize_evals(40_000, "GA",  30)   # → 40_000.0  (reference)
    >>> normalize_evals(50_010, "ACO", 30)   # → 1_500_300 (ACO is expensive)
    """
    factor = work_factor(algo, n)
    if isinstance(evals_cost, np.ndarray):
        return evals_cost.astype(float) * factor
    return float(evals_cost) * factor


def budget_for_algo(target_normalized: int, algo: str, n: int) -> int:
    """
    Given a target normalized budget, return the raw evals_cost limit
    you should configure for a given algorithm so it spends exactly that
    much normalized budget.

    Useful for setting n_ants*iters or max_iter so all algorithms start
    with the same normalized budget.

    Parameters
    ----------
    target_normalized : int
        Desired normalized eval budget (in O(n)-equivalent units).
    algo : str
        Algorithm name.
    n : int
        Problem size.

    Returns
    -------
    int
        Raw evals_cost cap to use when constructing the solver.

    Examples
    --------
    >>> budget_for_algo(40_000, "HC",  30)   # → 1_200_000  (HC is cheap, give it more)
    >>> budget_for_algo(40_000, "GA",  30)   # →    40_000  (unchanged)
    >>> budget_for_algo(40_000, "ACO", 30)   # →     1_333  (ACO is expensive)
    """
    factor = work_factor(algo, n)
    if factor <= 0:
        raise ValueError("work_factor must be positive")
    return max(1, int(round(target_normalized / factor)))


def add_normalized_evals(df, n_col: str = "n_cities") -> "pd.DataFrame":
    """
    Add a 'normalized_evals' column to a pandas DataFrame of run/trace rows.

    Expects columns: 'evals_cost', 'algorithm', and the column named by n_col.

    Parameters
    ----------
    df : pd.DataFrame
        Run log or trace log loaded from CSV.
    n_col : str
        Column name holding the problem size (default: "n_cities").

    Returns
    -------
    pd.DataFrame
        Same DataFrame with a new 'normalized_evals' float column appended.

    Example
    -------
    >>> import pandas as pd
    >>> from core.eval_norm import add_normalized_evals
    >>> runs = pd.read_csv("runs.csv")
    >>> runs = add_normalized_evals(runs, n_col="n_cities")
    >>> runs[["algorithm", "evals_cost", "normalized_evals"]].head()
    """
    import pandas as pd  # local import — pandas is optional

    df = df.copy()
    df["normalized_evals"] = df.apply(
        lambda row: normalize_evals(
            evals_cost=float(row["evals_cost"]),
            algo=str(row["algorithm"]),
            n=int(row[n_col]),
        ),
        axis=1,
    )
    return df


def budget_table(target_normalized: int, n: int) -> dict[str, dict]:
    """
    Print/return a comparison table showing the raw vs normalized budget
    for every algorithm at a given n and target budget.

    Parameters
    ----------
    target_normalized : int
        Normalized budget target (in O(n)-equivalent units).
    n : int
        Problem size.

    Returns
    -------
    dict
        Keys are algorithm names. Values are dicts with:
          raw_budget, work_factor, normalized_budget_check
    """
    result = {}
    for algo in _COMPLEXITY:
        raw = budget_for_algo(target_normalized, algo, n)
        wf  = work_factor(algo, n)
        result[algo] = {
            "work_factor":             wf,
            "raw_budget":              raw,
            "normalized_budget_check": raw * wf,
        }
    return result


# ── Pretty-print helper ───────────────────────────────────────────────────────

def print_budget_table(target_normalized: int, n: int) -> None:
    """
    Print a human-readable budget comparison table.

    Example output (target=40_000, n=30):
    ┌────────┬─────────────┬────────────┬───────────────────┐
    │ Algo   │ work_factor │ raw_budget │ normalized_budget │
    ├────────┼─────────────┼────────────┼───────────────────┤
    │ HC     │    0.0333   │  1200000   │       40000.0     │
    │ SA     │    0.0333   │  1200000   │       40000.0     │
    │ GA     │    1.0000   │    40000   │       40000.0     │
    │ TLBO   │    1.0000   │    40000   │       40000.0     │
    │ ACO    │   30.0000   │     1333   │       39990.0     │
    └────────┴─────────────┴────────────┴───────────────────┘
    """
    tbl = budget_table(target_normalized, n)
    header = f"\nBudget table — target_normalized={target_normalized:,}  n={n}\n"
    print(header)
    print(f"{'Algo':<8} {'work_factor':>12} {'raw_budget':>12} {'norm_budget_check':>18}")
    print("─" * 56)
    for algo, v in tbl.items():
        print(
            f"{algo:<8} {v['work_factor']:>12.4f} "
            f"{v['raw_budget']:>12,} "
            f"{v['normalized_budget_check']:>18,.1f}"
        )
    print()
