"""
core/eval_norm_knapsack.py
==========================
Normalized evaluation budget for 0/1 Knapsack algorithms — makes evals_cost
comparable across algorithms that have different per-eval computational costs.

--------------------------------------------------
All solvers increment evals_cost exactly once per problem.evaluate() call.
But the work done BEFORE evaluate() differs per algorithm:

  HC_KP   →  _flip(x, i)          →  O(1) flip, O(n) evaluate  → total O(n)
  SA_KP   →  _flip(x, i)          →  O(1) flip, O(n) evaluate  → total O(n)
  GA_KP   →  _crossover_1pt + mutate + evaluate
              crossover: O(n), mutate: O(n), evaluate: O(n)     → total O(n) × 3
  TLBO_KP →  _move_towards + evaluate
              move: O(n) diff-scan + mask, evaluate: O(n)       → total O(n) × 2
  ABC_KP  →  _neighbor_binary + evaluate
              neighbor: O(n) flatnonzero, evaluate: O(n)        → total O(n) × 2

Reference unit
--------------
1 kp_eval = one call to problem.evaluate(x)
          = O(n) dot products (n multiplications + n additions over item vector)

This is the natural reference for Knapsack — it is the minimum indivisible
unit of information gain (you learn the cost of exactly one solution).

Work factors (relative to 1 kp_eval = 1 evaluate() call)
----------------------------------------------------------
  HC_KP   →  1.0   flip is O(1), evaluate is the only O(n) work → factor 1
  SA_KP   →  1.0   same as HC
  GA_KP   →  3.0   crossover(n) + mutate(n) + evaluate(n) per logged eval
  TLBO_KP →  2.0   move_towards(n) + evaluate(n) per logged eval
  ABC_KP  →  2.0   neighbor_binary(n) + evaluate(n) per logged eval

Note on repair()
----------------
repair() costs O(n log n) (argsort + greedy removal loop) and is called by
ALL algorithms when feasible_only=True, BEFORE evaluate().  This cost is not
counted in evals_cost by any solver.  Because it affects all algorithms
equally (one repair per candidate, same problem), it does not change the
RELATIVE ordering of normalized_evals across algorithms — it would add a
constant multiplier to everyone.  We therefore exclude it from work_factor
for simplicity, but document it here so the user is aware.

Formula
-------
    normalized_evals = raw_evals_cost × work_factor(algo)

    (n does not appear — all KP operations are O(n) so the factor is constant)

Usage
-----
    from core.eval_norm_knapsack import (
        work_factor_kp, normalize_evals_kp, add_normalized_evals_kp,
        budget_for_algo_kp, print_budget_table_kp,
    )

    # Single value
    nev = normalize_evals_kp(evals_cost=50_000, algo="HC_KP")   # → 50_000.0
    nev = normalize_evals_kp(evals_cost=50_000, algo="GA_KP")   # → 150_000.0

    # Pandas DataFrame of trace/run rows
    df = add_normalized_evals_kp(df)

    # Raw budget to configure a solver so it hits a normalized target
    raw = budget_for_algo_kp(target_normalized=50_000, algo="GA_KP")  # → 16_667
"""

from __future__ import annotations

from typing import Union
import numpy as np


# ── Work factors ──────────────────────────────────────────────────────────────
#
# Key: algorithm name as logged by each solver (solver.name).
# Value: multiplier — how many O(n) evaluate()-equivalent units one raw
#        evals_cost unit actually represents.
#
# Accepted names are the canonical solver .name values AND short aliases so
# callers can use either "HC_KP" (from solver.name) or "HC" (short form).

_WORK_FACTORS: dict[str, float] = {
    # HC_KP: flip is O(1), only evaluate() is O(n) work → 1× reference
    "HC_KP": 1.0,
    "HC":    1.0,

    # SA_KP: identical move operator to HC (single bit-flip) → 1× reference
    "SA_KP": 1.0,
    "SA":    1.0,

    # GA_KP: crossover_1pt O(n) + mutate O(n) + evaluate O(n) = 3× per logged eval
    "GA_KP": 3.0,
    "GA":    3.0,

    # TLBO_KP: _move_towards O(n) diff-scan + mask + evaluate O(n) = 2× per logged eval
    "TLBO_KP": 2.0,
    "TLBO":    2.0,

    # ABC_KP: _neighbor_binary O(n) flatnonzero + evaluate O(n) = 2× per logged eval
    "ABC_KP": 2.0,
    "ABC":    2.0,
}


def work_factor_kp(algo: str) -> float:
    """
    Return the multiplier that converts raw evals_cost → normalized evals
    for a Knapsack algorithm, relative to one O(n) evaluate() call.

    Parameters
    ----------
    algo : str
        Algorithm name — accepts both full solver names (e.g. "HC_KP") and
        short aliases (e.g. "HC").  Case-insensitive.

    Returns
    -------
    float
        Multiplier such that:  normalized = raw_evals * work_factor_kp(algo)

    Raises
    ------
    ValueError
        If algo is not a known Knapsack algorithm.

    Examples
    --------
    >>> work_factor_kp("HC_KP")    # → 1.0
    >>> work_factor_kp("GA_KP")    # → 3.0
    >>> work_factor_kp("ABC_KP")   # → 2.0
    """
    key = algo.upper()
    if key not in _WORK_FACTORS:
        known = sorted(set(_WORK_FACTORS.keys()))
        raise ValueError(
            f"Unknown Knapsack algorithm '{algo}'. "
            f"Known names: {known}"
        )
    return _WORK_FACTORS[key]


def normalize_evals_kp(
    evals_cost: Union[int, float, np.ndarray],
    algo: str,
) -> Union[float, np.ndarray]:
    """
    Convert raw evals_cost → normalized evals (evaluate()-equivalent units)
    for a Knapsack algorithm.

    Parameters
    ----------
    evals_cost : int | float | np.ndarray
        Raw evaluation count(s) from RunResult or trace array.
    algo : str
        Algorithm name (e.g. "HC_KP", "GA_KP", "ABC_KP", or short "HC", "GA").

    Returns
    -------
    float or np.ndarray
        Normalized evaluation count(s).

    Examples
    --------
    >>> normalize_evals_kp(50_000, "HC_KP")    # → 50_000.0  (reference)
    >>> normalize_evals_kp(50_000, "GA_KP")    # → 150_000.0 (3× heavier per eval)
    >>> normalize_evals_kp(50_000, "TLBO_KP")  # → 100_000.0 (2× heavier)
    >>> normalize_evals_kp(50_000, "ABC_KP")   # → 100_000.0 (2× heavier)
    """
    factor = work_factor_kp(algo)
    if isinstance(evals_cost, np.ndarray):
        return evals_cost.astype(float) * factor
    return float(evals_cost) * factor


def budget_for_algo_kp(target_normalized: int, algo: str) -> int:
    """
    Given a target normalized budget (in evaluate()-equivalent units), return
    the raw evals_cost limit to configure the solver so it spends exactly that
    much normalized budget.

    Use this when setting iters / max_iter so all algorithms start with the
    same normalized budget and convergence curves are directly comparable.

    Parameters
    ----------
    target_normalized : int
        Desired normalized eval budget (in evaluate()-equivalent units).
    algo : str
        Algorithm name.

    Returns
    -------
    int
        Raw evals_cost cap to pass to the solver constructor.

    Examples
    --------
    >>> budget_for_algo_kp(50_000, "HC_KP")    # → 50_000  (unchanged — reference)
    >>> budget_for_algo_kp(50_000, "GA_KP")    # → 16_667  (gets fewer raw evals)
    >>> budget_for_algo_kp(50_000, "ABC_KP")   # → 25_000  (gets fewer raw evals)
    """
    factor = work_factor_kp(algo)
    return max(1, int(round(target_normalized / factor)))


def add_normalized_evals_kp(df, algo_col: str = "algorithm") -> "pd.DataFrame":
    """
    Add a 'normalized_evals' column to a pandas DataFrame of KP run/trace rows.

    Expects columns: 'evals_cost' and the column named by algo_col.
    Also adds 'normalized_evals_best_found' if 'evals_best_found' is present.

    Parameters
    ----------
    df : pd.DataFrame
        Run log or trace log loaded from CSV.
    algo_col : str
        Column name holding the algorithm name (default: "algorithm").

    Returns
    -------
    pd.DataFrame
        Same DataFrame with new float columns appended.

    Example
    -------
    >>> import pandas as pd
    >>> from core.eval_norm_knapsack import add_normalized_evals_kp
    >>> runs = pd.read_csv("runs_knapsack.csv")
    >>> runs = add_normalized_evals_kp(runs)
    >>> runs[["algorithm", "evals_cost", "normalized_evals"]].head()
    """
    import pandas as pd  # local import — pandas is optional

    df = df.copy()

    def _norm(row: "pd.Series") -> float:
        try:
            return float(
                normalize_evals_kp(float(row["evals_cost"]), str(row[algo_col]))
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
                    normalize_evals_kp(float(ebf), str(row[algo_col]))
                )
            except Exception:
                return float("nan")

        df["normalized_evals_best_found"] = df.apply(_norm_best, axis=1)

    return df


def budget_table_kp(target_normalized: int) -> dict[str, dict]:
    """
    Return a comparison table showing raw vs normalized budget for every
    KP algorithm at a given target.

    Parameters
    ----------
    target_normalized : int
        Normalized budget target (in evaluate()-equivalent units).

    Returns
    -------
    dict
        Keys are canonical algo names. Values are dicts with:
          work_factor, raw_budget, normalized_budget_check
    """
    # Use canonical (long) names only
    canonical = ["HC_KP", "SA_KP", "GA_KP", "TLBO_KP", "ABC_KP"]
    result = {}
    for algo in canonical:
        wf  = work_factor_kp(algo)
        raw = budget_for_algo_kp(target_normalized, algo)
        result[algo] = {
            "work_factor":             wf,
            "raw_budget":              raw,
            "normalized_budget_check": raw * wf,
        }
    return result


def print_budget_table_kp(target_normalized: int) -> None:
    """
    Print a human-readable budget comparison table for all KP algorithms.

    Example output (target=50_000):

    Budget table (Knapsack) — target_normalized=50,000
    ──────────────────────────────────────────────────────────
    Algo       work_factor   raw_budget   norm_budget_check
    ──────────────────────────────────────────────────────────
    HC_KP           1.0000       50,000            50,000.0
    SA_KP           1.0000       50,000            50,000.0
    GA_KP           3.0000       16,667            50,001.0  ← rounding
    TLBO_KP         2.0000       25,000            50,000.0
    ABC_KP          2.0000       25,000            50,000.0
    """
    tbl = budget_table_kp(target_normalized)
    print(f"\nBudget table (Knapsack) — target_normalized={target_normalized:,}\n")
    print(f"{'Algo':<10} {'work_factor':>12} {'raw_budget':>12} {'norm_budget_check':>18}")
    print("─" * 58)
    for algo, v in tbl.items():
        print(
            f"{algo:<10} {v['work_factor']:>12.4f} "
            f"{v['raw_budget']:>12,} "
            f"{v['normalized_budget_check']:>18,.1f}"
        )
    print()


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== work_factor_kp ===")
    for name in ["HC_KP", "SA_KP", "GA_KP", "TLBO_KP", "ABC_KP"]:
        print(f"  {name:<10}  factor={work_factor_kp(name):.1f}")

    print()
    print("=== normalize_evals_kp (raw=50,000) ===")
    for name in ["HC_KP", "SA_KP", "GA_KP", "TLBO_KP", "ABC_KP"]:
        print(f"  {name:<10}  normalized={normalize_evals_kp(50_000, name):>10,.1f}")

    print()
    print_budget_table_kp(50_000)

    print("=== short aliases ===")
    assert work_factor_kp("HC") == work_factor_kp("HC_KP")
    assert work_factor_kp("abc") == work_factor_kp("ABC_KP")
    print("  aliases OK")

    print()
    print("=== ValueError on unknown algo ===")
    try:
        work_factor_kp("ACO")
    except ValueError as e:
        print(f"  Caught: {e}")
