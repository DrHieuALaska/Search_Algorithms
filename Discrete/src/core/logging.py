from __future__ import annotations

import csv
import os
import re
from typing import Any, Dict, List, Optional

from core.eval_norm_tsp import normalize_evals
from core.eval_norm_knapsack import normalize_evals_kp
from core.eval_norm_gc import normalize_evals_gc

_KP_ALGOS: frozenset[str] = frozenset({
    "HC_KP", "SA_KP", "GA_KP", "TLBO_KP", "ABC_KP",
})

_GC_ALGOS: frozenset[str] = frozenset({
    "HC_GC", "SA_GC", "GA_GC", "ACO_GC", "DFS_GC",
})

def _infer_algo(rid: str) -> str:
    prefix = rid.split("[", 1)[0]              
    return re.sub(r"_n\d+.*$", "", prefix)

class RunLogger:
    """
    CSV logger:
    - run_csv: 1 row per run (stats)
    - trace_csv: checkpoint rows (convergence)

    Adds 'param_tag' column and auto-extracts it from run_id if not provided.
    Expected run_id format for sweeps:
      ALGO[tag]_n30_inst0_run0
    """

    _TAG_RE = re.compile(r"\[(.*?)\]")

    def __init__(self, run_csv: Optional[str] = None, trace_csv: Optional[str] = None):
        self.run_csv = run_csv
        self.trace_csv = trace_csv

        # ✅ Added param_tag right after run_id
        self.run_header: List[str] = [
            "experiment_id", "run_id", "param_tag", "timestamp_utc", "algorithm", "problem",
            "n_cities", "instance_seed", "coord_scale", "distance_type",
            "init_temp_T0", "min_temp_Tmin", "alpha", "steps_per_temp", "max_iter", "neighbor_operator",
            "iters", "evals_cost", "normalized_evals", "time_sec",
            "init_cost", "final_cost", "best_cost",
            "iter_best_found", "evals_best_found", "normalized_evals_best_found", "time_best_found_sec",
            "seed_algo", "code_version"
        ]

        # ✅ Added param_tag right after run_id
        self.trace_header: List[str] = [
            "experiment_id", "run_id", "param_tag",
            "iter", "evals_cost", "normalized_evals", "time_sec",
            "temp", "best_cost", "current_cost"
        ]

    @staticmethod
    def _append_row(path: str, header: List[str], row: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in header})

    def _ensure_param_tag(self, row: Dict[str, Any]) -> None:
        """
        If param_tag is missing/blank, try to extract it from run_id like ALGO[tag]_...
        If not found, set to 'base'.
        """
        tag = str(row.get("param_tag", "")).strip()
        if tag:
            return

        rid = str(row.get("run_id", ""))
        m = self._TAG_RE.search(rid)
        if m:
            row["param_tag"] = m.group(1)
        else:
            row["param_tag"] = "base"

    def _inject_normalized(self, row: Dict[str, Any]) -> None:
        """
        Routing
        -------
        KP  algos (HC_KP, SA_KP, GA_KP, TLBO_KP, ABC_KP):
            Uses eval_norm_knapsack.  Work factor is constant — n is NOT needed.

        GC  algos (HC_GC, SA_GC, GA_GC, ACO_GC, DFS_GC):
            Uses eval_norm_gc.  Requires n — parsed from n_cities column or
            from '_n{n}_' in the run_id.

        TSP algos (HC, SA, GA, TLBO, ACO):
            Uses eval_norm (TSP).  Requires n — same parsing as GC.
        """
        rid = str(row.get("run_id", ""))

        # ── resolve algo name ────────────────────────────────────────────────
        algo = str(row.get("algorithm", "")).strip()
        if not algo:
            if not rid:
                return
            algo = _infer_algo(rid)
        if not algo:
            return
        
        algo_upper = algo.upper()

        # --- n (cities) ---
        n_raw = row.get("n_cities", None)
        n = None
        if n_raw is not None and str(n_raw).strip() != "":
            try:
                n = int(float(n_raw))
            except ValueError:
                n = None

        if n is None and rid:
            m = re.search(r"_n(\d+)_", rid)
            if m:
                n = int(m.group(1))
        if n is None or n <= 0:
            return

        def _write(norm_fn):
            """Call norm_fn(raw) -> float and write to row if columns not already set."""
            try:
                if "evals_cost" in row and str(row.get("normalized_evals", "")).strip() == "":
                    row["normalized_evals"] = round(norm_fn(float(row["evals_cost"])), 2)

                if "evals_best_found" in row and str(row.get("normalized_evals_best_found", "")).strip() == "":
                    ebf = row.get("evals_best_found", "")
                    if str(ebf).strip() != "":
                        row["normalized_evals_best_found"] = round(norm_fn(float(ebf)), 2)
            except Exception:
                pass

        # ── KP path ──────────────────────────────────────────────────────────
        if algo_upper in _KP_ALGOS:
            _write(lambda raw: normalize_evals_kp(raw, algo_upper))
            return

        # ── TSP & GC path — needs n ───────────────────────────────────────────────
        n: Optional[int] = None

        n_raw = row.get("n_cities", None)
        if n_raw is not None and str(n_raw).strip() != "":
            try:
                n = int(float(n_raw))
            except (ValueError, TypeError):
                pass

        if n is None:
            m = re.search(r"_n(\d+)_", rid)
            if m:
                n = int(m.group(1))

        if n is None or n <= 0:
            return  # cannot normalize TSP without problem size

        # ── GC path ──────────────────────────────────────────────────────────
        if algo_upper in _GC_ALGOS:
            _write(lambda raw: normalize_evals_gc(raw, algo_upper, n))
            return

        # ── TSP path ─────────────────────────────────────────────────────────
        _write(lambda raw: normalize_evals(raw, algo, n))

    def log_run(self, row: Dict[str, Any]) -> None:
        if self.run_csv is None:
            return
        self._ensure_param_tag(row)
        self._inject_normalized(row)
        self._append_row(self.run_csv, self.run_header, row)

    def log_trace(self, row: Dict[str, Any]) -> None:
        if self.trace_csv is None:
            return
        self._ensure_param_tag(row)
        self._inject_normalized(row)
        self._append_row(self.trace_csv, self.trace_header, row)