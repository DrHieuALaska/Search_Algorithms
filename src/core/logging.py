from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional


class RunLogger:
    """
    CSV logger:
    - run_csv: 1 row per run (stats)
    - trace_csv: checkpoint rows (convergence)
    """

    def __init__(self, run_csv: Optional[str] = None, trace_csv: Optional[str] = None):
        self.run_csv = run_csv
        self.trace_csv = trace_csv

        # Minimal headers (no notes, no accept rate)
        self.run_header: List[str] = [
            "experiment_id","run_id","timestamp_utc","algorithm","problem",
            "n_cities","instance_seed","coord_scale","distance_type",
            "init_temp_T0","min_temp_Tmin","alpha","steps_per_temp","max_iter","neighbor_operator",
            "iters","evals_cost","time_sec",
            "init_cost","final_cost","best_cost",
            "iter_best_found","evals_best_found","time_best_found_sec",
            "seed_algo","code_version"
        ]

        self.trace_header: List[str] = [
            "experiment_id","run_id",
            "iter","evals_cost","time_sec",
            "temp","best_cost","current_cost"
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

    def log_run(self, row: Dict[str, Any]) -> None:
        if self.run_csv is None:
            return
        self._append_row(self.run_csv, self.run_header, row)

    def log_trace(self, row: Dict[str, Any]) -> None:
        if self.trace_csv is None:
            return
        self._append_row(self.trace_csv, self.trace_header, row)
