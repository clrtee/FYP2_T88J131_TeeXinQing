# src/utils_history.py
from __future__ import annotations

import os
import csv
from typing import Any, Dict, List, Tuple


def _dict_metrics_to_round_map(d: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Convert Flower History metrics dict format:
      {
        "accuracy": [(1, 0.3), (2, 0.4), ...],
        "f1":       [(1, 0.2), (2, 0.35), ...],
      }
    into:
      { 1: {"accuracy": 0.3, "f1": 0.2}, 2: {"accuracy": 0.4, "f1": 0.35}, ... }
    """
    out: Dict[int, Dict[str, Any]] = {}

    for metric_name, series in d.items():
        if series is None:
            continue

        # series should be list of (round, value)
        if isinstance(series, (list, tuple)):
            for item in series:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                try:
                    r = int(item[0])
                    v = item[1]
                except Exception:
                    continue

                if r not in out:
                    out[r] = {}
                out[r][metric_name] = v

    return out


def save_history_csv(history, out_path: str) -> None:
    """
    Save per-round CSV containing:
      - losses_distributed
      - metrics_distributed (evaluate)
      - metrics_distributed_fit (fit)
    Works with Flower History format where metrics_* are dict(metric_name -> [(round, value), ...]).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ---- Loss ----
    loss_by_round: Dict[int, float] = {}
    losses = getattr(history, "losses_distributed", None)
    if isinstance(losses, (list, tuple)):
        for item in losses:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    loss_by_round[int(item[0])] = float(item[1])
                except Exception:
                    pass

    # ---- Evaluate metrics ----
    eval_raw = getattr(history, "metrics_distributed", None)
    eval_metrics: Dict[int, Dict[str, Any]] = {}
    if isinstance(eval_raw, dict):
        eval_metrics = _dict_metrics_to_round_map(eval_raw)

    # ---- Fit metrics ----
    fit_raw = getattr(history, "metrics_distributed_fit", None)
    fit_metrics: Dict[int, Dict[str, Any]] = {}
    if isinstance(fit_raw, dict):
        fit_metrics = _dict_metrics_to_round_map(fit_raw)

    all_rounds = sorted(set(loss_by_round.keys()) | set(eval_metrics.keys()) | set(fit_metrics.keys()))
    if not all_rounds:
        with open(out_path, "w", newline="") as f:
            f.write("")
        return

    # Header keys
    keys = set()
    for r in all_rounds:
        keys.update(eval_metrics.get(r, {}).keys())
        keys.update(fit_metrics.get(r, {}).keys())

    preferred = [
        "round",
        "loss",
        "accuracy",
        "f1",
        "epsilon",
        "fit_wall_s",
        "fit_sim_s",
        "cpu_pct",
        "energy_wh",
        "device_key",
        "device_profile",
    ]
    remaining = sorted([k for k in keys if k not in preferred])
    header = [c for c in preferred if (c == "round" or c == "loss" or c in keys)] + remaining

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for r in all_rounds:
            row: Dict[str, Any] = {"round": int(r)}
            if r in loss_by_round:
                row["loss"] = loss_by_round[r]
            row.update(eval_metrics.get(r, {}))
            row.update(fit_metrics.get(r, {}))
            writer.writerow(row)
