from __future__ import annotations

from typing import List, Tuple, Dict, Any
import os
from datetime import datetime

import flwr as fl

from .config import TrainConfig, Paths
from .utils_history import save_history_csv
from .strategy_scaffold import ScaffoldStrategy
from .strategy_feddc import FedDCStrategy


def weighted_average_eval(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    total = sum(n for n, _ in metrics)
    total = max(1, total)
    acc = sum(n * float(m.get("accuracy", 0.0)) for n, m in metrics) / total
    f1 = sum(n * float(m.get("f1", 0.0)) for n, m in metrics) / total
    return {"accuracy": float(acc), "f1": float(f1)}


def weighted_average_fit(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Aggregate only numeric fit metrics.
    Ignore bytes like delta_c / gi (they stay in client metrics but won't break aggregation).
    """
    total = sum(n for n, _ in metrics)
    total = max(1, total)

    def wavg(key: str, default: float = 0.0) -> float:
        return sum(n * float(m.get(key, default)) for n, m in metrics) / total

    out: Dict[str, Any] = {}
    numeric_keys = ["fit_wall_s", "fit_sim_s", "cpu_pct", "energy_wh"]
    for k in numeric_keys:
        if any(k in m for _, m in metrics):
            out[k] = float(wavg(k, 0.0))

    # keep labels (first found)
    for label_key in ["device_key", "device_profile"]:
        for _, m in metrics:
            if label_key in m:
                out[label_key] = m[label_key]
                break

    return out


def run_server(algo: str, rounds: int = 50, prox_mu: float = 0.01, dataset: str = "uci"):
    cfg = TrainConfig()
    paths = Paths()

    algo = algo.lower().strip()
    dataset = dataset.lower().strip()

    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=int(cfg.num_clients),
        min_evaluate_clients=int(cfg.num_clients),
        min_available_clients=int(cfg.num_clients),
        evaluate_metrics_aggregation_fn=weighted_average_eval,
        fit_metrics_aggregation_fn=weighted_average_fit,   # ✅ crucial
    )

    if algo == "fedavg":
        strategy = fl.server.strategy.FedAvg(
            **common_kwargs,
            on_fit_config_fn=lambda r: {"algo": "fedavg"},
        )

    elif algo == "fedprox":
        strategy = fl.server.strategy.FedAvg(
            **common_kwargs,
            on_fit_config_fn=lambda r: {"algo": "fedprox", "prox_mu": float(prox_mu)},
        )

    elif algo == "scaffold":
        strategy = ScaffoldStrategy(**common_kwargs)

    elif algo == "feddc":
        strategy = FedDCStrategy(penalty_alpha=float(cfg.feddc_penalty_alpha), **common_kwargs)

    else:
        raise ValueError(f"Unknown algo: {algo}")

    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=int(rounds)),
        strategy=strategy,
    )

    if history is None:
        print("[CSV] No history returned (server may have stopped early).")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset}_{algo}_r{int(rounds)}_{ts}"
    out_path = os.path.join(paths.out_dir, f"{run_name}.csv")

    print("=== HISTORY ATTRS CHECK ===")
    for name in [
        "losses_distributed",
        "metrics_distributed",
        "metrics_distributed_fit",
        "metrics_fit_distributed",
        "metrics_centralized",
        "metrics_fit",
    ]:
        v = getattr(history, name, None)
        if v is None:
            print(name, "-> None")
        else:
            try:
                print(name, "->", type(v), "len=", len(v))
                if len(v) > 0:
                    print("  first item:", v[0])
            except Exception as e:
                print(name, "->", type(v), "cannot len()", e)
        
        save_history_csv(history, out_path)
        print(f"[CSV] History saved to: {out_path}")
