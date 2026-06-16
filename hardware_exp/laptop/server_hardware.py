"""
hardware_exp/laptop/server_hardware.py  —  run on LAPTOP
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))

from src.config import TrainConfig
from src.strategy_cdp import CentralDPStrategy
import flwr as fl
import csv
from typing import List, Tuple, Dict, Any

def weighted_average_eval(metrics):
    total = max(1, sum(n for n, _ in metrics))
    acc = sum(n * float(m.get("accuracy", 0.0)) for n, m in metrics) / total
    f1  = sum(n * float(m.get("f1",       0.0)) for n, m in metrics) / total
    return {"accuracy": float(acc), "f1": float(f1)}

def weighted_average_fit(metrics):
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    aggregated = {}
    for n, m in metrics:
        w = n / total
        for k, v in m.items():
            if isinstance(v, (int, float)):
                aggregated[k] = aggregated.get(k, 0.0) + v * w
    return aggregated

cfg = TrainConfig()
cfg.use_central_dp = False

strategy = CentralDPStrategy(
    cfg=cfg,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1,
    evaluate_metrics_aggregation_fn=weighted_average_eval,
    fit_metrics_aggregation_fn=weighted_average_fit,
    on_fit_config_fn=lambda r: {"algo": "fedprox", "prox_mu": 0.01}
)

print("Hardware experiment FL server starting...")
print("Rounds: 50 | Algorithm: FedProx | DP: False")
print("Waiting for RPi on 0.0.0.0:8080 ...")

class SavingStrategy(type(strategy)):
    def aggregate_fit(self, server_round, results, failures):
        agg = super().aggregate_fit(server_round, results, failures)
        if agg and server_round == 50:
            import torch
            from collections import OrderedDict
            from src.models import CNN1DClassifier
            params, _ = agg
            weights = fl.common.parameters_to_ndarrays(params)
            model = CNN1DClassifier(input_dim=9, num_classes=6)
            keys = list(model.state_dict().keys())
            state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, weights)})
            model.load_state_dict(state)
            torch.save(model.state_dict(), "models/fl_final_cnn_ucihar.pth")
            print("[server] Saved final model to models/fl_final_cnn_ucihar.pth")
        return agg

strategy.__class__ = SavingStrategy

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=200),
    strategy=strategy,
)
