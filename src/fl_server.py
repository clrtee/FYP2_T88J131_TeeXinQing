# src/fl_server.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np

from .config import TrainConfig
from .dp_utils import compute_epsilon_for_training
from .strategy_scaffold import ScaffoldStrategy
from .strategy_feddc import FedDCStrategy
from .strategy_cdp import CentralDPStrategy


# ── Metric aggregation helpers ─────────────────────────────────────────────

def weighted_average_eval(
    metrics: List[Tuple[int, Dict[str, Any]]]
) -> Dict[str, float]:
    total = max(1, sum(n for n, _ in metrics))
    return {
        "accuracy": sum(n * float(m.get("accuracy", 0.0)) for n, m in metrics) / total,
        "f1":       sum(n * float(m.get("f1",       0.0)) for n, m in metrics) / total,
    }


def weighted_average_fit(
    metrics: List[Tuple[int, Dict[str, Any]]]
) -> Dict[str, Any]:
    if not metrics:
        return {}
    total_samples = max(1, sum(n for n, _ in metrics))
    aggregated: Dict[str, float] = {}
    for n, m in metrics:
        weight = n / total_samples
        for key, val in m.items():
            if isinstance(val, (int, float)):
                aggregated[key] = aggregated.get(key, 0.0) + val * weight
    return aggregated


# ── Server entry point ─────────────────────────────────────────────────────

def run_server(
    algo: str,
    rounds: int = 150,
    prox_mu: float = 0.01,
    dataset: str = "uci",
    use_dp: bool = False,
):
    cfg = TrainConfig()
    cfg.use_dp = use_dp          # Sync CLI flag → config

    # ── Pre-training privacy budget estimate ──────────────────────────────
    if use_dp:
        # Rough estimate: assumes each client has ~dataset_size/num_clients samples.
        # Actual ε is tracked per-round via client metrics (dp_epsilon).
        approx_dataset_per_client = 700   # conservative estimate for UCI HAR
        est_eps = compute_epsilon_for_training(
            num_rounds=rounds,
            local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size,
            dataset_size=approx_dataset_per_client,
            noise_multiplier=cfg.dp_noise_multiplier,
            delta=cfg.dp_target_delta,
        )
        print(
            f"\n[DP-Prox] DP-SGD configuration"
            f"\n  Algo              : {algo}"
            f"\n  clip_norm  (C)    : {cfg.dp_clip_norm}"
            f"\n  noise_mult (σ)    : {cfg.dp_noise_multiplier}"
            f"\n  target ε          : {cfg.dp_target_epsilon}"
            f"\n  target δ          : {cfg.dp_target_delta}"
            f"\n  prox_mu    (μ)    : {prox_mu}"
            f"\n  rounds            : {rounds}"
            f"\n  local_epochs      : {cfg.local_epochs}"
            f"\n  batch_size        : {cfg.batch_size}"
            f"\n  Estimated total ε : {est_eps:.2f}  "
            f"(target ≤ {cfg.dp_target_epsilon})\n"
        )
        if est_eps > cfg.dp_target_epsilon * 1.5:
            print(
                f"  ⚠  Estimated ε ({est_eps:.2f}) exceeds target "
                f"({cfg.dp_target_epsilon}).  "
                f"Consider increasing noise_multiplier or reducing rounds.\n"
            )
    else:
        print(f"\n[SERVER] Algo={algo} | DP=OFF | Rounds={rounds}\n")

    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=int(cfg.num_clients),
        min_evaluate_clients=int(cfg.num_clients),
        min_available_clients=int(cfg.num_clients),
        evaluate_metrics_aggregation_fn=weighted_average_eval,
        fit_metrics_aggregation_fn=weighted_average_fit,
    )

    if algo.lower() in ["fedavg", "fedprox"]:
        strategy = CentralDPStrategy(
            cfg=cfg,
            **common_kwargs,
            on_fit_config_fn=lambda r: {
                "algo": algo,
                "prox_mu": float(prox_mu),
            },
        )
    elif algo.lower() == "scaffold":
        strategy = ScaffoldStrategy(cfg=cfg, **common_kwargs)
    elif algo.lower() == "feddc":
        strategy = FedDCStrategy(
            penalty_alpha=float(cfg.feddc_penalty_alpha),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=int(rounds)),
        strategy=strategy,
    )

    # 保存最终全局模型
    import torch
    from .models import CNN1DClassifier
    from collections import OrderedDict

    final_ndarrays = fl.common.parameters_to_ndarrays(strategy._last_params)
    model = CNN1DClassifier(input_dim=9, num_classes=6)
    keys = list(model.state_dict().keys())
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(keys, final_ndarrays)}
    )
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), "pretrained_9ch.pth")
    print("\n✅ 预训练模型已保存：pretrained_9ch.pth")

    return history
