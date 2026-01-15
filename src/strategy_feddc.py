# src/strategy_feddc.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import pickle
import numpy as np
import flwr as fl
from flwr.common import Parameters, FitRes, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


def pack_ndarrays(arrs: List[np.ndarray]) -> bytes:
    return pickle.dumps(arrs, protocol=pickle.HIGHEST_PROTOCOL)


def unpack_ndarrays(blob: bytes) -> List[np.ndarray]:
    return pickle.loads(blob)


def weighted_avg_ndarrays(items: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    total = sum(n for _, n in items)
    total = max(1, total)
    out: List[np.ndarray] = []
    for layer_idx in range(len(items[0][0])):
        s = None
        for arrs, n in items:
            a = arrs[layer_idx]
            if s is None:
                s = a.astype(np.float64) * n
            else:
                s += a.astype(np.float64) * n
        out.append((s / total).astype(items[0][0][layer_idx].dtype))
    return out


class FedDCStrategy(fl.server.strategy.FedAvg):
    """
    FedDC server (practical version for FYP):
    - maintains g_prev (average update from last round)
    - sends g_prev to clients each round
    - aggregates gi returned by clients to update g_prev
    """

    def __init__(self, penalty_alpha: float = 1.0, *args, **kwargs):
        self.penalty_alpha = penalty_alpha
        self.g_prev: Optional[List[np.ndarray]] = None
        super().__init__(*args, **kwargs)

    def initialize_parameters(self, client_manager):
        params = super().initialize_parameters(client_manager)
        if params is None:
            return None
        nds = parameters_to_ndarrays(params)
        self.g_prev = [np.zeros_like(x) for x in nds]
        return params

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        if self.g_prev is None:
            nds = parameters_to_ndarrays(parameters)
            self.g_prev = [np.zeros_like(x) for x in nds]

        g_blob = pack_ndarrays(self.g_prev)

        new_list = []
        for client, fit_ins in fit_ins_list:
            cfg = dict(fit_ins.config)
            cfg["algo"] = "feddc"
            cfg["feddc_penalty_alpha"] = float(self.penalty_alpha)
            cfg["g_prev"] = g_blob
            new_list.append((client, fl.common.FitIns(parameters, cfg)))
        return new_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        # 1) update g_prev from gi
        gi_items: List[Tuple[List[np.ndarray], int]] = []
        for _, fit_res in results:
            if fit_res.metrics is None:
                continue
            blob = fit_res.metrics.get("gi", None)
            if blob is None:
                continue
            arrs = unpack_ndarrays(blob)
            gi_items.append((arrs, fit_res.num_examples))

        if self.g_prev is not None and len(gi_items) > 0:
            gi_avg = weighted_avg_ndarrays(gi_items)
            # simplest: g_prev <- gi_avg (works as "last round average update")
            self.g_prev = gi_avg

        # 2) aggregate model parameters normally (FedAvg)
        return super().aggregate_fit(server_round, results, failures)
