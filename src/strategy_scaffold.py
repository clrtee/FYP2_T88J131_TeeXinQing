# src/strategy_scaffold.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import pickle
import numpy as np
import flwr as fl
from flwr.common import Parameters, FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


def pack_ndarrays(arrs: List[np.ndarray]) -> bytes:
    return pickle.dumps(arrs, protocol=pickle.HIGHEST_PROTOCOL)


def unpack_ndarrays(blob: bytes) -> List[np.ndarray]:
    return pickle.loads(blob)


def weighted_avg_ndarrays(items: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    # items: (arr_list, num_examples)
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


class ScaffoldStrategy(fl.server.strategy.FedAvg):
    """
    Full SCAFFOLD server:
    - maintains c_global
    - sends c_global to clients each round
    - aggregates delta_c from clients to update c_global
    """

    def __init__(self, *args, **kwargs):
        self.c_global: Optional[List[np.ndarray]] = None
        super().__init__(*args, **kwargs)

    def initialize_parameters(self, client_manager):
        params = super().initialize_parameters(client_manager)
        if params is None:
            return None
        nds = parameters_to_ndarrays(params)
        # init c_global zeros with same shapes
        self.c_global = [np.zeros_like(x) for x in nds]
        return params

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        if self.c_global is None:
            # fallback init
            nds = parameters_to_ndarrays(parameters)
            self.c_global = [np.zeros_like(x) for x in nds]

        c_blob = pack_ndarrays(self.c_global)

        # inject into fit config for each client
        new_list = []
        for client, fit_ins in fit_ins_list:
            cfg = dict(fit_ins.config)
            cfg["algo"] = "scaffold"
            cfg["c_global"] = c_blob  # bytes (Flower-safe)
            new_list.append((client, fl.common.FitIns(parameters, cfg)))
        return new_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        # 1) update c_global from delta_c
        delta_items: List[Tuple[List[np.ndarray], int]] = []
        for _, fit_res in results:
            if fit_res.metrics is None:
                continue
            blob = fit_res.metrics.get("delta_c", None)
            if blob is None:
                continue
            arrs = unpack_ndarrays(blob)
            delta_items.append((arrs, fit_res.num_examples))

        if self.c_global is not None and len(delta_items) > 0:
            delta_avg = weighted_avg_ndarrays(delta_items)
            self.c_global = [cg + dc for cg, dc in zip(self.c_global, delta_avg)]

        # 2) aggregate model parameters normally (FedAvg)
        return super().aggregate_fit(server_round, results, failures)
