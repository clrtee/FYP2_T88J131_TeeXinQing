# src/strategy_feddc.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
import flwr as fl
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy

from .fl_client import pack_tensors, unpack_tensors
from .dp_utils import compute_updates, add_central_dp_noise


class FedDCStrategy(fl.server.strategy.FedAvg):
    """
    FedDC Strategy with optional Central DP support.
    """

    def __init__(
        self,
        penalty_alpha: float = 0.01,
        use_central_dp: bool = False,
        cdp_clip_norm: float = 5.0,
        cdp_noise_multiplier: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.penalty_alpha = penalty_alpha
        self.g_prev: Optional[List[np.ndarray]] = None
        self.g_prev_blob: Optional[bytes] = None

        self.use_central_dp = use_central_dp
        self.cdp_clip_norm = cdp_clip_norm
        self.cdp_noise_multiplier = cdp_noise_multiplier
        self.last_global_params: Optional[List[np.ndarray]] = None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        config: Dict[str, Any] = {
            "algo": "feddc",
            "feddc_penalty_alpha": float(self.penalty_alpha),
        }

        if server_round > 1 and self.g_prev is not None:
            config["g_prev"] = self.g_prev_blob

        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        new_list = []
        for client, fit_ins in fit_ins_list:
            cfg = dict(fit_ins.config)
            cfg.update(config)
            new_list.append((client, fl.common.FitIns(fit_ins.parameters, cfg)))
        return new_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Any]]:
        if results:
            self.last_global_params = fl.common.parameters_to_ndarrays(results[0][1].parameters)

        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if not results or aggregated_parameters is None:
            return aggregated_parameters, metrics

        # Central DP noise injection
        if self.use_central_dp and self.last_global_params is not None:
            client_updates = []
            client_weights = []
            for _, fit_res in results:
                client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                update = [c - o for c, o in zip(client_params, self.last_global_params)]
                client_updates.append(update)
                client_weights.append(fit_res.num_examples)

            agg_params = fl.common.parameters_to_ndarrays(aggregated_parameters)
            noisy_avg_update = add_central_dp_noise(
                aggregated_ndarrays=agg_params,
                client_updates=client_updates,
                num_examples_list=client_weights,
                clip_norm=self.cdp_clip_norm,
                noise_multiplier=self.cdp_noise_multiplier,
            )
            new_params = [o + n for o, n in zip(self.last_global_params, noisy_avg_update)]
            aggregated_parameters = fl.common.ndarrays_to_parameters(new_params)

            sigma = self.cdp_clip_norm * self.cdp_noise_multiplier
            print(f"[Round {server_round}] FedDC Central DP noise added (σ = {sigma:.3f})")

        # Update FedDC auxiliary variable g_prev
        if results:
            ref_ndarrays = fl.common.parameters_to_ndarrays(results[0][1].parameters)
            ref_tensors = [torch.from_numpy(p).cpu() for p in ref_ndarrays]

            gi_items = []
            for _, fit_res in results:
                if fit_res.metrics and "gi" in fit_res.metrics:
                    blob = fit_res.metrics["gi"]
                    gi_tensors = unpack_tensors(blob, torch.device("cpu"), ref_tensors)
                    gi_np = [t.detach().cpu().numpy() for t in gi_tensors]
                    gi_items.append((gi_np, fit_res.num_examples))

            if gi_items:
                total = max(1, sum(n for _, n in gi_items))
                g_new: List[np.ndarray] = []
                for layer_idx in range(len(gi_items[0][0])):
                    s = sum(arrs[layer_idx].astype(np.float64) * n for arrs, n in gi_items)
                    g_new.append((s / total).astype(gi_items[0][0][layer_idx].dtype))
                self.g_prev = g_new
                self.g_prev_blob = pack_tensors([torch.from_numpy(p) for p in self.g_prev])

        return aggregated_parameters, metrics