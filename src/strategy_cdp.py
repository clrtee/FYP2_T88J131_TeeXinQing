from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np

from .config import TrainConfig
from .dp_utils import compute_epsilon_for_training


class CentralDPStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg aggregation strategy for DP-Prox.

    DP noise is injected client-side (see train_utils.train_dp_fedprox_local).
    The server only performs standard weighted-average aggregation.

    If ``cfg.use_central_dp`` is True (legacy mode), a warning is printed and
    the old server-side noise path is used for backward compatibility.
    """

    def __init__(self, cfg: TrainConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self._round = 0

        if cfg.use_central_dp and not cfg.use_dp:
            import warnings
            warnings.warn(
                "\n[DP-Prox] use_central_dp=True is the LEGACY mode (server-side noise).\n"
                "  It gives weaker privacy and lower accuracy than the client-side DP-SGD.\n"
                "  Set use_dp=True in TrainConfig to use the DP-Prox algorithm instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._legacy_cdp = True
            # Keep old parameters for backward compat
            self._old_parameters: Optional[List[np.ndarray]] = None
        else:
            self._legacy_cdp = False

    # ── Logging helper ────────────────────────────────────────────────────
    def _log_dp_info(self, server_round: int) -> None:
        if server_round == 1 and self.cfg.use_dp:
            print(
                f"\n[DP-Prox] CLIENT-SIDE DP-SGD enabled\n"
                f"          C  (clip_norm)      = {self.cfg.dp_clip_norm}\n"
                f"          σ  (noise_mult)     = {self.cfg.dp_noise_multiplier}\n"
                f"          ε  (target)         = {self.cfg.dp_target_epsilon}\n"
                f"          δ                   = {self.cfg.dp_target_delta}\n"
                f"          μ  (prox_mu)        = {self.cfg.prox_mu}\n"
                f"  Per-sample clipping + Gaussian noise applied on each client.\n"
                f"  Server aggregates privatised updates only (FedAvg).\n"
            )

    # ── Main aggregation ──────────────────────────────────────────────────
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[
            Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException
        ]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:

        self._round = server_round
        self._log_dp_info(server_round)

        # ── Legacy Central-DP path (backward compat) ──────────────────────
        if self._legacy_cdp:
            return self._aggregate_legacy_cdp(server_round, results, failures)

        # ── DP-Prox path: standard FedAvg (noise already on client) ───────
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None, {}

        params, metrics = aggregated

        # Attach useful DP metadata to metrics
        if self.cfg.use_dp:
            metrics["dp_clip_norm"] = float(self.cfg.dp_clip_norm)
            metrics["dp_noise_multiplier"] = float(self.cfg.dp_noise_multiplier)
            metrics["dp_target_epsilon"] = float(self.cfg.dp_target_epsilon)
            metrics["prox_mu"] = float(self.cfg.prox_mu)

            # Aggregate per-round epsilon reported by clients (if any)
            round_epsilons = [
                float(res.metrics.get("dp_epsilon", 0.0))
                for _, res in results
                if "dp_epsilon" in res.metrics
            ]
            if round_epsilons:
                metrics["dp_epsilon_mean"] = float(np.mean(round_epsilons))
                metrics["dp_epsilon_max"] = float(np.max(round_epsilons))
        self._last_params = params

        return params, metrics

    # ── Legacy CDP implementation (server-side noise) ──────────────────────
    def _aggregate_legacy_cdp(self, server_round, results, failures):
        """Old Central-DP: clip + noise on the server.  Kept for compat."""
        from .dp_utils import add_central_dp_noise

        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None, {}
        agg_params, metrics = aggregated
        metrics["noise_multiplier"] = float(self.cfg.cdp_noise_multiplier)

        if self._old_parameters is not None:
            client_params_list = [
                fl.common.parameters_to_ndarrays(res.parameters)
                for _, res in results
            ]
            num_examples = [res.num_examples for _, res in results]
            updates = [
                [c - o for c, o in zip(cp, self._old_parameters)]
                for cp in client_params_list
            ]
            old_agg = fl.common.parameters_to_ndarrays(agg_params)
            noisy_update = add_central_dp_noise(
                old_agg, updates, num_examples,
                self.cfg.cdp_clip_norm, self.cfg.cdp_noise_multiplier,
            )
            # Reconstruct noisy global model: old + noisy_update
            total = sum(num_examples)
            noisy_params = [o + u for o, u in zip(self._old_parameters, noisy_update)]
            agg_params = fl.common.ndarrays_to_parameters(noisy_params)

        self._old_parameters = fl.common.parameters_to_ndarrays(agg_params)
        return agg_params, metrics
