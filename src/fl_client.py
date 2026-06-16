# src/fl_client.py
# ─────────────────────────────────────────────────────────────────────────────
# HARClient — Flower NumPyClient for Human Activity Recognition.
#
# Supported algorithms
# ────────────────────
#  fedavg / fedprox  – with optional client-side DP-SGD (use_dp=True)
#  scaffold           – with optional client-side DP-SGD (use_dp=True)
#  feddc              – no DP (future work)
#
# DP paths
# ────────
#  FedProx + DP  → train_dp_fedprox_local()  (Algorithm 1 of DP-Prox paper)
#  SCAFFOLD + DP → train_dp_scaffold_local()  (DP-SGD + Option-II c_local update)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import flwr as fl

from .train_utils import (
    make_loader,
    train_one_client,
    train_dp_fedprox_local,
    train_dp_scaffold_local,
    train_scaffold,
    train_feddc,
    evaluate,
)
from .profiler import RoundProfiler
from .device_profiles import PROFILES


# ══════════════════════════════════════════════════════════════════════════════
# Parameter helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_parameters_torch(model: nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_parameters_torch(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(parameters) != len(keys):
        raise ValueError(
            f"Parameter length mismatch: got {len(parameters)}, expected {len(keys)}"
        )
    model.load_state_dict(
        {k: torch.tensor(arr, dtype=state_dict[k].dtype)
         for k, arr in zip(keys, parameters)},
        strict=True,
    )


def pack_tensors(tensors: List[torch.Tensor]) -> bytes:
    return pickle.dumps(
        [t.detach().cpu().numpy() for t in tensors],
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def unpack_tensors(
    blob: bytes,
    device: torch.device,
    ref_params: List[torch.Tensor],
) -> List[torch.Tensor]:
    return [
        torch.tensor(arr, dtype=ref.dtype, device=device).view_as(ref)
        for arr, ref in zip(pickle.loads(blob), ref_params)
    ]


@torch.no_grad()
def eval_with_loss(model: nn.Module, loader, device) -> Tuple[float, float, float]:
    model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_n = 0.0, 0
    ys, ps = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += float(loss_fn(logits, yb).item()) * yb.shape[0]
        total_n    += yb.shape[0]
        ps.extend(torch.argmax(logits, 1).cpu().numpy().tolist())
        ys.extend(yb.cpu().numpy().tolist())

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(ys, ps) if total_n > 0 else 0.0
    f1  = f1_score(ys, ps, average="macro") if total_n > 0 else 0.0
    return total_loss / max(1, total_n), acc, f1


def get_model_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def get_memory_usage_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


# ══════════════════════════════════════════════════════════════════════════════
# HARClient
# ══════════════════════════════════════════════════════════════════════════════

class HARClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        model_fn,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cfg,
        algo: str = "fedavg",
    ):
        self.cid  = cid
        self.cfg  = cfg
        self.algo = str(algo).lower()

        self.device = (
            torch.device("cuda")
            if getattr(cfg, "device", "cpu") == "cuda" and torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.model = model_fn()
        self.model.to(self.device)

        bs = int(getattr(cfg, "batch_size", 64))
        self.train_loader = make_loader(X_train, y_train, batch_size=bs, shuffle=True)
        self.test_loader  = make_loader(X_test,  y_test,  batch_size=bs, shuffle=False)

        self.lr     = float(getattr(cfg, "lr", 1e-3))
        self.epochs = int(getattr(cfg, "local_epochs", 1))

        # ── Class weights (fixes macro-F1 on imbalanced datasets like CASAS) ──
        # Uses sqrt-inverse frequency to avoid over-penalising rare classes.
        # Formula: w_c = 1 / sqrt(count_c),  then normalise so sum = n_classes
        n_classes = list(self.model.parameters())[-1].shape[0]
        counts    = np.bincount(y_train.astype(int), minlength=n_classes).astype(np.float32)
        counts    = np.where(counts == 0, 1.0, counts)          # avoid divide-by-zero
        raw_w     = 1.0 / np.sqrt(counts)                       # sqrt-inverse frequency
        raw_w     = raw_w / raw_w.sum() * n_classes             # normalise
        self.class_weights = torch.tensor(raw_w, dtype=torch.float32)
        # ───────────────────────────────────────────────────────────────────────

        # Algorithm state
        self.c_local: Optional[List[torch.Tensor]] = None   # SCAFFOLD
        self.h_local: Optional[List[torch.Tensor]] = None   # FedDC
        self.gi_prev: Optional[List[torch.Tensor]] = None   # FedDC

        self.base_bytes    = get_model_size_bytes(self.model)
        self._dataset_size = len(X_train)

    # ── Parameter I/O ─────────────────────────────────────────────────────

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        params = get_parameters_torch(self.model)
        if hasattr(self, "control_delta") and self.control_delta is not None:
            return params + [self.control_delta]
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        keys         = list(self.model.state_dict().keys())
        expected_len = len(keys)

        if len(parameters) == expected_len + 1:
            # SCAFFOLD: server appended c_global as an extra array
            self.c_global = parameters[-1]
            set_parameters_torch(self.model, parameters[:-1])
        else:
            set_parameters_torch(self.model, parameters)

    # ── Edge profiling ─────────────────────────────────────────────────────

    def _edge_enabled(self) -> bool:
        return getattr(self.cfg, "edge_profile", None) is not None

    def _edge_metrics(self, wall_s: float, cpu_pct: float) -> Dict[str, Any]:
        key     = getattr(self.cfg, "edge_profile", None)
        profile = PROFILES[key]
        sim_s   = float(wall_s) * float(profile.time_scale)
        return {
            "fit_wall_s":    float(wall_s),
            "fit_sim_s":     float(sim_s),
            "cpu_pct":       float(cpu_pct),
            "energy_wh":     float(profile.power_watt) * (sim_s / 3600.0),
            "device_key":    str(key),
            "device_profile":str(profile.name),
        }

    # ── Shared DP config reader ────────────────────────────────────────────

    def _dp_kwargs(self) -> Dict[str, Any]:
        return dict(
            clip_norm        = float(self.cfg.dp_clip_norm),
            noise_multiplier = float(self.cfg.dp_noise_multiplier),
            target_delta     = float(self.cfg.dp_target_delta),
        )

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        self.set_parameters(parameters)
        algo       = str(config.get("algo", self.algo)).lower()
        n_examples = len(self.train_loader.dataset)
        edge_on    = self._edge_enabled()
        use_dp     = bool(getattr(self.cfg, "use_dp", False))

        t0 = time.time()
        m: Dict[str, Any] = {
            "comm_bytes_down": self.base_bytes,
            "comm_bytes_up":   self.base_bytes,
            "memory_usage_mb": float(get_memory_usage_mb()),
        }

        # ══════════════════════════════════════════════════════════════════
        # FedAvg / FedProx
        # ══════════════════════════════════════════════════════════════════
        if algo in ["fedavg", "fedprox"]:
            prox_mu = float(config.get("prox_mu", 0.0))

            if use_dp:
                # ── DP-Prox: per-sample DP-SGD + proximal correction ──────
                global_params = [p.detach().clone() for p in self.model.parameters()]

                if not edge_on:
                    epsilon = train_dp_fedprox_local(
                        model=self.model, loader=self.train_loader,
                        epochs=self.epochs, lr=self.lr, device=self.device,
                        prox_mu=prox_mu, global_params=global_params,
                        class_weights=self.class_weights,
                        **self._dp_kwargs(),
                    )
                    m["fit_wall_s"] = float(time.time() - t0)
                    if epsilon is not None:
                        m["dp_epsilon"] = float(epsilon)
                    return self.get_parameters({}), n_examples, m

                prof = RoundProfiler(); prof.start()
                epsilon = train_dp_fedprox_local(
                    model=self.model, loader=self.train_loader,
                    epochs=self.epochs, lr=self.lr, device=self.device,
                    prox_mu=prox_mu, global_params=global_params,
                    class_weights=self.class_weights,
                    **self._dp_kwargs(),
                )
                wall_s, cpu_pct = prof.stop()
                if epsilon is not None:
                    m["dp_epsilon"] = float(epsilon)
                m.update(self._edge_metrics(wall_s, cpu_pct))
                return self.get_parameters({}), n_examples, m

            else:
                # ── Standard FedProx / FedAvg ─────────────────────────────
                global_params = (
                    [p.detach().clone() for p in self.model.parameters()]
                    if prox_mu > 0.0 else None
                )
                clip_norm = float(getattr(self.cfg, "cdp_clip_norm", 5.0))

                if not edge_on:
                    train_one_client(
                        model=self.model, loader=self.train_loader,
                        epochs=self.epochs, lr=self.lr, device=self.device,
                        prox_mu=prox_mu, global_params=global_params,
                        clip_norm=clip_norm,
                        class_weights=self.class_weights,
                    )
                    m["fit_wall_s"] = float(time.time() - t0)
                    return self.get_parameters({}), n_examples, m

                prof = RoundProfiler(); prof.start()
                train_one_client(
                    model=self.model, loader=self.train_loader,
                    epochs=self.epochs, lr=self.lr, device=self.device,
                    prox_mu=prox_mu, global_params=global_params,
                    clip_norm=clip_norm,
                    class_weights=self.class_weights,
                )
                wall_s, cpu_pct = prof.stop()
                m.update(self._edge_metrics(wall_s, cpu_pct))
                return self.get_parameters({}), n_examples, m

        # ══════════════════════════════════════════════════════════════════
        # SCAFFOLD  (with optional DP)
        # ══════════════════════════════════════════════════════════════════
        if algo == "scaffold":
            # SCAFFOLD sends extra comm: c_global down + delta_c up
            m["comm_bytes_down"] += self.base_bytes
            m["comm_bytes_up"]   += self.base_bytes

            # Initialise c_local to zeros on first round
            if self.c_local is None:
                self.c_local = [
                    torch.zeros_like(p, device=self.device)
                    for p in self.model.parameters()
                ]

            # Unpack c_global sent by the server (or zeros on round 0)
            c_global_blob = config.get("c_global", None)
            c_global = (
                unpack_tensors(c_global_blob, self.device,
                               list(self.model.parameters()))
                if c_global_blob is not None
                else [torch.zeros_like(p, device=self.device)
                      for p in self.model.parameters()]
            )

            prof = RoundProfiler() if edge_on else None
            if prof: prof.start()

            if use_dp:
                # ── DP-SCAFFOLD ───────────────────────────────────────────
                # Per-sample DP-SGD on task loss + SCAFFOLD correction.
                # c_local updated via Option II (parameter-difference, DP-safe).
                epsilon, new_c_local = train_dp_scaffold_local(
                    model=self.model, loader=self.train_loader,
                    epochs=self.epochs, lr=self.lr, device=self.device,
                    c_global=c_global, c_local=self.c_local,
                    class_weights=self.class_weights,
                    **self._dp_kwargs(),
                )
                if epsilon is not None:
                    m["dp_epsilon"] = float(epsilon)
            else:
                # ── Standard SCAFFOLD ─────────────────────────────────────
                # c_local updated via Option I (accumulated gradient).
                steps, true_grads = train_scaffold(
                    model=self.model, loader=self.train_loader,
                    epochs=self.epochs, lr=self.lr, device=self.device,
                    c_global=c_global, c_local=self.c_local,
                    class_weights=self.class_weights,
                )
                new_c_local = [tg.detach().clone() for tg in true_grads]

            if prof: wall_s, cpu_pct = prof.stop()

            # Compute delta_c = c_local_new − c_local_old  (sent to server)
            delta_c = [
                (nc - cl).detach()
                for nc, cl in zip(new_c_local, self.c_local)
            ]
            self.c_local = new_c_local   # persist updated control variate

            m["delta_c"]    = pack_tensors(delta_c)
            m["fit_wall_s"] = float(time.time() - t0)
            if edge_on: m.update(self._edge_metrics(wall_s, cpu_pct))
            return self.get_parameters({}), n_examples, m

        # ══════════════════════════════════════════════════════════════════
        # FedDC  (no DP currently)
        # ══════════════════════════════════════════════════════════════════
        if algo == "feddc":
            m["comm_bytes_down"] += self.base_bytes
            m["comm_bytes_up"]   += self.base_bytes

            penalty_alpha = float(
                config.get("feddc_penalty_alpha",
                           getattr(self.cfg, "feddc_penalty_alpha", 0.01))
            )
            if self.gi_prev is None:
                self.gi_prev = [
                    torch.zeros_like(p, device=self.device)
                    for p in self.model.parameters()
                ]

            g_prev_blob = config.get("g_prev", None)
            g_prev = (
                unpack_tensors(g_prev_blob, self.device,
                               list(self.model.parameters()))
                if g_prev_blob is not None
                else [torch.zeros_like(p, device=self.device)
                      for p in self.model.parameters()]
            )

            w_global    = [p.detach().clone() for p in self.model.parameters()]
            self.h_local = [
                (og - ogi).detach().clone()
                for og, ogi in zip(g_prev, self.gi_prev)
            ]

            prof = RoundProfiler() if edge_on else None
            if prof: prof.start()

            steps, true_grads = train_feddc(
                model=self.model, loader=self.train_loader,
                epochs=self.epochs, lr=self.lr, device=self.device,
                w_global=w_global, h_local=self.h_local,
                penalty_alpha=penalty_alpha,
            )

            if prof: wall_s, cpu_pct = prof.stop()

            self.gi_prev = [tg.detach().clone() for tg in true_grads]
            m["gi"]         = pack_tensors(self.gi_prev)
            m["K"]          = int(max(1, steps))
            m["fit_wall_s"] = float(time.time() - t0)
            if edge_on: m.update(self._edge_metrics(wall_s, cpu_pct))
            return self.get_parameters({}), n_examples, m

        raise ValueError(f"Unknown algo: {algo}")

    # ── evaluate ──────────────────────────────────────────────────────────

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        self.set_parameters(parameters)
        acc, f1 = evaluate(self.model, self.test_loader, self.device)
        loss, _, _ = eval_with_loss(self.model, self.test_loader, self.device)
        return float(loss), len(self.test_loader.dataset), {
            "accuracy": float(acc),
            "f1":       float(f1),
        }