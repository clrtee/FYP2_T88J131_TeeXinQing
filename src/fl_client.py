from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import pickle

import numpy as np
import torch
import torch.nn as nn
import flwr as fl

from .train_utils import make_loader, train_one_client, train_scaffold, train_feddc, evaluate
from .profiler import RoundProfiler
from .device_profiles import PROFILES


def get_parameters_torch(model: torch.nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_parameters_torch(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(parameters) != len(keys):
        raise ValueError(f"Parameter length mismatch: got {len(parameters)}, expected {len(keys)}")
    new_state = {}
    for k, arr in zip(keys, parameters):
        new_state[k] = torch.tensor(arr, dtype=state_dict[k].dtype)
    model.load_state_dict(new_state, strict=True)


def pack_tensors(tensors: List[torch.Tensor]) -> bytes:
    arrs = [t.detach().cpu().numpy() for t in tensors]
    return pickle.dumps(arrs, protocol=pickle.HIGHEST_PROTOCOL)


def unpack_tensors(blob: bytes, device: torch.device, ref_params: List[torch.Tensor]) -> List[torch.Tensor]:
    arrs = pickle.loads(blob)
    out: List[torch.Tensor] = []
    for arr, ref in zip(arrs, ref_params):
        t = torch.tensor(arr, dtype=ref.dtype, device=device).view_as(ref)
        out.append(t)
    return out


@torch.no_grad()
def eval_with_loss(model: torch.nn.Module, loader, device) -> Tuple[float, float, float]:
    model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss, total_n = 0.0, 0
    ys, ps = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = loss_fn(logits, yb)

        bs = yb.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

        pred = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        ys.extend(yb.detach().cpu().numpy().tolist())
        ps.extend(pred)

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(ys, ps) if total_n > 0 else 0.0
    f1 = f1_score(ys, ps, average="macro") if total_n > 0 else 0.0
    avg_loss = total_loss / max(1, total_n)
    return avg_loss, acc, f1


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
        self.cid = cid
        self.cfg = cfg
        self.algo = str(algo).lower()

        if getattr(cfg, "device", "cpu") == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model_fn()
        self.model.to(self.device)

        bs = int(getattr(cfg, "batch_size", 64))
        self.train_loader = make_loader(X_train, y_train, batch_size=bs, shuffle=True)
        self.test_loader = make_loader(X_test, y_test, batch_size=bs, shuffle=False)

        self.lr = float(getattr(cfg, "lr", 1e-3))
        self.epochs = int(getattr(cfg, "local_epochs", 1))

        self.c_local: Optional[List[torch.Tensor]] = None
        self.h_local: Optional[List[torch.Tensor]] = None
        self.gi_prev: Optional[List[torch.Tensor]] = None

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return get_parameters_torch(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_parameters_torch(self.model, parameters)

    def _edge_enabled(self) -> bool:
        return getattr(self.cfg, "edge_profile", None) is not None

    def _edge_metrics(self, wall_s: float, cpu_pct: float) -> Dict[str, Any]:
        edge_key = getattr(self.cfg, "edge_profile", None)
        profile = PROFILES[edge_key]
        sim_s = float(wall_s) * float(profile.time_scale)
        energy_wh = float(profile.power_watt) * (sim_s / 3600.0)
        return {
            "fit_wall_s": float(wall_s),
            "fit_sim_s": float(sim_s),
            "cpu_pct": float(cpu_pct),
            "energy_wh": float(energy_wh),
            "device_key": str(edge_key),
            "device_profile": str(profile.name),
        }

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        self.set_parameters(parameters)
        algo = str(config.get("algo", self.algo)).lower()
        num_examples = len(self.train_loader.dataset)
        edge_on = self._edge_enabled()

        # ---------------- FedAvg / FedProx ----------------
        if algo in ["fedavg", "fedprox"]:
            prox_mu = float(config.get("prox_mu", 0.0))
            global_params = None
            if prox_mu > 0.0:
                global_params = [p.detach().clone() for p in self.model.parameters()]

            dp_cfg = None
            if getattr(self.cfg, "use_dp", False):
                dp_cfg = {
                    "use_dp": True,
                    "noise_multiplier": float(getattr(self.cfg, "dp_noise_multiplier", 1.0)),
                    "max_grad_norm": float(getattr(self.cfg, "dp_max_grad_norm", 1.0)),
                }

            if not edge_on:
                train_one_client(
                    model=self.model,
                    loader=self.train_loader,
                    epochs=self.epochs,
                    lr=self.lr,
                    device=self.device,
                    prox_mu=prox_mu,
                    global_params=global_params,
                    dp_cfg=dp_cfg,
                )
                return self.get_parameters({}), num_examples, {}

            prof = RoundProfiler()
            prof.start()

            train_one_client(
                model=self.model,
                loader=self.train_loader,
                epochs=self.epochs,
                lr=self.lr,
                device=self.device,
                prox_mu=prox_mu,
                global_params=global_params,
                dp_cfg=dp_cfg,
            )

            wall_s, cpu_pct = prof.stop()
            return self.get_parameters({}), num_examples, self._edge_metrics(wall_s, cpu_pct)

        # ---------------- SCAFFOLD ----------------
        if algo == "scaffold":
            if self.c_local is None:
                self.c_local = [torch.zeros_like(p, device=self.device) for p in self.model.parameters()]

            c_global_blob = config.get("c_global", None)
            if c_global_blob is None:
                c_global = [torch.zeros_like(p, device=self.device) for p in self.model.parameters()]
            else:
                c_global = unpack_tensors(c_global_blob, self.device, list(self.model.parameters()))

            params_before = [p.detach().clone() for p in self.model.parameters()]

            if not edge_on:
                steps = train_scaffold(
                    model=self.model,
                    loader=self.train_loader,
                    epochs=self.epochs,
                    lr=self.lr,
                    device=self.device,
                    c_global=c_global,
                    c_local=self.c_local,
                )
                steps = max(1, int(steps))

                params_after = [p.detach().clone() for p in self.model.parameters()]
                delta_c: List[torch.Tensor] = []
                for pb, pa, cg in zip(params_before, params_after, c_global):
                    dc = (pb - pa) / (self.lr * steps) - cg
                    delta_c.append(dc.detach())
                self.c_local = [cl + dc for cl, dc in zip(self.c_local, delta_c)]
                return self.get_parameters({}), num_examples, {"delta_c": pack_tensors(delta_c)}

            prof = RoundProfiler()
            prof.start()

            steps = train_scaffold(
                model=self.model,
                loader=self.train_loader,
                epochs=self.epochs,
                lr=self.lr,
                device=self.device,
                c_global=c_global,
                c_local=self.c_local,
            )

            wall_s, cpu_pct = prof.stop()

            steps = max(1, int(steps))
            params_after = [p.detach().clone() for p in self.model.parameters()]

            delta_c = []
            for pb, pa, cg in zip(params_before, params_after, c_global):
                dc = (pb - pa) / (self.lr * steps) - cg
                delta_c.append(dc.detach())
            self.c_local = [cl + dc for cl, dc in zip(self.c_local, delta_c)]

            m = {"delta_c": pack_tensors(delta_c)}
            m.update(self._edge_metrics(wall_s, cpu_pct))
            return self.get_parameters({}), num_examples, m

        # ---------------- FedDC ----------------
        if algo == "feddc":
            penalty_alpha = float(config.get("feddc_penalty_alpha", getattr(self.cfg, "feddc_penalty_alpha", 1.0)))

            if self.h_local is None:
                self.h_local = [torch.zeros_like(p, device=self.device) for p in self.model.parameters()]
            if self.gi_prev is None:
                self.gi_prev = [torch.zeros_like(p, device=self.device) for p in self.model.parameters()]

            g_prev_blob = config.get("g_prev", None)
            if g_prev_blob is None:
                g_prev = [torch.zeros_like(p, device=self.device) for p in self.model.parameters()]
            else:
                g_prev = unpack_tensors(g_prev_blob, self.device, list(self.model.parameters()))

            w_global = [p.detach().clone() for p in self.model.parameters()]

            if not edge_on:
                K = train_feddc(
                    model=self.model,
                    loader=self.train_loader,
                    epochs=self.epochs,
                    lr=self.lr,
                    device=self.device,
                    w_global=w_global,
                    h_local=self.h_local,
                    gi_prev=self.gi_prev,
                    g_prev=g_prev,
                    penalty_alpha=penalty_alpha,
                )
                K = max(1, int(K))

                w_after = [p.detach().clone() for p in self.model.parameters()]
                gi = [(wa - wg).detach() for wa, wg in zip(w_after, w_global)]
                self.gi_prev = [g.detach() for g in gi]
                return self.get_parameters({}), num_examples, {"gi": pack_tensors(gi), "K": int(K)}

            prof = RoundProfiler()
            prof.start()

            K = train_feddc(
                model=self.model,
                loader=self.train_loader,
                epochs=self.epochs,
                lr=self.lr,
                device=self.device,
                w_global=w_global,
                h_local=self.h_local,
                gi_prev=self.gi_prev,
                g_prev=g_prev,
                penalty_alpha=penalty_alpha,
            )

            wall_s, cpu_pct = prof.stop()

            K = max(1, int(K))
            w_after = [p.detach().clone() for p in self.model.parameters()]
            gi = [(wa - wg).detach() for wa, wg in zip(w_after, w_global)]
            self.gi_prev = [g.detach() for g in gi]

            m = {"gi": pack_tensors(gi), "K": int(K)}
            m.update(self._edge_metrics(wall_s, cpu_pct))
            return self.get_parameters({}), num_examples, m

        raise ValueError(f"Unknown algo: {algo}")

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        self.set_parameters(parameters)
        acc, f1 = evaluate(self.model, self.test_loader, self.device)
        loss, _, _ = eval_with_loss(self.model, self.test_loader, self.device)
        num_examples = len(self.test_loader.dataset)
        return float(loss), num_examples, {"accuracy": float(acc), "f1": float(f1)}
