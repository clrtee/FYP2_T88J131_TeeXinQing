# src/train_utils.py
# ─────────────────────────────────────────────────────────────────────────────
# Training utilities for Federated Learning with optional Differential Privacy.
#
# Functions
# ─────────────────────────────────────────────────────────────────────────────
#  train_one_client          – standard FedAvg / FedProx (no DP)
#  train_dp_fedprox_local    – DP-SGD + FedProx (Algorithm 1, DP-Prox paper)
#  train_dp_scaffold_local   – DP-SGD + SCAFFOLD (Option-II c_local update)
#  train_scaffold            – standard SCAFFOLD (no DP)
#  train_feddc               – standard FedDC   (no DP)
#  evaluate                  – accuracy + macro-F1 on a DataLoader
#
# DP backends (both algorithms)
#   • Opacus  – primary  (efficient per-sample ops + RDP accounting)
#   • Manual  – fallback (per-sample loop, no Opacus required, slower)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import copy
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_loader(X, y, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds  = TensorDataset(X_t, y_t)
    # drop_last=True is required for DP-SGD (Poisson subsampling needs fixed B)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


# ══════════════════════════════════════════════════════════════════════════════
# Standard training  (FedAvg / FedProx without DP)
# ══════════════════════════════════════════════════════════════════════════════

def train_one_client(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    prox_mu: float = 0.0,
    global_params: Optional[List[torch.Tensor]] = None,
    clip_norm: float = 5.0,
    class_weights: Optional[torch.Tensor] = None,
) -> None:
    """Standard FedAvg / FedProx local training — no DP noise."""
    model.to(device)
    model.train()
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    # Use weighted loss if class_weights provided (helps imbalanced datasets)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    if prox_mu > 0.0 and global_params is not None:
        global_params = [p.detach().clone().to(device) for p in global_params]

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            if prox_mu > 0.0 and global_params is not None:
                prox = sum(
                    torch.sum((p - gp) ** 2)
                    for p, gp in zip(model.parameters(), global_params)
                )
                loss = loss + (prox_mu / 2.0) * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            opt.step()


# ══════════════════════════════════════════════════════════════════════════════
# DP-SGD + FedProx  (Algorithm 1 of the DP-Prox paper)
# ══════════════════════════════════════════════════════════════════════════════

def train_dp_fedprox_local(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    prox_mu: float,
    global_params: List[torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    target_delta: float = 1e-5,
    class_weights: Optional[torch.Tensor] = None,
) -> Optional[float]:
    """
    Client-side DP-SGD + FedProx proximal correction.

    Per-sample gradient clipping + Gaussian noise (Opacus or manual fallback),
    followed by a proximal pull toward the global model:
        θ ← θ − η (ḡ_DP  +  μ·(θ − L^t))

    Returns actual ε from RDP accounting (Opacus path), or None (manual path).
    """
    try:
        import opacus  # noqa: F401
        return _dp_fedprox_opacus(
            model, loader, epochs, lr, device,
            prox_mu, global_params, clip_norm, noise_multiplier, target_delta,
            class_weights=class_weights,
        )
    except ImportError:
        logger.warning(
            "[DP-FedProx] Opacus not installed — using slower manual per-sample DP-SGD. "
            "Install with: pip install opacus"
        )
        return _dp_fedprox_manual(
            model, loader, epochs, lr, device,
            prox_mu, global_params, clip_norm, noise_multiplier,
            class_weights=class_weights,
        )


def _dp_fedprox_opacus(
    model, loader, epochs, lr, device,
    prox_mu, global_params, clip_norm, noise_multiplier, target_delta,
    class_weights=None,
):
    from opacus import PrivacyEngine

    model_work = copy.deepcopy(model).to(device)
    model_work.train()
    global_p = [p.detach().clone().to(device) for p in global_params]
    loss_fn  = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(model_work.parameters(), lr=lr)
    privacy_engine = PrivacyEngine(accountant="rdp")

    # Try expanded-weights mode first (supports nn.LSTM without DPLSTM)
    try:
        model_dp, optimizer_dp, loader_dp = privacy_engine.make_private(
            module=model_work, optimizer=optimizer, data_loader=loader,
            noise_multiplier=noise_multiplier, max_grad_norm=clip_norm,
            grad_sample_mode="ew", poisson_sampling=True,
        )
        logger.debug("[DP-FedProx] Opacus: expanded-weights (ew) mode")
    except Exception:
        try:
            model_dp, optimizer_dp, loader_dp = privacy_engine.make_private(
                module=model_work, optimizer=optimizer, data_loader=loader,
                noise_multiplier=noise_multiplier, max_grad_norm=clip_norm,
                poisson_sampling=True,
            )
            logger.debug("[DP-FedProx] Opacus: hook-based mode")
        except Exception as exc:
            logger.warning(f"[DP-FedProx] Opacus make_private failed ({exc}), falling back to manual.")
            return _dp_fedprox_manual(
                model, loader, epochs, lr, device,
                prox_mu, global_params, clip_norm, noise_multiplier,
                class_weights=class_weights,
            )

    for _ in range(epochs):
        for xb, yb in loader_dp:
            xb, yb = xb.to(device), yb.to(device)
            optimizer_dp.zero_grad()
            loss_fn(model_dp(xb), yb).backward()
            optimizer_dp.step()           # per-sample clip + noise + update

            # Proximal correction (deterministic, applied after DP step)
            if prox_mu > 0.0:
                underlying = getattr(model_dp, "_module", model_dp)
                with torch.no_grad():
                    for p, gp in zip(underlying.parameters(), global_p):
                        p.data.sub_(lr * prox_mu * (p.data - gp))

    try:
        epsilon = privacy_engine.get_epsilon(delta=target_delta)
        logger.info(f"[DP-FedProx] ε = {epsilon:.4f}  (δ={target_delta})")
    except Exception:
        epsilon = None

    trained = getattr(model_dp, "_module", model_dp)
    model.load_state_dict(trained.state_dict())
    model.to(device)
    return epsilon


def _dp_fedprox_manual(
    model, loader, epochs, lr, device,
    prox_mu, global_params, clip_norm, noise_multiplier,
    class_weights=None,
):
    logger.warning("[DP-FedProx] Manual per-sample mode — ~B× slower. Install Opacus.")
    model.to(device)
    model.train()
    loss_fn    = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    global_p   = [p.detach().clone().to(device) for p in global_params]
    param_list = list(model.parameters())

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            B = xb.shape[0]
            if B == 0:
                continue

            sum_grads = [torch.zeros_like(p) for p in param_list]
            for i in range(B):
                model.zero_grad()
                loss_fn(model(xb[i:i+1]), yb[i:i+1]).backward()
                g_i = [p.grad.detach().clone() if p.grad is not None
                       else torch.zeros_like(p) for p in param_list]
                norm  = torch.cat([g.reshape(-1) for g in g_i]).norm(2).item()
                scale = min(1.0, clip_norm / (norm + 1e-8))
                for j, g in enumerate(g_i):
                    sum_grads[j].add_(g, alpha=scale)

            sigma = noise_multiplier * clip_norm
            noisy_avg = [
                (gs + torch.randn_like(gs).mul_(sigma)).div_(B)
                for gs in sum_grads
            ]

            model.zero_grad()
            with torch.no_grad():
                for p, g_dp, gp in zip(param_list, noisy_avg, global_p):
                    p.data.sub_(lr * (g_dp + prox_mu * (p.data - gp)))

    return None   # no RDP accounting in manual path


# ══════════════════════════════════════════════════════════════════════════════
# DP-SGD + SCAFFOLD
# ══════════════════════════════════════════════════════════════════════════════

def train_dp_scaffold_local(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    c_global: List[torch.Tensor],
    c_local: List[torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    target_delta: float = 1e-5,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[Optional[float], List[torch.Tensor]]:
    """
    DP-SCAFFOLD: per-sample DP-SGD on loss gradients + SCAFFOLD correction.

    Design decisions
    ────────────────
    • DP noise is applied ONLY to the task-loss gradient (per-sample clip +
      Gaussian noise via Opacus or manual loop).  The SCAFFOLD control-variate
      correction  (c_global − c_local)  is deterministic and does NOT need
      additional noise.
    • c_local is updated via Option II (parameter-difference formula) which is
      DP-safe because it uses only the final model weights w_T (already
      privatised by DP-SGD) and the initial weights w_0:
          c_local_new = c_local − c_global + (w_0 − w_T) / (K × lr)

    Returns
    ───────
    (epsilon_or_None, new_c_local_list)
    """
    try:
        import opacus  # noqa: F401
        return _dp_scaffold_opacus(
            model, loader, epochs, lr, device,
            c_global, c_local, clip_norm, noise_multiplier, target_delta,
            class_weights=class_weights,
        )
    except ImportError:
        logger.warning(
            "[DP-SCAFFOLD] Opacus not installed — using slower manual per-sample DP-SGD. "
            "Install with: pip install opacus"
        )
        return _dp_scaffold_manual(
            model, loader, epochs, lr, device,
            c_global, c_local, clip_norm, noise_multiplier,
            class_weights=class_weights,
        )


def _dp_scaffold_opacus(
    model, loader, epochs, lr, device,
    c_global, c_local, clip_norm, noise_multiplier, target_delta,
    class_weights=None,
):
    from opacus import PrivacyEngine

    # Snapshot initial weights for Option-II c_local update
    w_0   = [p.detach().clone() for p in model.parameters()]
    cg    = [c.detach().clone().to(device) for c in c_global]
    cl    = [c.detach().clone().to(device) for c in c_local]

    model_work = copy.deepcopy(model).to(device)
    model_work.train()
    loss_fn   = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.SGD(model_work.parameters(), lr=lr,
                                momentum=0.9, weight_decay=1e-4)
    privacy_engine = PrivacyEngine(accountant="rdp")

    try:
        model_dp, optimizer_dp, loader_dp = privacy_engine.make_private(
            module=model_work, optimizer=optimizer, data_loader=loader,
            noise_multiplier=noise_multiplier, max_grad_norm=clip_norm,
            grad_sample_mode="ew", poisson_sampling=True,
        )
        logger.debug("[DP-SCAFFOLD] Opacus: expanded-weights (ew) mode")
    except Exception:
        try:
            model_dp, optimizer_dp, loader_dp = privacy_engine.make_private(
                module=model_work, optimizer=optimizer, data_loader=loader,
                noise_multiplier=noise_multiplier, max_grad_norm=clip_norm,
                poisson_sampling=True,
            )
            logger.debug("[DP-SCAFFOLD] Opacus: hook-based mode")
        except Exception as exc:
            logger.warning(f"[DP-SCAFFOLD] Opacus make_private failed ({exc}), falling back to manual.")
            return _dp_scaffold_manual(
                model, loader, epochs, lr, device,
                c_global, c_local, clip_norm, noise_multiplier,
            )

    steps = 0
    for _ in range(epochs):
        for xb, yb in loader_dp:
            xb, yb = xb.to(device), yb.to(device)

            # ── Steps 1-4: DP-SGD on task loss (Opacus handles clip + noise) ──
            optimizer_dp.zero_grad()
            loss_fn(model_dp(xb), yb).backward()
            optimizer_dp.step()   # θ ← θ − lr · ḡ_DP

            # ── Step 5: SCAFFOLD correction (deterministic, no DP noise) ──────
            underlying = getattr(model_dp, "_module", model_dp)
            with torch.no_grad():
                for p, cg_i, cl_i in zip(underlying.parameters(), cg, cl):
                    p.data.sub_(lr * (cg_i - cl_i))

            steps += 1

    # ── Option-II c_local update (DP-safe) ───────────────────────────────────
    steps = max(1, steps)
    underlying = getattr(model_dp, "_module", model_dp)
    w_T = [p.detach().clone() for p in underlying.parameters()]

    new_c_local = [
        cl_i - cg_i + (w0_i.to(device) - wT_i) / (steps * lr)
        for cl_i, cg_i, w0_i, wT_i in zip(cl, cg, w_0, w_T)
    ]

    # Copy trained weights back to the original model
    model.load_state_dict(underlying.state_dict())
    model.to(device)

    try:
        epsilon = privacy_engine.get_epsilon(delta=target_delta)
        logger.info(f"[DP-SCAFFOLD] ε = {epsilon:.4f}  (δ={target_delta})")
    except Exception:
        epsilon = None

    return epsilon, new_c_local


def _dp_scaffold_manual(
    model, loader, epochs, lr, device,
    c_global, c_local, clip_norm, noise_multiplier,
    class_weights=None,
):
    logger.warning("[DP-SCAFFOLD] Manual per-sample mode — ~B× slower. Install Opacus.")
    model.to(device)
    model.train()
    loss_fn    = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    param_list = list(model.parameters())

    w_0 = [p.detach().clone() for p in param_list]
    cg  = [c.detach().clone().to(device) for c in c_global]
    cl  = [c.detach().clone().to(device) for c in c_local]

    steps = 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            B = xb.shape[0]
            if B == 0:
                continue

            # Per-sample clip
            sum_grads = [torch.zeros_like(p) for p in param_list]
            for i in range(B):
                model.zero_grad()
                loss_fn(model(xb[i:i+1]), yb[i:i+1]).backward()
                g_i = [p.grad.detach().clone() if p.grad is not None
                       else torch.zeros_like(p) for p in param_list]
                norm  = torch.cat([g.reshape(-1) for g in g_i]).norm(2).item()
                scale = min(1.0, clip_norm / (norm + 1e-8))
                for j, g in enumerate(g_i):
                    sum_grads[j].add_(g, alpha=scale)

            sigma = noise_multiplier * clip_norm
            noisy_avg = [
                (gs + torch.randn_like(gs).mul_(sigma)).div_(B)
                for gs in sum_grads
            ]

            # θ ← θ − lr · (ḡ_DP + c_global − c_local)
            model.zero_grad()
            with torch.no_grad():
                for p, g_dp, cg_i, cl_i in zip(param_list, noisy_avg, cg, cl):
                    p.data.sub_(lr * (g_dp + cg_i - cl_i))

            steps += 1

    # Option-II c_local update
    steps = max(1, steps)
    w_T = [p.detach().clone() for p in model.parameters()]
    new_c_local = [
        cl_i - cg_i + (w0_i - wT_i) / (steps * lr)
        for cl_i, cg_i, w0_i, wT_i in zip(cl, cg, w_0, w_T)
    ]

    return None, new_c_local   # no RDP accounting in manual path


# ══════════════════════════════════════════════════════════════════════════════
# Standard SCAFFOLD  (no DP)
# ══════════════════════════════════════════════════════════════════════════════

def train_scaffold(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    c_global: List[torch.Tensor],
    c_local: List[torch.Tensor],
    clip_norm: float = 5.0,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[int, List[torch.Tensor]]:
    """Standard SCAFFOLD local training (no DP). Returns (steps, true_grads)."""
    model.to(device)
    model.train()
    loss_fn   = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    opt       = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=1e-4)
    grad_sums = [torch.zeros_like(p) for p in model.parameters()]
    steps     = 0

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            with torch.no_grad():
                for gs, p in zip(grad_sums, model.parameters()):
                    if p.grad is not None:
                        gs.add_(p.grad.data)

            opt.step()

            with torch.no_grad():
                for p, cg, cl in zip(model.parameters(), c_global, c_local):
                    p.data.sub_(lr * (cg - cl))

            steps += 1

    true_grads = [gs / max(1, steps) for gs in grad_sums]
    return steps, true_grads


# ══════════════════════════════════════════════════════════════════════════════
# Standard FedDC  (no DP)
# ══════════════════════════════════════════════════════════════════════════════

def train_feddc(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    w_global: List[torch.Tensor],
    h_local: List[torch.Tensor],
    penalty_alpha: float,
    clip_norm: float = 5.0,
) -> Tuple[int, List[torch.Tensor]]:
    """Standard FedDC local training (no DP). Returns (steps, true_grads)."""
    model.to(device)
    model.train()
    loss_fn   = nn.CrossEntropyLoss()
    opt       = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=1e-4)
    grad_sums = [torch.zeros_like(p) for p in model.parameters()]
    steps     = 0

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()

            with torch.no_grad():
                for gs, p in zip(grad_sums, model.parameters()):
                    if p.grad is not None:
                        gs.add_(p.grad.data)
                for p, w in zip(model.parameters(), w_global):
                    if p.grad is not None:
                        p.grad.data.add_(p.data - w.data, alpha=penalty_alpha)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            opt.step()

            with torch.no_grad():
                for p, h in zip(model.parameters(), h_local):
                    p.data.sub_(lr * h)

            steps += 1

    true_grads = [gs / max(1, steps) for gs in grad_sums]
    return steps, true_grads


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (accuracy, macro-F1)."""
    model.to(device)
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = torch.argmax(model(xb), dim=1).cpu().numpy()
        ys.extend(yb.numpy().tolist())
        ps.extend(pred.tolist())
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")