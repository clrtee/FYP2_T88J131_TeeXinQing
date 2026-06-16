# src/dp_utils.py 
from __future__ import annotations
from typing import List, Optional
import math
import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════════════════
# RDP  ε  Accounting
# ══════════════════════════════════════════════════════════════════════════════

def _rdp_gaussian_single(alpha: float, sigma: float) -> float:
    """
    RDP of the pure Gaussian mechanism at order α.
    ε_RDP(α) = α / (2σ²)   (Mironov 2017, Proposition 3)
    """
    if alpha == float("inf"):
        return 0.5 / (sigma ** 2)
    return alpha / (2.0 * sigma ** 2)


def _rdp_sampled_gaussian(alpha: float, sigma: float, q: float) -> float:
    """
    RDP of Poisson-subsampled Gaussian mechanism at order α.
    Uses the tight bound from Wang et al. (2019) / Mironov et al. (2019):

        ε_RDP(α) ≤ log(1 + q²·α(α-1)/(2σ²)) / (α-1)

    which is valid for integer α ≥ 2 and small q.
    For non-integer α or α < 2 we fall back to a loose bound.
    """
    if q == 1.0:
        return _rdp_gaussian_single(alpha, sigma)
    if q == 0.0:
        return 0.0

    if alpha < 2:
        # Loose bound: ε_RDP(α) ≤ q² · ε_RDP(2) for α < 2
        return q ** 2 * _rdp_gaussian_single(2.0, sigma)

    # Tight subsampling bound
    log_term = math.log(
        1.0 + q ** 2 * alpha * (alpha - 1.0) / (2.0 * sigma ** 2)
    )
    return log_term / (alpha - 1.0)


def _rdp_to_approxdp(rdp_eps: float, alpha: float, delta: float) -> float:
    """
    Convert RDP guarantee  (α, ε_RDP)  to  (ε, δ)-DP.
    ε = ε_RDP + log(1/δ) / (α-1)          (Balle et al. 2020 / Canonne et al.)
    """
    if alpha == float("inf"):
        return rdp_eps
    return rdp_eps + math.log(1.0 / delta) / (alpha - 1.0)


def compute_epsilon_rdp(
    steps: int,
    noise_multiplier: float,
    sample_rate: float,
    delta: float,
    alphas: Optional[List[float]] = None,
) -> float:
    """
    Compute the  (ε, δ)-DP  privacy guarantee via Rényi DP composition.

    This replicates what Opacus's RDPAccountant does internally.

    Args
    ────
    steps           : total gradient-update steps
                      = rounds × local_epochs × ⌈dataset_size / batch_size⌉
    noise_multiplier: σ  (noise std = σ · clip_norm)
    sample_rate     : q  = batch_size / dataset_size   (Poisson subsampling ratio)
    delta           : target δ
    alphas          : RDP orders to search over (None → use a wide default grid)

    Returns
    ───────
    Minimum ε over the searched α orders.
    """
    if noise_multiplier == 0:
        return float("inf")

    if alphas is None:
        # Wide grid covering typical optimal α values
        alphas = (
            [1.0 + x * 0.1 for x in range(1, 100)]   # 1.1 … 10.9
            + list(range(11, 64))                       # 11 … 63
            + [128.0, 256.0, 512.0, 1024.0]
        )

    best_eps = float("inf")
    for alpha in alphas:
        rdp_per_step = _rdp_sampled_gaussian(alpha, noise_multiplier, sample_rate)
        rdp_total = steps * rdp_per_step          # Basic composition
        eps = _rdp_to_approxdp(rdp_total, alpha, delta)
        if eps < best_eps:
            best_eps = eps

    return max(0.0, best_eps)


def compute_epsilon_for_training(
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    dataset_size: int,
    noise_multiplier: float,
    delta: float,
) -> float:
    """
    Convenience wrapper: compute total ε for a full FL training run.

    In federated learning each client participates in every round, so:
        total_steps = rounds × epochs × ⌈dataset_size / batch_size⌉
    """
    if dataset_size == 0 or batch_size == 0:
        return float("inf")
    steps_per_epoch = max(1, dataset_size // batch_size)
    total_steps = num_rounds * local_epochs * steps_per_epoch
    sample_rate = min(1.0, batch_size / dataset_size)
    return compute_epsilon_rdp(total_steps, noise_multiplier, sample_rate, delta)


# ══════════════════════════════════════════════════════════════════════════════
# Per-Sample Clipping & Noise  (used by manual fallback in train_utils.py)
# ══════════════════════════════════════════════════════════════════════════════

def clip_per_sample_gradients(
    per_sample_grads: List[List[torch.Tensor]],
    clip_norm: float,
) -> List[List[torch.Tensor]]:
    """
    Clip each sample's gradient vector to L2 norm ≤ clip_norm.
    Implements Eq. (2) of the DP-Prox paper:

        g̃_i = g_i / max(1, ‖g_i‖₂ / C)

    Args
    ────
    per_sample_grads : list of B gradient lists, each of length num_params
    clip_norm        : C  (clipping threshold)

    Returns
    ───────
    Same structure as input, with each gradient vector clipped.
    """
    clipped = []
    for grads in per_sample_grads:
        flat = torch.cat([g.reshape(-1) for g in grads])
        norm = flat.norm(2).item()
        scale = min(1.0, clip_norm / (norm + 1e-8))
        clipped.append([g.mul(scale) for g in grads])
    return clipped


def add_gaussian_noise_to_sum(
    sum_grads: List[torch.Tensor],
    noise_multiplier: float,
    clip_norm: float,
    batch_size: int,
) -> List[torch.Tensor]:
    """
    Add calibrated Gaussian noise to the SUM of clipped gradients, then
    divide by batch_size to get the noisy average.  Implements Eqs. (3-4):

        ḡ_DP = (Σ g̃_i + N(0, σ²C²I)) / |B|

    Args
    ────
    sum_grads        : Σ g̃_i  per layer  (already summed, not averaged)
    noise_multiplier : σ
    clip_norm        : C
    batch_size       : |B|

    Returns
    ───────
    Noisy averaged gradients, one tensor per layer.
    """
    sigma = noise_multiplier * clip_norm
    noisy_avg = []
    for g_sum in sum_grads:
        noise = torch.randn_like(g_sum).mul_(sigma)
        noisy_avg.append((g_sum + noise).div_(batch_size))
    return noisy_avg


# ══════════════════════════════════════════════════════════════════════════════
# Legacy Central-DP  (server-side noise — kept for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def compute_updates(
    old_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
) -> List[List[np.ndarray]]:
    """Compute Δ_i = w_i − w_old for each client."""
    return [
        [c - o for c, o in zip(cp, old_params)]
        for cp in client_params_list
    ]


def add_central_dp_noise(
    aggregated_ndarrays: List[np.ndarray],
    client_updates: List[List[np.ndarray]],
    num_examples_list: List[int],
    clip_norm: float,
    noise_multiplier: float,
) -> List[np.ndarray]:
    """
    Legacy Central-DP: clip + aggregate + add ONE noise vector server-side.
    NOT recommended – use client-side DP-SGD (use_dp=True) instead.
    """
    if not client_updates:
        return aggregated_ndarrays

    total_samples = sum(num_examples_list)
    flat_updates = [
        np.concatenate([layer.reshape(-1) for layer in upd])
        for upd in client_updates
    ]

    # Clip each update
    clipped = []
    for flat in flat_updates:
        norm = np.linalg.norm(flat)
        clipped.append(flat * (clip_norm / norm) if norm > clip_norm else flat.copy())

    # Weighted average
    avg_flat = sum(f * n for f, n in zip(clipped, num_examples_list)) / total_samples

    # Add Gaussian noise
    sigma = clip_norm * noise_multiplier
    noisy_flat = avg_flat + np.random.normal(0.0, sigma, size=avg_flat.shape).astype(avg_flat.dtype)

    # Reshape back to layer structure
    out, offset = [], 0
    for ref in aggregated_ndarrays:
        sz = int(np.prod(ref.shape))
        out.append(
            noisy_flat[offset: offset + sz]
            .reshape(ref.shape)
            .astype(aggregated_ndarrays[0].dtype)
        )
        offset += sz
    return out