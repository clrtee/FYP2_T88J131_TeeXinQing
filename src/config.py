# src/config.py
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class TrainConfig:
    seed: int = 42

    # ── Core training ─────────────────────────────────────────────────────
    # Larger batch_size → stronger subsampling amplification → less noise
    # per effective update (paper uses batch tuned to saturate 8 GB device)
    batch_size: int = 64
    lr: float = 0.01
    # DP-SGD converges slower; use 2–3× more rounds than non-DP baseline
    local_epochs: int = 3          # was 3; more local work helps DP convergence
    rounds: int = 500              # was 500/50; tuned for DP (paper: T≈40 to converge)
    num_clients: int = 10
    partition_mode: str = "non_iid"
    dirichlet_alpha: float = 0.1   # Dirichlet α (paper uses 0.3; 0.1 is more heterogeneous)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    edge_profile: Optional[str] = None

    # ── FedProx ───────────────────────────────────────────────────────────
    # μ = 0.01 is the best from the paper's ablation sweep over {0,0.001,0.01,0.05}
    # With DP on, keep μ small so proximal pull does not over-regularise noisy grads
    prox_mu: float = 0.01
    scaffold_use_sgd: bool = True
    feddc_penalty_alpha: float = 0.01

    # ── DP-SGD  (client-side, Algorithm 1 of DP-Prox paper) ──────────────
    #
    #  Paper §IV.D:   C ≈ 1.0,  σ ≈ 0.8
    #  Paper §V.B:    ε = 5 is the "knee of the curve"
    #                 ε = 3 → F1 ≈ 0.74  |  ε = 5 → F1 ≈ 0.77  |  ε = 8 → F1 ≈ 0.79
    #
    use_dp: bool = False
    dp_clip_norm: float = 1.0          # C  – per-sample gradient clipping norm
    dp_noise_multiplier: float = 9.99   # σ  – Gaussian noise std = σ * C
    dp_target_epsilon: float = 6.0     # ε  – target privacy budget  (RDP "knee")
    dp_target_delta: float = 1e-5      # δ  – failure probability (= 1/N standard)
    # ──────────────────────────────────────────────────────────────────────

    # ── Legacy Central-DP  (backward compat only – NOT recommended) ───────
    # The old scheme adds noise SERVER-SIDE after aggregation.
    # This gives weaker privacy (server sees raw updates) and more accuracy
    # loss (noise scales with number of clients).  Use use_dp=True instead.
    use_central_dp: bool = False
    cdp_clip_norm: float = 1.0
    cdp_noise_multiplier: float = 0.8
    # ──────────────────────────────────────────────────────────────────────


@dataclass
class Paths:
    uci_root: str = "data/uci_har/UCI HAR Dataset"
    casas_file: str = "data/casas_aruba/aruba.csv"
    out_dir: str = "results"
