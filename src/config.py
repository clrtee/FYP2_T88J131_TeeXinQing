from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-3

    local_epochs: int = 5
    rounds: int = 50

    num_clients: int = 3
    partition_mode: str = "iid"   # "non_iid" or "iid"
    dirichlet_alpha: float = 0.1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ Edge simulation (None = disabled)
    edge_profile: Optional[str] = None

    # DP (optional)
    use_dp: bool = False
    dp_noise_multiplier: float = 1.0
    dp_max_grad_norm: float = 1.0

    # FedProx
    prox_mu: float = 0.01

    # SCAFFOLD: 建议用 SGD（更贴论文实现）
    scaffold_use_sgd: bool = True

    # FedDC
    feddc_penalty_alpha: float = 0.5


@dataclass
class Paths:
    uci_root: str = "data/uci_har/UCI HAR Dataset"
    casas_file: str = "data/casas_aruba/aruba.csv"
    out_dir: str = "results"
