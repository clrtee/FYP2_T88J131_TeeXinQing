# src/main.py
import argparse
import os

import numpy as np
import flwr as fl

from .config import TrainConfig, Paths
from .models import LSTMClassifier, CNN1DClassifier
from .data_uci_har import load_uci_har
from .data_casas_aruba_csv import load_casas_aruba_csv
from .partition import dirichlet_partition, iid_partition
from .fl_client import HARClient
from .fl_server import run_server
from .utils_history import save_history_csv
from .dp_utils import compute_epsilon_for_training


# ══════════════════════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════════════════════

def build_model(arch: str, input_dim: int, num_classes: int, use_dp: bool = False):
    """
    Return a zero-argument factory function that builds the chosen model.

    Parameters
    ──────────
    use_dp : True → LSTMClassifier uses DPLSTM (Opacus-compatible).
             Has no effect for CNN (Conv1d is natively supported by Opacus).
    """
    def fn():
        if arch == "cnn":
            return CNN1DClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                dropout=0.2,
            )
        else:
            return LSTMClassifier(
                input_dim=input_dim,
                hidden_dim=128,
                num_layers=2,
                num_classes=num_classes,
                dropout=0.2,
                use_dp=use_dp,        # ← KEY: enables DPLSTM when DP is on
            )
    return fn


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Federated Learning for HAR (DP-Prox)")
    parser.add_argument("--role", choices=["server", "client"], required=True)
    parser.add_argument("--cid", type=int, default=0)
    parser.add_argument("--dataset", choices=["uci", "casas"], required=True)
    parser.add_argument("--algo", choices=["fedavg", "fedprox", "scaffold", "feddc"],
                        default="fedprox")
    parser.add_argument("--arch", choices=["lstm", "cnn"], default="lstm")

    # FedProx
    parser.add_argument("--prox_mu", type=float, default=0.01,
                        help="FedProx proximal coefficient μ (paper recommends 0.01)")

    # DP-SGD (Algorithm 1 of DP-Prox paper)
    parser.add_argument("--dp", action="store_true",
                        help="Enable client-side DP-SGD (Algorithm 1 of DP-Prox paper)")
    parser.add_argument("--dp_clip", type=float, default=None,
                        help="Override dp_clip_norm C (default 1.0)")
    parser.add_argument("--dp_noise", type=float, default=None,
                        help="Override dp_noise_multiplier σ (default 0.8)")
    parser.add_argument("--dp_epsilon", type=float, default=None,
                        help="Override dp_target_epsilon (default 5.0)")

    # Training
    parser.add_argument("--rounds", type=int, default=None,
                        help="Number of FL rounds (default 150 with DP, 500 without)")
    parser.add_argument("--partition", choices=["iid", "non_iid"], default="non_iid")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Dirichlet concentration α (lower = more heterogeneous)")
    parser.add_argument("--edge", type=str, default=None)

    args = parser.parse_args()

    # ── Build config ───────────────────────────────────────────────────────
    cfg = TrainConfig()
    paths = Paths()

    cfg.use_dp = bool(args.dp)
    cfg.dirichlet_alpha = args.alpha
    cfg.edge_profile = args.edge
    cfg.prox_mu = args.prox_mu

    if args.dp_clip is not None:
        cfg.dp_clip_norm = args.dp_clip
    if args.dp_noise is not None:
        cfg.dp_noise_multiplier = args.dp_noise
    if args.dp_epsilon is not None:
        cfg.dp_target_epsilon = args.dp_epsilon

    # Default rounds: more rounds needed when DP is enabled
    if args.rounds is not None:
        cfg.rounds = args.rounds
    elif cfg.use_dp and cfg.rounds > 150:
        cfg.rounds = 150   # Cap at DP-tuned default

    # ── Data ───────────────────────────────────────────────────────────────
    if args.dataset == "uci":
        (Xtr, ytr), (Xte, yte), _ = load_uci_har(paths.uci_root)
    else:
        X, y, _, _ = load_casas_aruba_csv(
            paths.casas_file, window_len=50, stride=25
        )
        cut = int(0.8 * len(X))
        Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

    # ── Partition ──────────────────────────────────────────────────────────
    if args.partition == "iid":
        client_idxs = iid_partition(ytr, num_clients=cfg.num_clients, seed=cfg.seed)
    else:
        client_idxs = dirichlet_partition(
            ytr, num_clients=cfg.num_clients, alpha=cfg.dirichlet_alpha, seed=cfg.seed
        )

    # ── Model factory ──────────────────────────────────────────────────────
    input_dim = Xtr.shape[-1]
    num_classes = int(np.max(ytr) + 1)
    model_fn = build_model(args.arch, input_dim, num_classes, use_dp=cfg.use_dp)

    # ══════════════════════════════════════════════════════════════════════
    # SERVER
    # ══════════════════════════════════════════════════════════════════════
    if args.role == "server":
        print(
            f"\n[SERVER START]  model={args.arch} | algo={args.algo}"
            f" | DP={cfg.use_dp} | rounds={cfg.rounds}"
            f" | μ={cfg.prox_mu} | α={cfg.dirichlet_alpha}"
        )

        # Print expected privacy budget
        if cfg.use_dp:
            # Use average client dataset size for estimate
            avg_size = len(ytr) // cfg.num_clients
            est_eps = compute_epsilon_for_training(
                num_rounds=cfg.rounds,
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                dataset_size=avg_size,
                noise_multiplier=cfg.dp_noise_multiplier,
                delta=cfg.dp_target_delta,
            )
            print(
                f"  [DP] σ={cfg.dp_noise_multiplier}  C={cfg.dp_clip_norm}"
                f"  Estimated total ε ≈ {est_eps:.2f}  "
                f"(target ≤ {cfg.dp_target_epsilon})"
            )

        history = run_server(
            algo=args.algo,
            rounds=cfg.rounds,
            prox_mu=args.prox_mu,
            dataset=args.dataset,
            use_dp=cfg.use_dp,
        )

        if history:
            if cfg.use_dp:
                dp_tag = (
                    f"dpsgd_C{cfg.dp_clip_norm}_s{cfg.dp_noise_multiplier}"
                    f"_e{cfg.dp_target_epsilon}"
                )
            else:
                dp_tag = "nodp"

            filename = (
                f"{args.dataset}_{args.arch}_{args.algo}"
                f"_{args.partition}_a{args.alpha}"
                f"_r{cfg.rounds}_{dp_tag}.csv"
            )
            out_path = os.path.join(paths.out_dir, filename)
            save_history_csv(history, out_path)
            print(f"\n[SERVER] Results saved → {out_path}")
        return

    # ══════════════════════════════════════════════════════════════════════
    # CLIENT
    # ══════════════════════════════════════════════════════════════════════
    cid = args.cid
    idx = client_idxs[cid]
    Xc, yc = Xtr[idx], ytr[idx]

    print(
        f"[CLIENT {cid}]  samples={len(yc)} | DP={cfg.use_dp}"
        + (
            f" | C={cfg.dp_clip_norm} σ={cfg.dp_noise_multiplier}"
            if cfg.use_dp else ""
        )
    )

    client = HARClient(
        cid=str(cid),
        model_fn=model_fn,
        X_train=Xc,
        y_train=yc,
        X_test=Xte,
        y_test=yte,
        cfg=cfg,
        algo=args.algo,
    )

    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()