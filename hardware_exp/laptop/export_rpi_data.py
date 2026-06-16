"""
hardware_exp/laptop/export_rpi_data.py  —  run on LAPTOP

Exports one UCI-HAR client partition + fitted StandardScaler for the RPi.
Reuses your existing src/ code — no duplication.

Run from your project root:
    cd /home/clara/Research2.0/fyp_fl_smart_home
    python hardware_exp/laptop/export_rpi_data.py --client_id 0

Then copy outputs to RPi:
    scp rpi_X_train.npy rpi_y_train.npy rpi_X_test.npy rpi_y_test.npy rpi_scaler.pkl \
        pi@<RPI_IP>:~/hardware_exp/rpi/
"""

import argparse
import os
import pickle
import sys

import numpy as np

# ── Point to your existing src/ package ───────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

from src.data_uci_har import load_uci_har   # your existing function, untouched
from src.config import Paths, TrainConfig    # your existing config


def dirichlet_partition(X, y, num_clients: int, alpha: float,
                         client_id: int, seed: int = 42):
    """
    Same Dirichlet non-IID partition as your experiments (alpha=0.1).
    Isolated here because src/partition.py may use different signatures —
    check src/partition.py and replace this with your actual call if preferred.
    """
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(y))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(y == c)[0].copy()
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(class_idx)).astype(int)
        diff = len(class_idx) - proportions.sum()
        proportions[np.argmax(proportions)] += diff
        splits = np.split(class_idx, np.cumsum(proportions)[:-1])
        for i, s in enumerate(splits):
            client_indices[i].extend(s.tolist())

    idx = np.array(client_indices[client_id])
    return X[idx], y[idx]


def main(args):
    paths = Paths()
    cfg   = TrainConfig()

    print(f"Loading UCI-HAR from: {paths.uci_root}")
    (Xtr, ytr), (Xte, yte), scaler = load_uci_har(paths.uci_root)
    # NOTE: load_uci_har returns a scaler only if you added that return value.
    # If your version returns just (Xtr,ytr),(Xte,yte), add scaler fitting below.
    print(f"Full dataset — Train: {Xtr.shape}  Test: {Xte.shape}")

    # Partition training data for this RPi client
    X_client, y_client = dirichlet_partition(
        Xtr, ytr,
        num_clients=cfg.num_clients,       # 10, from your config
        alpha=cfg.dirichlet_alpha,          # 0.1, from your config
        client_id=args.client_id,
        seed=cfg.seed,
    )
    print(f"Client {args.client_id} — X: {X_client.shape}  y: {y_client.shape}")
    dist = {int(c): int((y_client == c).sum()) for c in np.unique(y_client)}
    print(f"Class distribution: {dist}")

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    np.save(os.path.join(out, "rpi_X_train.npy"), X_client.astype(np.float32))
    np.save(os.path.join(out, "rpi_y_train.npy"), y_client.astype(np.int64))
    np.save(os.path.join(out, "rpi_X_test.npy"),  Xte.astype(np.float32))
    np.save(os.path.join(out, "rpi_y_test.npy"),  yte.astype(np.int64))

    # Save scaler so RPi can apply same normalisation to live MPU6050 data
    with open(os.path.join(out, "rpi_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved to {out}/:")
    print("  rpi_X_train.npy  rpi_y_train.npy  — client training partition")
    print("  rpi_X_test.npy   rpi_y_test.npy   — full test set for evaluation")
    print("  rpi_scaler.pkl                     — StandardScaler for live MPU6050 mode")
    print(f"\nNext — copy to RPi:")
    print(f"  scp {out}/rpi_*.npy {out}/rpi_scaler.pkl pi@<RPI_IP>:~/hardware_exp/rpi/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, default=0,
                        help="Which of the 10 clients to assign to the RPi")
    parser.add_argument("--out_dir",   default="hardware_exp/rpi",
                        help="Where to save .npy and .pkl files")
    main(parser.parse_args())
