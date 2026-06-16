"""
rpi_client.py  —  run on RASPBERRY PI
Standalone FedProx client for hardware experiment.

This is your existing fl_client.py (HARClient) made standalone — no relative
imports, same model/training/evaluation logic preserved exactly.

The ONLY additions vs your original code:
  1. psutil.cpu_percent() added to the non-edge fedprox path (line marked ADD)
  2. MPU6050 live data collection (separate function, does not touch FL logic)
  3. Standalone imports replacing "from .module import ..."

Usage:
    # Simulate mode — uses UCI-HAR .npy files exported from laptop
    python rpi_client.py --mode simulate --server 192.168.x.x:8080

    # Live mode — collect real MPU6050 data for each activity
    python rpi_client.py --mode live --server 192.168.x.x:8080
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import flwr as fl
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — exact same values as your config.py / TrainConfig
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareConfig:
    batch_size:    int   = 64
    lr:            float = 0.01      # same as your TrainConfig
    local_epochs:  int   = 3
    prox_mu:       float = 0.01
    seed:          int   = 42
    device:        str   = "cpu"     # RPi has no GPU
    edge_profile:  Optional[str] = None   # None = real hardware (not simulated)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL  — exact copy of your models.CNN1DClassifier
# ══════════════════════════════════════════════════════════════════════════════

import torch.nn.functional as F

class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.2):
        super().__init__()
        self.conv1   = nn.Conv1d(in_channels=input_dim, out_channels=64,
                                  kernel_size=3, padding=1)
        self.conv2   = nn.Conv1d(in_channels=64, out_channels=128,
                                  kernel_size=3, padding=1)
        self.pool    = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)           # (N, 128, 9) → (N, 9, 128)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.mean(dim=-1)              # global average pooling
        return self.fc(x)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN UTILS  — exact copy of your train_utils.py
# ══════════════════════════════════════════════════════════════════════════════

def make_loader(X, y, batch_size=64, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_client(model, loader, epochs, lr, device,
                     prox_mu=0.0, global_params=None, clip_norm=5.0):
    """Exact copy of your train_utils.train_one_client."""
    model.to(device)
    model.train()
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    if prox_mu > 0.0 and global_params is not None:
        global_params = [p.detach().clone().to(device) for p in global_params]

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)

            if prox_mu > 0.0 and global_params is not None:
                prox = 0.0
                for p, gp in zip(model.parameters(), global_params):
                    prox += torch.sum((p - gp) ** 2)
                loss = loss + (prox_mu / 2.0) * prox

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            opt.step()
    return None


@torch.no_grad()
def evaluate(model, loader, device):
    """Exact copy of your train_utils.evaluate."""
    model.to(device)
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb    = xb.to(device)
        logits = model(xb)
        pred   = torch.argmax(logits, dim=1).cpu().numpy()
        ys.extend(yb.numpy().tolist())
        ps.extend(pred.tolist())
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


@torch.no_grad()
def eval_with_loss(model, loader, device):
    """Exact copy of your fl_client.eval_with_loss."""
    model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_n = 0.0, 0
    ys, ps = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits  = model(xb)
        loss    = loss_fn(logits, yb)
        bs      = yb.shape[0]
        total_loss += float(loss.item()) * bs
        total_n    += bs
        ps.extend(torch.argmax(logits, 1).cpu().numpy().tolist())
        ys.extend(yb.cpu().numpy().tolist())
    acc = accuracy_score(ys, ps) if total_n > 0 else 0.0
    f1  = f1_score(ys, ps, average="macro") if total_n > 0 else 0.0
    return total_loss / max(1, total_n), acc, f1


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS  — exact copy from your fl_client.py
# ══════════════════════════════════════════════════════════════════════════════

def get_parameters_torch(model: nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_parameters_torch(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(parameters) != len(keys):
        raise ValueError(
            f"Parameter length mismatch: got {len(parameters)}, expected {len(keys)}"
        )
    new_state = {k: torch.tensor(arr, dtype=state_dict[k].dtype)
                 for k, arr in zip(keys, parameters)}
    model.load_state_dict(new_state, strict=True)


def get_model_size_bytes(model: nn.Module) -> int:
    """Exact copy from your fl_client.py."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def get_memory_usage_mb() -> float:
    """Exact copy from your fl_client.py."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


# ══════════════════════════════════════════════════════════════════════════════
# HAR CLIENT  — your HARClient, fedprox path only, with cpu_pct added
# ══════════════════════════════════════════════════════════════════════════════

class HARClient(fl.client.NumPyClient):
    """
    Your existing HARClient, fedprox path extracted for hardware experiment.
    Single addition vs original: cpu_pct measurement in non-edge fit() path.
    """

    def __init__(
        self,
        cid:      str,
        X_train:  np.ndarray,
        y_train:  np.ndarray,
        X_test:   np.ndarray,
        y_test:   np.ndarray,
        cfg:      HardwareConfig,
        metrics_path: str = "rpi_metrics.csv",
    ):
        self.cid          = cid
        self.cfg          = cfg
        self.device       = torch.device(cfg.device)
        self.metrics_path = metrics_path
        self._round       = 0

        # Same model as your experiment: CNN1DClassifier(input_dim=9, num_classes=6)
        self.model = CNN1DClassifier(input_dim=9, num_classes=6).to(self.device)

        self.train_loader = make_loader(X_train, y_train,
                                        batch_size=cfg.batch_size, shuffle=True)
        self.test_loader  = make_loader(X_test,  y_test,
                                        batch_size=cfg.batch_size, shuffle=False)

        self.lr           = cfg.lr
        self.epochs       = cfg.local_epochs
        self.base_bytes   = get_model_size_bytes(self.model)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return get_parameters_torch(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_parameters_torch(self.model, parameters)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        self.set_parameters(parameters)
        self._round += 1

        algo     = str(config.get("algo", "fedprox")).lower()
        prox_mu  = float(config.get("prox_mu", self.cfg.prox_mu))
        num_examples = len(self.train_loader.dataset)

        m: Dict[str, Any] = {}
        m["comm_bytes_down"] = self.base_bytes   # same as your fl_client.py
        m["comm_bytes_up"]   = self.base_bytes
        m["memory_usage_mb"] = float(get_memory_usage_mb())

        global_params = None
        if prox_mu > 0.0:
            global_params = [p.detach().clone() for p in self.model.parameters()]

        # ── SAME as your non-edge fedprox path in fl_client.py ──
        # ADDITION: cpu_percent measured around training (justified — observe only)
        start_time = time.time()
        psutil.cpu_percent(interval=None)          # [ADD] reset/prime the counter

        _ = train_one_client(
            model=self.model,
            loader=self.train_loader,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device,
            prox_mu=prox_mu,
            global_params=global_params,
            clip_norm=float(getattr(self.cfg, "cdp_clip_norm", 5.0)),
        )

        m["fit_wall_s"] = float(time.time() - start_time)   # same as your code
        m["cpu_pct"]    = float(psutil.cpu_percent(interval=None))  # [ADD]

        self._log(m)
        print(f"[rpi] Round {self._round:3d} | "
              f"Latency: {m['fit_wall_s']*1000:.0f} ms | "
              f"CPU: {m['cpu_pct']:.1f}% | "
              f"RAM: {m['memory_usage_mb']:.0f} MB")

        return self.get_parameters({}), num_examples, m

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        """Exact copy of your HARClient.evaluate."""
        self.set_parameters(parameters)
        acc, f1   = evaluate(self.model, self.test_loader, self.device)
        loss, _, _ = eval_with_loss(self.model, self.test_loader, self.device)
        num_examples = len(self.test_loader.dataset)
        print(f"[rpi] Eval  | Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        return float(loss), num_examples, {"accuracy": float(acc), "f1": float(f1)}

    def _log(self, m: Dict):
        row = {"round": self._round,
               "fit_wall_s":     round(m.get("fit_wall_s",     0), 3),
               "cpu_pct":        round(m.get("cpu_pct",        0), 1),
               "memory_usage_mb":round(m.get("memory_usage_mb",0), 1),
               "comm_bytes_down":m.get("comm_bytes_down", 0),
               "comm_bytes_up":  m.get("comm_bytes_up",   0)}
        header = not os.path.exists(self.metrics_path)
        with open(self.metrics_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if header:
                w.writeheader()
            w.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_simulate_data(data_dir: str = "."):
    """
    Load .npy files exported by export_rpi_data.py.
    Shape: (N, 128, 9) — same as your experiment.
    """
    X_train = np.load(os.path.join(data_dir, "rpi_X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "rpi_y_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "rpi_X_test.npy"))
    y_test  = np.load(os.path.join(data_dir, "rpi_y_test.npy"))
    print(f"[data] Loaded — Train: {X_train.shape}, Test: {X_test.shape}")
    cls_dist = {int(c): int((y_train == c).sum()) for c in np.unique(y_train)}
    print(f"[data] Client class distribution: {cls_dist}")
    return X_train, y_train, X_test, y_test


def collect_live_data(scaler_path: str = "rpi_scaler.pkl",
                      windows_per_activity: int = 20,
                      X_test: Optional[np.ndarray] = None,
                      y_test: Optional[np.ndarray] = None):
    """
    Collect live MPU6050 data and preprocess to match your 9-channel format:
      Channel 0-2: body_acc_xyz  (raw accel minus gravity estimate)
      Channel 3-5: body_gyro_xyz (raw gyro)
      Channel 6-8: total_acc_xyz (raw accel, gravity included)
    Then apply the same StandardScaler fitted on your training data.
    """
    from mpu6050_reader import MPU6050

    ACTIVITIES = {
        0: "WALKING",
        1: "WALKING_UPSTAIRS",
        2: "WALKING_DOWNSTAIRS",
        3: "SITTING",
        4: "STANDING",
        5: "LAYING",
    }

    mpu = MPU6050()

    # Load scaler fitted on your training data
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[data] Loaded StandardScaler from {scaler_path}")

    # Calibrate gravity (stay still ~3 s)
    print("\n[data] Calibrating — keep sensor still for 3 seconds...")
    cal_window = mpu.collect_window(window_size=150, fs=50)   # (150, 6)
    gravity_estimate = cal_window[:, :3].mean(axis=0)          # mean accel = gravity
    print(f"[data] Gravity estimate: {gravity_estimate.round(3)}")

    X_all, y_all = [], []

    for label, activity in ACTIVITIES.items():
        print(f"\n>>> Activity {label}: {activity}")
        input("    Get into position, then press Enter to start...")

        windows = []
        for w in range(windows_per_activity):
            print(f"    Window {w+1}/{windows_per_activity}...", end="\r", flush=True)
            raw = mpu.collect_window(window_size=128, fs=50)  # (128, 6): accel_xyz + gyro_xyz

            total_acc = raw[:, :3]                            # (128, 3) raw accel
            body_acc  = raw[:, :3] - gravity_estimate        # (128, 3) gravity removed
            body_gyro = raw[:, 3:]                            # (128, 3)

            # Stack to (128, 9) — same channel order as your experiment
            window_9ch = np.concatenate([body_acc, body_gyro, total_acc], axis=1)
            windows.append(window_9ch)

        print(f"    Collected {windows_per_activity} windows for '{activity}'.")
        X_all.extend(windows)
        y_all.extend([label] * windows_per_activity)

    X = np.stack(X_all).astype(np.float32)   # (N, 128, 9)
    y = np.array(y_all, dtype=np.int64)

    # Apply same StandardScaler as your experiment
    N, T, F = X.shape
    X_2d  = X.reshape(N * T, F)
    X_2d  = scaler.transform(X_2d)           # same scaler as training
    X     = X_2d.reshape(N, T, F).astype(np.float32)

    np.save("mpu_X_live.npy", X)
    np.save("mpu_y_live.npy", y)
    print(f"\n[data] Saved live data: {X.shape}")

    # Use UCI-HAR test set for evaluation (or live data if no test set given)
    if X_test is None:
        X_test, y_test = X, y

    return X, y, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    cfg = HardwareConfig(
        batch_size=64,
        lr=0.01,
        local_epochs=3,
        prox_mu=0.01,
    )

    if args.mode == "simulate":
        X_train, y_train, X_test, y_test = load_simulate_data(args.data_dir)

    elif args.mode == "live":
        # Need test set for evaluation — load from .npy if available
        test_path = os.path.join(args.data_dir, "rpi_X_test.npy")
        if os.path.exists(test_path):
            X_test_ref = np.load(test_path)
            y_test_ref = np.load(os.path.join(args.data_dir, "rpi_y_test.npy"))
        else:
            X_test_ref, y_test_ref = None, None

        X_train, y_train, X_test, y_test = collect_live_data(
            scaler_path=os.path.join(args.data_dir, "rpi_scaler.pkl"),
            windows_per_activity=args.windows,
            X_test=X_test_ref,
            y_test=y_test_ref,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    client = HARClient(
        cid="rpi-0",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cfg=cfg,
        metrics_path=args.metrics,
    )

    print(f"\n[rpi] Connecting to {args.server} ...")
    print(f"[rpi] Config — lr={cfg.lr}, batch={cfg.batch_size}, "
          f"epochs={cfg.local_epochs}, prox_mu={cfg.prox_mu}")
    fl.client.start_numpy_client(server_address=args.server, client=client)
    print(f"[rpi] Done. Per-round metrics saved to {args.metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     choices=["simulate", "live"], default="simulate")
    parser.add_argument("--server",   default="192.168.1.100:8080",
                        help="Laptop IP:port running server_hardware.py")
    parser.add_argument("--data_dir", default=".",
                        help="Directory with rpi_X_train.npy etc.")
    parser.add_argument("--metrics",  default="rpi_metrics.csv")
    parser.add_argument("--windows",  type=int, default=20,
                        help="Windows per activity in live mode")
    main(parser.parse_args())
