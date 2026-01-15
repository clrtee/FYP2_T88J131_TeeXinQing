import argparse
import numpy as np
import flwr as fl

from .config import TrainConfig, Paths
from .models import LSTMClassifier
from .data_uci_har import load_uci_har
from .data_casas_aruba_csv import load_casas_aruba_csv
from .partition import dirichlet_partition, iid_partition   # ✅ 新增 iid_partition
from .fl_client import HARClient
from .fl_server import run_server


def build_model(input_dim: int, num_classes: int):
    def fn():
        return LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.2,
        )
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["server", "client"], required=True)
    parser.add_argument("--cid", type=int, default=0)
    parser.add_argument("--dataset", choices=["uci", "casas"], required=True)
    parser.add_argument("--algo", choices=["fedavg", "fedprox", "scaffold", "feddc"], default="fedavg")
    parser.add_argument("--prox_mu", type=float, default=0.01)
    parser.add_argument("--dp", action="store_true")

    # ✅ 新增：IID / non-IID 切换
    parser.add_argument(
        "--partition",
        choices=["iid", "non_iid"],
        default="non_iid",
        help="Data partition mode: iid or non_iid (Dirichlet)",
    )

    # non-IID 强度（只有 non_iid 时才用）
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Dirichlet alpha for non-IID partition",
    )

    # edge simulation optional
    parser.add_argument(
        "--edge",
        type=str,
        choices=["rpi4", "rpi5", "jetson_nano", "coral", "esp32"],
        default=None,
        help="Enable edge simulation (optional)",
    )

    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.use_dp = bool(args.dp)
    cfg.edge_profile = args.edge
    cfg.dirichlet_alpha = args.alpha  # 仍然保留原逻辑

    paths = Paths()

    # Load dataset
    if args.dataset == "uci":
        (Xtr, ytr), (Xte, yte) = load_uci_har(paths.uci_root)
    else:
        X, y, sensor2id, act2id = load_casas_aruba_csv(
            paths.casas_file, window_len=50, stride=25
        )
        n = len(X)
        cut = int(0.8 * n)
        Xtr, ytr = X[:cut], y[:cut]
        Xte, yte = X[cut:], y[cut:]

    # =====================================================
    # ✅ Partition train set into clients (IID / non-IID)
    # =====================================================
    if args.partition == "iid":
        client_idxs = iid_partition(
            ytr,
            num_clients=cfg.num_clients,
            seed=cfg.seed,
        )
    else:
        client_idxs = dirichlet_partition(
            ytr,
            num_clients=cfg.num_clients,
            alpha=cfg.dirichlet_alpha,
            seed=cfg.seed,
        )

    # ✅ 打印分布（论文 & debug 都很重要）
    print("\n=== Client label distribution ===")
    for cid, idx in enumerate(client_idxs):
        labels = ytr[idx]
        uniq, cnt = np.unique(labels, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(uniq, cnt)}
        print(f"Client {cid}: n={len(idx)} dist={dist}")
    print("================================\n")

    # Build model factory
    input_dim = Xtr.shape[-1]
    num_classes = int(np.max(ytr) + 1)
    model_fn = build_model(input_dim, num_classes)

    # Run server
    if args.role == "server":
        run_server(
            args.algo,
            rounds=cfg.rounds,
            prox_mu=args.prox_mu,
            dataset=args.dataset,
        )
        return

    # Run client
    cid = args.cid
    idx = client_idxs[cid]
    Xc, yc = Xtr[idx], ytr[idx]

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

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )


if __name__ == "__main__":
    main()
