# src/centralized.py
import argparse
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from .config import TrainConfig, Paths
from .models import LSTMClassifier, CNN1DClassifier
from .data_uci_har import load_uci_har
from .data_casas_aruba_csv import load_casas_aruba_csv
from .train_utils import FocalLoss, make_loader

def build_model(arch: str, input_dim: int, num_classes: int):
    if arch == "cnn":
        return CNN1DClassifier(input_dim=input_dim, num_classes=num_classes, dropout=0.2)
    else:
        return LSTMClassifier(input_dim=input_dim, hidden_dim=128, num_layers=2, num_classes=num_classes, dropout=0.2)

def main():
    parser = argparse.ArgumentParser(description="Centralized Training Benchmark (Full Epochs)")
    parser.add_argument("--dataset", choices=["uci", "casas"], required=True)
    parser.add_argument("--arch", choices=["lstm", "cnn"], default="lstm")
    parser.add_argument("--epochs", type=int, default=500, help="Total epochs to train")
    parser.add_argument("--focal", action="store_true", help="Use Focal Loss")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (Adam)")
    
    args = parser.parse_args()
    paths = Paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动集中式训练 (无早停) | 数据集: {args.dataset} | 模型: {args.arch} | Device: {device}")

    # 1. 加载完整数据集 (不进行任何 FL 分区)
    if args.dataset == "uci":
        (Xtr, ytr), (Xte, yte) = load_uci_har(paths.uci_root)
    else:
        X, y, sensor2id, act2id = load_casas_aruba_csv(paths.casas_file, window_len=50, stride=25)
        cut = int(0.8 * len(X))
        Xtr, ytr = X[:cut], y[:cut]
        Xte, yte = X[cut:], y[cut:]

    train_loader = make_loader(Xtr, ytr, batch_size=64, shuffle=True)
    test_loader = make_loader(Xte, yte, batch_size=64, shuffle=False)

    input_dim = Xtr.shape[-1]
    num_classes = int(np.max(ytr) + 1)

    # 2. 初始化模型和优化器
    model = build_model(args.arch, input_dim, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = FocalLoss(gamma=2.0) if args.focal else nn.CrossEntropyLoss()

    history = []
    
    print(f"📊 开始训练... 目标 Epoch: {args.epochs}")

    for epoch in range(1, args.epochs + 1):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        start_time = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_loader.dataset)
        wall_s = time.time() - start_time

        # --- 测试阶段 ---
        model.eval()
        test_loss = 0.0
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                test_loss += loss.item() * len(yb)
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                ys.extend(yb.cpu().numpy().tolist())
                ps.extend(preds.tolist())
                
        test_loss /= len(test_loader.dataset)
        acc = accuracy_score(ys, ps)
        f1 = f1_score(ys, ps, average="macro")

        # 保存本轮数据 (使用 "round" 键以兼容现有的画图脚本)
        history.append({
            "round": epoch,
            "train_loss": train_loss,
            "loss": test_loss,
            "accuracy": acc,
            "f1": f1,
            "fit_wall_s": wall_s
        })

        # 为了避免输出太吵，每 10 个 epoch 打印一次进度
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # 4. 保存为 CSV，格式与 FL 历史记录一致
    os.makedirs(paths.out_dir, exist_ok=True)
    filename = f"{args.dataset}_{args.arch}_centralized_benchmark.csv"
    out_path = os.path.join(paths.out_dir, filename)
    
    df = pd.DataFrame(history)
    df.to_csv(out_path, index=False)
    print(f"✅ 集中式训练完成！完整日志已保存至: {out_path}")

if __name__ == "__main__":
    main()