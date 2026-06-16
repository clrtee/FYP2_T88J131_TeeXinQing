"""
hardware_exp/laptop/save_pretrained.py  —  run on LAPTOP
Trains CNN1D on full UCI-HAR dataset and saves pretrained weights.
Reuses existing src/ code exactly — no duplication.

Run from project root:
    cd /home/clara/Research2.0/fyp_fl_smart_home
    python hardware_exp/laptop/save_pretrained.py
"""

import os, sys, torch, numpy as np
import torch.nn as nn

sys.path.insert(0, os.path.abspath("."))
from src.models import CNN1DClassifier
from src.data_uci_har import load_uci_har
from src.train_utils import make_loader
from src.config import Paths
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paths  = Paths()

print("Loading UCI-HAR...")
(Xtr, ytr), (Xte, yte), scaler = load_uci_har(paths.uci_root)

input_dim   = Xtr.shape[-1]   # 9
num_classes = int(ytr.max()) + 1   # 6

model    = CNN1DClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
opt      = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
loss_fn  = nn.CrossEntropyLoss()

train_loader = make_loader(Xtr, ytr, batch_size=64, shuffle=True)
test_loader  = make_loader(Xte, yte, batch_size=64, shuffle=False)

print(f"Training on {device} for 30 epochs...")
best_acc, best_epoch = 0.0, 0

for epoch in range(1, 31):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            ps.extend(model(xb.to(device)).argmax(1).cpu().numpy())
            ys.extend(yb.numpy())

    acc = accuracy_score(ys, ps)
    f1  = f1_score(ys, ps, average="macro")
    print(f"Epoch {epoch:2d}/30 | Acc: {acc:.4f} | F1: {f1:.4f}")

    if acc > best_acc:
        best_acc   = acc
        best_epoch = epoch
        torch.save(model.state_dict(), "models/pretrained_cnn_ucihar.pth")

print(f"\nBest accuracy: {best_acc:.4f} at epoch {best_epoch}")
print("Saved: models/pretrained_cnn_ucihar.pth")
print("\nNext — copy to RPi:")
print("  scp models/pretrained_cnn_ucihar.pth pi@10.104.245.68:~/hardware_exp/rpi/")
