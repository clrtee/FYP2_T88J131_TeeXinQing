import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import CNN1D
from load_ucihar import load_ucihar

print("📦 加载 UCI-HAR 数据...")
X_train, y_train, X_test, y_test = load_ucihar()

X_tr = torch.tensor(X_train, dtype=torch.float32)
y_tr = torch.tensor(y_train, dtype=torch.long)
X_te = torch.tensor(X_test,  dtype=torch.float32)
y_te = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = CNN1D().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

print(f"🚀 开始训练（设备：{device}）...")
for epoch in range(50):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                correct += (model(xb.to(device)).argmax(1) == yb.to(device)).sum().item()
        acc = correct / len(y_test)
        print(f"Epoch {epoch+1}/50 | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.4f}")

torch.save(model.state_dict(), "pretrained_9ch.pth")
print("✅ 预训练模型已保存：pretrained_9ch.pth")
