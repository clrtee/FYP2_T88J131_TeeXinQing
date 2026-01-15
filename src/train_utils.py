import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

def make_loader(X, y, batch_size=64, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def train_one_client(model, loader, epochs, lr, device, prox_mu=0.0, global_params=None, dp_cfg=None):
    """
    prox_mu>0 => FedProx: add (mu/2)*||w - w_global||^2
    dp_cfg: dict with keys use_dp, noise_multiplier, max_grad_norm
    """
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    privacy_engine = None
    if dp_cfg and dp_cfg.get("use_dp", False):
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine()
        model, opt, loader = privacy_engine.make_private(
            module=model,
            optimizer=opt,
            data_loader=loader,
            noise_multiplier=dp_cfg["noise_multiplier"],
            max_grad_norm=dp_cfg["max_grad_norm"],
        )

    # cache global params for prox
    if prox_mu > 0.0 and global_params is not None:
        global_params = [p.detach().clone().to(device) for p in global_params]

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)

            if prox_mu > 0.0 and global_params is not None:
                prox = 0.0
                for p, gp in zip(model.parameters(), global_params):
                    prox += torch.sum((p - gp) ** 2)
                loss = loss + (prox_mu / 2.0) * prox


            loss.backward()
            opt.step()

    eps = None
    if privacy_engine is not None:
        eps = privacy_engine.get_epsilon(delta=1e-5)

    return eps

@torch.no_grad()
def evaluate(model, loader, device):
    model.to(device)
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.extend(yb.numpy().tolist())
        ps.extend(pred.tolist())
    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="macro")
    return acc, f1

def train_scaffold(
    model,
    loader,
    epochs: int,
    lr: float,
    device: str,
    c_global,     # List[Tensor] same shapes as params
    c_local,      # List[Tensor]
):
    """
    SCAFFOLD local update:
      w <- w - lr * (grad - c_local + c_global)

    Returns:
      steps (K): number of optimizer steps performed
    """
    model.to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    steps = 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # forward
            logits = model(xb)
            loss = loss_fn(logits, yb)

            # backward
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss.backward()

            # manual SGD step with correction
            with torch.no_grad():
                for p, cg, cl in zip(model.parameters(), c_global, c_local):
                    if p.grad is None:
                        continue
                    corrected_grad = p.grad - cl + cg
                    p -= lr * corrected_grad

            steps += 1

    return steps

def train_feddc(
    model,
    loader,
    epochs: int,
    lr: float,
    device: str,
    w_global,           # List[Tensor] global params (fixed during local training)
    h_local,            # List[Tensor] drift variable hi
    gi_prev,            # List[Tensor] last round local update gi
    g_prev,             # List[Tensor] last round average update g
    penalty_alpha: float,
):
    """
    FedDC objective (Eq.4 in paper):
      L_i(theta) + (alpha/2)||h + theta - w||^2 + (1/(eta*K)) <theta, gi - g>

    SGD update uses gradient:
      grad = grad_L + alpha*(h + theta - w) + (1/(eta*K))*(gi_prev - g_prev)

    Returns:
      steps K
    """
    model.to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    # total steps K (epochs * num_batches)
    K = epochs * len(loader)
    if K <= 0:
        K = 1

    steps = 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss.backward()

            with torch.no_grad():
                for p, w, h, gi, g in zip(model.parameters(), w_global, h_local, gi_prev, g_prev):
                    if p.grad is None:
                        continue
                    # penalty grad: alpha*(h + theta - w)
                    penalty_grad = penalty_alpha * (h + p - w)
                    # gradient correction term: (1/(eta*K))*(gi - g)
                    corr_grad = (gi - g) / (lr * K)
                    total_grad = p.grad + penalty_grad + corr_grad
                    p -= lr * total_grad

            steps += 1

    return K