"""Small PyTorch MLP baseline for tabular data."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLPRegressorTorch(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        d_prev = n_in
        for h in hidden:
            layers.extend([nn.Linear(d_prev, h), nn.ReLU(), nn.Dropout(dropout)])
            d_prev = h
        layers.append(nn.Linear(d_prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_mlp_torch(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    seed: int = 0,
) -> tuple[MLPRegressorTorch, dict[str, float]]:
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    X_t = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=torch.float32, device=dev)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev)
    mean, std = X_t.mean(0), X_t.std(0).clamp_min(1e-6)
    Xn = (X_t - mean) / std

    ds = TensorDataset(Xn, y_t)
    dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    model = MLPRegressorTorch(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    meta = {"mean": mean.detach().cpu().numpy(), "std": std.detach().cpu().numpy()}
    return model, meta


@torch.no_grad()
def predict_mlp_torch(model: MLPRegressorTorch, meta: dict[str, float], X: np.ndarray, device: str | None = None) -> np.ndarray:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    mean = torch.tensor(meta["mean"], dtype=torch.float32, device=dev)
    std = torch.tensor(meta["std"], dtype=torch.float32, device=dev)
    X_t = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=torch.float32, device=dev)
    Xn = (X_t - mean) / std
    return model(Xn).detach().cpu().numpy()
