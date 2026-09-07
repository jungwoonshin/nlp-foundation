"""Minimal MLP classifier on tutorial/sample_data.csv (XOR-style 2D points)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = Path(__file__).resolve().parent / "sample_data.csv"


def load_tensors(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    rows = path.read_text(encoding="utf-8").strip().splitlines()[1:]
    features: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        x1, x2, y = row.split(",")
        features.append([float(x1), float(x2)])
        labels.append(int(y))
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


class MLP(nn.Module):
    def __init__(self, in_features: int = 2, hidden: int = 8, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean())


def main() -> None:
    torch.manual_seed(0)
    x, y = load_tensors(DATA_PATH)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)

    model = MLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    model.train()
    for epoch in range(1, 201):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)

        if epoch in {1, 50, 100, 200}:
            model.eval()
            with torch.no_grad():
                acc = accuracy(model(x), y)
            model.train()
            print(f"epoch {epoch:3d}  loss={epoch_loss:.4f}  acc={acc:.2f}")

    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    print("labels:", y.tolist())
    print("preds: ", pred.tolist())


if __name__ == "__main__":
    main()
