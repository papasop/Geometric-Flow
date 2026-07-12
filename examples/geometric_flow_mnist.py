"""Minimal MNIST-style training loop for GeometricOptimizer.

This file uses synthetic MNIST-shaped tensors by default so the smoke path does
not require network downloads. Replace the dataset with torchvision MNIST for a
real first-week proof run.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from geometric_flow import GeoMLP, GeometricOptimizer, phase_diagram_scanner, write_phase_diagram


def make_synthetic_mnist(n: int = 256):
    x = torch.randn(n, 1, 28, 28)
    y = (x[:, :, 10:18, 10:18].mean(dim=(1, 2, 3)) > 0).long()
    return TensorDataset(x, y)


def main() -> None:
    torch.manual_seed(7)
    model = GeoMLP(hidden_dim=64, output_dim=2)
    loader = DataLoader(make_synthetic_mnist(), batch_size=64, shuffle=True)
    optimizer = GeometricOptimizer(
        model.parameters(),
        lr=0.3,
        damping=1e-2,
        curvature_interval=1,
        cg_max_iter=8,
        trace_samples=2,
        max_update_norm=1.0,
        path_smoothing=0.1,
    )

    batch = None
    for epoch in range(3):
        for batch in loader:
            x, y = batch

            def closure():
                return F.cross_entropy(model(x), y)

            loss = optimizer.step(closure)
        print(f"epoch={epoch} loss={float(loss):.4f} mode={optimizer.topography_log[-1]['mode']}")

    x, y = batch
    points = phase_diagram_scanner(
        model,
        lambda: F.cross_entropy(model(x), y),
        param_range=[-2, -1, 0, 1, 2],
        probes=3,
    )
    write_phase_diagram(points, "geometric_phase_diagram.json")
    print("wrote geometric_phase_diagram.json")


if __name__ == "__main__":
    main()
