# Geometric Flow Codex

`geometric_flow` is a geometry-first PyTorch optimization layer. It treats the
loss landscape as a local differentiable manifold:

- `geo.measure()` builds an implicit Hessian/Fisher curvature operator with HVP.
- `geo.navigate()` solves `A * step = -grad` with conjugate gradients.
- `geo.plot_boundary()` probes curvature regimes and emits phase-map JSON.
- `geo.embed()` inserts parameter-free geometric rotation layers after linear
  blocks to reshape information flow without increasing parameter count.

Minimal optimizer usage:

```python
import torch.nn.functional as F
from geometric_flow import GeoMLP, GeometricOptimizer

model = GeoMLP(output_dim=10)
opt = GeometricOptimizer(model.parameters(), lr=0.3, damping=1e-2)

def closure():
    return F.cross_entropy(model(x), y)

loss = opt.step(closure)
print(opt.topography_log[-1])
```

The optimizer falls back to SGD when local curvature is non-positive,
ill-conditioned, or too expensive to trust.
