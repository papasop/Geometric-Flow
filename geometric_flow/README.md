# Geometric Flow Codex

`geometric_flow` is a geometry-first PyTorch optimization layer. It treats the
loss landscape as a local differentiable manifold:

- `geo.measure()` builds an implicit Hessian/Fisher curvature operator with HVP.
- `geo.navigate()` solves `A * step = -grad` with conjugate gradients.
- `geo.plot_boundary()` probes curvature regimes and emits phase-map JSON.
- `geo.plot_boundary_2d()` runs a learning-rate/damping style grid scan and
  writes `param1,param2,final_loss,avg_trace` CSV data.
- `geo.embed()` inserts learnable geometric phase rotations after linear blocks
  to reshape information flow without changing feature dimensionality.

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

Each optimizer step records a topography row with `trace_estimate`,
`rayleigh_grad`, `update_norm`, and cumulative `geodesic_distance`, which acts as
the model's geometric mileage through parameter space.
