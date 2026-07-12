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

def closure(backward=False):
    loss = F.cross_entropy(model(x), y)
    if backward:
        loss.backward()
    return loss

loss = opt.step(closure)
print(opt.topography_log[-1])
```

The optimizer falls back to SGD when local curvature is non-positive,
ill-conditioned, or too expensive to trust.
`GeometricOptimizer` calls `closure(backward=False)` when supported and owns the
single `loss.backward()` call internally, so HVP curvature probes and ordinary
gradients share one retained computation graph safely.

Each optimizer step records a topography row with `trace_estimate`,
`rayleigh_grad`, `update_norm`, and cumulative `geodesic_distance`, which acts as
the model's geometric mileage through parameter space.

Scale-curve experiment:

```bash
python experiments/scale_curve.py --widths 64,128,256,512
```

The script trains Adam and `GeometricOptimizer` to a shared target accuracy,
then writes CSV/JSON metrics plus SVG plots under `artifacts/scale_curve/`,
including `geometric_speedup.svg`.
