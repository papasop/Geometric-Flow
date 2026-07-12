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
opt = GeometricOptimizer(
    model.parameters(),
    lr=3e-3,
    damping=1e-3,
    max_grad_norm=1.0,
    regularization=1e-3,
    warmup_steps=10,
    curvature_reuse=5,
    lr_scale=3.0,
    grad_smoothing=0.9,
    preconditioner_scale=0.5,
    mode="hybrid",
    adam_warmup_steps=25,
)

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
Early steps run in SGD warm-up mode, gradients are clipped, and damping adapts
inside `[1e-3, 1.0]` to reduce trust in noisy curvature when gradients are large.
Curvature is refreshed every `curvature_reuse` steps and reused between refreshes
to keep HVP overhead under control.
Geometric steps use `lr_scale` to exploit the preconditioned direction, while
`grad_smoothing` applies EMA smoothing to the preconditioned update.
Use `mode="hybrid"` to run Adam for `adam_warmup_steps` steps before switching
to CG or diagonal geometric updates. This is the recommended starting point for
small CNN classification tasks because Adam quickly finds a stable early basin
and the geometric phase then takes over with curvature-aware steps. For short
CIFAR-style smoke tests, late switches such as 48 Adam steps in a 50-step run are
more stable than early switches while the geometric CNN direction is still being
tuned. Use
`mode="adam"` for a pure Adam baseline and `mode="geometric"` for pure
geometry-first training with the SGD warm-up path.
For small CNNs or noisy batches, lower `preconditioner_scale` to keep the
preconditioned/raw gradient ratio in a stable range.
For classification, try `curvature_kind="fisher"` or `preconditioner="diagonal"`
when Hessian-CG directions are noisy.

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

Hyperparameter tuning:

```bash
python experiments/tune_geometric_optimizer.py --steps 200
```

GeoCNN CIFAR-style baseline:

```bash
python experiments/train_cifar10_geo.py --dataset synthetic
python experiments/train_cifar10_geo.py --dataset synthetic --use-fisher --preconditioner diagonal
python experiments/train_cifar10_geo.py --dataset synthetic --mode hybrid --adam-warmup-steps 48 --use-fisher --preconditioner diagonal --precond-scale 0.5 --max-grad-norm 2.0 --grad-smoothing 0.0
python experiments/train_cifar10_geo.py --dataset cifar10 --data-root ./data
```

Pass `verbose=True` to `GeometricOptimizer.step(...)` or construct the optimizer
with `verbose=True` to print per-step mode/reuse diagnostics and write a CSV row
every 10 steps.
