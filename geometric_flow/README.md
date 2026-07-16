# GeoFlow Codex

This package-level README documents legacy import paths and early optimizer
APIs. For the current project positioning, functional-time framework, quotient
capacity flow, and evidence boundaries, see the repository-level
[`README.md`](../README.md).

`geometric_flow` is the import-compatible Python package for GeoFlow, a
geometry-first PyTorch optimization layer. It treats the
loss landscape as a local differentiable manifold:

- Legacy `geo.measure()` builds an implicit Hessian/grad-square curvature operator with HVP.
- `geo.navigate()` solves `A * step = -grad` with conjugate gradients.
- `functional_geometry.FunctionalMap` builds `Phi(theta; X_probe)` and its
  dense Jacobian for small theory tests.
- `mode="functional_geoflow"` uses SVD projectors `P_T/P_N` and a functional
  response operator instead of the diagonal grad-square heuristic.
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
and the geometric phase then takes over with curvature-aware steps. A practical
starting configuration is `mode="hybrid", adam_warmup_steps=30` with grad-square
diagonal preconditioning. This is a legacy heuristic baseline, exposed as
`preconditioner="diagonal_grad_square"` for clarity. In a 50-step synthetic CIFAR smoke test, a late-switch
hybrid configuration reached 52.3% accuracy versus Adam's 51.6%, with a
preconditioned/raw ratio of 0.458. Use
`mode="adam"` for a pure Adam baseline and `mode="geometric"` for pure
geometry-first training with the SGD warm-up path.
For small CNNs or noisy batches, lower `preconditioner_scale` to keep the
preconditioned/raw gradient ratio in a stable range.
For classification, try `curvature_kind="grad_square"` or
`preconditioner="diagonal"` when Hessian-CG directions are noisy. The legacy
`curvature_kind="fisher"` spelling is accepted as a compatibility alias, but the
current implementation is a batch gradient-square diagonal rather than a true
empirical Fisher.

For the theory-aligned path, use `mode="functional_geoflow"` with
`functional_model=model` and a fixed `functional_probe` tensor. The first dense
implementation constructs `Phi(theta)=vec(model(X_probe))`, `J_phi`, `P_T` onto
`ker(J_phi)`, `P_N=I-P_T`, and the Gauss-Newton response `J_phi^T J_phi`.
It is intended for small MLP/toy validation and is not yet an efficiency
replacement for Adam.
Null-space selection supports `absolute`, `relative`, `spectral_gap`, and
`energy_fraction` modes, with diagnostics for selected threshold, spectral-gap
index, normal condition number, and retained energy fraction.
Functional response solving supports `dense`, `low_rank`, and `implicit_cg`.
The low-rank path uses truncated SVD of dense `J_phi` as a first prototype and
avoids materializing full `A_resp`; the implicit path uses VJP probes to build a
matrix-free normal basis and JVP/VJP matvecs for the constrained solve. It does
not use dense `J_phi` or dense `P_N` in the solve, but remains a small-model
prototype.

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
python experiments/tune_geometric_optimizer.py --modes geometric,adam,hybrid --adam-warmup-steps-list 10,30,50,80
```

GeoCNN CIFAR-style baseline:

```bash
python experiments/train_cifar10_geo.py --dataset synthetic
python experiments/train_cifar10_geo.py --dataset synthetic --use-grad-square --preconditioner diagonal
python experiments/train_cifar10_geo.py --dataset synthetic --mode all --trials 3 --use-grad-square --preconditioner diagonal
python experiments/train_cifar10_geo.py --dataset synthetic --mode all --trials 1 --steps 50 --adam-warmup-steps 48 --seed 32 --use-grad-square --preconditioner diagonal --precond-scale 0.5 --max-grad-norm 2.0 --grad-smoothing 0.0
python experiments/train_cifar10_geo.py --dataset synthetic --mode hybrid --adam-warmup-steps 30 --use-grad-square --preconditioner diagonal --precond-scale 0.5 --max-grad-norm 2.0 --grad-smoothing 0.0
python experiments/train_cifar10_geo.py --dataset synthetic --mode hybrid --auto-warmup --auto-warmup-steps 30,50,80 --conv-layers 6 --use-grad-square --preconditioner diagonal
python experiments/train_cifar10_geo.py --dataset synthetic --mode switch_compare --adam-warmup-steps 50 --use-grad-square --preconditioner diagonal
python experiments/train_cifar10_geo.py --dataset cifar10 --data-root ./data
```

Full CIFAR-10 benchmark:

```bash
python experiments/run_cifar10_benchmark.py --download --steps 500 --trials 3 --hybrid-warmup-steps 10,30,50,80
python experiments/run_cifar10_benchmark.py --config hybrid_diagonal_500 --download
python experiments/run_cifar10_benchmark.py --download --precond-scales 0.35,0.5,0.75 --grad-smoothing-values 0.0,0.5
python experiments/plot_comparison.py artifacts/cifar10_benchmark.csv --out artifacts/adam_vs_hybrid.svg
python experiments/plot_comparison.py artifacts/cifar10_geo_diagnostics.csv --ratio-out artifacts/ratio_over_time.svg
python experiments/normal_projection_toy.py --out artifacts/normal_projection_toy.csv
python experiments/functional_projection_toy.py --response-solver dense
python experiments/functional_projection_toy.py --response-solver low_rank
python experiments/functional_projection_toy.py --response-solver implicit_cg
python experiments/run_functional_switch_validation.py --trials 5 --steps 200
python experiments/reparameterization_stress_test.py
python experiments/noisy_redundancy_validation.py
python experiments/near_null_stress_test.py
```

Pass `verbose=True` to `GeometricOptimizer.step(...)` or construct the optimizer
with `verbose=True` to print per-step mode/reuse diagnostics and write a CSV row
every 10 steps.
