# GeoFlow for PyTorch

GeoFlow is a research toolkit for geometry-aware optimization under redundant
neural-network parameterizations. Its central rule is simple:

> Optimizer outputs are proposals, not automatically final updates.

For LoRA-style factors, many parameter pairs represent the same functional
matrix:

```text
M = B A
```

GeoFlow studies optimizers that filter, project, or lift update proposals
through this geometry before applying them. The current library contains:

- `FixedRankFunctionalAdam`: an opt-in product-coordinate backend that keeps
  Adam state on explicit invariant product matrices `M`, projects proposals
  into the fixed-rank tangent space, and retracts back to rank `r`;
- `SubsteppedQuotientFlow`: an experimental factorized quotient-flow integrator
  with fresh gradients at every local substep and no Adam-style moments;
- functional stable-neutral geometry tools for dense, low-rank, and
  matrix-free response-direction experiments;
- legacy CIFAR and diagonal `grad_square` baselines retained for comparison.

This is not a production large-model optimizer and not evidence of universal
superiority over Adam. The strongest current results are structural: improved
LoRA gauge robustness, tangent suppression, and controlled benchmark parity or
small task improvements in specific settings.

## Installation

```bash
git clone https://github.com/papasop/Geometric-Flow.git
cd Geometric-Flow
pip install -e .
```

Run the test suite:

```bash
python -m pytest -q
python -m compileall -q geometric_flow experiments tests
```

## What GeoFlow Does

Conventional optimizers update coordinates. GeoFlow asks whether those
coordinate updates are meaningful in the represented function.

For LoRA,

```text
A -> S A
B -> B S^{-1}
```

leaves `B A` unchanged. Motions along this gauge orbit change the
representation without changing the represented model update. A
geometry-aware optimizer should therefore:

1. generate a candidate direction;
2. identify directions associated with equivalent representations;
3. remove, quotient, or avoid those ineffective components;
4. apply an update in the transverse functional space.

A Euclidean projection of an Adam update can remove an explicit tangent
component, but it does not generally make Adam gauge-invariant. Adam is
coordinate-dependent, and Euclidean normal spaces also change under
non-orthogonal LoRA reparameterizations. Exact gauge-aware formulations operate
on invariant product variables `M = B A`, or use a quotient-compatible metric.

## Optimizers

| method | core idea | status |
| :--- | :--- | :--- |
| `adam` / `adam_raw` | First-order coordinate baseline | Control baseline |
| `diagonal_grad_square` | Legacy diagonal heuristic | Diagnostic baseline, not full stable-neutral GeoFlow |
| `functional_geoflow` | `J_Phi`-based stable-neutral response directions | Research reference |
| `FixedRankFunctionalAdam` | Adam proposals in explicit product coordinates, tangent projection, rank-preserving retraction | Experimental backend |
| `SubsteppedQuotientFlow` | Gauge-equivariant Gram-preconditioned factor flow with fresh-gradient substeps | Experimental integrator |

### FixedRankFunctionalAdam

`FixedRankFunctionalAdam` treats each low-rank product matrix `M` as the
optimizer state, rather than updating hidden LoRA factors directly. Its update
pipeline is:

```text
product gradient -> Adam proposal in M-space
                 -> fixed-rank tangent projection
                 -> optional held-out trust calibration
                 -> max-norm bounding
                 -> rank-r SVD retraction
```

This backend is intentionally separate from `GeometricOptimizer`; it is not a
pile of extra flags on the legacy optimizer. The model forward pass must
explicitly consume the same product tensor whose gradient is passed to the
optimizer. See `experiments/d7_fixed_rank_tangent_benchmark.py` for a runnable
example.

Minimal smoke:

```bash
python experiments/d7_fixed_rank_tangent_benchmark.py \
  --seeds 101 \
  --representations 2 \
  --steps 5 \
  --out-dir artifacts/d7_smoke
```

Reference D7 result on a small synthetic Transformer benchmark:

| metric | result |
| --- | ---: |
| `rank_tangent_trust` mean loss | `1.710495` |
| factor Adam mean loss | `1.711612` |
| paired loss gap | `-0.001117` |
| 95% CI | `[-0.003562, 0.000925]` |
| logit sensitivity ratio | `7.3e-5` |
| structural win rate | `1.0` |
| tangent residual | `~3e-6` |
| rank violations | `0` |

This establishes task parity and structural invariance in that benchmark, not
general task superiority.

### SubsteppedQuotientFlow

`SubsteppedQuotientFlow` integrates a factorized quotient-compatible vector
field for LoRA-style modules exposing trainable `A` and `B` parameters with
shapes `(rank, input_dim)` and `(output_dim, rank)`.

For a macro step with `K` substeps:

```text
local_lr = macro_lr / K
```

For `M = B A`, each local quotient-preconditioned step is:

```math
\Delta A =
-\eta_{\mathrm{local}}
(B^\top B)^{-1}\nabla_A L,
\qquad
\Delta B =
-\eta_{\mathrm{local}}
\nabla_B L(AA^\top)^{-1}.
```

Under the gauge transformation

```math
A \mapsto S A,
\qquad
B \mapsto B S^{-1},
```

the directions transform covariantly:

```math
\Delta A \mapsto S\Delta A,
\qquad
\Delta B \mapsto \Delta B S^{-1}.
```

For full-rank factors on the ordinary-inverse branch, the represented product
trajectory is gauge-equivariant in exact arithmetic. When an ill-conditioned
Gram matrix triggers the Moore-Penrose pseudoinverse fallback, exact covariance
is not generally guaranteed under arbitrary non-orthogonal gauge transforms.

Each macro step should recompute gradients at every local substep:

```python
from geometric_flow import SubsteppedQuotientFlow

optimizer = SubsteppedQuotientFlow(
    factor_modules,
    macro_lr=2.6,
    substeps=16,
    clip_norm=None,
    balance_after_substep=True,
)

def closure():
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    return loss

loss = optimizer.macro_step(closure)
```

These values reproduce the H10.6 fast-screening configuration and are not
universal defaults.

The integrator has no Adam-style first or second moments. It stores scalar
runtime diagnostics such as `condition_max`, `fallback_count`,
`balance_residual_max`, `last_update_norm`, and `last_clip_scale`; these
counters are not guaranteed to persist across optimizer checkpoint restoration.
Temporary rank-by-rank Gram matrices are used to compute directions.

The optional canonicalization step is product-preserving QR normalization: it
keeps `B_new A_new = B A`, but it does not explicitly guarantee
`B^T B = A A^T`.

H10 evidence on a small GPT-2 LoRA benchmark:

- H10.4/H10.5 reached Adam-scale progress with approximately `7.29x` best fast
  gauge reduction.
- Increasing macro LR above `3` did not improve the progress-versus-gauge
  trade-off.
- H10.6 introduced progress-budgeted stopping and expanded substeps to `K=8`
  and `K=16`.
- With fixed `macro_lr=2.6`, `K=16` across three seeds, H10.6 obtained
  mean loss-progress ratio `1.846`, mean product-displacement ratio `0.773`,
  and geometric-mean gauge suppression `15.23x`.
- The strict `10x` fast-benchmark gate passed on all aggregate criteria, with
  no pseudoinverse fallback and product-preserving balance passing.
- This remains a small fast benchmark and requires held-out-seed confirmation.

Detailed H10.6 metrics:

- geometric-mean gauge-divergence ratio: `0.0656`;
- mean loss-progress ratio: `1.846`;
- mean product-displacement ratio: `0.773`;
- matched-progress pass on all three seeds;
- `H106_MEAN_GAUGE_SUPPRESSION_10X_PASS=True`;
- `H106_NO_FALLBACK_PASS=True`;
- `H106_BALANCE_PASS=True`.

These settings are diagnostics, not universal defaults. The method is best
described as a gauge-equivariant, quotient-compatible Gram-preconditioned
factor-flow integrator; the repository does not prove it is the unique
quotient-Riemannian gradient, a strict horizontal lift, or the standard
fixed-rank quotient-manifold optimizer.

## Functional Geometry Tools

The theory-aligned functional path builds a configurable functional map

```text
Phi(theta; X_probe) = vec(model(X_probe))
```

and uses its Jacobian `J_Phi` to separate neutral reparameterization directions
from normal functional directions. The library includes:

- dense small-model `J_Phi` construction and SVD projectors `P_T/P_N`;
- low-rank response solves from truncated SVD;
- matrix-free JVP/VJP prototypes with randomized VJP basis estimation,
  cached normal bases, warm-start CG, and explicit per-step budgets.

These tools are correctness and scaling prototypes. They should be read as
research diagnostics until larger LoRA and language-model benchmarks justify
broader claims. See [docs/functional_geometry.md](docs/functional_geometry.md).

## Quick Examples

Run the fixed-rank D7 smoke:

```bash
python experiments/d7_fixed_rank_tangent_benchmark.py \
  --seeds 101 \
  --representations 2 \
  --steps 5 \
  --out-dir artifacts/d7_smoke
```

Run the substepped quotient-flow tests:

```bash
python -m pytest -q tests/test_fixed_rank_optimizer.py
```

Run the legacy CIFAR smoke, retained as a historical baseline:

```bash
python experiments/train_cifar10_geo.py \
  --dataset synthetic \
  --mode all \
  --trials 1 \
  --steps 50 \
  --adam-warmup-steps 48 \
  --seed 32 \
  --train-samples 256 \
  --eval-samples 128 \
  --batch-size 32 \
  --channels 16 \
  --use-grad-square \
  --preconditioner diagonal \
  --precond-scale 0.5 \
  --max-grad-norm 2.0 \
  --grad-smoothing 0.0 \
  --out artifacts/synthetic_cifar_milestone.csv
```

Open the historical CIFAR notebook:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/papasop/Geometric-Flow/blob/main/notebooks/run_cifar10_benchmark.ipynb)

## Evidence And Limits

| milestone | structural result | task result | status |
| :--- | :--- | :--- | :--- |
| Legacy CIFAR hybrid | Useful diagnostic ratio, late-switch hybrid smoke | Small synthetic edge only | Historical |
| Phase F LoRA | Gauge sensitivity, tangent drift, and near-null amplification reduced | No task superiority | Archived |
| Phase G matched step | Strong corrected structural CI in B2 | Task gap worsened in controlled setting | Archived |
| Transformer layerwise projection | Small controlled LoRA projection did not harm training | Slight mean loss/accuracy improvement | Bounded evidence |
| D7 fixed-rank backend | Near-exact gauge invariance and rank preservation | Task parity in small synthetic Transformer | Experimental |
| H10 quotient flow | H10.6 progress-budgeted fast run reached `15.23x` geometric-mean gauge suppression | Small GPT-2 LoRA benchmark; held-out seeds still needed | Experimental |

Established:

- product-coordinate tangent projection and rank-preserving retraction work in
  controlled tests;
- functional quotient tools reduce LoRA gauge sensitivity and suppress tangent
  or near-null motion in controlled settings;
- matrix-free functional quotient directions match dense small-toy references
  under regression tests;
- `SubsteppedQuotientFlow` has a machine-precision ordinary-inverse gauge
  covariance test and fresh-gradient substep coverage.

Not established:

- broad task superiority over Adam or AdamW;
- production large-model scalability;
- GPT-2 or LLM performance claims;
- a universal recommendation to replace existing optimizers;
- held-out-seed confirmation of the H10.6 strict `10x` result;
- broad robustness across models, ranks, datasets, and LoRA target modules.

## Reproduce Key Benchmarks

| benchmark | command |
| :--- | :--- |
| D7 fixed-rank tangent benchmark | `python experiments/d7_fixed_rank_tangent_benchmark.py --seeds 101,211,307 --representations 4 --steps 80 --out-dir artifacts/d7_fixed_rank` |
| H10 progress-budget benchmark | `python experiments/h10_progress_budget_benchmark.py --macro-lr 2.6 --substeps 16 --out-dir artifacts/h10_progress_budget` |
| Phase G matched-step benchmark | `python experiments/lora_matched_step_benchmark.py --trials 5 --steps 200 --representations 5 --train-scope lora_only --functional-map hidden --out artifacts/lora_matched_step.csv` |
| CIFAR legacy benchmark | `python experiments/run_cifar10_benchmark.py --config hybrid_diagonal_500 --download --out artifacts/cifar10_benchmark_results.csv` |
| Functional solver toy | `python experiments/functional_projection_toy.py --response-solver implicit_cg` |
| Structural pressure tests | `python experiments/reparameterization_stress_test.py && python experiments/noisy_redundancy_validation.py && python experiments/near_null_stress_test.py` |

Longer command sets and CSV schemas live in:

- [docs/cifar_benchmarks.md](docs/cifar_benchmarks.md)
- [docs/functional_geometry.md](docs/functional_geometry.md)
- [docs/research_history.md](docs/research_history.md)

## Testing

```bash
python -m pytest -q
python -m pytest -q tests/test_d7_core_audit.py
python -m compileall -q geometric_flow experiments tests
```

The D7 core audit checks production edge cases such as final update norm
bounding, partial product gradients, empty-gradient no-ops, and trust-region
candidate consistency.

## Documentation

- [docs/research_history.md](docs/research_history.md): archived Phase F/G,
  D7, H10, and Transformer experiment notes.
- [docs/cifar_benchmarks.md](docs/cifar_benchmarks.md): legacy CIFAR smoke and
  full benchmark commands.
- [docs/functional_geometry.md](docs/functional_geometry.md): functional map,
  projector, low-rank, and matrix-free solver details.
- `experiments/d7_fixed_rank_tangent_benchmark.py`: D7 reproduction and
  scientific regression benchmark.
- `experiments/lora_matched_step_benchmark.py`: Phase G matched functional-step
  calibration.
- `experiments/lora_reparameterization_benchmark.py`: controlled LoRA gauge
  sensitivity benchmark.

## Further Reading

- https://zenodo.org/records/21329073 Computation as Geometric Flow
- Theory direction: *Computation as GeoFlow*, a local stable-neutral
  formulation of implementation manifolds.
