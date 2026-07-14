# GeoFlow for PyTorch

GeoFlow is a research toolkit for geometry-aware optimization under redundant
neural-network parameterizations.

> Optimizer outputs are proposals, not automatically final updates.

For LoRA-style factors, many parameter pairs represent the same functional
matrix:

```text
M = B A
```

The library currently includes:

- `FixedRankFunctionalAdam`: Adam in explicit product coordinates, followed by
  fixed-rank tangent projection and rank-preserving retraction;
- `SubsteppedQuotientFlow`: a factorized, quotient-compatible integrator with
  fresh gradients at each local substep and no Adam-style moments;
- `CapacityAdaptiveQuotientFlow`: the same quotient-flow vector field with a
  product-space capacity controller that chooses local substeps at runtime;
- dense, low-rank, and matrix-free functional-geometry research tools;
- legacy CIFAR and diagonal `grad_square` baselines retained for comparison.

GeoFlow is experimental. The strongest evidence is structural: improved LoRA
gauge robustness, tangent suppression, rank preservation, and task parity or
small task improvements in controlled settings. It is not a production
large-model optimizer and does not establish universal superiority over Adam.

## Installation

```bash
git clone https://github.com/papasop/Geometric-Flow.git
cd Geometric-Flow
pip install -e .
```

Run the tests:

```bash
python -m pytest -q
python -m compileall -q geometric_flow experiments tests
```

## Core Geometry

For any invertible matrix `S`, the LoRA transformation

```text
A -> S A
B -> B S^{-1}
```

leaves `B A` unchanged. Motions along this gauge orbit alter the parameter
representation without changing the represented update.

A geometry-aware optimizer should therefore distinguish functional motion from
redundant coordinate motion. A Euclidean projection can remove one explicit
tangent component, but it does not generally make Adam gauge-invariant because
Adam and Euclidean normal spaces are coordinate-dependent under non-orthogonal
reparameterizations.

## Optimizers

| method | core idea | status |
| :--- | :--- | :--- |
| `adam` / `adam_raw` | First-order coordinate baseline | Control |
| `diagonal_grad_square` | Legacy diagonal preconditioner | Historical diagnostic |
| `functional_geoflow` | `J_Phi`-based stable-neutral response directions | Research reference |
| `FixedRankFunctionalAdam` | Product-coordinate Adam, tangent projection, rank-`r` retraction | Experimental backend |
| `SubsteppedQuotientFlow` | Gauge-equivariant Gram-preconditioned factor flow with fresh-gradient substeps | Experimental integrator |
| `CapacityAdaptiveQuotientFlow` | Quotient flow with product-space capacity-controlled local step size | Experimental integrator |

### FixedRankFunctionalAdam

`FixedRankFunctionalAdam` treats each low-rank product matrix `M` as the
optimizer state:

```text
product gradient
    -> Adam proposal in M-space
    -> fixed-rank tangent projection
    -> optional held-out trust calibration
    -> max-norm bounding
    -> rank-r SVD retraction
```

The model forward pass must explicitly consume the same product tensor whose
gradient is passed to the optimizer.

Minimal smoke:

```bash
python experiments/d7_fixed_rank_tangent_benchmark.py \
  --seeds 101 \
  --representations 2 \
  --steps 5 \
  --out-dir artifacts/d7_smoke
```

Reference D7 result on a small synthetic Transformer:

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

For `M = B A`, with

```text
A: rank x input_dim
B: output_dim x rank
```

a macro step is split into `K` local steps:

```text
local_lr = macro_lr / K
```

Each local quotient-preconditioned direction is:

```math
\Delta A =
-\eta_{\mathrm{local}}
(B^\top B)^{-1}\nabla_A L,
\qquad
\Delta B =
-\eta_{\mathrm{local}}
\nabla_B L(AA^\top)^{-1}.
```

Under

```math
A \mapsto S A,
\qquad
B \mapsto B S^{-1},
```

the ordinary-inverse, full-rank directions transform covariantly:

```math
\Delta A \mapsto S\Delta A,
\qquad
\Delta B \mapsto \Delta B S^{-1}.
```

Thus the represented product trajectory is gauge-equivariant in exact
arithmetic on the ordinary-inverse branch. If an ill-conditioned Gram matrix
triggers the Moore-Penrose pseudoinverse fallback, exact covariance is not
generally guaranteed for arbitrary non-orthogonal gauge transforms.

Usage:

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

The values above reproduce the H10.6 fast-screening configuration; they are not
universal defaults.

The integrator:

- recomputes gradients at every local substep;
- stores no Adam-style first or second moments;
- uses temporary rank-by-rank Gram matrices;
- reports diagnostics including `condition_max`, `fallback_count`,
  `balance_residual_max`, `last_update_norm`, and `last_clip_scale`;
- optionally applies product-preserving QR canonicalization satisfying
  `B_new A_new = B A`, without claiming `B^T B = A A^T`.

Runtime diagnostic counters are not guaranteed to persist across optimizer
checkpoint restoration.

#### H10 Evidence

On a small GPT-2 LoRA benchmark:

- H10.4/H10.5 reached Adam-scale progress with about `7.29x` best fast gauge
  reduction;
- H10.6 fixed `macro_lr=2.6`, `K=16` and obtained `15.23x` geometric-mean gauge
  suppression across three seeds;
- H10.7 tested the same configuration on five held-out seeds, matched progress
  on all five, improved gauge divergence on all five, and obtained `12.84x`
  geometric-mean suppression;
- `60%` of H10.7 seeds individually exceeded `10x`;
- the H10.7 bootstrap 95% suppression interval was approximately
  `[8.21x, 20.08x]`.

Therefore, the mean `10x` effect was reproduced, while stricter per-seed and
bootstrap-CI confirmation gates were not passed.

The method is best described as a gauge-equivariant, quotient-compatible,
Gram-preconditioned factor-flow integrator. The repository does not prove that
it is the unique quotient-Riemannian gradient, a strict horizontal lift, or the
standard fixed-rank quotient-manifold optimizer.

`experiments/h10_progress_budget_benchmark.py` is a tiny GPT-style regression
benchmark for mechanism and gate checks. It does not instantiate Hugging Face
GPT-2 and does not exactly reproduce the GPT-2 H10.6/H10.7 runs above.

### CapacityAdaptiveQuotientFlow

`CapacityAdaptiveQuotientFlow` replaces the manually selected substep count with
a local product-space capacity controller. It uses the same LoRA convention
`M = B A`, with `A: rank x input_dim` and `B: output_dim x rank`.

The quotient direction is:

```math
V_A = -(B^\top B)^{-1}\nabla_A L,
\qquad
V_B = -\nabla_B L(AA^\top)^{-1}.
```

The product-space capacity is:

```math
H_{\mathrm{opt}}
=
\sqrt{
\sum_\ell
\|V_{B,\ell}A_\ell+B_\ell V_{A,\ell}\|_F^2
}.
```

Each local flow step uses:

```math
d\tau
=
\min\left(
T_{\mathrm{remaining}},
\frac{\varepsilon_\Phi}{H_{\mathrm{opt}}}
\right).
```

Thus geometry determines direction, capacity determines local step size, and
`macro_flow_time` determines total macro progress. Here
`H_opt * d_tau` is the first-order predicted product-space displacement; since
both factors are updated simultaneously, the exact finite product increment
also contains the second-order term `d_tau^2 * V_B @ V_A`. The realized substep
count is generated at runtime rather than provided as a user hyperparameter.

Usage:

```python
from geometric_flow import CapacityAdaptiveQuotientFlow

optimizer = CapacityAdaptiveQuotientFlow(
    factor_modules,
    macro_flow_time=2.6,
    local_function_tolerance=0.05,
    max_flow_dt=None,
    balance_after_substep=True,
)

loss = optimizer.macro_step(closure)
```

The two primary controls are `macro_flow_time` and
`local_function_tolerance`. Other arguments, such as `max_auto_substeps`,
`max_flow_dt`, `balance_after_substep`, and `gram_condition_limit`, are
numerical safeguards or representation-management options.

On the ordinary-inverse branch with full-rank factors, the direction is
gauge-equivariant in exact arithmetic. As with `SubsteppedQuotientFlow`,
Moore-Penrose pseudoinverse fallback is a numerical safeguard and should not be
interpreted as exact covariance under arbitrary non-orthogonal gauges.

#### H10.11 Held-Out Confirmation

In a fixed ten-seed held-out GPT-2 LoRA confirmation, the controller:

- generated between `5` and `13` substeps per macro step;
- matched Adam-scale progress on all ten seeds;
- reduced gauge divergence relative to factor Adam on all ten seeds;
- obtained `11.07x` geometric-mean gauge suppression;
- obtained at least `7.45x` suppression on every seed;
- produced a bootstrap 95% suppression interval of approximately
  `[9.09x, 13.97x]`;
- used no pseudoinverse fallback and no flow-step cap.

This is bounded experimental evidence, not a claim of universal optimizer
superiority or per-seed `10x` suppression.

See [docs/capacity_adaptive_flow.md](docs/capacity_adaptive_flow.md) for shape
conventions, numerical safeguards, and the evidence boundary.

## Functional Geometry Tools

The functional path defines

```text
Phi(theta; X_probe) = vec(model(X_probe))
```

and uses `J_Phi` to separate neutral reparameterization directions from normal
functional directions. Available research paths include:

- dense `J_Phi` construction and SVD projectors `P_T/P_N`;
- low-rank response solves from truncated SVD;
- matrix-free JVP/VJP prototypes with randomized basis estimation, cached
  normal bases, warm-start CG, and explicit budgets.

See [docs/functional_geometry.md](docs/functional_geometry.md).

## Evidence And Limits

| milestone | structural result | task result | status |
| :--- | :--- | :--- | :--- |
| Phase F LoRA | Gauge sensitivity, tangent drift, and near-null amplification reduced | No task superiority | Archived |
| Phase G matched step | Strong corrected structural CI in B2 | Task gap worsened | Archived |
| Transformer layerwise projection | Projection did not harm controlled training | Small mean improvement | Bounded evidence |
| D7 fixed-rank backend | Near-exact gauge invariance and rank preservation | Task parity | Experimental |
| H10 quotient flow | H10.7 reached `12.84x` geometric-mean suppression at matched progress | Mean `10x` reproduced; stricter gates failed | Experimental |
| Capacity-adaptive quotient flow | Ten-seed held-out GPT-2 LoRA run reached `11.07x` geometric-mean suppression | Adam-scale progress matched; per-seed `10x` not established | Experimental |

Established in controlled tests:

- product-coordinate tangent projection and rank-preserving retraction;
- reduced LoRA gauge sensitivity and tangent/near-null motion;
- agreement between matrix-free and dense small-toy response directions;
- machine-precision ordinary-inverse covariance and fresh-gradient substep tests
  for `SubsteppedQuotientFlow`;
- runtime capacity control for quotient flow with zero-capacity and dynamic
  substep regression tests.

Not established:

- broad superiority over Adam or AdamW;
- production large-model scalability;
- GPT-2 or LLM task-performance gains;
- a universal recommendation to replace existing optimizers;
- per-seed and bootstrap-CI confirmation of strict `10x` H10 suppression;
- robustness across models, ranks, datasets, and LoRA target modules.

## Reproduce Key Benchmarks

| benchmark | command |
| :--- | :--- |
| D7 fixed-rank benchmark | `python experiments/d7_fixed_rank_tangent_benchmark.py --seeds 101,211,307 --representations 4 --steps 80 --out-dir artifacts/d7_fixed_rank` |
| H10 tiny-model regression | `python experiments/h10_progress_budget_benchmark.py --macro-lr 2.6 --substeps 16 --out-dir artifacts/h10_progress_budget` |
| Capacity-adaptive smoke | `python experiments/capacity_adaptive_quotient_smoke.py --seeds 101,211,307 --macro-flow-time 2.6 --local-function-tolerance 0.05 --out-dir artifacts/capacity_adaptive_smoke` |
| H10.11/H10.12 research archive | `experiments/archive/` contains non-API GPT-2 LoRA confirmation scripts |
| Phase G matched-step benchmark | `python experiments/lora_matched_step_benchmark.py --trials 5 --steps 200 --representations 5 --train-scope lora_only --functional-map hidden --out artifacts/lora_matched_step.csv` |
| Functional solver toy | `python experiments/functional_projection_toy.py --response-solver implicit_cg` |
| CIFAR legacy benchmark | `python experiments/run_cifar10_benchmark.py --config hybrid_diagonal_500 --download --out artifacts/cifar10_benchmark_results.csv` |

Longer commands and archived results:

- [docs/research_history.md](docs/research_history.md)
- [docs/cifar_benchmarks.md](docs/cifar_benchmarks.md)
- [docs/functional_geometry.md](docs/functional_geometry.md)
- [docs/capacity_adaptive_flow.md](docs/capacity_adaptive_flow.md)

## Testing

```bash
python -m pytest -q
python -m pytest -q tests/test_d7_core_audit.py
python -m compileall -q geometric_flow experiments tests
```

## Further Reading

- [Computation as Geometric Flow](https://zenodo.org/records/21329073)
- Theory direction: *Computation as GeoFlow*, a local stable-neutral
  formulation of implementation manifolds.
