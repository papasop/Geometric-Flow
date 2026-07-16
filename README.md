# GeoFlow for PyTorch

Geometric-Flow studies optimization when model parameters are redundant:
different parameter coordinates may represent the same functional model.
Instead of defining direction and step size only in parameter space, the
framework introduces a functional map, a quotient-aware direction, a functional
capacity, and a functional-time controller.

> Optimizer outputs are proposals, not automatically final updates.

The current implementation specializes this functional-time framework to
low-rank products and LoRA adapters. In this setting, the functional state is
the product

```text
M = B A
```

and the local functional velocity is

```text
dM = V_B A + B V_A.
```

LoRA is therefore the first complete realization, not the boundary of the
theory.

The library currently contains:

- fixed-rank product-state optimization;
- quotient-compatible low-rank factor flow;
- fixed and adaptive functional-time capacity controllers;
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

## Functional-Time Framework

The general object is a functional map:

```math
\Phi = \pi(\theta).
```

If two parameter states satisfy `pi(theta) = pi(theta')`, they belong to the
same functional equivalence class. The guiding question is:

> Should optimization depend on arbitrary parameter coordinates, or on
> functional state?

GeoFlow organizes optimization around four components:

| component | role |
| :--- | :--- |
| functional state `Phi = pi(theta)` | the represented model state |
| quotient-aware direction `V_Q` | a proposal direction compatible with the equivalence class |
| functional capacity `H_Phi = ||D pi_theta[V_Q]||` | the speed of actual functional motion |
| functional time `d_tau = epsilon_Phi / H_Phi` | a local time step bounded by functional displacement |

An optional controller adapts `epsilon_Phi` from functional progress,
prediction error, and limiter state. K1 is one such controller; it is not the
framework itself and should not be read as a conventional learning-rate
scheduler.

## Variational Motivation

Geometric-Flow is motivated by a functional steepest-descent principle:
optimization should maximize task improvement per unit motion of the realized
model function, rather than per unit motion of an arbitrary parameter
representation.

Let

```math
\pi:\Theta\rightarrow\mathcal F
```

map parameters `theta` to a functional state `Phi = pi(theta)`. A
parameter-space direction `V` induces the functional velocity

```math
\dot\Phi = D\pi_\theta[V].
```

A natural local variational problem is

```math
V^\star
=
\arg\max_{V\in\mathcal H_\theta}
\left\{
-\mathrm dL_\theta[V]
:
\|D\pi_\theta[V]\|_\Phi\le1
\right\},
```

where `H_theta` excludes directions that move only along a
representation-equivalence orbit.

For a chosen quotient-aware direction `V_Q`, GeoFlow defines

```math
H_\Phi(V_Q)=\|D\pi_\theta[V_Q]\|_\Phi,
\qquad
d\tau=\frac{\epsilon_\Phi}{H_\Phi(V_Q)}.
```

Thus, `epsilon_Phi` specifies an allowed functional-displacement budget, while
`H_Phi` measures the rate at which the current direction consumes that budget.

In the current low-rank realization,

```math
\pi(A,B)=BA,
\qquad
D\pi_{(A,B)}[V_A,V_B]=V_BA+BV_A.
```

The implemented full-rank quotient direction is gauge covariant, but it has not
yet been proven to be the exact optimizer of the full product-space
variational problem above. Determining the precise functional metric and
horizontal constraint under which the implemented direction is a true
steepest-descent direction remains an open theoretical objective.

## Low-Rank Quotient Instance

The current implementation targets low-rank products and LoRA adapters. For any
invertible matrix `S`, the transformation

```text
A -> S A
B -> B S^{-1}
```

leaves the functional state `B A` unchanged. Motions along this gauge orbit
alter the parameter representation without changing the represented product.

A geometry-aware optimizer should therefore distinguish functional motion from
redundant coordinate motion. A Euclidean projection can remove one explicit
tangent component, but it does not generally make Adam gauge-invariant because
Adam and Euclidean normal spaces are coordinate-dependent under non-orthogonal
reparameterizations.

In this low-rank instance, the quotient direction, functional capacity, and
functional time become concrete:

```math
V_A = -(B^\top B)^{-1}\nabla_A L,
\qquad
V_B = -\nabla_B L(AA^\top)^{-1},
```

```math
H_{\Phi}
=
\left(
\sum_\ell
\|V_{B,\ell}A_\ell+B_\ell V_{A,\ell}\|_F^2
\right)^{1/2},
```

```math
d\tau
=
\min\left(
T_{\mathrm{remaining}},
\frac{\epsilon_\Phi}{H_\Phi}
\right).
```

`H_Phi` is not a coordinate gradient norm. It measures how fast the represented
product moves in functional space. Geometric-Flow therefore does not merely
choose a coordinate learning rate; it reparameterizes optimization time by
functional motion.

## Optimizers

| method | core idea | status |
| :--- | :--- | :--- |
| `adam` / `adam_raw` | First-order coordinate baseline | Control |
| `diagonal_grad_square` | Legacy diagonal preconditioner | Historical diagnostic |
| `functional_geoflow` | `J_Phi`-based stable-neutral response directions | Research reference |
| `FixedRankFunctionalAdam` | Product-coordinate Adam, tangent projection, rank-`r` retraction | Experimental backend |
| `SubsteppedQuotientFlow` | Ordinary-inverse gauge-equivariant Gram-preconditioned factor flow with fresh-gradient substeps | Experimental integrator |
| `CapacityAdaptiveQuotientFlow` | Quotient flow with adaptive functional-time capacity control; direction covariance still belongs to the full-rank ordinary-inverse quotient field | Experimental integrator |

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

and for losses depending on the represented product, `L = L(BA)`, the
factor gradients transform as:

```math
\nabla_{A'} L = S^{-\top}\nabla_A L,
\qquad
\nabla_{B'} L = \nabla_B L\,S^\top.
```

The ordinary-inverse, full-rank directions transform covariantly:

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

To avoid manually tuning the substep-count hyperparameter,
`CapacityAdaptiveQuotientFlow` adds a product-space capacity controller that
chooses the number of local steps at runtime from the current quotient-flow
geometry. It uses the same LoRA convention `M = B A`, with
`A: rank x input_dim` and `B: output_dim x rank`.

The quotient direction is:

```math
V_A = -(B^\top B)^{-1}\nabla_A L,
\qquad
V_B = -\nabla_B L(AA^\top)^{-1}.
```

Interpretation: the optimizer first builds a gauge-aware local vector field for
the two LoRA factors, rather than stepping directly in raw factor coordinates.

The product-space capacity is:

```math
H_{\mathrm{opt}}
=
\sqrt{
\sum_\ell
\|V_{B,\ell}A_\ell+B_\ell V_{A,\ell}\|_F^2
}.
```

Interpretation: `H_opt` measures the first-order size of the represented
product motion produced by simultaneously changing `A` and `B` across all
target modules.

Each local flow step uses:

```math
d\tau
=
\min\left(
T_{\mathrm{remaining}},
\frac{\varepsilon_\Phi}{H_{\mathrm{opt}}}
\right).
```

Interpretation: `d_tau` is the local flow time chosen so that the first-order
product-space motion stays within the configured tolerance.

Thus geometry determines direction, capacity determines local step size, and
`macro_flow_time` determines total macro progress. Here
`H_opt * d_tau` is the first-order predicted product-space displacement; since
both factors are updated simultaneously, the exact finite product increment
also contains the second-order term `d_tau^2 * V_B @ V_A`. The realized substep
count is generated at runtime rather than provided as a user hyperparameter.

The two primary controls are:

| parameter | meaning |
| :--- | :--- |
| `macro_flow_time` | total quotient-flow time per macro step |
| `local_function_tolerance` | first-order product-motion budget per local step |

Other arguments, such as `max_auto_substeps`, `max_flow_dt`,
`balance_after_substep`, and `gram_condition_limit`, are numerical safeguards or
representation-management options.

Controller interpretation:

- fixed Capacity uses a fixed `epsilon_Phi` and is the strongest current
  structural mode for long-horizon gauge robustness;
- K1 adapts the active `epsilon_Phi` from functional progress and prediction
  feedback, improving early target-quality efficiency in tested runs;
- K1 exposes an efficiency-equivariance tradeoff: it should not be described as
  preserving exact long-horizon gauge equivariance.

#### Usage

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

On the ordinary-inverse branch with full-rank factors, the direction is
gauge-equivariant in exact arithmetic. As with `SubsteppedQuotientFlow`,
Moore-Penrose pseudoinverse fallback is a numerical safeguard and should not be
interpreted as exact covariance under arbitrary non-orthogonal gauges.

#### H10.11 Held-Out Confirmation

Core finding: in a fixed ten-seed held-out GPT-2 LoRA confirmation, the
controller matched Adam-scale progress on every seed while reaching `11.07x`
geometric-mean gauge suppression.

<details>
<summary>Detailed H10.11 statistics</summary>

- generated between `5` and `13` substeps per macro step;
- matched Adam-scale progress on all ten seeds;
- reduced gauge divergence relative to factor Adam on all ten seeds;
- obtained `11.07x` geometric-mean gauge suppression;
- obtained at least `7.45x` suppression on every seed;
- produced a bootstrap 95% suppression interval of approximately
  `[9.09x, 13.97x]`;
- used no pseudoinverse fallback and no flow-step cap.

</details>

This is bounded experimental evidence, not a claim of universal optimizer
superiority or per-seed `10x` suppression.

See [docs/capacity_adaptive_flow.md](docs/capacity_adaptive_flow.md) for shape
conventions, numerical safeguards, and the evidence boundary.

### H13.4 Full-Product Gauge-Dynamics Audit

H13.4 tests whether gauge-equivalent LoRA factorizations that represent the
same initial product `M = B A` produce the same learning dynamics in the
complete adapted product space.

For the transformation

```math
(A,B)\mapsto(SA,BS^{-1}),
\qquad
(BS^{-1})(SA)=BA,
```

the audit measures update-direction error, log-magnitude error, and trajectory
gap using the full `B @ A` product of every adapted module, not sampled
coordinates.

Across three locked-batch seeds and `kappa in {5,10,100,1000}`, the initial
product mismatch stayed below `6.7e-8`. Capacity kept full-product update
direction numerically indistinguishable and final trajectory errors near
`1e-5`; AdamW showed monotone product-space divergence, with mean final
trajectory gap increasing from approximately `0.538` at `kappa=5` to `4.901`
at `kappa=1000`.

This is empirical full-product gauge-equivariance evidence under the tested
GPT-2 LoRA conditions, not a proof of exact mathematical gauge invariance.

Reproduce:

```bash
python experiments/h134_full_product_audit.py
```

Actual machine-readable Colab outputs should be imported with
`tools/import_h134_results.py`; result CSV/JSON files are not reconstructed from
console prose.

### H13.5 Naive Rebalancing Counterfactual

H13.5 asks whether Capacity's full-product gauge robustness can be explained by
product-preserving factor rebalancing alone.

The audit compares six methods under identical seeds, locked mini-batches, and
complete LoRA products:

- AdamW;
- AdamW with rebalancing after every step;
- AdamW with rebalancing every ten steps;
- SGD;
- SGD with rebalancing after every step;
- `CapacityAdaptiveQuotientFlow`.

The counterfactual rebalance uses thin QR factorizations and a rank-by-rank
core SVD to preserve each represented product `B A` while balancing factor
Gramians. AdamW moment states are intentionally not transported, so this is a
naive factor-canonicalization counterfactual rather than a covariant AdamW
state transformation.

Across three seeds, thirty steps, and `kappa in {5,100,1000}`, naive
rebalancing reduced some coordinate-optimizer trajectory errors but did not
restore Capacity-like full-product gauge dynamics. Per-step AdamW rebalancing
retained mean final trajectory gaps from approximately `0.495` to `1.772`.
Per-step SGD rebalancing was stronger but still grew from approximately
`8.88e-4` at `kappa=5` to `0.212` at `kappa=1000`.
`CapacityAdaptiveQuotientFlow`, executed through the K1-enabled resumable
capacity stepper used in the H13 series, remained near `1e-5` at every tested
condition number. The K1 controller adapts the active local product-motion
tolerance but does not alter the underlying quotient-preconditioned direction.

These results rule out naive factor rebalancing as a sufficient explanation
for Capacity's observed full-product gauge robustness. They do not prove that
Capacity is mathematically unique or exactly gauge invariant.

Reproduce:

```bash
python experiments/h135_rebalance_counterfactual.py
```

Import genuine Colab output files with `tools/import_h135_results.py`.

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
| Capacity-adaptive quotient flow | Ten-seed held-out GPT-2 LoRA run reached `11.07x` geometric-mean suppression | Mean `10x` suppression reproduced; per-seed confirmation not strict | Experimental |
| H13.4 full-product audit | Capacity full-product trajectory gaps stayed near `1e-5` across `kappa=5..1000` | Mechanism audit; not a task-superiority claim | Empirical audit |
| H13.5 rebalance counterfactual | Naive rebalancing reduced coordinate-optimizer divergence but did not match Capacity | Mechanism audit; not a task-superiority claim | Empirical audit |

Confirmed in controlled tests:

- product-coordinate tangent projection and rank-preserving retraction;
- reduced LoRA gauge sensitivity and tangent/near-null motion;
- agreement between matrix-free and dense small-toy response directions;
- machine-precision ordinary-inverse covariance and fresh-gradient substep tests
  for `SubsteppedQuotientFlow`;
- runtime capacity control for quotient flow with zero-capacity and dynamic
  substep regression tests;
- fixed-tolerance Capacity can preserve near-`1e-5` full-product trajectory gaps
  in the tested full-rank GPT-2 LoRA audit conditions;
- K1 improves target-quality efficiency relative to fixed Capacity on reached
  targets in the reported H13 series, while increasing long-horizon gauge gap;
- no-balance and balance task behavior was close in the tested H13 setup, while
  balancing mainly improved structural precision.

Open claims and limits:

- broad superiority over Adam or AdamW;
- production large-model scalability;
- robust GPT-2 or LLM task-performance gains;
- a universal recommendation to replace existing optimizers;
- per-seed and bootstrap-CI confirmation of strict `10x` H10 suppression;
- strict long-horizon gauge equivariance for K1;
- strict arbitrary non-orthogonal gauge covariance on the pseudoinverse branch;
- robustness across models, ranks, datasets, and LoRA target modules;
- total cloud-cost, energy-cost, or distributed-training advantage;
- full-parameter pretraining applicability;
- a checked-in H13.6 matched-resource efficiency frontier; current H13.4/H13.5
  scripts audit gauge dynamics and mechanism counterfactuals, not equal-cost
  training efficiency.

Engineering status:

- H13.4/H13.5 experiment files are Colab-oriented audit scripts that still
  vendor archived H12/H13.2 helper routines. Their active script entrypoints are
  `h134_main()` and `h135_main()`, but the files should be split into shared
  `experiments/common/` utilities before being treated as clean experiment
  templates.
- Continuous integration runs syntax and unit-test checks, but heavyweight
  GPT-2/WikiText audits remain manual or scheduled experiments.

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
- [docs/PAPER_H134_UPDATE.md](docs/PAPER_H134_UPDATE.md)
- [docs/PAPER_H135_UPDATE.md](docs/PAPER_H135_UPDATE.md)

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
