# GeoFlow for PyTorch

Geometric-Flow studies optimization when model parameters are redundant:
different parameter coordinates may represent the same functional model. The
current low-rank direction is a gauge-covariant local steepest descent per unit
channel-resolved executed functional information, and it is uniformly
near-optimal relative to net product-space Frobenius displacement with a sharp
local efficiency floor of `2*sqrt(2)/3`.

These are local variational results. They do not prove a globally shortest
training path, universal optimizer superiority, or guaranteed long-horizon
advantage over AdamW.

Instead of defining direction and step size only in parameter space, the
framework introduces a functional map, a quotient-aware direction, executed
functional information, and a functional-time controller.

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
| chosen capacity `H_func(V_Q)` | the selected measure of functional motion for a controller |
| functional time `d_tau = epsilon_func / H_func` | a local time step bounded by the chosen functional-displacement budget |

An optional controller adapts `epsilon_Phi` from functional progress,
prediction error, and limiter state. K1 is one such controller; it is not the
framework itself and should not be read as a conventional learning-rate
scheduler.

## Variational Foundation: Executed-Information Steepest Descent

Geometric-Flow is motivated by a functional steepest-descent principle:
optimization should maximize task improvement per unit motion actually executed
by the represented model function, rather than per unit motion of an arbitrary
parameter representation.

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

For low-rank factors

```math
M=BA,
\qquad
B\in\mathbb R^{m\times r},
\qquad
A\in\mathbb R^{r\times n},
```

a factor tangent vector `V=(V_A,V_B)` has two executed functional channels:

```math
\delta M_A = BV_A,
\qquad
\delta M_B = V_BA.
```

H13.9D identifies the channel-resolved executed-information metric

```math
g_{\mathrm{split}}(V,W)
=
\langle BV_A,BW_A\rangle_F
+
\langle V_BA,W_BA\rangle_F,
```

with norm

```math
\|V\|_{\mathrm{split}}^2
=
\|BV_A\|_F^2+\|V_BA\|_F^2.
```

This metric counts the functional motion executed by each factor channel. If
`BV_A` is nearly cancelled by `V_BA`, the net product displacement can be small,
but both channels still moved; the split metric does not erase that internal
execution cost.

> **Theorem-style statement.** For full-rank `A` and `B`, the implemented
> inverse-Gram GeoFlow direction
>
> ```math
> V_A=-(B^\top B)^{-1}\nabla_A L,
> \qquad
> V_B=-\nabla_B L(AA^\top)^{-1}
> ```
>
> is the unique normalized solution of
>
> ```math
> \max_V -\mathrm dL[V]
> \quad\text{subject to}\quad
> \|BV_A\|_F^2+\|V_BA\|_F^2\le1.
> ```
>
> Equivalently, it is the negative split-metric gradient direction:
>
> ```math
> V^\star
> =
> -\frac{\operatorname{grad}_{\mathrm{split}}L}
> {\|\operatorname{grad}_{\mathrm{split}}L\|_{\mathrm{split}}},
> \qquad
> \operatorname{grad}_{\mathrm{split}}L
> =
> \left(
> (B^\top B)^{-1}\nabla_A L,\,
> \nabla_B L(AA^\top)^{-1}
> \right).
> ```

Proof structure: Riesz representation under `g_split` defines the split-metric
gradient; expanding `g_split(grad L,V)=dL[V]` yields the two Gram equations;
Cauchy-Schwarz, or equivalently KKT stationarity on the unit information ball,
gives the unique full-rank optimum.

A theoretically aligned split-information clock for a chosen quotient-aware
direction `V_Q` would use

```math
H_{\mathrm{split}}(V_Q)
=
\|V_Q\|_{\mathrm{split}},
\qquad
d\tau_{\mathrm{split}}
=
\frac{\epsilon_{\mathrm{split}}}{H_{\mathrm{split}}(V_Q)}.
```

Thus, `epsilon_split` would specify an allowed executed-information budget,
while `H_split` measures the rate at which the current direction consumes that
budget. The public `CapacityAdaptiveQuotientFlow` controller instead uses the
net product-displacement clock `d_tau_product = epsilon_product / H_product`
defined below. This is a local variational result. It is not a proof of a
globally optimal information brachistochrone or a shortest nonlinear training
path.

### Gauge Covariance

For any invertible `S in GL(r)`, define

```math
A'=SA,
\qquad
B'=BS^{-1},
```

and transform tangent vectors by

```math
V_A'=SV_A,
\qquad
V_B'=V_BS^{-1}.
```

Then

```math
B'V_A'=BV_A,
\qquad
V_B'A'=V_BA,
```

so

```math
g'_{\mathrm{split}}(V',W')=g_{\mathrm{split}}(V,W).
```

The executed-information metric is therefore invariant under internal
full-rank gauge changes. Direction covariance additionally assumes that the
loss is gauge invariant, as in `L = L(BA)`, so that factor gradients transform
covariantly. With rank-deficient factors, the Moore-Penrose pseudoinverse
exposes a visible quotient direction, but the factor lift can add zero-cost null
directions and is no longer unique.

### Split Executed Information vs Net Product Displacement

The net product velocity is

```math
D=V_BA+BV_A.
```

If the metric is the net Frobenius displacement `||D||_F`, the rank-`r`
tangent-space steepest direction for a product gradient `G = \nabla_M L` is

```math
D_\star
=
-\left(P_BG+GP_A-P_BGP_A\right),
```

where

```math
P_B=B(B^\top B)^{-1}B^\top,
\qquad
P_A=A^\top(AA^\top)^{-1}A.
```

The current inverse-Gram GeoFlow direction induces

```math
D_{\mathrm{cur}}
=
-\left(P_BG+GP_A\right).
```

Therefore the current direction is not the exact steepest direction under the
net full-product Frobenius metric. H13.9 shows the sharper statement: it is
exact under the split executed-information metric, and uniformly near-optimal
under the net product metric:

```math
\frac{\eta(D_{\mathrm{cur}})}{\eta(D_\star)}
\ge
\frac{2\sqrt2}{3}
\approx 0.942809,
\qquad
\eta(D)=\frac{-\langle G,D\rangle_F}{\|D\|_F}.
```

This bound is independent of dimension, rank, and factor conditioning; a
constructive worst case can approach equality.

See [docs/variational_foundation.md](docs/variational_foundation.md) for the
Riesz proof, gauge-covariance derivation, rank-deficient pseudoinverse boundary,
and H13.9/H13.9D audit details.

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

In this low-rank instance, the quotient direction, executed-information
capacity, and functional time become concrete:

```math
V_A = -(B^\top B)^{-1}\nabla_A L,
\qquad
V_B = -\nabla_B L(AA^\top)^{-1},
```

```math
H_{\mathrm{split}}
=
\left(
\sum_\ell
\|B_\ell V_{A,\ell}\|_F^2
+
\|V_{B,\ell}A_\ell\|_F^2
\right)^{1/2},
```

```math
d\tau
=
\min\left(
T_{\mathrm{remaining}},
\frac{\epsilon_{\mathrm{split}}}{H_{\mathrm{split}}}
\right).
```

This is the theory-level split-information clock. The current public capacity
controller does not use this exact clock; it uses `H_product`, the net first-
order product-displacement capacity in the `CapacityAdaptiveQuotientFlow`
section. Geometric-Flow therefore does not merely choose a coordinate learning
rate, but the present implementation still distinguishes the variational
direction metric from the finite-step controller metric.

## Optimizers

| method | core idea | status |
| :--- | :--- | :--- |
| `adam` / `adam_raw` | First-order coordinate baseline | Control |
| `diagonal_grad_square` | Legacy diagonal preconditioner | Historical diagnostic |
| `functional_geoflow` | `J_Phi`-based stable-neutral response directions | Research reference |
| `FixedRankFunctionalAdam` | Product-coordinate Adam, tangent projection, rank-`r` retraction | Experimental backend |
| `SubsteppedQuotientFlow` | Full-rank inverse-Gram direction is exact split executed-information steepest flow | Experimental integrator |
| `CapacityAdaptiveQuotientFlow` | Same split-steepest direction with a net product-displacement capacity controller | Experimental integrator |

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
Gram-preconditioned factor-flow integrator. H13.9D gives it an exact local
steepest-descent characterization under the split executed-information metric;
that does not make it the standard fixed-rank quotient-manifold optimizer or a
global shortest-path training method.

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

The current finite-step capacity controller uses the net product-displacement
capacity:

```math
H_{\mathrm{product}}
=
H_{\mathrm{opt}}
=
\sqrt{
\sum_\ell
\|V_{B,\ell}A_\ell+B_\ell V_{A,\ell}\|_F^2
}.
```

Interpretation: `H_product`, exposed in the code as `H_opt`, measures the
first-order size of the net represented product motion produced by
simultaneously changing `A` and `B` across all target modules.

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

The variational direction metric and the current finite-step capacity
controller are related but distinct objects: direction geometry uses the split
executed-information metric, while the public `CapacityAdaptiveQuotientFlow`
controller currently bounds first-order net product displacement. Whether these
should be unified into one executed-information time functional remains an open
research question.

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

### H13.8-H13.9: Direction Theory and Controller Tradeoffs

H13.8 reframed balancing and K1 as separate mechanisms. In the tested GPT-2
LoRA setting:

- per-step balancing was not the main task-performance bottleneck;
- no-balance saved about `3%`-`4%` wall time relative to balance;
- fixed no-balance Capacity retained an approximately `8.35e-5` gauge
  trajectory gap at `1000` backward calls;
- legacy K1 reduced backward calls for the deep `0.40` target relative to fixed
  Capacity, but its long-horizon trajectory gap grew to about `9.15e-2`;
- invariant K1 kept a lower gauge gap but did not improve training efficiency;
- AdamW reached stronger long-horizon validation improvement in the benchmark,
  while remaining strongly gauge dependent.

The supported conclusion is: GeoFlow provides strong representation covariance
and competitive short-budget efficiency, while AdamW remains stronger at long
horizon in the current benchmark.

H13.9 and H13.9D clarified the local direction theory. The implemented
inverse-Gram direction is not merely a heuristic preconditioner: under the
split executed-information metric it is the exact local steepest-descent
direction in the full-rank branch. Under the different net full-product
Frobenius metric, it is not exact but has a sharp uniform efficiency guarantee.

Numerical H13.9 audit:

- random audit mean `eta(D_cur)/eta(D_star) ~= 0.9795`;
- median `~= 0.9810`;
- constructive worst case `0.942809115`, matching the theoretical
  `2*sqrt(2)/3` floor.

H13.9D theorem audit:

- full-rank trials: `12,000`;
- rank-deficient trials: `200`;
- ranks: `1, 2, 4, 8`;
- factor condition scales: `1, 10, 100, 1000, 10000`;
- gauge condition scales: `1, 10, 1000`.

Key audit flags:

```text
PASS_RIESZ_REPRESENTATION = true
PASS_KKT_STATIONARITY = true
PASS_UNIT_INFORMATION_BUDGET = true
PASS_CAUCHY_SCHWARZ_EQUALITY = true
PASS_RANDOM_FEASIBLE_OPTIMALITY = true
PASS_FULL_RANK_UNIQUENESS_PROBES = true
PASS_GAUGE_METRIC_TYPICAL = true
PASS_GAUGE_METRIC_P99 = true
PASS_GAUGE_METRIC_EXTREME = true
PASS_GAUGE_PRODUCT_COVARIANCE = true
PASS_GAUGE_DIRECTION_COVARIANCE = true
PASS_RANK_DEFICIENT_VISIBLE_RIESZ = true
PASS_RANK_DEFICIENT_NULL_COST = true
PASS_RANK_DEFICIENT_PSEUDOINVERSE = true
PASS_ALL = true
```

Representative residuals:

- max Riesz residual: `2.81e-12`;
- max KKT residual: `7.62e-15`;
- max unit-budget residual: `5.55e-16`;
- gauge metric median: `3.66e-15`;
- gauge metric p99: `2.55e-8`;
- gauge metric max: `6.16e-7`.

The larger gauge residuals occur in extreme ill-conditioned floating-point
settings; they are numerical audit residuals, not an analytic failure of gauge
invariance.

H13.9C tested the exact net-product Frobenius tangent correction in a real
GPT-2/LoRA setting. The correction was nontrivial: direction cosine was about
`0.991`, and correction fraction was about `0.13`-`0.14`. However, replacing the
channel-resolved split direction with the exact net-product Frobenius tangent
direction did not improve long-horizon validation improvement in the completed
tests and added wall-time overhead. Full-product correction combined with the
legacy K1 controller also did not outperform the original legacy K1 path.

This supports a useful boundary: local net-displacement steepest descent and
long-horizon stochastic optimization are not equivalent objectives.

### H13.10: Matched-Budget Stochastic Variance Decomposition

H13.10 tested whether the local covariance theorem survives stochastic
minibatch probing under a matched first-order product-displacement budget. The
final audit used six trials, 100 training steps, rank 4, batch size 64, and a
batch x gauge two-way decomposition. Exact covariance probes used the full-rank
ordinary-inverse branch with zero ridge; the practical ridge branch was reported
separately.

For product-space directions indexed by minibatch `b` and gauge representation
`s`, the decomposition was

```math
D_{b,s}=\mu+F_b+G_s+C_{b,s},
```

with

```math
V_F=\mathbb E_b\|F_b\|_F^2,
\qquad
V_G=\mathbb E_s\|G_s\|_F^2,
\qquad
V_C=\mathbb E_{b,s}\|C_{b,s}\|_F^2.
```

Here `F_b` is the minibatch functional main effect, `G_s` is the gauge
representation main effect, and `C_{b,s}` is the batch-gauge interaction.

| method | final loss | `V_F` | `V_G` | `V_C` | product gauge p99 | alignment |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW factor | 9.8968 | `2.40e4` | `1.33e5` | `8.31e4` | 0.973 | 0.9745 |
| Fixed split | 8.6235 | 0.3927 | `1.52e-23` | `3.81e-24` | `2.66e-12` | 0.9038 |
| Legacy K1, matched budget | 8.6200 | 0.3822 | `4.67e-23` | `1.44e-23` | `2.84e-12` | 0.9049 |
| Full-product corrected | 8.5053 | 0.2819 | `1.35e-24` | `6.48e-25` | `1.89e-12` | 0.8613 |
| Factor EMA split | 8.3236 | 0.01611 | `6.66e-24` | `3.12e-26` | `1.78e-12` | 0.9901 |

Core H13.10 gates passed:

```text
PASS_MATCHED_PRODUCT_STEP = true
PASS_SPLIT_CHANNEL_A_COVARIANCE = true
PASS_SPLIT_CHANNEL_B_COVARIANCE = true
PASS_SPLIT_PRODUCT_COVARIANCE = true
PASS_FULL_PRODUCT_COVARIANCE = true
PASS_TWO_WAY_DECOMPOSITION = true
PASS_NO_EXACT_PROBE_SKIPS = true
PASS_ALL_METHODS_IMPROVE_LOSS = true
PASS_FINITE_RESULTS = true
PASS_CORE = true
```

Under the matched budget, exact split GeoFlow remained gauge covariant to
approximately `1e-12`; gauge main-effect and batch-by-gauge interaction
variance were numerically negligible for the GeoFlow directions. The practical
ridge branch showed product-gauge residuals around `1e-4`, dominated by the
non-covariant isotropic factor-coordinate ridge `lambda I`, not by failure of
the exact inverse-Gram direction.

Factor EMA split, which stores temporal averaging in factor-gradient
coordinates and explicitly transports history during probes, reduced
functional direction variance from about `0.393` to `0.0161`, improved
full-batch alignment from about `0.904` to `0.990`, and lowered final loss in
all six trials relative to the memoryless split flow. The exact full-product
tangent correction gave moderate variance and loss improvement, but
substantially less than temporal averaging. With the realized product-step
budget strictly matched, legacy K1 was nearly identical to fixed split flow;
its larger advantage in earlier unmatched audits came primarily from allocating
a larger functional displacement budget, not from changing the underlying local
direction.

### H13.11: Gauge-Covariant Channel-History Momentum

H13.11 moved optimizer first-moment history out of raw factor coordinates and
into executed split channels:

```math
C_{A,t}=B_tV_{A,t},
\qquad
C_{B,t}=V_{B,t}A_t.
```

The intrinsic channel moments are

```math
U_{A,t}
=
\beta_1 U_{A,t-1}
+
(1-\beta_1)C_{A,t},
\qquad
U_{B,t}
=
\beta_1 U_{B,t-1}
+
(1-\beta_1)C_{B,t}.
```

Because `B'V_A'=BV_A` and `V_B'A'=V_BA` on the full-rank ordinary-inverse
branch for gauge-invariant losses, `(U_A,U_B)` is a gauge-visible,
gauge-invariant history state. It is lifted back to factor velocities by

```math
V_A=B^+U_A,
\qquad
V_B=U_BA^+,
```

or, on the full-rank ordinary-inverse branch,

```math
V_A=(B^\top B)^{-1}B^\top U_A,
\qquad
V_B=U_BA^\top(AA^\top)^{-1}.
```

The compared methods were `adamw_factor`, `fixed_capacity_split`,
`factor_ema_split`, `channel_momentum_geoflow`,
`channel_adaptive_geoflow`, and `full_product_corrected`.

| method | final loss | `V_F` | product gauge p99 | alignment |
| :--- | ---: | ---: | ---: | ---: |
| AdamW factor | 9.2839 | `2.21e4` | 0.969 | 0.9746 |
| Fixed split | 7.8189 | 0.3869 | `2.09e-12` | 0.9039 |
| Factor EMA split | 7.5226 | 0.01597 | `2.35e-12` | 0.9905 |
| Channel momentum GeoFlow | 7.5048 | 0.01557 | `3.51e-12` | 0.9910 |
| Channel-adaptive GeoFlow | 7.5107 | 0.03756 | `3.89e-12` | 0.9902 |
| Full-product corrected | 7.7056 | 0.2789 | `1.02e-12` | 0.8597 |

Core H13.11 gates passed:

```text
PASS_MATCHED_PRODUCT_STEP = true
PASS_CHANNEL_MOMENTUM_PRODUCT_COVARIANCE = true
PASS_CHANNEL_ADAPTIVE_PRODUCT_COVARIANCE = true
PASS_CHANNEL_MOMENTUM_CHANNEL_COVARIANCE = true
PASS_CHANNEL_ADAPTIVE_CHANNEL_COVARIANCE = true
PASS_TWO_WAY_DECOMPOSITION = true
PASS_ALL_METHODS_IMPROVE = true
PASS_FINITE = true
PASS_CORE = true
```

Channel-space momentum preserved product and individual channel covariance to
approximately `1e-12` while reducing functional direction variance by about
`96%` relative to the memoryless split flow:

```math
1-\frac{0.01557}{0.3869}\approx95.98\%.
```

Under the matched product-displacement budget, channel momentum improved mean
final loss from `7.8189` to `7.5048` and increased full-batch alignment from
`0.9039` to `0.9910`. It also matched factor EMA and was slightly better on
the six-trial mean, but it was lower loss in only two of six trials; the main
advance is intrinsic gauge-invariant representation of optimizer history, not
statistically established task superiority over factor EMA.

The channel-adaptive variant used scalar channel second moments

```math
q_{A,t}
=
\beta_2q_{A,t-1}
+
(1-\beta_2)\|C_{A,t}\|_F^2,
\qquad
q_{B,t}
=
\beta_2q_{B,t-1}
+
(1-\beta_2)\|C_{B,t}\|_F^2.
```

This did not improve the momentum method: `V_F` increased from `0.01557` to
`0.03756`, and mean loss increased from `7.5048` to `7.5107`. Gauge invariance
alone is therefore insufficient to define a geometry-compatible adaptive
second moment.

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
| H13.8 controller audit | Fixed Capacity strongest structurally; K1 exposes efficiency-equivariance tradeoff | AdamW stronger at long horizon in current benchmark | Empirical audit |
| H13.9 direction audit | Split executed-information steepest descent proven and numerically audited | Local theorem; not a global training optimum | Theorem audit |
| H13.9C full-product correction | Net-product tangent correction is nontrivial but costly | No long-horizon improvement in completed GPT-2/LoRA tests | Empirical observation |
| H13.9D variational audit | `PASS_ALL=true` across full-rank and rank-deficient theorem checks | Numerical verification of local theorem | Theorem audit |
| H13.10 stochastic decomposition | Exact stochastic covariance and batch x gauge decomposition under matched product budget | Factor EMA reduced `V_F` by about `96%`; full-product correction was weaker | Controlled mechanism audit |
| H13.11 channel-history momentum | Gauge-invariant optimizer history stored in executed channels | Mean loss slightly below factor EMA, but only 2/6 seed wins | Experimental optimizer mechanism |
| H13.11 scalar channel adaptation | Gauge-covariant scalar channel second moments | No benefit; functional variance increased | Negative result |

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
  balancing mainly improved structural precision;
- the inverse-Gram low-rank direction is the exact split executed-information
  steepest direction on the full-rank ordinary-inverse branch;
- relative to the net full-product Frobenius metric, the same direction has a
  sharp local efficiency floor of `2*sqrt(2)/3`;
- H13.9C found that the exact net-product tangent correction did not improve
  long-horizon GPT-2/LoRA validation improvement in the completed tests.
- exact split and full-product directions retain approximately `1e-12` product
  covariance under stochastic minibatch and multi-gauge probing;
- batch-by-gauge interaction variance is numerically negligible for the tested
  GeoFlow directions;
- temporal averaging is substantially more effective than the instantaneous
  full-product tangent correction in the controlled matched-budget benchmark;
- optimizer first-moment history can be stored intrinsically in the two
  gauge-invariant executed channels;
- channel-space momentum reduces functional direction variance by about `96%`
  relative to the memoryless split flow;
- independent scalar second-moment normalization of the two channels does not
  improve the tested momentum method.

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
  training efficiency;
- a global information brachistochrone or shortest nonlinear training path;
- statistically significant superiority of channel momentum over factor EMA;
- production Transformer or LLM validation of channel momentum;
- global optimality of the channel first-moment construction;
- a geometry-compatible operator-valued second moment;
- a gauge-covariant practical regularizer replacing isotropic ridge;
- gauge-covariant second-moment estimation that improves over channel momentum;
- an invariant K1 controller that preserves fixed-Capacity-style long-horizon
  gauge robustness while retaining K1's efficiency gains.

Engineering status:

- H13.4/H13.5 experiment files are Colab-oriented audit scripts that still
  vendor archived H12/H13.2 helper routines. Their active script entrypoints are
  `h134_main()` and `h135_main()`, but the files should be split into shared
  `experiments/common/` utilities before being treated as clean experiment
  templates.
- Continuous integration runs syntax and unit-test checks, but heavyweight
  GPT-2/WikiText audits remain manual or scheduled experiments.

## Next Research Direction: Coupled Channel Covariance

The main open question is no longer whether the inverse-Gram direction has a
local variational basis. H13.9D supplies that basis under the split
executed-information metric, and H13.11 shows that optimizer history can be
stored in gauge-invariant executed channels. The next question is:

> What is the geometry-compatible second moment for coupled executed channels?

A proposed H13.12 direction is to estimate the coupled channel covariance

```math
\Sigma_t
=
\begin{pmatrix}
\langle C_A,C_A\rangle_F &
\langle C_A,C_B\rangle_F\\
\langle C_B,C_A\rangle_F &
\langle C_B,C_B\rangle_F
\end{pmatrix},
```

and use a joint preconditioner

```math
\begin{pmatrix}
\widetilde U_A\\
\widetilde U_B
\end{pmatrix}
=
(\Sigma_t+\epsilon I)^{-1/2}
\begin{pmatrix}
U_A\\
U_B
\end{pmatrix}.
```

This is a proposed next experiment, not yet validated. It should be evaluated
against channel momentum, factor EMA, and the negative scalar-channel
normalization result from H13.11.

## Reproduce Key Benchmarks

| benchmark | command |
| :--- | :--- |
| D7 fixed-rank benchmark | `python experiments/d7_fixed_rank_tangent_benchmark.py --seeds 101,211,307 --representations 4 --steps 80 --out-dir artifacts/d7_fixed_rank` |
| H10 tiny-model regression | `python experiments/h10_progress_budget_benchmark.py --macro-lr 2.6 --substeps 16 --out-dir artifacts/h10_progress_budget` |
| Capacity-adaptive smoke | `python experiments/capacity_adaptive_quotient_smoke.py --seeds 101,211,307 --macro-flow-time 2.6 --local-function-tolerance 0.05 --out-dir artifacts/capacity_adaptive_smoke` |
| H10.11/H10.12 research archive | `experiments/archive/` contains non-API GPT-2 LoRA confirmation scripts |
| H13.9 split-vs-product direction audit | `python experiments/h139_functional_steepest_descent.py --trials-per-setting 50 --out-dir artifacts/h139_functional_steepest` |
| H13.9D variational theorem audit | `python experiments/h139d_direct_variational_proof.py --trials-per-setting 50 --random-feasible-samples 50 --no-plots --out-dir artifacts/h139d_variational_proof` |
| H13.10 matched-budget stochastic decomposition | `python experiments/h1310_final_matched_budget_two_way.py --trials 2 --steps 30 --probe-batches 6 --probe-gauges 4 --probe-steps 0,10,20,29 --no-plots --output-dir artifacts/h1310_smoke` |
| H13.11 gauge-covariant channel momentum | `python experiments/h1311_gauge_covariant_momentum.py --trials 2 --steps 30 --probe-batches 6 --probe-gauges 4 --probe-steps 0,10,20,29 --no-plots --output-dir artifacts/h1311_smoke` |
| Phase G matched-step benchmark | `python experiments/lora_matched_step_benchmark.py --trials 5 --steps 200 --representations 5 --train-scope lora_only --functional-map hidden --out artifacts/lora_matched_step.csv` |
| Functional solver toy | `python experiments/functional_projection_toy.py --response-solver implicit_cg` |
| CIFAR legacy benchmark | `python experiments/run_cifar10_benchmark.py --config hybrid_diagonal_500 --download --out artifacts/cifar10_benchmark_results.csv` |

H13.9 and H13.9D are included as runnable local theorem/direction audits.
H13.9C remains a bounded GPT-2/LoRA empirical observation rather than a
production example.

Longer commands and archived results:

- [docs/research_history.md](docs/research_history.md)
- [docs/cifar_benchmarks.md](docs/cifar_benchmarks.md)
- [docs/functional_geometry.md](docs/functional_geometry.md)
- [docs/variational_foundation.md](docs/variational_foundation.md)
- [docs/stochastic_history.md](docs/stochastic_history.md)
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
