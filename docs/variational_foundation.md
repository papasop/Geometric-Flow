# Variational Foundation of GeoFlow

This document records the H13.9/H13.9D theoretical update: for full-rank
low-rank factors, the existing inverse-Gram GeoFlow direction is the exact local
steepest-descent direction under a channel-resolved executed-information metric.

This is a local variational statement. It is not a proof of a globally shortest
training path, universal optimizer superiority, or guaranteed long-horizon
advantage over AdamW.

## Low-Rank Quotient Representation

GeoFlow uses the factor convention

```math
M=BA,
\qquad
B\in\mathbb R^{m\times r},
\qquad
A\in\mathbb R^{r\times n}.
```

For a factor tangent vector

```math
V=(V_A,V_B),
```

the net product velocity is

```math
D\pi_{(A,B)}[V]=V_BA+BV_A.
```

The two executed channel motions are

```math
\delta M_A = BV_A,
\qquad
\delta M_B = V_BA.
```

`BV_A` is functional motion executed through the `A`-factor channel, and `V_BA`
is functional motion executed through the `B`-factor channel. If

```math
BV_A\approx -V_BA,
```

then the net product velocity can be small even though both channels executed
nonzero motion. The split executed-information metric keeps this internal
execution cost visible.

## Split Executed-Information Metric

Define

```math
g_{\mathrm{split}}(V,W)
=
\langle BV_A,BW_A\rangle_F
+
\langle V_BA,W_BA\rangle_F.
```

The associated norm is

```math
\|V\|_{\mathrm{split}}^2
=
\|BV_A\|_F^2+\|V_BA\|_F^2.
```

Equivalently, the infinitesimal executed-information budget is

```math
dI_{\mathrm{exec}}^2
=
\|BV_A\|_F^2+\|V_BA\|_F^2.
```

This metric is not claimed to be the unique universal information metric. It is
the metric under which the current inverse-Gram low-rank direction has an exact
local steepest-descent characterization.

## Theorem: Exact Executed-Information Steepest Descent

Assume `A` has full row rank and `B` has full column rank, so `AA^T` and `B^T B`
are positive definite. For any differentiable loss `L(A,B)`, define

```math
\operatorname{grad}_{\mathrm{split}}L
=
\left(
(B^\top B)^{-1}\nabla_A L,\,
\nabla_B L(AA^\top)^{-1}
\right).
```

Then the existing GeoFlow direction

```math
V_A^\star=-(B^\top B)^{-1}\nabla_A L,
\qquad
V_B^\star=-\nabla_B L(AA^\top)^{-1}
```

is the negative split-metric gradient direction.

The normalized direction is the unique optimizer of

```math
\max_V -\mathrm dL[V]
\quad\text{subject to}\quad
\|BV_A\|_F^2+\|V_BA\|_F^2\le1.
```

Equivalently,

```math
V^\star
=
-\frac{\operatorname{grad}_{\mathrm{split}}L}
{\|\operatorname{grad}_{\mathrm{split}}L\|_{\mathrm{split}}}.
```

The unnormalized direction used by the optimizer differs from this normalized
solution only by a positive scalar.

## Proof Sketch

The Riesz representation condition requires

```math
g_{\mathrm{split}}(\operatorname{grad}_{\mathrm{split}}L,V)
=
\mathrm dL[V]
```

for all `V=(V_A,V_B)`. Write

```math
\operatorname{grad}_{\mathrm{split}}L=(Z_A,Z_B).
```

Expanding the metric gives

```math
\langle BZ_A,BV_A\rangle_F
+
\langle Z_BA,V_BA\rangle_F
=
\langle\nabla_A L,V_A\rangle_F
+
\langle\nabla_B L,V_B\rangle_F.
```

Using Frobenius identities,

```math
\langle BZ_A,BV_A\rangle_F
=
\langle B^\top BZ_A,V_A\rangle_F,
```

and

```math
\langle Z_BA,V_BA\rangle_F
=
\langle Z_BAA^\top,V_B\rangle_F.
```

Therefore

```math
B^\top BZ_A=\nabla_A L,
\qquad
Z_BAA^\top=\nabla_B L,
```

so

```math
Z_A=(B^\top B)^{-1}\nabla_A L,
\qquad
Z_B=\nabla_B L(AA^\top)^{-1}.
```

For any feasible `V`,

```math
-\mathrm dL[V]
=
-g_{\mathrm{split}}(\operatorname{grad}_{\mathrm{split}}L,V)
\le
\|\operatorname{grad}_{\mathrm{split}}L\|_{\mathrm{split}}
\|V\|_{\mathrm{split}}.
```

On the unit executed-information ball, the maximum is
`\|\operatorname{grad}_{\mathrm{split}}L\|_{\mathrm{split}}`, reached by the
negative normalized split gradient. Since the full-rank metric is positive
definite, the normalized optimizer is unique.

## Gauge Covariance

For any `S in GL(r)`, define

```math
A'=SA,
\qquad
B'=BS^{-1}.
```

Tangent vectors transform by

```math
V_A'=SV_A,
\qquad
V_B'=V_BS^{-1}.
```

Then

```math
B'V_A'=BV_A,
\qquad
V_B'A'=V_BA.
```

Thus

```math
g'_{\mathrm{split}}(V',W')=g_{\mathrm{split}}(V,W).
```

The split metric is gauge invariant independently of the loss. Direction
covariance additionally assumes a gauge-invariant objective, as in `L = L(BA)`,
so that factor gradients transform covariantly. Under this condition, the
quotient direction and induced product velocity are gauge-covariant on the
full-rank ordinary-inverse branch. Floating-point audits under extreme
condition numbers measure numerical residuals, not analytic failure of this
identity.

## Rank-Deficient Extension

If `A` or `B` is rank deficient, `AA^T` or `B^T B` is not invertible. The
Moore-Penrose pseudoinverse gives the minimum-norm visible quotient direction:

```math
(B^\top B)^+,
\qquad
(AA^\top)^+.
```

Null directions satisfy

```math
BV_A=0,
\qquad
V_BA=0.
```

They have zero split norm, zero first-order loss differential, and no product
velocity. Therefore full rank gives a unique factor-space normalized optimizer,
while rank deficiency gives a unique visible quotient direction but a nonunique
factor lift modulo zero-cost null directions.

This does not imply arbitrary non-orthogonal gauge covariance of the
pseudoinverse branch without additional conditions.

## Net Product Frobenius Comparison

Assume `L=L(M)` and let `G=\nabla_M L`. Define projectors

```math
P_B=B(B^\top B)^{-1}B^\top,
\qquad
P_A=A^\top(AA^\top)^{-1}A.
```

The current split direction induces

```math
D_{\mathrm{cur}}
=
-\left(P_BG+GP_A\right).
```

The exact rank-`r` tangent-space steepest direction under the net product
Frobenius metric is

```math
D_\star
=
-\left(P_BG+GP_A-P_BGP_A\right).
```

The split direction double-counts the overlapping `P_BGP_A` block relative to
the net product tangent projection. Therefore the current direction is exact
for the split executed-information metric and generally not exact for the net
product Frobenius metric.

However, H13.9 gives the sharp local efficiency bound

```math
\frac{\eta(D_{\mathrm{cur}})}{\eta(D_\star)}
\ge
\frac{2\sqrt2}{3}
\approx0.942809,
\qquad
\eta(D)=\frac{-\langle G,D\rangle_F}{\|D\|_F}.
```

The bound is independent of dimension, rank, and factor condition number, and a
constructive worst case can approach equality.

## Numerical Audits

H13.9 audited the split direction against the net product Frobenius optimum in
random matrix settings:

- matrix dimensions: `32 x 40`;
- ranks: `1, 2, 4, 8`;
- factor condition scales: `1, 10, 100, 1000, 10000`;
- gauge condition scales: `1, 10, 1000`;
- trials: `12000`.

Key results:

- minimum steepness ratio: `0.949021`;
- mean ratio: `0.979507`;
- median ratio: `0.981007`;
- p5 ratio: `0.956326`;
- constructive worst case: `0.942809115`;
- theoretical bound: `0.942809042`.

H13.9D directly audited the variational theorem:

- full-rank trials: `12000`;
- rank-deficient trials: `200`;
- max Riesz residual: `2.8133e-12`;
- max KKT residual: `7.6224e-15`;
- max unit-budget residual: `5.5511e-16`;
- max Cauchy-Schwarz residual: `8.2845e-13`;
- gauge metric median: `3.6609e-15`;
- gauge metric p99: `2.5509e-8`;
- gauge metric max: `6.1636e-7`.

The H13.9D gate summary was:

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

These numerical audits verify the closed-form identities and stress the
floating-point implementation. They are not a substitute for the analytic proof.

## Limits

- The local theorem is not a global information brachistochrone.
- It does not prove universal optimizer superiority.
- It does not prove long-horizon stochastic optimality.
- The split executed-information metric and the current net product-capacity
  controller are related but distinct.
- The physical uniqueness of the split metric remains an open modeling
  question.
- History-aware quotient momentum, functional EMA, and gauge-covariant
  second-moment estimation remain open research directions.
