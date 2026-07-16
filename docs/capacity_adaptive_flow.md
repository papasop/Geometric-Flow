# Capacity-Adaptive Quotient Flow

`CapacityAdaptiveQuotientFlow` is an experimental controller for factorized
LoRA-style updates. It keeps the repository convention

```math
M = B A,
```

where `A` has shape `(rank, in_features)` and `B` has shape
`(out_features, rank)`.

## Local Quotient Direction

For each factor pair, the unit quotient-preconditioned vector field is

```math
V_A = -(B^\top B)^{-1}\nabla_A L,
\qquad
V_B = -\nabla_B L(AA^\top)^{-1}.
```

The corresponding first-order product velocity is

```math
V_M = V_B A + B V_A.
```

H13.9D identifies this same inverse-Gram direction as the exact local
steepest-descent direction under the split executed-information metric

```math
\|V\|_{\mathrm{split}}^2
=
\|BV_A\|_F^2+\|V_BA\|_F^2.
```

The public capacity controller below uses the net product velocity `V_M` to
bound first-order product displacement. Thus the variational direction metric
and the finite-step controller metric are related but distinct. The public
implementation intentionally does not use the transposed GPT-2 Conv1D layout
found in some research scripts.

## Capacity Controller

For multiple LoRA targets, capacity is the joint product-space speed

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

This is `H_product`, not the split executed-information norm
`\|V\|_split`. It is exposed in diagnostics as `H_opt` and is the controller
quantity currently used by
`CapacityAdaptiveQuotientFlow`.

Each local step advances by

```math
d\tau
=
\min\left(
T_{\mathrm{remaining}},
\frac{\varepsilon_\Phi}{H_{\mathrm{opt}}}
\right),
```

with an optional `max_flow_dt` cap. Directions are computed for all modules
before any module is mutated, and every module shares the same `d_tau`.

Here, `H_opt * d_tau` is the first-order predicted product-space displacement.
Because both factors are updated simultaneously,

```math
(B + d\tau V_B)(A + d\tau V_A) - BA
=
d\tau(V_BA + BV_A) + d\tau^2 V_BV_A.
```

The capacity controller bounds the first-order term; smaller local steps reduce
the second-order residual.

If `H_opt` is numerically zero, the controller consumes the remaining macro
flow time in one no-op local step.

## Gauge Boundary

On the full-rank ordinary-inverse branch, the direction is gauge-equivariant in
exact arithmetic under

```math
A \mapsto S A,\qquad B \mapsto B S^{-1}.
```

When an ill-conditioned Gram matrix triggers the Moore-Penrose pseudoinverse
fallback, exact covariance is not generally guaranteed under arbitrary
non-orthogonal gauges. The fallback is a numerical safety path and is reported
through `fallback_count` and `condition_max`.

## Numerical Safeguards

- `gram_condition_limit` chooses ordinary inverse versus pseudoinverse.
- `balance_after_substep` performs product-preserving QR canonicalization.
- `max_auto_substeps` prevents runaway local integration.
- `max_flow_dt` can cap the local flow step.

QR canonicalization preserves `B @ A`; it should not be counted as functional
product displacement.

## Diagnostics

Runtime diagnostics include:

- `last_auto_substeps`
- `last_capacity`
- `last_flow_dt`
- `last_predicted_local_dphi`
- `last_factor_update_norm`
- `last_predicted_product_motion`
- `condition_max`
- `fallback_count`
- `balance_residual_max`

Scalar diagnostics are runtime counters and are not guaranteed to persist
across optimizer checkpoint restoration.

## Evidence Boundary

H10.10/H10.11 used heavier GPT-2 LoRA research scripts that are intentionally
not part of the public optimizer API. In a fixed ten-seed held-out GPT-2 LoRA
confirmation, the controller generated between 5 and 13 local substeps,
matched Adam-scale progress on all seeds, reduced gauge divergence on all
seeds, and obtained 11.07x geometric-mean gauge suppression. Every seed
achieved approximately at least 7.45x suppression, while the bootstrap 95%
interval was approximately
`[9.09x, 13.97x]`.

This is bounded experimental evidence. It does not establish universal task
superiority or per-seed 10x suppression.
