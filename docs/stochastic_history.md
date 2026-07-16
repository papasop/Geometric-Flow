# Stochastic Variance and Gauge-Covariant History

This document summarizes the H13.10 and H13.11 audits. These experiments are
controlled mechanism tests for low-rank GeoFlow, not production LLM optimizer
claims.

## H13.10 Motivation

H13.9D established that the inverse-Gram low-rank direction is the exact local
steepest descent direction under the split executed-information metric. H13.10
asks whether that covariance survives stochastic minibatch probing and whether
the remaining weakness is local direction quality or missing optimizer history.

The final H13.10 script is:

```bash
python experiments/h1310_final_matched_budget_two_way.py
```

The early files `h1310_stochastic_variance_decomposition.py` and
`h1310_fix_matched_budget_two_way.py` are superseded diagnostics and are not the
final H13.10 result.

## Matched Product-Displacement Protocol

All compared methods were normalized to the same first-order product-step
budget. This prevents K1 or momentum-like methods from appearing better merely
because they move farther in represented product space.

The final H13.10 setting used six trials, 100 training steps, rank 4, batch size
64, a matched first-order product-displacement budget, and batch x gauge
probing. Exact covariance probes used the full-rank ordinary-inverse branch with
zero ridge.

## Batch x Gauge Two-Way Decomposition

For each minibatch `b` and gauge representation `s`, H13.10 decomposes the
product-space direction as

```math
D_{b,s}=\mu+F_b+G_s+C_{b,s}.
```

The reported components are

```math
V_F=\mathbb E_b\|F_b\|_F^2,
\qquad
V_G=\mathbb E_s\|G_s\|_F^2,
\qquad
V_C=\mathbb E_{b,s}\|C_{b,s}\|_F^2.
```

`F_b` is minibatch functional variance, `G_s` is gauge-representation main
effect, and `C_{b,s}` is batch-gauge interaction.

## Exact vs Practical Covariance

The exact ordinary-inverse branch showed product covariance around
`epsilon_D,p99 ~= 1e-12`.

The practical ridge branch showed product covariance around
`epsilon_D,p99 ~= 1e-4`.

The practical gap is dominated by the non-covariant isotropic factor-coordinate
ridge regularizer `lambda I`, not by failure of the exact inverse-Gram
direction. A numerically stable gauge-covariant replacement for isotropic ridge
regularization remains open engineering work.

## H13.10 Results

| method | final loss | `V_F` | `V_G` | `V_C` | product gauge p99 | alignment |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW factor | 9.8968 | `2.40e4` | `1.33e5` | `8.31e4` | 0.973 | 0.9745 |
| Fixed split | 8.6235 | 0.3927 | `1.52e-23` | `3.81e-24` | `2.66e-12` | 0.9038 |
| Legacy K1, matched budget | 8.6200 | 0.3822 | `4.67e-23` | `1.44e-23` | `2.84e-12` | 0.9049 |
| Full-product corrected | 8.5053 | 0.2819 | `1.35e-24` | `6.48e-25` | `1.89e-12` | 0.8613 |
| Factor EMA split | 8.3236 | 0.01611 | `6.66e-24` | `3.12e-26` | `1.78e-12` | 0.9901 |

Core gates:

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

Under a matched first-order product-displacement budget, the exact split
GeoFlow direction remained gauge covariant to approximately `1e-12` under
stochastic minibatch probing. Gauge main-effect and batch-by-gauge interaction
variance were numerically negligible for the GeoFlow directions.

## Factor EMA Interpretation

The H13.10 method named `ema_geoflow_split` is factor-gradient EMA. It should
not be described as intrinsic quotient-history momentum. It stores history in
factor coordinates and explicitly transports that state during probes.

Factor EMA reduced functional direction variance from about `0.393` to
`0.0161`, improved full-batch alignment from about `0.904` to `0.990`, and
lowered final loss in all six trials relative to the memoryless split flow.

Legacy K1 performed almost identically to fixed split flow once the realized
product-step budget was strictly matched. Its larger advantage in earlier
unmatched audits primarily came from allocating a larger functional
displacement budget, not from changing the local direction.

The exact full-product tangent correction produced a moderate variance and loss
improvement, but substantially less than temporal averaging in this benchmark.

## H13.11 Executed-Channel History

H13.11 stores optimizer first-moment history directly in the executed split
channels

```math
C_{A,t}=B_tV_{A,t},
\qquad
C_{B,t}=V_{B,t}A_t.
```

The intrinsic channel first moments are

```math
U_{A,t}
=
\beta_1U_{A,t-1}
+
(1-\beta_1)C_{A,t},
\qquad
U_{B,t}
=
\beta_1U_{B,t-1}
+
(1-\beta_1)C_{B,t}.
```

Because `B'V_A'=BV_A` and `V_B'A'=V_BA`, `(U_A,U_B)` is gauge-visible and
gauge-invariant on the full-rank ordinary-inverse branch for gauge-invariant
losses.

## Channel-Space Lift

Channel history is lifted back to factor velocities by

```math
V_A=B^+U_A,
\qquad
V_B=U_BA^+.
```

On the full-rank ordinary-inverse branch:

```math
V_A=(B^\top B)^{-1}B^\top U_A,
\qquad
V_B=U_BA^\top(AA^\top)^{-1}.
```

The optimizer history is stored in executed functional channels rather than raw
factor coordinates.

## H13.11 Results

| method | final loss | `V_F` | product gauge p99 | alignment |
| :--- | ---: | ---: | ---: | ---: |
| AdamW factor | 9.2839 | `2.21e4` | 0.969 | 0.9746 |
| Fixed split | 7.8189 | 0.3869 | `2.09e-12` | 0.9039 |
| Factor EMA split | 7.5226 | 0.01597 | `2.35e-12` | 0.9905 |
| Channel momentum GeoFlow | 7.5048 | 0.01557 | `3.51e-12` | 0.9910 |
| Channel-adaptive GeoFlow | 7.5107 | 0.03756 | `3.89e-12` | 0.9902 |
| Full-product corrected | 7.7056 | 0.2789 | `1.02e-12` | 0.8597 |

Core gates:

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
`96%` relative to memoryless split flow:

```math
1-\frac{0.01557}{0.3869}\approx95.98\%.
```

Under the matched product-displacement budget, channel momentum improved mean
final loss from `7.8189` to `7.5048` and increased full-batch alignment from
`0.9039` to `0.9910`.

## Factor EMA Boundary

Channel momentum matched factor EMA and was slightly better on the six-trial
mean. However, channel momentum had lower loss in only two of six trials;
factor EMA was lower in four of six. The strongest established advantage is the
intrinsic representation of optimizer history, not statistically decisive task
superiority over factor EMA.

## Scalar Second-Moment Negative Result

H13.11 also tested scalar channel second moments:

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

This did not improve channel momentum. Functional variance increased from
`0.01557` to `0.03756`, and mean loss increased from `7.5048` to `7.5107`.
Gauge invariance alone is therefore insufficient to define a
geometry-compatible adaptive second moment.

## Limits

- No claim of universal superiority over AdamW.
- No production Transformer or LLM validation of channel momentum yet.
- No statistically significant superiority of channel momentum over factor EMA.
- No proof that channel first-moment history is globally optimal.
- No geometry-compatible operator-valued second moment yet.
- No gauge-covariant practical regularizer replacing isotropic ridge.
- No global convergence theorem.
- No globally shortest functional-time path.
- No full-parameter pretraining validation.
- No distributed or memory-cost analysis.

## H13.12 Proposed Coupled Covariance

A proposed next experiment is to estimate the coupled channel covariance

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

and apply

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

This is proposed next work and has not been validated.
