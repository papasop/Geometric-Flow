# H13.4 paper update

## Claim to use

For gauge-equivalent LoRA factorizations representing the same initial product
\(BA\), AdamW generated systematically different full-product update
directions, magnitudes, and trajectories, with deviations increasing
monotonically with gauge condition number. Capacity preserved full-product
update direction to numerical precision and maintained magnitude and
trajectory discrepancies near \(10^{-5}\) across
\(\kappa=5,10,100,1000\).

## Claim not to use

Do not write that Capacity is mathematically or exactly gauge invariant.
H13.4 is empirical evidence from GPT-2, WikiText-2, rank-4 LoRA, two adapted
attention modules, three seeds, and 40 optimization steps.

## Result table

| Optimizer | kappa | Direction error | Log-magnitude error | Final trajectory gap |
|---|---:|---:|---:|---:|
| AdamW | 5 | 0.1525 | 0.1458 | 0.5383 |
| AdamW | 10 | 0.2270 | 0.2692 | 0.7156 |
| AdamW | 100 | 0.5080 | 0.9024 | 1.8297 |
| AdamW | 1000 | 0.7247 | 1.8387 | 4.9009 |
| Capacity | 5 | 0.0 | 1.322e-5 | 1.250e-5 |
| Capacity | 10 | 0.0 | 6.665e-6 | 1.078e-5 |
| Capacity | 100 | 0.0 | 2.081e-5 | 1.603e-5 |
| Capacity | 1000 | 0.0 | 1.671e-5 | 1.259e-5 |

The direction error displayed as zero is the result of cosine clamping at
floating-point precision; describe it as “numerically indistinguishable from
zero,” not as an exact symbolic zero.
