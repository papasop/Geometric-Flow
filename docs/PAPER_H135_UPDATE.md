# H13.5 paper update

## Supported conclusion

Product-preserving factor rebalancing reduced gauge-induced trajectory
divergence for coordinate optimizers, especially stateless SGD, but was not
sufficient to restore full-product gauge-equivariant learning dynamics.
AdamW remained strongly representation-dependent even with per-step
rebalancing, consistent with its untransported coordinate-dependent moment
states. Capacity was the only tested method whose complete product-space
direction and trajectory remained numerically stable across
\(\kappa=5,100,1000\).

## Mean final trajectory gaps

| Method | kappa=5 | kappa=100 | kappa=1000 |
|---|---:|---:|---:|
| AdamW | 0.5756 | 2.2167 | 6.2820 |
| AdamW + rebalance every step | 0.4948 | 0.8339 | 1.7722 |
| AdamW + rebalance every 10 steps | 0.6027 | 1.7465 | 4.9815 |
| SGD | 0.01391 | 0.3199 | 2.4718 |
| SGD + rebalance every step | 8.879e-4 | 0.02102 | 0.2121 |
| Capacity | 1.231e-5 | 1.552e-5 | 1.191e-5 |

## Wording boundaries

Use:

- empirical full-product gauge equivariance;
- naive product-preserving rebalancing counterfactual;
- evidence that quotient-aware vector fields matter beyond canonicalization.

Do not use:

- exact mathematical proof;
- universal optimizer superiority;
- proof that factor balancing is irrelevant;
- proof that AdamW moment transport cannot work.

The experiment shows only that naive rebalancing without optimizer-state
transport is insufficient.
