# H13.12 Coupled Channel Covariance — Final Results

## Status

```text
PASS_MATCHED_PRODUCT_STEP = true
PASS_COUPLED_PRODUCT_COVARIANCE = true
PASS_COUPLED_CHANNEL_COVARIANCE = true
PASS_TWO_WAY_DECOMPOSITION = true
PASS_COVARIANCE_CONDITION_FINITE = true
PASS_ALL_METHODS_IMPROVE = true
PASS_FINITE = true
PASS_CORE = true
```

Empirical hypothesis results:

```text
HYPOTHESIS_COUPLED_REDUCES_VARIANCE = false
HYPOTHESIS_COUPLED_IMPROVES_ALIGNMENT = false
HYPOTHESIS_COUPLED_IMPROVES_LOSS = true
HYPOTHESIS_COUPLED_BEATS_SCALAR_ADAPTIVE = true
HYPOTHESIS_COUPLED_BEATS_FACTOR_EMA = true
COUPLED_WINS_VS_MOMENTUM = 6
COUPLED_WINS_VS_SCALAR = 6
COUPLED_STRICT_MAJORITY_THRESHOLD = 4
HYPOTHESIS_COUPLED_MAJORITY_SEED_WINS = true
```

## Method summary

| method | final loss | matched step | functional variance | product gauge p99 | alignment | mean abs channel correlation | mean covariance condition |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW factor | 9.283884 | 0.050116 | 2.208e4 | 0.9687 | 0.9746 | 0.423 | 3.01 |
| Fixed split | 7.818892 | 0.050266 | 0.3869 | 2.087e-12 | 0.9039 | 0.540 | 4.23 |
| Factor EMA split | 7.522583 | 0.050218 | 0.01597 | 2.348e-12 | 0.9905 | 0.549 | 4.31 |
| Channel momentum | 7.504751 | 0.050226 | 0.01557 | 3.513e-12 | 0.9910 | 0.547 | 4.34 |
| Scalar channel adaptive | 7.510665 | 0.050229 | 0.03756 | 3.889e-12 | 0.9902 | 0.543 | 4.31 |
| Coupled channel covariance | 7.290806 | 0.050237 | 0.03487 | 1.036e-12 | 0.9903 | 0.597 | 5.61 |
| Full-product corrected | 7.716226 | 0.050104 | 0.2836 | 6.571e-12 | 0.8587 | 0.633 | 6.68 |

## Main conclusion

The coupled 2x2 channel covariance method preserved exact product and channel
gauge covariance to approximately 1e-12 and achieved the lowest mean final loss.

It beat channel momentum in 6/6 trials and scalar channel adaptation in 6/6
trials under a matched first-order product-displacement budget.

The gain did not come from lower functional variance or higher full-batch
alignment. Instead, the coupled preconditioner appears to improve the relative
allocation of motion across the two executed channels.

## Important boundary

Do not claim universal optimizer superiority. This remains a controlled
low-rank regression mechanism audit. No Transformer or production LoRA result
has yet been established for H13.12.
