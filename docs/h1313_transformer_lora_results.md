# H13.13 Transformer/LoRA Validation Results

## Scope

H13.13 transfers the H13.12 coupled executed-channel covariance mechanism from
single low-rank matrix regression to a genuine multi-layer causal Transformer
with frozen base weights and LoRA-only training.

The experiments are offline teacher-student audits. They are not GPT-2,
WikiText, pretrained-language-model, or production LLM validations.

## H13.13A smoke audit

Configuration:

- two-layer causal Transformer;
- LoRA rank 4;
- all-linear LoRA scope;
- 2 trials;
- 80 steps;
- shared LoRA product initialization;
- shared minibatch schedule;
- matched global realized LoRA-product displacement.

Structural result:

```text
PASS_SAME_INITIAL_PRODUCT = true
PASS_SAME_BATCH_SCHEDULE = true
PASS_MATCHED_PRODUCT_STEP = true
PASS_COUPLED_GAUGE_COVARIANCE = true
PASS_FINITE_TRAINING = true
PASS_NO_LAYER_SKIPS = true
PASS_CORE = true
```

Mean validation losses:

| method | validation loss |
|---|---:|
| AdamW LoRA | 0.1111003 |
| Fixed split | 0.1089613 |
| Factor EMA | 0.0927855 |
| Channel momentum | 0.0914212 |
| Scalar channel adaptive | 0.0908687 |
| Coupled channel covariance | 0.0881435 |

Coupled covariance beat channel momentum in 2/2 trials.

## H13.13B formal validation

Configuration:

- two-layer causal Transformer;
- hidden size 96;
- four attention heads;
- LoRA rank 4;
- all-linear LoRA scope;
- 6 trials;
- 200 steps;
- 1024 training samples;
- 256 validation samples;
- shared LoRA product initialization;
- shared minibatch schedule;
- matched global realized LoRA-product displacement.

Structural gates:

```text
PASS_SAME_INITIAL_PRODUCT = true
PASS_SAME_BATCH_SCHEDULE = true
PASS_MATCHED_PRODUCT_STEP = true
PASS_COUPLED_GAUGE_COVARIANCE = true
PASS_FINITE_TRAINING = true
PASS_NO_LAYER_SKIPS = true
PASS_CORE = true
```

Mean validation losses:

| method | validation loss | product gauge p99 | mean abs channel correlation | mean covariance condition | wall time |
|---|---:|---:|---:|---:|---:|
| AdamW LoRA | 0.06413484 | 0.9670 | 0.206 | 1.79 | 23.8 s |
| Fixed split | 0.07986790 | 3.763e-14 | 0.284 | 4.01 | 24.3 s |
| Factor EMA | 0.05023424 | 3.132e-13 | 0.209 | 2.04 | 24.4 s |
| Channel momentum | 0.04852739 | 6.157e-14 | 0.215 | 1.98 | 26.0 s |
| Scalar channel adaptive | 0.04783990 | 5.970e-14 | 0.207 | 1.78 | 26.2 s |
| Coupled channel covariance | 0.04542299 | 1.810e-13 | 0.185 | 1.77 | 26.7 s |

Paired result against channel momentum:

```text
COUPLED_WINS_VS_MOMENTUM = 6
COUPLED_MAJORITY_THRESHOLD = 4
HYPOTHESIS_COUPLED_MAJORITY_SEED_WINS = true
PAIRED_MEAN_VAL_LOSS_ADVANTAGE = 0.003104392770511896
PAIRED_MEDIAN_VAL_LOSS_ADVANTAGE = 0.002480712198447751
PAIRED_STANDARDIZED_EFFECT = 1.4586202941737914
HYPOTHESIS_POSITIVE_PAIRED_EFFECT = true
```

Relative mean improvement of coupled covariance over channel momentum:

```math
\frac{0.04852739-0.04542299}{0.04852739}
\approx 6.40\%.
```

Relative mean improvement over factor EMA:

```math
\frac{0.05023424-0.04542299}{0.05023424}
\approx 9.58\%.
```

## Interpretation

The H13.12 coupled-channel mechanism transferred beyond single-matrix
regression. Under identical represented initialization, shared batch schedules,
and matched functional displacement, coupled covariance achieved lower
validation loss than channel momentum in all six Transformer trials while
retaining layerwise gauge covariance near machine precision.

Scalar channel adaptation also slightly improved over channel momentum in this
Transformer setting. Therefore the correct conclusion is not that scalar
adaptation is universally harmful. The stronger observed conclusion is that
retaining cross-channel covariance was more consistently effective.

The lower mean absolute channel correlation and lower covariance condition
observed for the coupled method are suggestive of dynamic channel
decorrelation, but they do not yet establish causality.

## Limits

Do not claim:

- universal superiority over AdamW;
- production Transformer or LLM validation;
- GPT-2 or WikiText validation;
- proof of globally optimal operator-valued moments;
- statistical finality from six seeds;
- a proven causal link between lower channel correlation and lower loss.

## Next ablations

1. `attention_only`, six trials;
2. `qkv_only`, six trials;
3. rank sweep `2,4,8`;
4. checkpoint-wise channel correlation, condition number, and progress analysis;
5. GPT-2 small / WikiText-2 only after the controlled ablations are complete.
