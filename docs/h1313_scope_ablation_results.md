# H13.13C/D Transformer/LoRA Scope Ablations

## Scope

These experiments localize the H13.13 coupled-channel advantage inside the
attention stack.

- H13.13C trains attention `qkv` and attention output `proj` LoRA modules.
- H13.13D-FIX trains only the attention `qkv` LoRA modules.
- Both use six paired seeds, 200 steps, shared represented initialization,
  shared minibatch schedules, and matched global realized LoRA-product
  displacement.

The first attempted H13.13D script was invalid because a global text replacement
duplicated the `qkv_only` branch and accidentally executed the
`attention_only` scope. Those duplicated numerical results must not be committed
or cited. Only `h1313d_fix_qkv_only_ablation.py` and its results are valid.

## H13.13C attention-only ablation

Actual active modules:

```text
blocks.*.qkv
blocks.*.proj
```

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
| AdamW LoRA | 0.02335560 | 0.9670 | 0.198 | 1.68 | 13.2 s |
| Fixed split | 0.03548125 | 2.855e-14 | 0.245 | 3.21 | 13.3 s |
| Factor EMA | 0.01535773 | 3.415e-13 | 0.210 | 1.76 | 13.7 s |
| Channel momentum | 0.01541905 | 7.315e-14 | 0.200 | 1.76 | 14.2 s |
| Scalar channel adaptive | 0.01507937 | 7.921e-14 | 0.201 | 1.67 | 14.0 s |
| Coupled channel covariance | 0.01387520 | 9.127e-14 | 0.191 | 1.64 | 14.2 s |

Paired result against channel momentum:

```text
COUPLED_WINS_VS_MOMENTUM = 6
COUPLED_MAJORITY_THRESHOLD = 4
HYPOTHESIS_COUPLED_MAJORITY_SEED_WINS = true
PAIRED_MEAN_VAL_LOSS_ADVANTAGE = 0.0015438482080169927
PAIRED_MEDIAN_VAL_LOSS_ADVANTAGE = 0.001606890178719905
PAIRED_STANDARDIZED_EFFECT = 2.230602901110854
HYPOTHESIS_POSITIVE_PAIRED_EFFECT = true
```

Relative mean improvement over channel momentum:

```math
\frac{0.01541905-0.01387520}{0.01541905}
\approx 10.01\%.
```

Interpretation:

- the coupled-channel advantage is present in attention-only LoRA;
- it does not require MLP or output-head adaptation;
- the six-seed consistency is stronger than in the all-linear audit.

## H13.13D-FIX qkv-only ablation

Actual active modules:

```text
blocks.0.qkv
blocks.1.qkv
N_ACTIVE_LORA_LAYERS = 2
```

Structural gates:

```text
PASS_SAME_INITIAL_PRODUCT = true
PASS_SAME_BATCH_SCHEDULE = true
PASS_MATCHED_PRODUCT_STEP = true
PASS_COUPLED_GAUGE_COVARIANCE = true
PASS_FINITE_TRAINING = true
PASS_NO_LAYER_SKIPS = true
PASS_QKV_SCOPE_EXACT = true
PASS_CORE = true
```

Mean validation losses:

| method | validation loss | product gauge p99 | mean abs channel correlation | mean covariance condition | wall time |
|---|---:|---:|---:|---:|---:|
| AdamW LoRA | 0.01116509 | 0.9564 | 0.193 | 1.81 | 9.3 s |
| Fixed split | 0.01494303 | 2.439e-14 | 0.278 | 2.65 | 9.4 s |
| Factor EMA | 0.005239425 | 3.882e-14 | 0.240 | 1.87 | 9.4 s |
| Channel momentum | 0.005061230 | 2.540e-14 | 0.251 | 1.93 | 9.9 s |
| Scalar channel adaptive | 0.005350084 | 3.288e-14 | 0.232 | 1.69 | 9.7 s |
| Coupled channel covariance | 0.004449028 | 8.841e-14 | 0.229 | 1.67 | 9.8 s |

Paired result against channel momentum:

```text
COUPLED_WINS_VS_MOMENTUM = 5
COUPLED_MAJORITY_THRESHOLD = 4
HYPOTHESIS_COUPLED_MAJORITY_SEED_WINS = true
PAIRED_MEAN_VAL_LOSS_ADVANTAGE = 0.0006122018334238747
PAIRED_MEDIAN_VAL_LOSS_ADVANTAGE = 0.000482815703360109
PAIRED_STANDARDIZED_EFFECT = 0.9458912131900352
HYPOTHESIS_POSITIVE_PAIRED_EFFECT = true
```

Relative mean improvement over channel momentum:

```math
\frac{0.005061230-0.004449028}{0.005061230}
\approx 12.10\%.
```

Interpretation:

- the coupled-channel advantage is already present in QKV-only LoRA;
- the attention output projection is not necessary for the effect to appear;
- adding the output projection appears to improve seed-wise consistency:
  `5/6` for QKV-only versus `6/6` for attention-only;
- this does not establish that output projection causally stabilizes the method.

## Scope comparison

| scope | wins vs momentum | relative mean improvement | paired standardized effect |
|---|---:|---:|---:|
| all-linear H13.13B | 6/6 | 6.40% | 1.46 |
| attention-only H13.13C | 6/6 | 10.01% | 2.23 |
| qkv-only H13.13D-FIX | 5/6 | 12.10% | 0.95 |

## Updated conclusion

The coupled-channel advantage does not depend on MLP or language-head LoRA.
It is already visible in QKV-only adaptation. Including the attention output
projection increased seed-wise consistency in the tested six-seed audit.

Across all three Transformer scopes, coupled covariance preserved exact
product-gauge covariance near machine precision and achieved lower mean
validation loss than channel momentum.

## Limits

Do not claim:

- that output projection is proven to causally stabilize coupled covariance;
- universal superiority over AdamW;
- pretrained-model, GPT-2, WikiText, or production LLM validation;
- statistical finality from six seeds;
- broad rank-independence;
- that the invalid first H13.13D run was a genuine QKV result.

## Next research direction

Run an attention-only LoRA rank sweep at ranks `2,4,8`, followed by
checkpoint-wise channel-correlation, covariance-condition, and progress
analysis.
