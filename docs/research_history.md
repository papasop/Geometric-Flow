# Research History And Archived Results

This file preserves the research trail behind the current GeoFlow README. These
results are useful for auditability, but they are not the fastest onboarding
path for new users.

## Gauge Sensitivity Convention

For a fixed seed and optimizer, gauge sensitivity is the mean pairwise distance
between final functional representations produced from gauge-equivalent
parameterizations:

```text
S(seed, optimizer)
  = mean_{i < j} ||Phi(seed, optimizer, representation_i)
                  - Phi(seed, optimizer, representation_j)||_2
```

Ratios are formed within seed and then summarized across seeds. Function
distances between different seeds are not gauge sensitivity, because those runs
may differ in model initialization, data, and task realization.

## Phase F: LoRA Gauge Stability

Phase F tested functional quotient geometry on small LoRA adapters with exact
gauge-equivalent initializations:

```text
A -> S A
B -> B S^{-1}
```

Observed structural results:

- 12 targeted configurations.
- 5 seeds and 5 gauge-equivalent representations.
- 900 total runs.
- Initial functional equivalence residuals below the `1e-7` scale.
- Matched per-seed functional/diagonal sensitivity ratio about `0.536`.
- Matched 95% confidence interval entirely below `1`.
- Tangent drift ratio about `0.316`.
- Near-null amplification ratio about `0.361`.

Negative result: Phase F did not establish task superiority. Functional loss
was higher than the diagonal baseline, functional accuracy was lower, and
wall-clock cost was much higher. Phase F establishes LoRA gauge stability,
tangent suppression, and near-null suppression.

## Controlled LoRA Architecture

The controlled LoRA experiments use:

```text
z(x) = (W0 + B A) x
h(x) = tanh(z(x))
f_theta(x) = W_head h(x) + b_head
```

`W0` is frozen. `A` has shape `rank x input_dim`; `B` has shape
`hidden_dim x rank`. The dense output head is `W_head, b_head`.

Training scopes:

| scope | trainable parameters |
| --- | --- |
| `lora_only` | `A, B` only |
| `head_only` | output head only |
| `lora_and_head` | both LoRA factors and head |

Functional maps:

| functional map | definition |
| --- | --- |
| `lora_output` | `z(x)` |
| `hidden` | `h(x)` |
| `logits` | `f_theta(x)` |
| `logits_hidden` | concatenation of logits and hidden features |

Phase G uses `lora_only` as the primary setting because the gauge symmetry
belongs to `B A`, not the dense head.

## Phase G: Matched Functional-Step Calibration

Phase G compares actual movement in function space:

```text
functional_step_norm = ||Phi(theta_after) - Phi(theta_before)||_2
```

The benchmark calibrates the functional GeoFlow update so its observed
functional displacement matches a reference optimizer. Calibration is done on
training/probe batches, never test loss.

Corrected smoke result:

| metric | value |
| --- | ---: |
| matched-step within-seed sensitivity | `0.00399` |
| diagonal within-seed sensitivity | `0.00459` |
| mean matched/diagonal sensitivity ratio | `0.939` |
| structural seed win rate | `0.50` |
| mean functional-step calibration error | `0.000727` |
| mean null leakage | `2.0e-08` |

The corrected smoke supports calibration and null control, but not robust
structural or task advantage.

Corrected Phase G B2 long-run:

| metric | result |
| --- | ---: |
| matched-step sensitivity | `0.2772` |
| diagonal sensitivity | `0.8154` |
| mean matched/diagonal ratio | `0.3385` |
| 95% CI | `[0.2760, 0.4329]` |
| structural seed win rate | `1.00` |
| tangent suppression | passed |
| calibration error | `0.000848` |
| null leakage | `1.68e-08` |
| matched/diagonal wall-clock ratio | `1.38` |

Structural gates passed:

- `STRUCTURAL_SENSITIVITY_PASS=True`
- `STRUCTURAL_WIN_RATE_PASS=True`
- `STRICT_STRUCTURAL_CI_PASS=True`
- `TANGENT_SUPPRESSION_PASS=True`

Task-level gates failed:

- `TASK_GAP_REDUCED_PASS=False`
- `TASK_PARITY_PASS=False`
- `TASK_ADVANTAGE_PASS=False`

The supported conclusion is structural: matched-step GeoFlow reduces
sensitivity to LoRA gauge parameterization, but this did not translate into
better task optimization in that controlled setting.

Reanalyze Phase G artifacts:

```bash
python experiments/analyze_phase_g_results.py \
  --artifact-dir artifacts/phase_g_formal \
  --out artifacts/phase_g_formal/reanalysis
```

## Transformer Layerwise Projection

A small causal Transformer with 2 layers, 4 heads, `d_model=24`, and LoRA rank
`3` was tested on a synthetic next-token task.

| optimizer | mean loss | mean accuracy | mean alpha | step loss change |
| :--- | ---: | ---: | ---: | ---: |
| `adam_raw` | `1.7021` | `79.45%` | `0.000` | `-0.0171` |
| `layerwise_projected` | `1.7003` | `79.60%` | `1.000` | `-0.0175` |
| `hybrid_fixed` | `1.7011` | `79.56%` | `0.500` | `-0.0173` |
| `hybrid_loss_aware` | `1.6999` | `79.59%` | `0.826` | `-0.0175` |

This is bounded evidence: useful small controlled task behavior, not a
large-language-model claim.

## D7 Fixed-Rank Backend

D7 validates `FixedRankFunctionalAdam` as a scientific regression benchmark.

| metric | D7 result |
| --- | ---: |
| `rank_tangent_trust` mean loss | `1.710495` |
| factor Adam mean loss | `1.711612` |
| paired loss gap | `-0.001117` |
| 95% CI | `[-0.003562, 0.000925]` |
| logit sensitivity ratio | `7.3e-5` |
| structural win rate | `1.0` |
| tangent residual | `~3e-6` |
| rank violations | `0` |

Quick smoke:

```bash
python experiments/d7_fixed_rank_tangent_benchmark.py \
  --seeds 101 \
  --representations 2 \
  --steps 5 \
  --out-dir artifacts/d7_smoke
```

Full D7-style run:

```bash
python experiments/d7_fixed_rank_tangent_benchmark.py \
  --seeds 101,211,307 \
  --representations 4 \
  --steps 80 \
  --out-dir artifacts/d7_fixed_rank
```

## H10 Substepped Quotient Flow

H10 introduced `SubsteppedQuotientFlow`, a gauge-equivariant
Gram-preconditioned factor flow with fresh-gradient substep integration.

H10.4/H10.5 established Adam-scale progress and mapped the early
progress-versus-gauge trade-off. Before progress-budgeted stopping, the best
fast aggregate gauge reduction was about `7.29x`; `K=4` was more favorable than
`K=2`; and higher LR often improved product-space progress while worsening
gauge suppression.

H10.6 introduced progress-budgeted stopping so quotient-flow and factor-Adam
trajectories were compared at comparable functional progress. With fixed
`macro_lr=2.6` and `substeps=16` across three seeds, the fast benchmark
obtained:

- mean loss-progress ratio: `1.846`;
- mean product-displacement ratio: `0.773`;
- geometric-mean gauge-divergence ratio: `0.0656`;
- geometric-mean gauge suppression: `15.23x`;
- matched-progress pass on all three seeds;
- no pseudoinverse fallback;
- product-preserving balance pass.

This is the first fast H10 configuration to pass the strict `10x`
gauge-suppression gate. The method remains experimental and opt-in, and the
result required held-out-seed confirmation before broader claims.

H10.7 held-out confirmation used the same fixed configuration
(`macro_lr=2.6`, `substeps=16`) on five previously unseen seeds. It obtained:

- matched progress on all five seeds;
- lower gauge divergence than factor Adam on all five seeds;
- geometric-mean gauge suppression of `12.84x`;
- `60%` of seeds individually exceeding `10x`;
- bootstrap 95% CI for suppression of approximately `[8.21x, 20.08x]`.

Thus the mean `10x` gate was confirmed, but the stricter per-seed and
bootstrap-CI confirmation gates were not passed.

Reproduce the progress-budgeted screen:

```bash
python experiments/h10_progress_budget_benchmark.py \
  --macro-lr 2.6 \
  --substeps 16 \
  --out-dir artifacts/h10_progress_budget
```

This repository script uses a tiny GPT-style LoRA model and validates the
mechanism, matched-progress gate, product/logit gauge metrics, and summary
format. It does not instantiate Hugging Face GPT-2 and should not be described
as an exact reproduction of the GPT-2 H10.6/H10.7 runs above.
