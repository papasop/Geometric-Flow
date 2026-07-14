# GeoFlow for PyTorch

GeoFlow is a research toolkit for geometry-aware optimization under redundant
neural-network parameterizations.

Its central principle is that an optimizer output is a proposal, not
automatically an admissible functional update. The library contains:

- functional stable-neutral geometry based on the Jacobian of a configurable
  function-space representation;
- matrix-free research solvers for projected response directions;
- an opt-in fixed-rank backend that performs Adam updates in invariant product
  coordinates, projects them into the fixed-rank tangent space, and applies
  rank-preserving retraction;
- an experimental substepped quotient-flow integrator that recomputes gradients
  across local LoRA factor updates without Adam-style moment tensors.

The fixed-rank backend has demonstrated near-exact LoRA gauge invariance and
task parity in a small synthetic Transformer benchmark. It is experimental, not
a production large-model optimizer, and not evidence of universal superiority
over Adam.

## Why GeoFlow?

Most deep learning optimizers, including SGD and Adam, navigate parameter space
using gradients alone. They know which way is downhill, but they do not directly
measure how the terrain bends.

The legacy optimizer measures local curvature with Hessian or gradient-square
approximations. The theory-aligned path instead builds a functional map
`Phi(theta; X_probe) = vec(model(X_probe))`, computes its Jacobian, separates
neutral reparameterization directions from normal functional directions, and
updates only through the normal response operator.

The project was motivated in part by geometry-aware methods in quantum control,
but the PyTorch results in this repository are evaluated independently and
should not be interpreted as quantum-computing performance claims.

## Core Principle: Optimizer Outputs Are Proposals, Not Final Updates

Conventional optimizers treat their parameter-space output as the final
executable update. GeoFlow treats that output as a **proposal** that should be
interpreted through the geometry of functional equivalence classes before it is
applied.

For LoRA fine-tuning, multiple parameter pairs `(A, B)` may represent the same
functional update matrix:

```text
M = B A
```

Infinitesimal motions along the reparameterization orbit,

```text
delta A = Omega A
delta B = -B Omega
```

change the parameter representation without changing `M`, and therefore do not
change the represented model function.

A geometry-aware optimizer should therefore:

1. Generate a candidate direction using a conventional optimizer.
2. Identify components associated with functionally equivalent
   reparameterizations.
3. Remove, quotient out, or avoid those ineffective directions.
4. Apply an update defined in the transverse functional space.

The methodological shift is:

> The optimizer output is not assumed to be the final update. It is a proposal
> that must be filtered, projected, or lifted through the geometry of functional
> equivalence classes.

### Important Distinction

A simple Euclidean projection of an Adam update can remove an explicit
tangential component, but it does **not** in general make Adam gauge-invariant.
Adam itself is coordinate-dependent, and the Euclidean normal space also changes
under non-orthogonal LoRA reparameterizations.

For exact LoRA gauge invariance, the preferred formulations operate directly on
the invariant product `M = B A`, or use a gauge-invariant quotient metric.

In short: traditional optimizers update parameters; this framework seeks to
update the function represented by those parameters.

## Experimental Backend: Fixed-Rank Functional Adam [Status: Opt-In]

The D7 kernel adds an experimental optimizer backend for LoRA-style fixed-rank
product states:

```text
FixedRankFunctionalAdam
```

Instead of optimizing factor coordinates directly, this backend treats the
invariant product `M = B A` as the optimizer state. Each product-space Adam
proposal is projected into the tangent space of the fixed-rank manifold and then
returned to rank `r` by a rank-preserving SVD retraction.

| feature | benefit |
| :--- | :--- |
| Invariant product state | Optimizer moments live in `M = B A` coordinates, not gauge-dependent factor coordinates. |
| Fixed-rank tangent projection | Ambient proposals are filtered before they become executable updates. |
| Rank-preserving retraction | `M + D_tangent` is returned to rank `r` with best-rank SVD truncation. |
| Held-out trust calibration | Optional scale selection evaluates candidate product updates on caller-supplied calibration data. |

| method | structural advantage | task advantage | speed | verdict |
| :--- | :--- | :--- | :--- | :--- |
| `factor_adam` / `adam_raw` | No | Baseline | Fastest | Good control baseline. |
| Euclidean factor-space projection | Removes an explicit tangent component but is not gauge invariant | Slight mean task improvement in one controlled benchmark; not established as a general advantage | Moderate | Historical baseline. |
| `FixedRankFunctionalAdam` | Stronger by construction | D7 showed task parity only in a small synthetic Transformer benchmark | Experimental | Opt-in backend for fixed-rank product-coordinate experiments. |
| `SubsteppedQuotientFlow` | Factorized quotient vector field with gradient recomputation per substep | H10.4 reached Adam-scale progress but missed the strict 10x gauge-suppression gate | Experimental | Opt-in integrator; no Adam moments. |
| Full quotient-space methods | Strong | Not yet | Slower | Research reference, not a production default. |

This path is experimental and opt-in. It does not change the default behavior
of `GeometricOptimizer`, and it does not automatically rewrite arbitrary LoRA
factor modules to consume explicit product states. The model forward pass must
explicitly consume the same product tensor `M` whose gradient is passed to the
optimizer. See `experiments/d7_fixed_rank_tangent_benchmark.py` for a complete
runnable example.

## Experimental: Substepped Quotient Flow [Status: Opt-In]

`SubsteppedQuotientFlow` integrates the factorized quotient vector field using
repeated small factor-space substeps. It is intended for LoRA-style modules that
expose trainable `A` and `B` parameters with shapes `(rank, input_dim)` and
`(output_dim, rank)`.

For a macro step with `K` substeps, the local learning rate is:

```text
local_lr = macro_lr / K
```

For the LoRA convention `M = B A`, where `A` has shape
`rank x input_dim` and `B` has shape `output_dim x rank`, each local
quotient-preconditioned step is:

```math
\Delta A =
-\eta_{\mathrm{local}}
(B^\top B)^{-1}\nabla_A L,
\qquad
\Delta B =
-\eta_{\mathrm{local}}
\nabla_B L(AA^\top)^{-1}.
```

Under the gauge transformation

```math
A \mapsto S A,
\qquad
B \mapsto B S^{-1},
```

the directions transform covariantly:

```math
\Delta A \mapsto S\Delta A,
\qquad
\Delta B \mapsto \Delta B S^{-1}.
```

For full-rank factors on the ordinary-inverse branch, the represented product
trajectory is gauge-equivariant in exact arithmetic. When an ill-conditioned
Gram matrix triggers the Moore-Penrose pseudoinverse fallback, exact covariance
is not generally guaranteed under arbitrary non-orthogonal gauge transforms.
In finite precision, diagnostics such as `condition_max`, `fallback_count`,
and `balance_residual_max` should still be monitored.

Each substep uses freshly supplied factor gradients, applies the quotient
preconditioned directions, optionally clips the global quotient update, updates
the factors, and optionally applies product-preserving QR canonicalization
without changing the represented product `B A`.

This optimizer has **no Adam-style persistent first or second moments**.
It stores scalar diagnostics only, such as `condition_max`, `fallback_count`,
`balance_residual_max`, `last_update_norm`, and `last_clip_scale`. Temporary
rank-by-rank Gram matrices are used to compute the quotient direction. Scalar
diagnostics are runtime counters and are not guaranteed to persist across
optimizer checkpoint restoration.

Example using benchmark-scale values from H10.4, not universal defaults:

```python
optimizer = SubsteppedQuotientFlow(
    factor_modules,
    macro_lr=3.0,
    substeps=4,
    clip_norm=1.0,
    balance_after_substep=True,
)

def closure():
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    return loss

loss = optimizer.macro_step(closure)
```

H10.4 on a small GPT-2 LoRA benchmark obtained Adam-scale loss progress and
product displacement. Mean gauge divergence was approximately 6-7x lower than
factor Adam, but the strict 10x gauge-suppression gate was not passed. The best
configurations were often at the macro-LR search boundary. This feature is
therefore experimental and opt-in; no production or generalization claim is
made.

Subsequent H10.5 tuning refined, but did not overturn, that status. Increasing
macro learning rates above 3 did not remove the progress-versus-gauge trade-off;
`K=4` substeps were more favorable for gauge suppression than `K=2`; the best
fast aggregate remained approximately 7.29x gauge reduction; and higher LR
usually improved product-space progress while worsening gauge suppression.
These settings are benchmark diagnostics, not universal defaults.

Terminology boundary: this is a gauge-equivariant, quotient-compatible
Gram-preconditioned factor-flow integrator. The repository does not yet prove
that it is the unique quotient-Riemannian gradient, a strict horizontal lift, or
the standard fixed-rank quotient-manifold optimizer.

## Transformer-Ready Geometry [Status: Small Controlled Evidence]

A new experimental path applies the stable-neutral decomposition inside a small
Transformer's LoRA layers. Instead of using a global geometric optimizer, this
method modifies each LoRA update at the layer level by projecting out
gauge-equivalent, redundant parameter directions.

The benchmark uses a small causal Transformer with 2 layers, 4 heads,
`d_model=24`, and LoRA rank `3` on a synthetic next-token task.

| mode | description |
| :--- | :--- |
| `adam_raw` | Standard Adam baseline with no geometry |
| `layerwise_projected` | Full layerwise normal projection, `alpha=1` |
| `hybrid_fixed` | Adam plus projected direction with fixed `alpha=0.5` |
| `hybrid_loss_aware` | Adam plus projected direction with `alpha` chosen per step to minimize batch loss |

Observed results from 3 seeds and 4 gauge-equivalent representations:

| optimizer | mean loss | mean accuracy | mean alpha | step loss change |
| :--- | ---: | ---: | ---: | ---: |
| `adam_raw` | `1.7021` | `79.45%` | `0.000` | `-0.0171` |
| `layerwise_projected` | `1.7003` | `79.60%` | `1.000` | `-0.0175` |
| `hybrid_fixed` | `1.7011` | `79.56%` | `0.500` | `-0.0173` |
| `hybrid_loss_aware` | `1.6999` | `79.59%` | `0.826` | `-0.0175` |

Key takeaways:

- **Layerwise projection did not harm training.** Full projection matched or
  slightly improved over Adam in both loss and accuracy.
- **Loss-aware adaptive mixing worked in this controlled setting.**
  `hybrid_loss_aware` reached the lowest mean loss, and `alpha ~= 0.83`
  suggests a stable preference for geometry-dominant updates.
- **The claim is still bounded.** This is a small controlled Transformer result,
  not a large-language-model or broad optimizer-superiority claim.

## Claims Boundary

**Established so far:**

- **Fixed-rank kernel:** product-coordinate tangent projection and
  rank-preserving retraction are available as an experimental backend.
- **Small Transformer task result:** layerwise LoRA gauge projection with
  loss-aware mixing improved mean loss in the controlled synthetic
  next-token benchmark.
- **Structural result:** functional quotient methods reduce LoRA gauge
  sensitivity and suppress tangent / near-null motion in controlled settings.
- **Solver result:** matrix-free functional quotient directions match dense
  small-toy references under regression tests.

**Not established:**

- **General task superiority:** results do not prove a universally better
  optimizer.
- **Broad AdamW competitiveness:** broad task and hyperparameter comparisons are
  still missing.
- **Large-model scalability:** no GPT-2, LLM, or production large-model claim is
  made here.
- **Functional-step task improvement:** corrected Phase G matched-step
  calibration improved structure but did not reduce the task gap.
- **D7 general task superiority:** the fixed-rank backend has not established
  universal superiority; D7 demonstrated task parity only in a small synthetic
  Transformer benchmark.
- **H10.4/H10.5 strict gauge gate:** `SubsteppedQuotientFlow` reached
  Adam-scale progress in a small GPT-2 LoRA benchmark and H10.5 tuning improved
  the trade-off map, but the strict 10x gauge-suppression gate remains unmet.
- **Strict Transformer structural pass:** the Transformer layerwise projection
  path has task evidence, but not a strict Phase G structural CI pass.

Avoid interpreting these experiments as a production-ready large-model
optimizer, a proven generalization improvement, or a quantum advantage claim.

## One-Command Quickstart

### Experimental Path: Fixed-Rank Product Updates

For new fixed-rank experiments, start from explicit product-coordinate state.
The optimizer expects gradients on product matrices `M`, not hidden gradients on
factor tensors `A` and `B`. The model forward pass must explicitly consume the
same product tensor `M`; `ProductState.from_lora_modules()` can create product
variables, but it does not automatically rewire third-party LoRA modules.

Run the complete fixed-rank example:

```bash
python experiments/d7_fixed_rank_tangent_benchmark.py \
  --seeds 101 \
  --representations 2 \
  --steps 5 \
  --out-dir artifacts/d7_smoke
```

Use the historical scripts below when you want to reproduce the research trail,
diagnostics, or legacy baselines.

### Run In Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/papasop/Geometric-Flow/blob/main/notebooks/run_cifar10_benchmark.ipynb)

Open the notebook and run the first cell. It clones the repository, installs
dependencies, downloads CIFAR-10, and writes benchmark results to
`artifacts/cifar10_benchmark_results.csv`.

### Run Locally

Fast 50-step synthetic CIFAR-10 smoke test, with no dataset download:

```bash
git clone https://github.com/papasop/Geometric-Flow.git
cd Geometric-Flow
pip install -e .
python experiments/train_cifar10_geo.py \
  --dataset synthetic \
  --mode all \
  --trials 1 \
  --steps 50 \
  --adam-warmup-steps 48 \
  --seed 32 \
  --train-samples 256 \
  --eval-samples 128 \
  --batch-size 32 \
  --channels 16 \
  --use-grad-square \
  --preconditioner diagonal \
  --precond-scale 0.5 \
  --max-grad-norm 2.0 \
  --grad-smoothing 0.0 \
  --out artifacts/synthetic_cifar_milestone.csv
```

Expected shape of the output:

```text
optimizer  mean_acc  std_acc  mean_loss  std_loss  mean_sec  ratio
adam          ...
geometric     ...
hybrid        ...
wrote artifacts/synthetic_cifar_milestone.csv
```

With the fixed seed above, the smoke test should visibly exercise the hybrid
path and show Hybrid above Adam. Exact numbers can vary by PyTorch version and
hardware.

## Implementation Tiers

| tier | name | status |
| --- | --- | --- |
| Baseline | `adam` | First-order optimizer control |
| Legacy heuristic | `diagonal_grad_square` | Stable diagnostic baseline, not full stable-neutral GeoFlow |
| Theory-aligned reference | `functional_geoflow` + `response_solver="dense"` | Dense small-model implementation of `J_phi`, `P_T/P_N`, and `P_N A_resp P_N` |
| Matrix-free prototype | `functional_geoflow` + `response_solver="implicit_cg"` | JVP/VJP normal-space solve with randomized VJP basis, Q cache, warm-start CG, and explicit per-step budgets |
| Experimental invariant backend | `FixedRankFunctionalAdam` | Adam moments in explicit product coordinates, followed by fixed-rank tangent projection, final candidate bounding, and rank-preserving retraction |
| Experimental quotient integrator | `SubsteppedQuotientFlow` | Repeated quotient-flow substeps over LoRA factors with closure-based gradient recomputation, optional clipping, and no Adam moments |

`functional_geoflow` is experimental. The dense solver is the correctness
reference for small MLPs and toy networks. The implicit solver is the
scaling-oriented matrix-free prototype, but it should be treated as a
controlled research path until LoRA and larger-model benchmarks support broader
claims.

`FixedRankFunctionalAdam` is a validated research backend. It is opt-in and is
not a default production optimizer.

`SubsteppedQuotientFlow` is an experimental H10.4 integrator. It is opt-in,
does not replace existing optimizers, and should not be interpreted as
production-ready.

## Legacy CIFAR Smoke Benchmark [Status: Historical Baseline]

The older CIFAR experiments compare three training modes:

| mode | description |
| --- | --- |
| `adam` | Standard Adam baseline |
| `geometric` | Pure geometric preconditioning |
| `hybrid` | Adam warm-up, then geometric updates |

Reference synthetic CIFAR-10 smoke milestone from a late-switch hybrid run:

| optimizer | accuracy | loss | ratio |
| --- | ---: | ---: | ---: |
| Adam | 51.6% | 1.8246 | - |
| Hybrid | 52.3% | 1.8397 | 0.458 |

`ratio` is `mean_preconditioned_to_raw_ratio`, a diagnostic for how strongly the
geometric direction is being used. This result is retained as a historical
baseline; it is not the current recommended path.

## Full CIFAR-10 Benchmark

For a more robust comparison, run the full benchmark on real CIFAR-10. The
dataset is downloaded automatically:

```bash
pip install -e . torchvision
python experiments/run_cifar10_benchmark.py \
  --config hybrid_diagonal_500 \
  --download \
  --out artifacts/cifar10_benchmark_results.csv
```

Output format:

```text
best=<optimizer> mean_acc=<score> delta_vs_adam=<signed_delta>
wrote artifacts/cifar10_benchmark_results.csv
```

The goal is to verify whether the hybrid edge remains stable over 200-500+
training steps, not to assume the synthetic smoke result will transfer
unchanged.

## Customize The Benchmark

| argument | description | default |
| --- | --- | --- |
| `--steps` | Training steps per trial | `500` |
| `--trials` | Number of independent runs | `3` |
| `--conv-layers` | Number of GeoCNN convolution layers | `3` |
| `--hybrid-warmup-steps` | Warm-up steps for hybrid mode | `10,30,50,80` |
| `--auto-warmup` | Try several hybrid warm-up settings in `train_cifar10_geo.py` | off |
| `--preconditioner` | `cg` or `diagonal` | `diagonal` |
| `--use-grad-square` / `--no-grad-square` | Use `grad_square` diagonal instead of Hessian | grad-square on |
| `--config` | Load a recommended CIFAR-10 config from `experiments/cifar10_configs.py` | none |
| `--precond-scales` | Optional sensitivity scan over preconditioner scale values | current value |
| `--grad-smoothing-values` | Optional sensitivity scan over smoothing values | current value |

Scan different warm-up steps in the tuning script:

```bash
python experiments/tune_geometric_optimizer.py \
  --modes geometric,adam,hybrid \
  --adam-warmup-steps-list "10,30,50,80" \
  --use-grad-square \
  --preconditioner diagonal
```

Auto-scan warm-up steps in the CIFAR trainer and keep all rows in one CSV:

```bash
python experiments/train_cifar10_geo.py \
  --dataset synthetic \
  --mode hybrid \
  --auto-warmup \
  --auto-warmup-steps "30,50,80" \
  --conv-layers 6 \
  --use-grad-square \
  --preconditioner diagonal \
  --out artifacts/auto_warmup.csv
```

Run the matched switch-control experiment. Both branches share the same Adam
warm-up state and batch sequence before splitting into `adam_continue` and
`hybrid_geometric`:

```bash
python experiments/train_cifar10_geo.py \
  --dataset synthetic \
  --mode switch_compare \
  --adam-warmup-steps 50 \
  --use-grad-square \
  --preconditioner diagonal \
  --out artifacts/switch_compare.csv
```

Run a longer benchmark with more trials:

```bash
python experiments/run_cifar10_benchmark.py \
  --download \
  --steps 1000 \
  --trials 5 \
  --hybrid-warmup-steps "30,80,150"
```

Run a small sensitivity scan:

```bash
python experiments/run_cifar10_benchmark.py \
  --download \
  --steps 500 \
  --trials 3 \
  --precond-scales "0.35,0.5,0.75" \
  --grad-smoothing-values "0.0,0.5"
```

Generate an SVG comparison chart from any benchmark CSV:

```bash
python experiments/plot_comparison.py \
  artifacts/cifar10_benchmark_results.csv \
  --out artifacts/adam_vs_hybrid.svg
```

Generate a ratio-over-time SVG from a diagnostic CSV:

```bash
python experiments/plot_comparison.py \
  artifacts/cifar10_geo_diagnostics.csv \
  --ratio-out artifacts/ratio_over_time.svg
```

Run the two-layer linear normal-projection toy benchmark:

```bash
python experiments/normal_projection_toy.py --out artifacts/normal_projection_toy.csv
```

Run the functional stable-neutral toy benchmark:

```bash
python experiments/functional_projection_toy.py --response-solver dense
python experiments/functional_projection_toy.py --response-solver low_rank
python experiments/functional_projection_toy.py --response-solver implicit_cg
python experiments/functional_projection_toy.py \
  --response-solver implicit_cg \
  --production-mode \
  --max-basis-rank 16 \
  --max-vjp-probes 24
```

Run a matched small-MLP validation that forks from the same Adam warm-up state
and compares Adam continuation, the legacy diagonal heuristic, and functional
GeoFlow:

```bash
python experiments/run_functional_switch_validation.py --trials 5 --steps 200
```

Run structural pressure tests. These are designed to separate parameterization
invariance from final accuracy:

```bash
python experiments/reparameterization_stress_test.py
python experiments/noisy_redundancy_validation.py
python experiments/near_null_stress_test.py
```

Run the small controlled LoRA bridge benchmark. It tests the LoRA
reparameterization symmetry `A -> S A, B -> B S^{-1}` before attempting larger
language-model adapters:

```bash
python experiments/lora_reparameterization_benchmark.py \
  --trials 3 \
  --steps 80 \
  --representations 4 \
  --out artifacts/lora_reparameterization.csv
```

Run the Phase G matched functional-step benchmark. This keeps the Phase F
benchmark intact and asks whether equalizing observed functional displacement
shrinks the task gap while preserving LoRA gauge stability:

```bash
python experiments/lora_matched_step_benchmark.py \
  --trials 5 \
  --steps 200 \
  --representations 5 \
  --train-scope lora_only \
  --functional-map hidden \
  --out artifacts/lora_matched_step.csv
```

## Output CSV Format

`experiments/run_cifar10_benchmark.py` writes:

| column | description |
| --- | --- |
| `optimizer` | `adam`, `geometric`, or `hybrid_<warmup_steps>` |
| `trials` | Number of independent runs |
| `mean_accuracy` / `std_accuracy` | Accuracy mean and standard deviation |
| `mean_loss` / `std_loss` | Loss mean and standard deviation |
| `mean_generalization_loss_gap` | Test loss minus train loss |
| `mean_generalization_accuracy_gap` | Train accuracy minus test accuracy |
| `mean_seconds` / `std_seconds` | Training time mean and standard deviation |
| `mean_preconditioned_to_raw_ratio` | Geometric direction strength diagnostic |
| `steps` | Training steps per trial |

## Theory-First Safety Checks

- Geometric updates are gated by the descent condition `g^T d < 0`; otherwise
  the optimizer falls back to a gradient step.
- The old `fisher` name is treated as a compatibility alias. The current
  positive diagonal approximation is named `grad_square`; true empirical Fisher
  remains a future extension.
- `experiments/normal_projection_toy.py` constructs the tangent space of the
  two-layer linear reparameterization symmetry and reports `P_N H P_N` normal
  curvature diagnostics.
- `geometric_flow.functional_geometry` constructs `J_phi`, SVD projectors
  `P_T/P_N`, the Gauss-Newton response `J_phi^T J_phi`, and the projected
  direction `d = -pinv(P_N A_resp P_N + damping P_N) P_N g`.
- `response_solver="low_rank"` uses a truncated SVD of dense `J_phi` and solves
  in retained right-singular directions without constructing full `A_resp`.
- `response_solver="implicit_cg"` uses VJP probes to estimate
  `range(J_phi^T)` and solves with JVP/VJP matvecs inside that matrix-free
  normal subspace. It no longer depends on dense `J_phi` or dense `P_N` for the
  solve. In production mode it uses randomized VJP probes, caches the normal
  basis for `refresh_interval` steps, warm-starts CG from the previous
  direction, and reports `jvp_count`, `vjp_count`, `peak_memory_bytes`,
  `null_leakage`, and wall-clock diagnostics. This is not yet a claim of
  large-model scalability.
- `experiments/run_functional_switch_validation.py` saves raw per-seed rows and
  reports win rate, gate accept rate, fallback rate, functional drift, update
  norm, and wall-clock time. Current output should be read as diagnostics, not a
  success claim.
- `experiments/reparameterization_stress_test.py` generates functionally
  equivalent hidden-basis representations and reports
  `reparameterization_sensitivity`; lower values mean the optimizer is less
  dependent on arbitrary parameterization.
- `experiments/noisy_redundancy_validation.py` decomposes injected gradient and
  parameter noise into tangent/normal components and records how much tangent
  noise remains after updates.
- `experiments/near_null_stress_test.py` appends an epsilon-weighted auxiliary
  parameter observable to create weakly broken null directions and stress-test
  threshold selection. This is for structural diagnostics, not accuracy claims.
- `experiments/lora_reparameterization_benchmark.py` is the next bridge from
  hand-built linear redundancy to modern low-rank adapter structure. Its primary
  metric is reparameterization sensitivity, not final accuracy.

## Historical Experiment Log [Status: Historical / Structural Only]

The following sections preserve the research trail. They are useful for
auditing how the current recommendation emerged, but they are not the fastest
path for new users.

### Statistical Convention For Gauge Sensitivity

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

### Phase F: LoRA Gauge Stability [Status: Historical / Structural Only]

Phase F tested functional quotient geometry on small LoRA adapters with exact
gauge-equivalent initializations:

```text
A -> S A
B -> B S^{-1}
```

Observed structural results from the Phase F sweep:

- 12 targeted configurations.
- 5 seeds and 5 gauge-equivalent representations.
- 900 total runs.
- Initial functional equivalence residuals were below the `1e-7` scale.
- The primary gauge metric is computed within each seed across
  gauge-equivalent representations.
- Matched per-seed functional/diagonal sensitivity ratio was about `0.536`.
- Its matched 95% confidence interval was entirely below `1`.
- Tangent drift ratio was about `0.316`.
- Near-null amplification ratio was about `0.361`.
- An earlier aggregate-all-pairs summary produced a ratio near `0.863`, but
  that quantity mixed cross-seed function distances and is retained only as a
  historical diagnostic, not as evidence for gauge invariance.

Negative result, stated plainly: Phase F did not establish task superiority.
Functional loss was higher than the diagonal baseline, functional accuracy was
lower than the diagonal baseline, and wall-clock cost was roughly tens of times
higher. Phase F establishes LoRA gauge stability, tangent suppression, and
near-null suppression, not a generally better optimizer.

### Controlled LoRA Architecture [Status: Reference]

The controlled LoRA experiments use a deliberately small network:

```text
z(x) = (W0 + B A) x
h(x) = tanh(z(x))
f_theta(x) = W_head h(x) + b_head
```

Here `W0` is a frozen base weight. `A` has shape
`rank x input_dim`, and `B` has shape `hidden_dim x rank`; these are the
trainable LoRA factors. The dense output head is `W_head, b_head`. The LoRA
product `B A` is invariant under the gauge transform:

```text
A -> S A
B -> B S^{-1}
```

for any invertible `S`.

The benchmark supports three training scopes:

| scope | trainable parameters |
| --- | --- |
| `lora_only` | `A, B` only |
| `head_only` | output head only |
| `lora_and_head` | both LoRA factors and head |

Phase G uses `lora_only` as the primary setting because the gauge symmetry
belongs to the factorization `B A`, not to the dense output head.

The functional map `Phi` may be chosen at different network levels:

| functional map | definition |
| --- | --- |
| `lora_output` | `z(x)` |
| `hidden` | `h(x)` |
| `logits` | `f_theta(x)` |
| `logits_hidden` | concatenation of logits and hidden features |

Changing the functional map changes the Jacobian `J_Phi`, and therefore changes
which parameter directions are classified as neutral or functional.

### Phase G: Matched Functional-Step Calibration [Status: Historical / Structural Only]

Equal parameter-space learning rates are not equal functional-space step sizes.
Phase G compares actual movement in function space:

```text
functional_step_norm = ||Phi(theta_after) - Phi(theta_before)||_2
```

The matched-step benchmark calibrates the functional GeoFlow update so its
initial observed functional displacement matches a reference optimizer
(`diagonal_grad_square` by default, or `adamw`). Calibration is done only on
training batches and the probe batch; it never uses test loss.

Phase G separately evaluates `lora_only`, `head_only`, and `lora_and_head`
training scopes. The primary configuration is `lora_only`, because the LoRA
gauge symmetry belongs to `A/B`, not the dense head. It also compares functional
maps over logits, LoRA output, hidden features, and logits+hidden.

Phase G gauge sensitivity is computed strictly within each seed across
gauge-equivalent representations. Cross-seed function distances are reported
only as `cross_seed_mixed_pairwise_distance`; they are not used as gauge
sensitivity and should not support gauge-invariance claims.

The relevant question is not whether accuracy can be tuned upward in one run.
It is whether the task gap shrinks after functional-step calibration while LoRA
gauge stability, low tangent drift, and low null leakage survive.

Current Stage A observation: the calibration mechanism works, but task behavior
has not improved yet. Functional-step calibration error was about `1e-3` or
lower, null leakage remained small, and fixed-lr and matched-step results were
close. In the current `lora_only` Stage A sweep, matched calibration did not
improve the fixed-lr task gap.

#### Corrected Phase G Smoke Result

A corrected within-seed reanalysis of the controlled smoke run found:

| metric | value |
| --- | ---: |
| matched-step within-seed sensitivity | `0.00399` |
| diagonal within-seed sensitivity | `0.00459` |
| mean matched/diagonal sensitivity ratio | `0.939` |
| structural seed win rate | `0.50` |
| mean functional-step calibration error | `0.000727` |
| mean null leakage | `2.0e-08` |

The corrected smoke supports the calibration and null-control mechanisms:

- `FUNCTIONAL_STEP_MATCH_PASS=True`
- `NULL_LEAKAGE_PASS=True`

It does not establish a robust structural or task advantage:

- `STRUCTURAL_WIN_RATE_PASS=False`
- `TASK_GAP_REDUCED_PASS=False`
- `TASK_ADVANTAGE_PASS=False`

The smoke result is diagnostic only.

#### Corrected Phase G B2 Long-Run Result

The long-run confirmation used:

- 8 independent seeds.
- 600 training steps.
- 5 gauge-equivalent representations per seed.
- `train_scope=lora_only`.
- `functional_map=logits_hidden`.

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

The corrected long-run structural gates passed:

- `STRUCTURAL_SENSITIVITY_PASS=True`
- `STRUCTURAL_WIN_RATE_PASS=True`
- `STRICT_STRUCTURAL_CI_PASS=True`
- `TANGENT_SUPPRESSION_PASS=True`

Task-level gates failed:

- `TASK_GAP_REDUCED_PASS=False`
- `TASK_PARITY_PASS=False`
- `TASK_ADVANTAGE_PASS=False`

The matched-step loss exceeded both the fixed-lr functional path and the
diagonal baseline. The calibration-improvement confidence interval was
`[-0.0416, -0.0126]`, indicating that matched functional-step calibration
worsened the task gap in this controlled setting.

The supported conclusion is structural: matched-step GeoFlow substantially
reduces sensitivity to LoRA gauge parameterization, but this structural
robustness does not translate into better task optimization here.

Existing Phase G artifacts can be reanalyzed without retraining:

```bash
python experiments/analyze_phase_g_results.py \
  --artifact-dir artifacts/phase_g_formal \
  --out artifacts/phase_g_formal/reanalysis
```

The analyzer skips incomplete Stage B or non-run CSV files with a warning. It
uses within-seed gauge sensitivity, matched seed sensitivity ratios, and paired
bootstrap confidence intervals for task gates.

## Reproducibility

The benchmark used to validate the experimental fixed-rank tangent optimizer is
available at:

```text
experiments/d7_fixed_rank_tangent_benchmark.py
```

It compares `factor_adam`, `explicit_product_adam`, `rank_tangent_sgd`,
`rank_tangent_adam`, and `rank_tangent_trust` across gauge-equivalent LoRA
representations in a small synthetic Transformer task.

The benchmark is intentionally kept in `experiments/`, not imported by the core
library. Its role is to reproduce the D7 milestone, provide a scientific
regression target, and guard against future implementations drifting away from
the fixed-rank tangent mechanism.

Reference D7 result:

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

These values establish task parity and structural invariance in this benchmark,
not universal task superiority.

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

The report includes gates such as `D7_TANGENT_RESIDUAL_PASS`,
`D7_RANK_PRESERVATION_PASS`, `D7_NEAR_EXACT_GAUGE_INVARIANCE`, and
`D7_TANGENT_TRUST_TASK_PARITY_PASS`. These gates are regression diagnostics, not
claims of universal optimizer superiority.

## Testing

```bash
python -m pytest -q
python -m pytest -q tests/test_d7_core_audit.py
python -m compileall -q geometric_flow experiments tests
```

The clean-clone audit at commit `cd81224f` passed the complete test suite, the
D7 core audit, the D7 smoke benchmark, and bytecode compilation.

## Documentation Index

- `experiments/d7_fixed_rank_tangent_benchmark.py`: full D7 reproduction and
  scientific regression benchmark.
- `experiments/lora_matched_step_benchmark.py`: Phase G matched functional-step
  calibration.
- `experiments/lora_reparameterization_benchmark.py`: controlled LoRA gauge
  sensitivity benchmark.
- `experiments/run_cifar10_benchmark.py`: legacy CIFAR benchmark harness.
- Historical Phase F/G notes remain below as archived context.

## Further Reading

- https://zenodo.org/records/21329073 Computation as Geometric Flow
- Theory direction: *Computation as GeoFlow*, a local
  stable-neutral formulation of implementation manifolds.
