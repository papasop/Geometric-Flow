# GeoFlow for PyTorch

A geometry-first optimization toolkit for PyTorch, inspired by quantum-control
manifolds. The repository now separates an older diagonal gradient-square
heuristic from a theory-aligned functional GeoFlow path that explicitly builds a
stable/neutral decomposition in function space.

## Core Experiment: CIFAR-10 Benchmark

We compare three training modes:

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
geometric direction is being used.

## Why GeoFlow?

Most deep learning optimizers, including SGD and Adam, navigate parameter space
using gradients alone. They know which way is downhill, but they do not directly
measure how the terrain bends.

The legacy optimizer measures local curvature with Hessian or gradient-square
approximations. The theory-aligned path instead builds a functional map
`Phi(theta; X_probe) = vec(model(X_probe))`, computes its Jacobian, separates
neutral reparameterization directions from normal functional directions, and
updates only through the normal response operator.

The idea comes from quantum-control experiments, where geometry-aware updates
reduced evaluations by 56% and saved 30% of physical qubits. This repository
brings that geometry-first philosophy into PyTorch deep learning.

## One-Command Quickstart

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

`functional_geoflow` is experimental. The dense solver is the correctness
reference for small MLPs and toy networks. The implicit solver is the
production-oriented path, but it should be treated as a controlled prototype
until LoRA and larger-model benchmarks support broader claims.

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

## Phase F: LoRA Gauge Stability

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
- Config-level functional/diagonal sensitivity ratio mean was about `0.863`.
- Matched per-seed functional/diagonal sensitivity ratio was about `0.536`.
- The matched 95% confidence interval was entirely below `1`.
- Tangent drift ratio was about `0.316`.
- Near-null amplification ratio was about `0.361`.
- Every tested configuration had config-level sensitivity ratio below `1`.

These are two different statistics and should not be mixed: the first is the
ratio of aggregated sensitivities, while the second is the mean of matched
per-seed sensitivity ratios.

Negative result, stated plainly: Phase F did not establish task superiority.
Functional loss was higher than the diagonal baseline, functional accuracy was
lower than the diagonal baseline, and wall-clock cost was roughly tens of times
higher. Phase F establishes LoRA gauge stability, tangent suppression, and
near-null suppression, not a generally better optimizer.

## Controlled LoRA Architecture

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

## Phase G: Matched Functional-Step Calibration

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

The relevant question is not whether accuracy can be tuned upward in one run.
It is whether the task gap shrinks after functional-step calibration while LoRA
gauge stability, low tangent drift, and low null leakage survive.

## Claims Boundary

Established so far:

- Matrix-free functional quotient directions with dense small-toy regression.
- LoRA gauge sensitivity reduction.
- Tangent and near-null suppression in controlled settings.

Not established:

- General task superiority.
- AdamW competitiveness.
- Large-model scalability.
- GPT-2 or other language-model results.

Avoid interpreting these experiments as a universally better optimizer,
production-ready large-model optimizer, proven generalization improvement, or
quantum advantage claim.

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

## Further Reading

- https://zenodo.org/records/21329073 Computation as Geometric Flow
- Theory direction: *Computation as GeoFlow*, a local
  stable-neutral formulation of implementation manifolds.
