# Legacy CIFAR Benchmarks

The CIFAR scripts are retained as historical baselines for the older
`GeometricOptimizer` path. They are useful for regression and diagnostics, but
they are not the current recommended GeoFlow path.

## Synthetic Smoke

```bash
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

Reference late-switch hybrid smoke:

| optimizer | accuracy | loss | ratio |
| --- | ---: | ---: | ---: |
| Adam | `51.6%` | `1.8246` | - |
| Hybrid | `52.3%` | `1.8397` | `0.458` |

`ratio` is `mean_preconditioned_to_raw_ratio`, a diagnostic for how strongly
the geometric direction is used.

## Full CIFAR-10 Benchmark

```bash
pip install -e . torchvision
python experiments/run_cifar10_benchmark.py \
  --config hybrid_diagonal_500 \
  --download \
  --out artifacts/cifar10_benchmark_results.csv
```

The goal is to test whether a synthetic hybrid edge survives over 200-500+
steps on real CIFAR-10, not to assume transfer.

## Tuning Commands

Scan warm-up steps:

```bash
python experiments/tune_geometric_optimizer.py \
  --modes geometric,adam,hybrid \
  --adam-warmup-steps-list "10,30,50,80" \
  --use-grad-square \
  --preconditioner diagonal
```

Auto-scan warm-up steps:

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

Matched switch-control experiment:

```bash
python experiments/train_cifar10_geo.py \
  --dataset synthetic \
  --mode switch_compare \
  --adam-warmup-steps 50 \
  --use-grad-square \
  --preconditioner diagonal \
  --out artifacts/switch_compare.csv
```

Longer benchmark:

```bash
python experiments/run_cifar10_benchmark.py \
  --download \
  --steps 1000 \
  --trials 5 \
  --hybrid-warmup-steps "30,80,150"
```

Sensitivity scan:

```bash
python experiments/run_cifar10_benchmark.py \
  --download \
  --steps 500 \
  --trials 3 \
  --precond-scales "0.35,0.5,0.75" \
  --grad-smoothing-values "0.0,0.5"
```

## Plots

```bash
python experiments/plot_comparison.py \
  artifacts/cifar10_benchmark_results.csv \
  --out artifacts/adam_vs_hybrid.svg

python experiments/plot_comparison.py \
  artifacts/cifar10_geo_diagnostics.csv \
  --ratio-out artifacts/ratio_over_time.svg
```

## Output CSV Columns

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
