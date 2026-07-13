# GeoFlow for PyTorch

A geometry-first optimization toolkit for PyTorch, inspired by quantum-control
manifolds. This library implements curvature-aware preconditioning with
Hessian/Fisher information, moving beyond pure gradient descent.

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

GeoFlow also measures local curvature with Hessian/Fisher information.
Like a hiker who can see both slope and terrain shape, a geometric optimizer can
precondition its steps and choose a more informed path through the loss
landscape.

The idea comes from quantum-control experiments, where geometry-aware updates
reduced evaluations by 56% and saved 30% of physical qubits. This repository
brings that geometry-first philosophy into PyTorch deep learning.

## One-Command Quickstart

### Run In Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/papasop/AGI/blob/main/notebooks/run_cifar10_benchmark.ipynb)

Open the notebook and run the first cell. It clones the repository, installs
dependencies, downloads CIFAR-10, and writes benchmark results to
`artifacts/cifar10_benchmark_results.csv`.

### Run Locally

Fast 50-step synthetic CIFAR-10 smoke test, with no dataset download:

```bash
git clone https://github.com/papasop/AGI.git
cd AGI
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
  --use-fisher \
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

## Full CIFAR-10 Benchmark

For a more robust comparison, run the full benchmark on real CIFAR-10. The
dataset is downloaded automatically:

```bash
pip install -e . torchvision
python experiments/run_cifar10_benchmark.py \
  --download \
  --steps 500 \
  --trials 3 \
  --hybrid-warmup-steps "10,30,50,80" \
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
| `--hybrid-warmup-steps` | Warm-up steps for hybrid mode | `10,30,50,80` |
| `--preconditioner` | `cg` or `diagonal` | `diagonal` |
| `--use-fisher` / `--no-fisher` | Use Fisher instead of Hessian | Fisher on |

Scan different warm-up steps:

```bash
python experiments/tune_geometric_optimizer.py \
  --modes geometric,adam,hybrid \
  --adam-warmup-steps-list "10,30,50,80" \
  --use-fisher \
  --preconditioner diagonal
```

Run a longer benchmark with more trials:

```bash
python experiments/run_cifar10_benchmark.py \
  --download \
  --steps 1000 \
  --trials 5 \
  --hybrid-warmup-steps "30,80,150"
```

## Output CSV Format

`experiments/run_cifar10_benchmark.py` writes:

| column | description |
| --- | --- |
| `optimizer` | `adam`, `geometric`, or `hybrid_<warmup_steps>` |
| `trials` | Number of independent runs |
| `mean_accuracy` / `std_accuracy` | Accuracy mean and standard deviation |
| `mean_loss` / `std_loss` | Loss mean and standard deviation |
| `mean_seconds` / `std_seconds` | Training time mean and standard deviation |
| `mean_preconditioned_to_raw_ratio` | Geometric direction strength diagnostic |
| `steps` | Training steps per trial |

## Further Reading

- Package details: https://zenodo.org/records/21329073
- Theory direction: *Computation as GeoFlow*, a local
  stable-neutral formulation of implementation manifolds.
