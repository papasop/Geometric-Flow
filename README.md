# AGI Geometric Flow

Geometry-first optimization tools for PyTorch. The core experiment compares
three CIFAR-10 training modes:

- `adam`: standard Adam baseline.
- `geometric`: pure geometric preconditioning.
- `hybrid`: Adam warm-up followed by geometric updates.

Reference synthetic CIFAR-10 smoke milestone from the late-switch hybrid run:

| optimizer | accuracy | loss | ratio |
| --- | ---: | ---: | ---: |
| Adam | 51.6% | 1.8246 | 0.000 |
| Hybrid | 52.3% | 1.8397 | 0.458 |

The copy-paste smoke command below fixes `--seed 32` so contributors get a
short visible check of the hybrid path.

## Run In Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/papasop/AGI/blob/main/notebooks/run_cifar10_benchmark.ipynb)

Open the notebook and run the first cell. It clones the repository, installs the
package plus `torchvision`, downloads CIFAR-10, and writes results to
`artifacts/cifar10_benchmark_results.csv`.

## One-Copy Local Test

Fast synthetic milestone smoke test, no dataset download:

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

On the current fair multi-run harness, this smoke command should show Hybrid
well above Adam for that fixed seed. For longer 500-step CIFAR-10 runs, start with
`adam_warmup_steps=30` or scan `10,30,50,80`.

Full CIFAR-10 benchmark:

```bash
git clone https://github.com/papasop/AGI.git
cd AGI
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

Actual numbers depend on hardware, PyTorch version, and random seeds. The goal
of the full benchmark is to verify whether the hybrid edge persists for
200-500+ training steps, not to assume it.

## Results CSV

`experiments/run_cifar10_benchmark.py` writes:

| column | meaning |
| --- | --- |
| `optimizer` | `adam`, `geometric`, or `hybrid_<warmup_steps>` |
| `trials` | independent runs per configuration |
| `mean_accuracy` / `std_accuracy` | accuracy mean and standard deviation |
| `mean_loss` / `std_loss` | loss mean and standard deviation |
| `mean_seconds` / `std_seconds` | training time mean and standard deviation |
| `mean_preconditioned_to_raw_ratio` | geometric direction strength diagnostic |
| `steps` | training steps per trial |

## Useful Variants

Scan hybrid warm-up settings:

```bash
python experiments/tune_geometric_optimizer.py \
  --modes geometric,adam,hybrid \
  --adam-warmup-steps-list "10,30,50,80" \
  --use-fisher \
  --preconditioner diagonal
```

Run a longer benchmark:

```bash
python experiments/run_cifar10_benchmark.py \
  --download \
  --steps 1000 \
  --trials 5 \
  --hybrid-warmup-steps "30,80,150"
```

Package-level technical details live in
[`geometric_flow/README.md`](geometric_flow/README.md).
