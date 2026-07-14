# Functional Geometry Notes

This document collects the lower-level stable-neutral machinery that is too
detailed for the README.

## Functional Map

The theory-aligned path constructs a configurable functional map:

```text
Phi(theta; X_probe) = vec(model(X_probe))
```

The dense path explicitly builds `J_Phi`, computes SVD projectors `P_T/P_N`,
constructs the Gauss-Newton response `J_Phi^T J_Phi`, and solves

```text
d = -pinv(P_N A_resp P_N + damping P_N) P_N g
```

for small toy models and probe batches.

## Safety Checks

- Geometric updates are gated by `g^T d < 0`; otherwise the optimizer falls
  back.
- The old `fisher` name is treated as a compatibility alias. The current
  positive diagonal approximation is named `grad_square`; true empirical Fisher
  remains future work.
- `experiments/normal_projection_toy.py` constructs the two-layer linear
  reparameterization tangent space and reports `P_N H P_N` diagnostics.

## Response Solvers

| solver | description |
| :--- | :--- |
| `dense` | Explicit `J_Phi`, SVD projectors, and dense response solve for small models |
| `low_rank` | Truncated SVD of dense `J_Phi`; solves in retained right-singular directions |
| `implicit_cg` | JVP/VJP matvecs inside a randomized matrix-free normal subspace |

The matrix-free prototype uses randomized VJP probes, caches the normal basis
for `refresh_interval` steps, warm-starts CG from the previous direction, and
reports `jvp_count`, `vjp_count`, `peak_memory_bytes`, `null_leakage`, and
wall-clock diagnostics. This is not yet a large-model scalability claim.

## Useful Commands

Functional projection toy:

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

Matched small-MLP validation:

```bash
python experiments/run_functional_switch_validation.py --trials 5 --steps 200
```

Structural pressure tests:

```bash
python experiments/reparameterization_stress_test.py
python experiments/noisy_redundancy_validation.py
python experiments/near_null_stress_test.py
```

Controlled LoRA bridge:

```bash
python experiments/lora_reparameterization_benchmark.py \
  --trials 3 \
  --steps 80 \
  --representations 4 \
  --out artifacts/lora_reparameterization.csv
```

Phase G matched functional-step benchmark:

```bash
python experiments/lora_matched_step_benchmark.py \
  --trials 5 \
  --steps 200 \
  --representations 5 \
  --train-scope lora_only \
  --functional-map hidden \
  --out artifacts/lora_matched_step.csv
```
