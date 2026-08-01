# Neural Response-Fibre Experiments

This directory is an isolated numerical extension of the frozen quantum-control
v0.9.3 release. It does **not** modify the theorem-bearing `src/`, inputs,
certificates, paper scope, or release tag.

For a declared neural response map `R(theta)` and an independent objective
`V(theta)`, the intrinsic branch discretizes

```text
d theta / dt = -P_ker(DR(theta)) grad V(theta)
```

by SVD tangent projection followed by SVD-Newton response retraction. These are
floating-point retracted-Euler experiments, not validated continuous-time ODEs.

## Complete positive/negative audit chain

| audit | role | frozen outcome |
| --- | --- | --- |
| v0.12.0 | prospective response enrichment / jet filtration | 3/3 supported |
| v0.13.0 | prospective equal-dimension jet efficiency and original-baseline advantage | not supported |
| v0.14.0 | development-only fair non-geometric comparator qualification | 3/3 qualified |
| v0.14.1 | prospective fair-baseline task advantage, two co-primary responses | 3/3 and 3/3 supported |

The v0.13.0 negative result is retained deliberately. Its fixed-step penalty
baselines had no positive response-feasible checkpoint. v0.14.0 repaired that
comparison using a declared-response backtracking firewall and augmented-
Lagrangian comparators. Held-out responses and labels never enter step
acceptance. v0.14.1 then froze entirely new seeds and required both `R_value72`
and `R_jet72` to pass on at least two of three seeds.

Every v0.14.1 seed passed both co-primary gates. The six observed ratios of
intrinsic objective reduction to the best response-feasible non-geometric
checkpoint ranged from `6.783709599169901` to `8.346997345961455`, against a
predeclared threshold of `1.25`.

## Reproduction

Python 3.12 is recommended.

```bash
python -m pip install -r experiments/neural_response_fibres/requirements-neural.txt
python experiments/neural_response_fibres/src/response_fibre_nn_jet_filtration_v0_12_0.py
python experiments/neural_response_fibres/src/response_fibre_nn_dimension_matched_task_advantage_v0_13_0.py
python experiments/neural_response_fibres/src/response_fibre_nn_fair_baseline_pareto_v0_14_0.py
python experiments/neural_response_fibres/src/response_fibre_nn_prospective_task_advantage_v0_14_1.py
```

The prospective cohorts have already been exposed. Reruns are reproducibility
checks only and must not be described as new prospective evidence.

Verify the committed experiment artifacts without rerunning training:

```bash
python experiments/neural_response_fibres/verify_neural_artifacts.py
sha256sum -c experiments/neural_response_fibres/SHA256SUMS.txt
```

## Frozen identifiers

- v0.14.0 protocol: `83e88e44fd207be14950c9005f93a0ef2bb7cc97549dd38967e24a0cce44ffc4`
- v0.14.0 certificate: `ad71c665c30dd8e697da125c06c9e79cbf356be7ca8b59b00a7fbc288ee7908e`
- v0.14.1 protocol: `de264dd5be7d681e00e2e40ec3f188823d6861fc3aa4ccb5de467eb16ee68d8d`
- v0.14.1 certificate: `acc06340c1750f8a225b539527b0bd4e493c2fa6a92bbe395fcd9bf0b64a6bbe`

## Claim boundary

The supported claim is limited to a frozen small synthetic residual-CNN
experiment. No interval arithmetic, neural ODE existence theorem, real-data
generalization result, large-model result, quantum cloud result, or QPU claim
is made here. The theorem-bearing statement of the repository remains the
v0.9.3 Arb-certified local quantum-control ODE microstep.
