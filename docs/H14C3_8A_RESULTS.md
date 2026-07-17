# H14C3-8A — Transported Accepted-Secant Response

## Milestone

H14C3-8A is the first intrinsic response mechanism in the H14 series that consistently improves the plain compact Hamiltonian across all tested conditioning regimes.

\[
s_t=\mathcal T_{t\to t+1}(\Delta\tau_t v_t),
\qquad
y_t=g_{t+1}-\mathcal T_{t\to t+1}(g_t).
\]

Only accepted steps enter the history, and historical secants are transported into the current tangent frame before fitting the response operator.

## Verified configuration

- matrix size: 128 x 96
- rank: 4
- conditions: 1, 100, 10000
- trials per condition: 6
- total paired cases: 18
- train/validation samples: 2048/1024
- steps: 400
- dtype: float64
- hardware: NVIDIA A100-SXM4-40GB

## Mean final validation loss

| condition | plain compact | diagonal response | full-core response | tuned AdamW |
|---:|---:|---:|---:|---:|
| 1 | 2.01942720e-04 | 2.00920463e-04 | 2.00941047e-04 | 2.00882522e-04 |
| 100 | 2.01335530e-04 | 2.01208128e-04 | 2.01207145e-04 | 2.01067509e-04 |
| 10000 | 2.01160590e-04 | 2.00988368e-04 | 2.00988442e-04 | 2.00853935e-04 |

Both response variants improve the plain compact mean at every tested condition number.

## Gap reduction

Cross-condition mean gap to tuned AdamW:

- plain compact: approximately 5.45e-7
- diagonal response: approximately 1.04e-7
- full-core response: approximately 1.11e-7

The diagonal response closes approximately 80.9% of the measured plain-compact gap to AdamW.

## Mechanism evidence

- minimum raw/preconditioned cosine: 0.7956228216
- maximum relative direction change: 1.3481501079
- maximum response condition: 4.9584345141
- maximum KLR constraint residual: 1.0797832013e-16

The diagonal response primarily changes effective magnitude. The full-core response genuinely changes direction.

## Raw gates and derived interpretation

The raw run returned `PASS_RESPONSE_MECHANISM_ACTIVE=false`. This aggregate gate required both magnitude and directional activation for every response variant. Preserve this raw result in `experiments/h14c3_8a/verified/raw_gates.json`.

The derived interpretation is stored separately in `experiments/h14c3_8a/verified/derived_verdict.json`:

```text
DERIVED_MAGNITUDE_RESPONSE_ACTIVE = true
DERIVED_FULL_CORE_DIRECTION_ACTIVE = true
```

## Claims supported

1. Transport-consistent accepted secants provide an active intrinsic response signal.
2. Both tested response variants improve the plain compact baseline in mean final validation loss at cond=1,100,10000.
3. The response mechanism closes most of the measured loss gap between plain compact Hamiltonian dynamics and tuned AdamW.
4. KLR constraints remain at numerical precision.

## Claims not supported

Do not claim that H14C3-8A beats AdamW, wins paired trials, proves full-core rotation is superior to magnitude control, provides a reliable full Hessian estimate, or generalizes beyond controlled low-rank regression.

AdamW remained better in all 18/18 paired cases.

## Reproduce

```bash
python experiments/h14c3_8a/h14c3_8a_transported_accepted_secant_response_audit.py
```
