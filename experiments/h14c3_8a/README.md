# H14C3-8A Experiment

This directory contains the transported accepted-secant response audit.

\[
s_t=\mathcal T_{t\to t+1}(\Delta\tau_t v_t),
\qquad
y_t=g_{t+1}-\mathcal T_{t\to t+1}(g_t).
\]

Methods:

- `plain_compact`
- `diagonal_response_8a`
- `full_core_response_8a`
- `factor_adamw`

Both response variants improve the plain compact mean final validation loss at condition numbers 1, 100, and 10000. The diagonal response reduces the cross-condition mean gap to AdamW by approximately 80.9%, but AdamW remains better in all 18/18 paired cases.

Run:

```bash
python experiments/h14c3_8a/h14c3_8a_transported_accepted_secant_response_audit.py
```
