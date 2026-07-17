# H14C3-6 — Compact USV–KLR Intrinsic Hamiltonian

## Milestone

H14C3-6 replaces the persistent dense product/momentum state with the compact
intrinsic representation

\[
M = U\operatorname{diag}(S)V^T,
\]

\[
P = UKV^T + LV^T + UR,
\]

subject to

\[
U^T L = 0, \qquad RV = 0.
\]

The compact implementation reproduces the dense intrinsic Hamiltonian
trajectory to machine precision.

## Verified run

Configuration:

- matrix size: 128 x 96
- rank: 4
- target condition number: 100
- train samples: 2048
- validation samples: 1024
- steps: 400
- trials: 6
- dtype: float64
- hardware: NVIDIA A100-SXM4-40GB

Results:

- maximum compact/dense trajectory residual: `8.259418311551157e-14`
- maximum KLR constraint residual: `6.884811598472147e-17`
- maximum rank tail: `9.179450981494967e-16`
- maximum gauge residual: `1.2430207345664703e-13`
- compact persistent-state elements: `1812`
- dense persistent-state elements: `13188`
- estimated AdamW persistent-state elements: `3584`
- compact/dense state ratio: `0.1373976342`
- compact/AdamW estimated-state ratio: `0.5055803571`
- `PASS_CORE: true`

## Claims supported

The verified result supports the following claims:

1. The USV–KLR state is dynamically equivalent to the dense intrinsic
   Hamiltonian implementation up to numerical precision.
2. Rank, tangent constraints, and gauge invariance are preserved.
3. Persistent optimizer-state storage is reduced by approximately 86.3%
   relative to the dense intrinsic implementation.
4. Under the stated counting convention, compact persistent state is about
   50.6% of the estimated LoRA AdamW state.

## Claims not yet supported

Do not claim that H14C3-6:

- has lower peak GPU memory than AdamW;
- is faster than AdamW;
- uses fewer FLOPs than AdamW;
- has better final validation loss than tuned AdamW.

The present implementation still constructs transient dense matrices during
information velocity and vector transport. H14C3-7 should remove those dense
temporary objects and benchmark actual peak memory and wall time.

## Run

```bash
python experiments/h14c3_6/h14c3_6_compact_usv_klr_audit.py
```

Expected output directory:

```text
h14c3_6_compact_usv_klr_audit/
```
