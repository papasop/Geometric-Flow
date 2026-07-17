# Verified Run Summary

- GPU: NVIDIA A100-SXM4-40GB
- dtype: float64
- matrix: 128 x 96
- rank: 4
- steps: 400
- conditions: 1, 100, 10000
- trials per condition: 6
- total paired cases: 18

## Main findings

- diagonal and full-core response improve plain compact mean loss at all three conditions;
- diagonal response closes approximately 80.9% of the cross-condition mean plain-to-AdamW gap;
- full-core minimum raw/preconditioned cosine: 0.7956228216;
- maximum relative direction change: 1.3481501079;
- maximum KLR constraint residual: 1.0797832013e-16;
- AdamW remains better in all 18 paired cases.

Milestone classification: mechanism milestone, not AdamW-superiority milestone.
