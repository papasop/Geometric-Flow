# H13.14 GPT-2 Full-Depth Validation

All experiments adapt all 12 GPT-2-small attention `c_attn` modules with rank-4 LoRA, frozen base weights, shared represented initialization, shared minibatch schedules, and matched realized LoRA-product displacement.

## H13.14C — 200 steps, two seeds

| method | validation loss | perplexity | mean `|rho_AB|` | mean condition |
|---|---:|---:|---:|---:|
| Channel momentum | 4.133092 | 62.371 | 0.193 | 4.27 |
| Coupled covariance | 4.124508 | 61.837 | 0.233 | 2.86 |

Coupled won 2/2. Mean loss advantage: 0.008584. This is about 0.208% lower cross-entropy and 0.86% lower perplexity.

## H13.14E — traditional baselines

The six-seed run compared AdamW LoRA, factor EMA, channel momentum, and coupled covariance. The supplied transcript contains five complete seeds with coupled lower than every baseline. Do not claim 6/6 until the sixth coupled row and final gates are archived.

## H13.14F — 1000 steps, three seeds

| method | validation loss | mean `|rho_AB|` | mean condition |
|---|---:|---:|---:|
| Factor EMA | 3.894558 | 0.151 | 31.6 |
| Channel momentum | 3.896769 | 0.156 | 27.8 |
| Coupled covariance | 3.892090 | 0.085 | 1.68 |

Coupled won 3/3 against factor EMA and momentum. One seed had gauge p99 1.815e-5, above the preregistered per-seed float32 threshold 1e-5; report both mean and maximum and do not retroactively relax the threshold.

These are controlled GPT-2-small results, not production-scale LLM validation and not evidence of universal optimizer superiority.

## External reproduction

```bash
python scripts/run_external_gpt2_validation.py --mode smoke --install-deps
python scripts/run_external_gpt2_validation.py --mode formal
python scripts/run_external_gpt2_validation.py --mode long
```
