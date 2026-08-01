# Computation as Geometric Flow

## An Arb-Certified Local Intrinsic ODE on a Quantum-Control Response Fibre

[![Full joint verification](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/papasop/Geometric-Flow/blob/main/notebooks/reproduce_joint_v093.ipynb)
[![reproduce-joint-geometric-flow](https://github.com/papasop/Geometric-Flow/actions/workflows/reproduce-joint-geometric-flow.yml/badge.svg)](https://github.com/papasop/Geometric-Flow/actions/workflows/reproduce-joint-geometric-flow.yml)

Frozen research-software release **v0.9.3** for
[`papasop/Geometric-Flow`](https://github.com/papasop/Geometric-Flow).

This repository studies a 14-phase driven-qubit model with response constraint

$$
R_3(\theta)=(\Re a_0,\Re a_1,\Re a_2,\Re a_3,
             \Im a_0,\Im a_1,\Im a_2,\Im a_3)
$$

and independent objective $L_6$. The declared ambient metric is Euclidean in
the fourteen phase coordinates. All theorem-bearing enclosures use
outward-rounded Arb arithmetic at 192-bit precision; floating midpoint SVDs
and inverses are frozen preconditioners only.

## Strongest certified statement

For child 15 of chart 9, subdivision 32, v0.9.3 constructs a local fibre chart

$$
\theta(a)=\theta_0+Ta+N\psi(a), \qquad a\in\mathbb R^6,
$$

certifies the complex parametric graph $b=\psi(a)$, pulls the Euclidean
metric back through $W=T+N D\psi$, and validates one intrinsic normalized
projected-gradient ODE microstep:

$$
\dot a=-\frac{H^{-1}W^\top\nabla L_6}
{\sqrt{(W^\top\nabla L_6)^\top H^{-1}(W^\top\nabla L_6)}},
\qquad H=W^\top W.
$$

The reference report has status:

```text
VALIDATED_INTRINSIC_RESPONSE_FIBRE_ODE_MICROSTEP_CERTIFIED
```

It formally certifies, on the declared microstep $0\le t\le10^{-14}$:

- existence and uniqueness of the intrinsic six-dimensional ODE solution;
- exact response preservation $R_3(\theta(t))=R_3(\theta(0))$;
- nonstationarity of the intrinsic projected gradient; and
- uniform strict descent
  $$
  \frac{dL_6}{dt}\le-0.6419529191591549<0.
  $$

| quantity | certified value |
| --- | ---: |
| complex-graph Krawczyk utilization | `0.02316105051443677` |
| pullback-metric Neumann defect | `0.000695359681037727` |
| intrinsic projected-gradient norm lower bound | `0.6419529191591549` |
| Picard contraction factor | `0.00348785915675938` |
| Picard self-mapping utilization | `0.0005813098594598967` |
| certified time step | `1e-14` |

The protocol SHA-256 is
`6d0aaefabd71f1d2986515ed84673f0083ae90d0344b9a1e92d7697ac08d061a`.

## Two distinct certified layers

The release preserves v0.7.4 because its theorem and the v0.9.3 theorem have
different scopes:

| layer | certified result | coverage |
| --- | --- | --- |
| v0.7.4 | rank, response tangency, projected-gradient nonstationarity and strict atlas descent | one complete 1/64 parent box, covered by 16 children |
| v0.9.3 | existence, uniqueness, exact response preservation and strict descent for the intrinsic ODE | one (10^{-14}) microstep near child 15 |

The broader v0.7.4 result is not an ODE theorem. The v0.9.3 ODE theorem is not
a complete-parent-box result.

## What v0.9.3 does not claim

- no validated traversal of the complete child interval;
- no complete ten-chart flow;
- no global connection between arbitrary response-equivalent controls;
- no global response-fibre, holonomy, neural-network, cloud, or QPU theorem;
- no practically large integration time—the certified step is deliberately
  microscopic and establishes the first local theorem unit.

See [`docs/CLAIM_SCOPE.md`](docs/CLAIM_SCOPE.md) and
[`docs/PAPER_WORDING.md`](docs/PAPER_WORDING.md).

## Isolated neural-response experiments

The directory
[`experiments/neural_response_fibres/`](experiments/neural_response_fibres/)
contains a separate floating-point residual-CNN experiment chain. Its frozen
v0.12.0--v0.14.1 records include both negative and positive prospective
audits. In particular, v0.14.1 reports a prospective task-advantage result on
a small synthetic model after the non-geometric comparator architecture was
qualified in v0.14.0.

This extension is **not** part of the v0.9.3 Arb theorem: it is not interval
certified, is not a continuous neural ODE theorem, and makes no real-data or
large-model claim. The theorem-bearing quantum-control sources, inputs,
certificates, requirements and reproduction workflows remain unchanged.

## Repository layout

```text
src/
  response_fibre_arb_kkt_witness_alignment_v0_7_4.py
  response_fibre_centered_mean_value_krawczyk_v0_9_2.py
  response_fibre_intrinsic_picard_microstep_v0_9_3.py
inputs/
  response_fibre_v0_6_2_backend_inputs.zip
results/
  reference_run_summary.json
  reference/
    protocol.json
    certificate.json
    report.json
  v0_9_3_reference/
    protocol.json
    intrinsic_picard_microstep_certificate.json
    report.json
docs/
  CLAIM_SCOPE.md
  NEURAL_NETWORK_RESPONSE_FIBRES.md
  PAPER_WORDING.md
experiments/
  neural_response_fibres/
    README.md
    verify_neural_artifacts.py
    src/
    results/
notebooks/
  reproduce_joint_v093.ipynb
  reproduce_v093.ipynb
tools/
  verify_release.py
.github/workflows/
  reproduce-joint-geometric-flow.yml
  structural-checks.yml
  reproduce-validated-ode.yml
```

## Install

Python 3.12 is recommended.

```bash
python -m pip install -r requirements.txt
```

## One-click external reproduction

- Recommended: Full geometry + ODE verification:
  [Open in Colab](https://colab.research.google.com/github/papasop/Geometric-Flow/blob/main/notebooks/reproduce_joint_v093.ipynb)
  or
  [![reproduce-joint-geometric-flow](https://github.com/papasop/Geometric-Flow/actions/workflows/reproduce-joint-geometric-flow.yml/badge.svg)](https://github.com/papasop/Geometric-Flow/actions/workflows/reproduce-joint-geometric-flow.yml)
- Quick: Local ODE microstep only:
  [Open in Colab](https://colab.research.google.com/github/papasop/Geometric-Flow/blob/main/notebooks/reproduce_v093.ipynb)
  or
  [![reproduce-validated-ode](https://github.com/papasop/Geometric-Flow/actions/workflows/reproduce-validated-ode.yml/badge.svg)](https://github.com/papasop/Geometric-Flow/actions/workflows/reproduce-validated-ode.yml)
- Frozen structural verification: `python tools/verify_release.py`

A successful joint reproduction must end with:

```text
STAGE_A_PARENT_BOX_GEOMETRY_CERTIFIED = true
STAGE_B_LOCAL_ODE_MICROSTEP_CERTIFIED = true
JOINT_GEOMETRIC_FLOW_RELEASE_GATE = true
GLOBAL_FLOW_CLAIMED = false
```

The default `structural-checks` workflow verifies committed files and hashes.
It is not an ODE reproduction. The `reproduce-joint-geometric-flow` workflow
reruns the v0.7.4 complete-parent-box geometry certificate and the v0.9.3
local ODE microstep as separate stages, then writes a joint summary without
merging the original certificates or claiming a global flow. The
`reproduce-validated-ode` workflow remains as a faster ODE-microstep-only
entry point.

## Full v0.9.3 Arb reproduction

```bash
python src/response_fibre_intrinsic_picard_microstep_v0_9_3.py \
  --inputs-zip inputs/response_fibre_v0_6_2_backend_inputs.zip \
  --v074-source src/response_fibre_arb_kkt_witness_alignment_v0_7_4.py \
  --no-download \
  --output results/v0_9_3_reproduction
```

Expected high-level fields:

```text
all_gates_pass = true
validated_ODE_claimed = true
ODE_existence_certified = true
ODE_uniqueness_certified = true
exact_response_preservation_certified = true
uniform_L6_descent_certified_for_validated_solution = true
global_flow_claimed = false
```

The final report binds the generator source hash. Notebook-cell execution can
leave `generator_source_sha256` null, so release-grade reproduction should run
the saved `.py` file as shown above.

## Reproduce the broader v0.7.4 descent audit

```bash
python src/response_fibre_arb_kkt_witness_alignment_v0_7_4.py \
  --inputs-zip inputs/response_fibre_v0_6_2_backend_inputs.zip \
  --chart 9 \
  --subdivision 32 \
  --output results/v0_7_4_reproduction
```

## Verify the frozen package

```bash
python tools/verify_release.py
sha256sum -c SHA256SUMS.txt
```

## 中文说明

v0.9.3 首次在响应纤维的六维内蕴坐标中，以 192-bit Arb 区间算术认证了
一个投影梯度 ODE 微步：解存在且唯一，响应 $R_3$ 精确保持，并且
$dL_6/dt\le-0.6419529191591549<0$。认证时间仅为 $10^{-14}$，因此这是
严格的局部微步定理，不是完整 child、十图或全局几何流定理。

## Citation and license

Use [`CITATION.cff`](CITATION.cff) and cite the exact v0.9.3 release or commit.
The code is released under the MIT License.
