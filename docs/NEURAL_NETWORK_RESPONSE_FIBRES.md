# Neural-Network Response Fibres

This note records the theoretical motivation for a neural response-fibre
extension. It is not a result of the theorem-bearing v0.7.4 or v0.9.3
quantum-control release. The separate floating-point experiment chain and its
frozen positive and negative outcomes are documented in
[`../experiments/neural_response_fibres/README.md`](../experiments/neural_response_fibres/README.md).

For a parameterized model, let $R_{\mathrm{train}}(\theta)$ represent a
declared collection of training responses, such as logits, predictions, or
input jets. When

$$
\operatorname{rank}DR_{\mathrm{train}}(\theta)=\dim R_{\mathrm{train}},
$$

the matched-response set is locally a smooth fibre near regular points:

$$
\mathcal F_r=R_{\mathrm{train}}^{-1}(r).
$$

Let $G(\theta)$ be a separate robustness or generalization objective. One may
then study constrained descent of the form

$$
\dot{\theta}
=-P_{\ker DR_{\mathrm{train}}(\theta)}^{\,g}\nabla_g G(\theta).
$$

The descended quantity here is the independent objective $G$, not the training
loss itself. If $L$ is the training loss and the motion is restricted to a loss
level set, then

$$
L|_{L^{-1}(c)}=c
\quad\Longrightarrow\quad
\nabla_{L^{-1}(c)}L=0,
$$

so the intrinsic descent problem for $L$ on its own level set is degenerate.
The response-fibre question is instead whether one can preserve a declared
training response while descending a distinct objective.

Different jet-matching orders may define different notions of model
equivalence. A future theory would also need to specify the metric $g$, such
as a Fisher-type or declared response metric, and prove regularity,
stability, and computable certification conditions.

No neural-network theorem is claimed here. The numerical experiments support
specific finite-dimensional mechanism and task-advantage statements under
their declared synthetic protocols, but they do not transfer the
projective-jet no-go theorem, the Arb certificate, or the validated local ODE
theorem to neural networks or to the NTK regime. There is also no scalable Arb
certification method here for high-dimensional neural networks.
