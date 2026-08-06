# Proposed Replacement for Sections 5.1 and 5.3

## 5.1. Data-Generating Process

We formulate the simulation design through three components: a covariate
distribution \(P_X\), a behavior policy \(\pi_0(a\mid x)\), and a conditional
outcome kernel \(P_\vartheta(Y\mid X,A)\). This formulation accommodates the
different outcome specifications required by the experiments below while
holding the sampling and policy-learning framework fixed.

In all experiments, \(X\) is sampled uniformly from the closed unit ball in
\(\mathbb{R}^5\), and the action set is \(\mathcal{A}=\{0,1\}\). We consider
both a linear score,

\[
s_{\mathrm{L}}(x_1,x_2)=x_1-0.8x_2,
\]

and a nonlinear score \(s_{\mathrm{NL}}(x_1,x_2)\). The zero contour of the
latter is the S-shaped boundary formed by two semicircles of radius \(1/2\):

\[
s_{\mathrm{NL}}(x_1,x_2)
=
\begin{cases}
x_1-\sqrt{1/4-(x_2-1/2)^2}, & x_2\geq 0,\\
x_1+\sqrt{1/4-(x_2+1/2)^2}, & x_2<0.
\end{cases}
\]

For a generic score \(s\), let \(a^\dagger(x)=\mathbb{I}\{s(x)\geq 0\}\).
The behavior policy assigns the score-preferred action with probability
\(0.75\):

\[
\pi_0(a\mid x)
=
\begin{cases}
0.75, & a=a^\dagger(x),\\
0.25, & a\neq a^\dagger(x).
\end{cases}
\]

We use two conditional outcome specifications within this common framework.
The first is the smooth-outcome specification used in Sections 5.2 and 5.4.
Its latent conditional means are

\[
\mu_a(x)
=m_0+(2a-1)\delta\tanh\{\gamma s(x)\},
\qquad a\in\{0,1\},
\]

where \(m_0=0\), \(\delta=1.5\), and \(\gamma=6\). Potential outcomes are
generated as

\[
Y(a)=\operatorname{sigmoid}\{\mu_a(X)+\epsilon_a\},
\qquad
\epsilon_a\overset{\mathrm{i.i.d.}}{\sim}N(0,0.3^2).
\]

The second is a lower-tail-risk specification used for both panels of Table
2. Here \(a_R(x)=a^\dagger(x)\) denotes the risky action and
\(a_S(x)=1-a_R(x)\) the stable action. Under the nominal environment,

\[
Y\{a_R(x)\}
=
\begin{cases}
0, & \text{with probability }p_0=0.10,\\
0.90, & \text{with probability }1-p_0=0.90,
\end{cases}
\qquad
Y\{a_S(x)\}=0.79.
\]

Thus, the risky action has the larger nominal conditional mean,
\(0.90(1-p_0)=0.81>0.79\), but is exposed to a low-probability adverse
outcome. This specification separates nominal mean performance from
lower-tail stability, which is the mechanism examined in the
robust-versus-non-robust comparison.

Each experiment is repeated 100 times. In every replication, the training
sample contains 10,000 observations and the evaluation sample contains
2,500 observations. All methods within a replication use the same training
and evaluation samples.

## 5.3. Robust Versus Non-Robust Policy Learning

We compare the robust policy learner with its non-robust counterpart under
covariate shift and concept drift. Both panels of Table 2 use the lower-tail
baseline model defined in Section 5.1. The two policies are trained once per
replication and then evaluated under both types of shift. Consequently, the
score function, action boundary, behavior policy, outcome support, training
sample, and learned policies are identical across the two panels. The robust
learner uses the same training ambiguity radius, \(\eta=0.2\), for all shift
levels, whereas the non-robust learner optimizes the nominal empirical
objective.

### Covariate shift

Covariate shift changes the deployment distribution of \(X\) while leaving
the conditional outcome kernel fixed. For environment \(m\) and evaluation
observation \(i\), draw an auxiliary tilting score \(z_{im}\sim N(0,1)\) and
define

\[
q_{im}(\lambda_1)
=
\frac{\exp(\lambda_1 z_{im})}
{\sum_{j=1}^{n_{\mathrm{test}}}\exp(\lambda_1 z_{jm})}.
\]

We construct an environment by drawing \(n_{\mathrm{test}}\) observations
with replacement from the nominal evaluation sample using probabilities
\(\{q_{im}(\lambda_1)\}_{i=1}^{n_{\mathrm{test}}}\). Larger
\(\lambda_1\) produces a more concentrated deployment distribution. This
resampling changes the empirical covariate distribution but does not alter
\(P(Y\mid X,A)\).

### Concept drift

Concept drift leaves \(P_X\) unchanged and modifies only the catastrophe
probability of the risky action. Let

\[
\phi(x)
=
\left(
1,\,
\widetilde{x}_1,\,
\widetilde{x}_2,\,
\widetilde{x_1x_2},\,
\widetilde{x_1^2-x_2^2},\,
\widetilde{\sin(\pi x_1)},\,
\widetilde{\cos(\pi x_2)}
\right)^\top,
\]

where the nonconstant components are standardized in the evaluation sample.
For each deployment environment \(m\), draw
\(\theta_m\sim N(0,I_7)\), normalize it to unit Euclidean length, and define
the heterogeneous susceptibility

\[
h_m(x)
=\exp\!\left[
0.35\operatorname{clip}
\{\theta_m^\top\phi(x),-2,2\}
\right].
\]

The catastrophe probability in environment \(m\) at severity
\(\lambda_2\) is

\[
p_{m,\lambda_2}(x)
=
\operatorname{logit}^{-1}
\left\{
\operatorname{logit}(p_0)
+\tau_{m,\lambda_2}h_m(x)
\right\},
\]

where \(\tau_{m,\lambda_2}\geq 0\) is chosen so that

\[
\frac{1}{n_{\mathrm{test}}}
\sum_{i=1}^{n_{\mathrm{test}}}
D_{\mathrm{KL}}
\left[
\operatorname{Bern}\{p_{m,\lambda_2}(X_i)\}
\;\middle\|\;
\operatorname{Bern}(p_0)
\right]
=
\eta_{\mathrm{test}}(\lambda_2),
\qquad
\eta_{\mathrm{test}}(\lambda_2)
=\frac{1}{2}(0.2\lambda_2)^2.
\]

The stable outcome remains \(0.79\), and the risky outcome remains supported
on \(\{0,0.90\}\). Hence concept drift changes only the conditional
probability of the adverse outcome, not the covariate distribution, action
boundary, or outcome support. We use common catastrophe uniforms across
severities and environments within each replication to reduce simulation
noise.

For each replication, shift type, and severity, we generate \(M=100\)
deployment environments. Let

\[
Q_m(\widehat{\pi})
=
\frac{1}{n_{\mathrm{test}}}
\sum_{i=1}^{n_{\mathrm{test}}}
Y_{im}\{\widehat{\pi}(X_{im})\}
\]

denote the value of learned policy \(\widehat{\pi}\) in environment \(m\).
We report its worst-environment value,

\[
Q_{\min}(\widehat{\pi})
=\min_{1\leq m\leq M}Q_m(\widehat{\pi}),
\]

averaged over 100 replications. Parentheses in Table 2 contain Monte Carlo
standard errors. The relative improvement is

\[
100\times
\frac{
\overline{Q}_{\min}(\widehat{\pi}_{\mathrm{rob}})
-\overline{Q}_{\min}(\widehat{\pi}_{\mathrm{non}})
}{
\overline{Q}_{\min}(\widehat{\pi}_{\mathrm{non}})
}.
\]

**Table 2. Worst-environment policy value under distribution shift**

| Shift type | Severity | Robust \(Q_{\min}\) | Non-robust \(Q_{\min}\) | Improvement |
|---|---:|---:|---:|---:|
| Covariate shift | \(\lambda_1=0.5\) | 0.7870 (0.0004) | 0.7907 (0.0005) | -0.47% |
|  | \(\lambda_1=1.0\) | 0.7819 (0.0004) | 0.7834 (0.0006) | -0.19% |
|  | \(\lambda_1=1.5\) | 0.7631 (0.0015) | 0.7529 (0.0022) | 1.36% |
|  | \(\lambda_1=2.0\) | 0.7193 (0.0045) | 0.6723 (0.0071) | 6.99% |
|  | \(\lambda_1=2.5\) | 0.6488 (0.0092) | 0.5491 (0.0124) | 18.15% |
|  | \(\lambda_1=3.0\) | 0.5678 (0.0135) | 0.4228 (0.0153) | 34.30% |
| Concept drift | \(\lambda_2=0.5\) | 0.7869 (0.0003) | 0.7825 (0.0005) | 0.56% |
|  | \(\lambda_2=1.0\) | 0.7792 (0.0005) | 0.7612 (0.0006) | 2.36% |
|  | \(\lambda_2=1.5\) | 0.7713 (0.0008) | 0.7390 (0.0008) | 4.36% |
|  | \(\lambda_2=2.0\) | 0.7627 (0.0011) | 0.7158 (0.0011) | 6.56% |
|  | \(\lambda_2=2.5\) | 0.7540 (0.0015) | 0.6909 (0.0014) | 9.14% |
|  | \(\lambda_2=3.0\) | 0.7448 (0.0018) | 0.6653 (0.0017) | 11.94% |

Under mild covariate shift, the two methods have similar worst-environment
values, with differences below one-half of one percent. The robust advantage
becomes positive at \(\lambda_1=1.5\) and increases to \(34.30\%\) at
\(\lambda_1=3\). Under concept drift, the robust advantage increases
monotonically from \(0.56\%\) to \(11.94\%\). Both policies lose value as the
catastrophe probability increases, but the robust policy deteriorates more
slowly. The learned robust policies choose the risky action for \(21.0\%\) of
evaluation observations on average, compared with \(61.9\%\) for the
non-robust policies. This difference in lower-tail exposure explains why the
performance gap expands as concept drift becomes more severe.

## Revision Note

This replacement defines one general simulation framework and treats the
smooth-outcome and lower-tail-risk models as two conditional outcome
specifications within that framework. Both panels of Table 2 now use the same
lower-tail baseline DGP and the same learned policies. The covariate-shift
panel has therefore been rerun and replaced. The concept-drift panel is
unchanged from the previously verified 100-replication result. Table 1, Table
3, Figures 3--4, and their underlying simulations are unchanged.
