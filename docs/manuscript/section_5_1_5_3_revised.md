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
The first is the smooth-outcome specification used in Sections 5.2 and 5.4
and in the covariate-shift panel of Table 2. Its latent conditional means are

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

The second is a lower-tail-risk specification used for the concept-drift
panel of Table 2. Here \(a_R(x)=a^\dagger(x)\) denotes the risky action and
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
covariate shift and concept drift. The covariate-shift panel retains the
smooth-outcome specification, whereas the concept-drift panel uses the
lower-tail-risk specification defined in Section 5.1. The policy-learning
configuration and worst-environment evaluation criterion are otherwise
unchanged. Within each panel and replication, the robust and non-robust
methods use the same training and evaluation samples. The robust learner uses
the same training ambiguity radius, \(\eta=0.2\), for all shift levels,
whereas the non-robust learner optimizes the nominal empirical objective.

### Covariate shift

Covariate shift changes the deployment distribution of \(X\) while leaving
the conditional outcome kernel fixed. Let \(s(x_1,x_2)\) denote the score
defining the nominal treatment boundary and define the middle-margin score

\[
h(x)=|s(x_1,x_2)|\exp\{-\gamma|s(x_1,x_2)|\}.
\]

This score is zero at the treatment boundary, reaches its maximum at
\(|s(x_1,x_2)|=1/\gamma\), and decreases far from the boundary. For evaluation
observation \(i\), define

\[
q_i(\lambda_1)
=
\frac{\exp\{\beta(\lambda_1)h(X_i)\}}
{\sum_{j=1}^{n_{\mathrm{test}}}\exp\{\beta(\lambda_1)h(X_j)\}}.
\]

For each replication and severity, \(\beta(\lambda_1)\) is calibrated so that
\(D_{\mathrm{KL}}\{q(\lambda_1)\|p_0\}\) equals, respectively,
\(\{0.005,0.02,0.045,0.08,0.125,0.18\}\) for
\(\lambda_1\in\{0.5,1,1.5,2,2.5,3\}\), where \(p_0\) is uniform on the
nominal evaluation sample. We construct each of 100 environments by drawing
\(n_{\mathrm{test}}\) observations with replacement using probabilities
\(\{q_i(\lambda_1)\}_{i=1}^{n_{\mathrm{test}}}\). Larger \(\lambda_1\)
concentrates the deployment distribution in the intermediate-margin region.
This resampling changes the empirical covariate distribution but does not alter
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
| Covariate shift | \(\lambda_1=0.5\) | 0.7592 (0.0006) | 0.7436 (0.0006) | 2.09% |
|  | \(\lambda_1=1.0\) | 0.7570 (0.0006) | 0.7401 (0.0006) | 2.29% |
|  | \(\lambda_1=1.5\) | 0.7552 (0.0007) | 0.7366 (0.0006) | 2.53% |
|  | \(\lambda_1=2.0\) | 0.7534 (0.0007) | 0.7329 (0.0007) | 2.79% |
|  | \(\lambda_1=2.5\) | 0.7512 (0.0008) | 0.7291 (0.0008) | 3.04% |
|  | \(\lambda_1=3.0\) | 0.7495 (0.0008) | 0.7256 (0.0008) | 3.29% |
| Concept drift | \(\lambda_2=0.5\) | 0.7887 (0.0002) | 0.7816 (0.0006) | 0.90% |
|  | \(\lambda_2=1.0\) | 0.7858 (0.0002) | 0.7573 (0.0007) | 3.76% |
|  | \(\lambda_2=1.5\) | 0.7829 (0.0002) | 0.7327 (0.0009) | 6.84% |
|  | \(\lambda_2=2.0\) | 0.7796 (0.0003) | 0.7067 (0.0011) | 10.31% |
|  | \(\lambda_2=2.5\) | 0.7762 (0.0004) | 0.6793 (0.0014) | 14.26% |
|  | \(\lambda_2=3.0\) | 0.7729 (0.0004) | 0.6506 (0.0017) | 18.79% |

Under covariate shift, the average robust improvement increases monotonically
from \(2.09\%\) at \(\lambda_1=0.5\) to \(3.29\%\) at \(\lambda_1=3\).
Both policy values decline as deployment mass shifts toward the middle-margin
region, but the robust policy deteriorates more slowly. Under the revised
concept drift, the robust advantage increases monotonically from \(0.90\%\) to
\(18.79\%\). Both concept-drift policies lose value as the catastrophe
probability increases, but the robust policy deteriorates substantially more
slowly. The expanding gap is consistent with the robust objective placing more
weight on lower-tail outcomes during policy learning.

## Revision Note

This replacement defines one general simulation framework and treats the
smooth-outcome and lower-tail-risk models as two conditional outcome
specifications within that framework. The covariate-shift panel retains the
smooth-outcome equation but replaces the random tilt with the middle-margin
covariate shift and reports the corresponding 100-replication results. The
concept-drift panel retains the revised lower-tail-risk DGP. Table 1, Table 3,
Figures 3--4, and their underlying simulations are unchanged.
