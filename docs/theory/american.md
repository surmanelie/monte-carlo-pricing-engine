# American options

The holder chooses the exercise time $\tau$ among the exercise dates $t_1 < \dots < t_M$:

$$ V_0 = \sup_{\tau}\ \mathbb E\big[e^{-r\tau}h(S_\tau)\big]. $$

With finitely many dates the option is *Bermudan*; as $M\to\infty$ it converges to the
American price.

## Binomial tree (reference)

The Cox, Ross & Rubinstein (1979) tree uses $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$ and
$p = (e^{(r-q)\Delta t} - d)/(u - d)$, and rolls back

$$
V_{i,j} = \max\Big(h(S_{i,j}),\ e^{-r\Delta t}\big(pV_{i+1,j+1} + (1-p)V_{i+1,j}\big)\Big)
$$

at exercise dates (only the continuation value elsewhere), vectorised over nodes. Plain
CRR prices oscillate with $N$. Following Broadie & Detemple (1996), the continuation value
at the last step is replaced by the Black-Scholes price (**BBS**), and two-point Richardson
extrapolation $2V(N) - V(N/2)$ cancels the leading $O(1/N)$ term (**BBSR**). The American
put's early-exercise boundary is read off the tree as the highest node where exercising is
optimal.

## Longstaff-Schwartz least-squares Monte Carlo

**Regression pass.** On training paths, going backwards over the exercise dates, the
realised discounted cash flow $Y$ of the current policy is regressed, **on in-the-money
paths only**, on basis functions of the moneyness $x = S/K$:

$$
\hat C_k(S) = \sum_j\hat\beta_{k,j}\psi_j(S/K),\qquad
\psi \in \{1,\ e^{-x/2}L_0(x),\ e^{-x/2}L_1(x),\ \dots\}
$$

(weighted Laguerre polynomials, as in Longstaff & Schwartz, 2001; monomials are also
available), solved with `numpy.linalg.lstsq`. The policy exercises when
$h(S) \ge \hat C_k(S)$.

**Pricing pass on independent paths.** Applying the frozen policy to fresh paths gives an
unbiased estimate of the value of a feasible, hence sub-optimal, stopping rule. The
estimator is therefore **low-biased**, $\mathbb E[\hat V] \le V$. Reusing the training
paths instead adds an upward "foresight" bias, since the policy is fitted to the paths it
is evaluated on. That in-sample value is reported as a diagnostic. The low bias shrinks
with a richer basis and more training paths (Clément, Lamberton & Protter, 2002).
Glasserman (2003, Section 8.6) discusses both biases.

**Validation.** The table reproduces the grid of Longstaff & Schwartz (2001, Table 1)
($S_0\in\{36,40,44\}$, $\sigma\in\{0.2,0.4\}$, $T\in\{1,2\}$, $K = 40$, $r = 0.06$,
50 exercise dates per year). The reference is our own **Bermudan** CRR tree with the same
exercise dates and 5,000 steps per year, which compares like with like. An American call
on a non-dividend-paying stock is never exercised early, so its LSM price is compared with
Black-Scholes.
