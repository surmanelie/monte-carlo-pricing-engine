# Monte Carlo estimators

We estimate $V_0 = e^{-rT}\,\mathbb E^{\mathbb Q}[h(S_{t_1},\dots,S_{t_m})]$ by averaging
$n$ discounted payoffs $Y_i$. By the central limit theorem,

$$
\hat V_n = \frac1n\sum_{i=1}^n Y_i,\qquad
\sqrt n(\hat V_n - V_0) \Rightarrow N(0, \sigma_Y^2),\qquad
\mathrm{SE} = \frac{s_Y}{\sqrt n},
$$

so the error decays like $n^{-1/2}$ and a 95 % confidence interval is
$\hat V_n \pm 1.96\,\mathrm{SE}$ (Glasserman, 2003, Ch. 1).

## Streaming statistics

Paths are simulated in chunks. Each chunk's mean $\bar x_b$ and centred co-moment matrix
$C_b$ are merged into the running totals with the update of Chan, Golub & LeVeque (1979),

$$
n = n_a + n_b,\quad \delta = \bar x_b - \bar x_a,\quad
\bar x = \bar x_a + \delta\frac{n_b}{n},\quad
C = C_a + C_b + \delta\delta^\top\frac{n_a n_b}{n},
$$

so $10^7$ paths run in constant memory and the result does not depend on the chunk size.

## Antithetic variates

Pairs $(Z, -Z)$ give $\tfrac12(Y(Z) + Y(-Z))$, whose variance is
$\tfrac12\sigma_Y^2(1 + \rho)$ with $\rho = \mathrm{Corr}(Y(Z), Y(-Z))$. It helps when the
payoff is monotone in $Z$. The standard error is computed from the $n/2$ independent pair
means.

## Control variates

With a control $X$ of known mean $\mathbb E X$,

$$
\hat V_{cv} = \bar Y - \hat\beta(\bar X - \mathbb E X),\qquad
\hat\beta = \frac{\widehat{\mathrm{Cov}}(Y, X)}{\widehat{\mathrm{Var}}(X)},\qquad
\frac{\mathrm{Var}(\hat V_{cv})}{\mathrm{Var}(\bar Y)} = 1 - \rho_{XY}^2 .
$$

Estimating $\hat\beta$ on the same sample introduces an $O(1/n)$ bias, which is negligible
at the sample sizes used (Glasserman, 2003, Section 4.1). The default controls are:

- vanilla, digital, barrier payoffs: the discounted terminal price $e^{-rT}S_T$, mean $S_0e^{-qT}$;
- arithmetic Asian under GBM: the discounted **geometric** Asian payoff with its
  Kemna-Vorst price as mean (Kemna & Vorst, 1990).

The engine reports $\hat\beta$, the correlation and the variance-reduction factor
$\widehat{\mathrm{Var}}(Y)/\widehat{\mathrm{Var}}(Y - \hat\beta X)$.

## Importance sampling

Draw the normals driving the price from $N(c_k, 1)$ instead of $N(0,1)$, with the
constant-drift (Girsanov) shift $c_k = \mu\sqrt{\Delta t_k/T}$, which moves $W_T$ by
$\mu\sqrt T$, and weight each payoff by the likelihood ratio

$$
L = \exp\Big(\sum_k -c_kZ_k + \tfrac12 c_k^2\Big),\qquad
\mathbb E_{\mathbb P}[Y] = \mathbb E_{\mathbb Q_\mu}[Y L].
$$

The default $\mu = (\ln(K/S_0) - (r - q - \tfrac12\sigma^2)T)/(\sigma\sqrt T)$ moves the
median of $S_T$ to the strike, so deep out-of-the-money payoffs become frequent events
(Glasserman, 2003, Section 4.6). The engine also estimates the variance plain Monte Carlo
would have had, $\mathbb E_{\mathbb Q}[Y^2L] - V^2$, and reports the variance-reduction factor.

## Randomised quasi-Monte Carlo

Scrambled Sobol points (Sobol', 1967; Owen, 1997) fill the unit cube more evenly than
random points. They are drawn in powers of two, mapped to normals by the inverse CDF, and
assigned by a **Brownian bridge**: the first (best distributed) coordinate builds
$W_T$, the next ones the midpoints, and so on,

$$
W_{t_j} = \frac{(t_r - t_j)W_{t_l} + (t_j - t_l)W_{t_r}}{t_r - t_l}
+ \sqrt{\frac{(t_j - t_l)(t_r - t_j)}{t_r - t_l}}\,Z_j,
$$

which concentrates the variance in the first dimensions (Moskowitz & Caflisch, 1996). The
map from bridge normals to increments is orthogonal, so the increments remain i.i.d.
$N(0,1)$ and every model can use them. $R$ independent scramblings give $R$ i.i.d.
unbiased estimates. Their standard deviation over $\sqrt R$ is an honest standard error
(L'Ecuyer & Lemieux, 2002).

## Efficiency

Estimators are compared by $\mathrm{Var}\times\text{time}$, the inverse of the
work-normalised precision (Glasserman, 2003, Section 1.1.3). See the error-vs-time figure
and the efficiency table on the [Results](../results.md) page.
