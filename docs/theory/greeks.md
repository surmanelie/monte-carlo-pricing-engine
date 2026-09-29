# Greeks

Under GBM, write $S_T = S_0\exp\big((r-q-\tfrac12\sigma^2)T + \sigma\sqrt TZ\big)$ and
$Y = e^{-rT}f(S_T)$. Each estimator averages i.i.d. per-path quantities, so every Greek
comes with a standard error (Glasserman, 2003, Ch. 7).

## Bump and revalue with common random numbers

$$
\hat\Delta = \frac{Y(S_0+h) - Y(S_0-h)}{2h},\quad
\hat\Gamma = \frac{Y(S_0+h) - 2Y(S_0) + Y(S_0-h)}{h^2},\quad
\hat{\mathcal V} = \frac{Y(\sigma+h_\sigma) - Y(\sigma-h_\sigma)}{2h_\sigma},
$$

evaluated on the **same** normals, so the difference of nearly identical quantities is
not swamped by sampling noise. The bias is $O(h^2)$. For discontinuous payoffs the
variance of $\hat\Gamma$ grows like $h^{-3}$.

## Pathwise derivatives

Differentiate the payoff along the path (Broadie & Glasserman, 1996):

$$
\Delta^{PW} = e^{-rT}f'(S_T)\frac{S_T}{S_0},\qquad
\mathcal V^{PW} = e^{-rT}f'(S_T)\frac{S_T}{\sigma}\Big(\ln\frac{S_T}{S_0} - (r - q + \tfrac12\sigma^2)T\Big).
$$

This is valid for payoffs that are Lipschitz in $S_T$. The vanilla $f'$ jumps at the
strike, so gamma uses the mixed pathwise/likelihood-ratio estimator
$\Gamma = e^{-rT}K\,\mathbf 1\{S_T > K\}\,Z/(S_0^2\sigma\sqrt T)$ (call). **For the
digital option $f' = 0$ almost everywhere.** The pathwise delta, gamma and vega are then
identically zero, which is wrong: the derivative of the expectation comes entirely from
the discontinuity, which differentiating inside the expectation cannot see.

## Likelihood ratio

Differentiate the density instead,
$\partial_\theta\mathbb E[Y] = \mathbb E[Y\,\partial_\theta\ln p_\theta(S_T)]$:

$$
w_\Delta = \frac{Z}{S_0\sigma\sqrt T},\quad
w_\Gamma = \frac{Z^2 - 1}{S_0^2\sigma^2T} - \frac{Z}{S_0^2\sigma\sqrt T},\quad
w_{\mathcal V} = \frac{Z^2-1}{\sigma} - Z\sqrt T .
$$

No smoothness of the payoff is needed, so it works for the digital. The price is higher
variance for smooth payoffs. The Greeks figure on the [Results](../results.md) page shows
all three estimators side by side.
