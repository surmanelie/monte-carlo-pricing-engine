# Path-dependent products

## Asian options (discrete monitoring)

The payoff depends on the average of $S$ at fixing dates $t_i = iT/m$.

**Geometric average: closed form.** Under GBM, $\ln G = \frac1m\sum_i\ln S_{t_i}$ is
normal with

$$
\mu_G = \ln S_0 + (r - q - \tfrac12\sigma^2)\,\bar t,\qquad
\sigma_G^2 = \frac{\sigma^2}{m^2}\sum_{i,j}\min(t_i, t_j),
$$

so $C_G = e^{-rT}\big(e^{\mu_G + \sigma_G^2/2}N(d_1) - KN(d_2)\big)$ with
$d_1 = (\mu_G - \ln K + \sigma_G^2)/\sigma_G$ and $d_2 = d_1 - \sigma_G$ (Kemna & Vorst,
1990, discrete version).

**Arithmetic average: control variate.** The geometric payoff is almost perfectly
correlated with the arithmetic one and has a known mean. It is the classic control
variate, and the variance-reduction factor is reported by the engine.

**Arithmetic average: independent reference.** There is no closed form, so Monte Carlo
is validated against a *recursive density convolution* (Carverhill & Clewlow, 1990;
Benhamou, 2002). With $R_k = S_{t_k}/S_{t_{k-1}}$, $Z_m = R_m$ and
$Z_k = R_k(1 + Z_{k+1})$, we have $\sum_i S_{t_i} = S_0Z_1$. The density of
$X_k = \ln Z_k$ follows from that of $X_{k+1}$ by the change of variables
$Y = \ln(1 + e^{X_{k+1}})$,

$$ g(y) = f_{k+1}\big(\ln(e^y - 1)\big)\frac{1}{1 - e^{-y}},\quad y > 0, $$

and an FFT convolution with the Gaussian density of $\ln R_k$. Linear interpolation and
the trapezoidal rule give an $O(\Delta x^2)$ error, which Richardson extrapolation over
two grids removes.

## Barrier options

A knock-out option dies (a knock-in option is activated) when the price crosses the
barrier $H$. Knock-in payoffs are computed as `vanilla - knock-out` path by path, so
**in-out parity holds exactly for every estimator**.

**Continuous-monitoring reference.** The Reiner & Rubinstein (1991) formulas give all eight
single-barrier contracts (down/up, in/out, call/put) as combinations of four terms
$A, B, C, D$ (notation of Haug, 2007). For example, with $K > H$,
$C_{do} = A - C$ and $C_{di} = C$.

**Discrete monitoring bias.** Monitoring on $m$ dates misses crossings between dates.
The discrete price converges to the continuous one only at rate $O(\sqrt{\Delta t})$.

**Broadie-Glasserman-Kou continuity correction.** A discretely monitored option with
barrier $H$ is worth approximately a continuously monitored one with barrier
$He^{\pm\beta\sigma\sqrt{\Delta t}}$, where $\beta = -\zeta(1/2)/\sqrt{2\pi}$ and the shift
is away from the spot (Broadie, Glasserman & Kou, 1997). Conversely, a discrete simulation
with the barrier moved *towards* the spot estimates the continuous price.

**Brownian-bridge estimator.** Given two consecutive simulated log-prices, the path in
between is a Brownian bridge, whose probability of not crossing $H$ is

$$
1 - \exp\Big(-\frac{2\ln(S_i/H)\ln(S_{i+1}/H)}{\sigma^2\Delta t}\Big)
$$

(Glasserman, 2003, Section 6.4). Weighting the vanilla payoff by the product of these
survival probabilities gives an **unbiased** estimator of the continuously monitored price
under GBM. This is how the barrier rows of the validation table are computed.
