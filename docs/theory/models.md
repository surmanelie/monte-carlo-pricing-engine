# Models

All models are risk-neutral with constant rate $r$ and dividend yield $q$. They map a
tensor of independent standard normals to price paths.

## Black-Scholes GBM

$$dS_t = (r - q)S_t\,dt + \sigma S_t\,dW_t$$

is simulated exactly on any grid (Black & Scholes, 1973; Glasserman, 2003, Section 3.2):

$$
S_{t_{k+1}} = S_{t_k}\exp\big((r - q - \tfrac12\sigma^2)\Delta t_k + \sigma\sqrt{\Delta t_k}Z_k\big).
$$

## Heston

$$
dS_t = (r-q)S_t\,dt + \sqrt{V_t}S_t\,dW^S_t,\quad
dV_t = \kappa(\theta - V_t)\,dt + \xi\sqrt{V_t}\,dW^V_t,\quad
d\langle W^S, W^V\rangle_t = \rho\,dt
$$

(Heston, 1993). The variance hits zero unless the Feller condition
$2\kappa\theta \ge \xi^2$ holds. Calibrated parameters often violate it, and naive
discretisations then become badly biased.

### Andersen's quadratic-exponential (QE) scheme

Given $V_t$, the conditional law of $V_{t+\Delta}$ (a scaled non-central $\chi^2$) has mean
and variance

$$
m = \theta + (V_t - \theta)e^{-\kappa\Delta},\quad
s^2 = \frac{V_t\xi^2e^{-\kappa\Delta}}{\kappa}(1 - e^{-\kappa\Delta})
+ \frac{\theta\xi^2}{2\kappa}(1 - e^{-\kappa\Delta})^2 .
$$

With $\psi = s^2/m^2$ and $\psi_c = 1.5$ (Andersen, 2008):

- $\psi \le \psi_c$: $V_{t+\Delta} = a(b + Z_V)^2$ with $b^2 = 2/\psi - 1 + \sqrt{2/\psi}\sqrt{2/\psi - 1}$, $a = m/(1+b^2)$;
- $\psi > \psi_c$: a point mass $p = (\psi-1)/(\psi+1)$ at zero plus an exponential tail,
  $V_{t+\Delta} = \beta^{-1}\ln\frac{1-p}{1-U}$ for $U > p$, $\beta = (1-p)/m$.

The log-price uses the central discretisation of $\int V\,ds$,

$$
\ln S_{t+\Delta} = \ln S_t + (r-q)\Delta + K_0^* + K_1V_t + K_2V_{t+\Delta}
+ \sqrt{K_3V_t + K_4V_{t+\Delta}}\,Z,
$$

$K_1 = \tfrac12\Delta(\kappa\rho/\xi - \tfrac12) - \rho/\xi$,
$K_2 = \tfrac12\Delta(\kappa\rho/\xi - \tfrac12) + \rho/\xi$,
$K_3 = K_4 = \tfrac12\Delta(1-\rho^2)$. The **martingale correction** $K_0^*$ is chosen so
that $\mathbb E[S_{t+\Delta}\mid S_t, V_t] = S_te^{(r-q)\Delta}$ exactly:

$$
K_0^* = -\frac{A b^2 a}{1 - 2Aa} + \tfrac12\ln(1 - 2Aa) - (K_1 + \tfrac12K_3)V_t
\quad\text{or}\quad
K_0^* = -\ln\Big(p + \frac{\beta(1-p)}{\beta - A}\Big) - (K_1 + \tfrac12K_3)V_t,
$$

with $A = K_2 + \tfrac12K_4$.

### Full-truncation Euler

For comparison (Lord, Koekkoek & van Dijk, 2010), with $V^+ = \max(V, 0)$:

$$
V_{t+\Delta} = V_t + \kappa(\theta - V_t^+)\Delta + \xi\sqrt{V_t^+\Delta}\,Z_V,\qquad
\ln S_{t+\Delta} = \ln S_t + (r - q - \tfrac12V_t^+)\Delta + \sqrt{V_t^+\Delta}\,Z_S .
$$

Its bias is $O(\Delta)$ and large when the Feller condition fails. The figure on the
[Results](../results.md) page compares both schemes as $\Delta$ shrinks.

### Characteristic function ("little Heston trap")

$$
\varphi(u) = \exp\Big(iu(\ln S_0 + (r-q)T)
+ \frac{\kappa\theta}{\xi^2}\big[(\kappa - \rho\xi iu - d)T - 2\ln\tfrac{1 - ge^{-dT}}{1-g}\big]
+ \frac{v_0}{\xi^2}(\kappa - \rho\xi iu - d)\frac{1 - e^{-dT}}{1 - ge^{-dT}}\Big),
$$

with $d = \sqrt{(\rho\xi iu - \kappa)^2 + \xi^2(iu + u^2)}$ and
$g = (\kappa - \rho\xi iu - d)/(\kappa - \rho\xi iu + d)$. This form keeps the complex
logarithm on its principal branch for all maturities (Albrecher et al., 2007).

## Merton jump-diffusion

$$
\frac{dS_t}{S_{t^-}} = (r - q - \lambda\bar k)\,dt + \sigma\,dW_t + (J - 1)\,dN_t,\quad
\ln J\sim N(\mu_J, \delta^2),\quad \bar k = e^{\mu_J + \delta^2/2} - 1
$$

(Merton, 1976). The simulation is exact: over a step the number of jumps is
$\mathrm{Poisson}(\lambda\Delta)$ (drawn by inversion of $\Phi(Z_2)$), and given $n$ jumps
the sum of log-jumps is exactly $N(n\mu_J, n\delta^2)$. The reference price is the
Poisson mixture of Black-Scholes prices

$$
V = \sum_{n\ge0} e^{-\lambda'T}\frac{(\lambda'T)^n}{n!}\,\mathrm{BS}(S_0, K, T, r_n, \sigma_n),\quad
\lambda' = \lambda(1+\bar k),\ \sigma_n^2 = \sigma^2 + \frac{n\delta^2}{T},\
r_n = r - \lambda\bar k + \frac{n\ln(1+\bar k)}{T},
$$

truncated once the remaining Poisson mass is below a tolerance $\varepsilon$ (error at
most $\varepsilon\max(S_0, K)$).
