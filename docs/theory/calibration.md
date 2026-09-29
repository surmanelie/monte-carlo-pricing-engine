# Calibration

The Heston parameters $\Theta = (v_0, \kappa, \theta, \xi, \rho)$ solve the bounded
weighted least-squares problem in implied-volatility units,

$$
\min_{\Theta\in B}\ \sum_i w_i\big(\sigma^{\mathrm{model}}_i(\Theta) - \sigma^{\mathrm{mkt}}_i\big)^2,
$$

where the model volatility inverts the Heston price (vectorised Gil-Pelaez quadrature, one
call per maturity; out-of-the-money options are inverted). Working in volatility units
makes errors comparable across strikes and maturities (Gatheral, 2006, Ch. 3). The solver
is SciPy's bounded trust-region reflective `least_squares` (Branch, Coleman & Li, 1999),
started from several points, keeping the best fit.

**Feller condition.** $2\kappa\theta \ge \xi^2$ is *reported*, not imposed: it is
routinely violated by fits to market smiles, and the QE scheme handles that regime.

**Synthetic test.** A surface generated from known parameters, optionally with Gaussian
noise added to the implied volatilities, checks that the calibration recovers the
parameters. Without noise it does so to solver precision; with noise, the residual RMSE
should be of the order of the noise level. The measured parameters, residuals and heatmap
are on the [Results](../results.md) page.

**Real data (optional).** `mcengine calibrate --csv chain.csv --spot S --rate r` loads an
option chain that *you* provide. Nothing is downloaded or scraped. Schema:

```text
Required columns (header row, comma separated):
  maturity      time to expiry in years (> 0)
  strike        strike price (> 0)
and either
  implied_vol   Black-Scholes implied volatility (decimal, e.g. 0.215)
or both
  price         option mid price
  option_type   "call" or "put"
Optional column:
  weight        non-negative least-squares weight (default 1)
Spot, rate and dividend yield are passed as arguments.
```
