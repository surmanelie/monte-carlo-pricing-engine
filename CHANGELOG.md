# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-09-29

### Added
- Discretely monitored Asian options (arithmetic and geometric averages).
- Kemna-Vorst closed form for the discrete geometric Asian; geometric-Asian control
  variate for the arithmetic Asian, with the variance-reduction factor reported.
- Recursive density convolution (Carverhill-Clewlow / Benhamou) as an independent
  reference for discrete arithmetic Asians, with Richardson extrapolation over grids.
- Single-barrier options (all 8 knock-in/knock-out call/put types) with discrete
  monitoring, the Broadie-Glasserman-Kou continuity correction and a Brownian-bridge
  estimator of continuous monitoring.
- Reiner-Rubinstein continuous-barrier formulas and the BGK discrete approximation.
- Figure: barrier price vs number of monitoring dates.

## [0.1.0] - 2026-09-29

### Added
- Core engine: `PricingResult` / `GreeksResult` frozen dataclasses, streaming mean and
  covariance (Chan-Golub-LeVeque merge), confidence intervals, injectable RNG streams.
- Black-Scholes GBM model with exact simulation on arbitrary grids and its
  characteristic function.
- European call/put and cash-or-nothing digital products.
- Black-Scholes closed-form prices and Greeks (delta, gamma, vega, theta, rho; digital
  delta/gamma/vega).
- Monte Carlo engine with plain, antithetic and control-variate estimators, chunked
  constant-memory execution and chunk-size-invariant results under a fixed seed.
- Validation registry (`scripts/validate.py`) with a confidence-interval coverage study,
  and figures (`scripts/make_figures.py`): convergence with 95 % band, RMSE vs n.
