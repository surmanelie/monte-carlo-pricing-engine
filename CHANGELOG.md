# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
