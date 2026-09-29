# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-09-29

### Added
- Heston model with the Andersen (2008) QE scheme (martingale correction, branch-wise
  vectorised step) and full-truncation Euler for comparison; "little Heston trap"
  characteristic function (Albrecher et al. 2007); Feller-ratio diagnostic.
- Merton jump-diffusion with exact simulation (Poisson counts by inversion, exact
  conditional jump sums) and the Merton series formula with an explicit truncation
  tolerance.
- Fourier pricing: adaptive Gil-Pelaez inversion (reference), vectorised
  Gauss-Legendre Gil-Pelaez for many strikes, and Carr-Madan FFT with configurable
  damping.
- Implied volatility: vectorised Newton with vega, Brent fallback, no-arbitrage bound
  checks.
- Validation rows for Heston (QE and Euler) against Gil-Pelaez and for Merton against the
  series formula.
- Figures: Heston/Merton implied-volatility smiles with Monte Carlo points, and QE vs
  Euler discretisation bias vs time step in a Feller-violating case.

## [0.3.0] - 2026-09-29

### Added
- `AmericanOption` product (Bermudan exercise on an equally spaced grid of dates).
- Cox-Ross-Rubinstein binomial tree for European, American and Bermudan exercise, with
  Black-Scholes smoothing and Richardson extrapolation (BBSR, Broadie-Detemple 1996) and
  early-exercise boundary extraction.
- Longstaff-Schwartz least-squares Monte Carlo: in-the-money regression, weighted
  Laguerre or monomial basis, independent pricing paths (low-biased estimator), chunked
  pricing pass, in-sample value reported as a diagnostic, training-support-aware
  exercise boundary.
- Validation on the Longstaff & Schwartz (2001) Table 1 grid against a Bermudan CRR tree
  (N = 5000 steps per year) and American call = European call.
- Figure: early-exercise boundary, LSM vs tree.

### Changed
- `price_mc` rejects early-exercise products and points to `price_lsm`.

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
