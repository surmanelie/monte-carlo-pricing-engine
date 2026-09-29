# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-09-29

### Added
- Typer command-line interface: `mcengine price | validate | figures | benchmark |
  calibrate | version`, with Rich tables and an independent reference next to every
  price when one exists.
- Streamlit dashboard (`app/streamlit_app.py`): price with confidence interval,
  reference and error in standard errors, convergence plot, sample paths, implied-volatility
  smile and hedging P&L; cached computations; Streamlit Community Cloud instructions.
- MkDocs Material documentation site with MathJax theory pages, API reference generated
  from the docstrings, and a results page embedding the generated tables and figures;
  deployed to GitHub Pages by `docs.yml`.
- Three executed notebooks (walkthrough, Heston smile and calibration, Greeks and
  hedging), built by `scripts/build_notebooks.py`.
- `scripts/render_readme.py`: fills every numeric section of the README and the docs
  summary tables from the generated results.

## [0.7.0] - 2026-09-29

### Added
- Heston calibration to an implied-volatility surface: weighted least squares on implied
  volatilities (vectorised Gil-Pelaez pricing per maturity, vectorised implied-vol
  inversion), bounded trust-region reflective solver with multiple starts, Feller-ratio
  diagnostic.
- Synthetic surfaces generated from known parameters plus Gaussian noise, used to test
  parameter recovery.
- Optional loader for a user-provided option-chain CSV (documented schema: maturity,
  strike, implied_vol or price + option_type, optional weight); nothing is downloaded
  or scraped.
- Figure: market vs calibrated implied-volatility smiles and residual heatmap.

## [0.6.0] - 2026-09-29

### Added
- Scrambled Sobol sequences with Brownian-bridge path construction and randomised QMC
  (independent scramblings, honest standard errors) as `method="qmc"`.
- Importance sampling by a constant Girsanov drift shift with likelihood-ratio weights
  (`method="is"`), default shift centring the terminal price at the strike, and a
  variance-reduction diagnostic.
- Optional Numba backend (`[fast]` extra) for the Heston QE scheme and the LSM pricing
  pass, parallel over paths, identical to the NumPy implementation for the same seed.
- Benchmarks (`scripts/benchmark.py`, pytest-benchmark suite in `tests/benchmarks`):
  throughput per model/backend, RMSE vs wall-clock time per method, efficiency
  (variance x time) relative to plain Monte Carlo.
- Validation rows for QMC, importance sampling and the Numba backend.
- Figure: error vs wall-clock time for all methods.

### Changed
- Merton simulation builds one Poisson inverse-CDF table per distinct step length.

## [0.5.0] - 2026-09-29

### Added
- Monte Carlo Greeks under GBM (delta, gamma, vega) with standard errors: bump and
  revalue with common random numbers, pathwise derivatives (mixed pathwise-LR gamma for
  vanillas) and likelihood-ratio weights; the digital option illustrates why pathwise
  estimators fail on discontinuous payoffs while the likelihood ratio works.
- Discrete delta-hedging simulator: short option sold at the Black-Scholes price, N
  rebalancing dates on common paths, proportional transaction costs, dividends, GBM and
  Heston (misspecified hedge) dynamics, fitted log-log slope of the hedging-error std.
- Validation rows for Monte Carlo Greeks against Black-Scholes Greeks.
- Figures: Greeks estimators vs closed forms, hedging P&L histograms, hedging-error std
  vs rebalancing frequency.

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
