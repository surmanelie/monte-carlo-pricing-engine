# Getting started

## Installation

```bash
git clone https://github.com/surmanelie/monte-carlo-pricing-engine.git
cd monte-carlo-pricing-engine
pip install -e ".[dev]"          # library, CLI, tests and linters
pip install -e ".[fast]"         # optional Numba kernels (Heston QE, LSM)
pip install -e ".[app]"          # Streamlit dashboard
pip install -e ".[docs]"         # this documentation
```

Python 3.11 or newer is required. The core dependencies are NumPy, SciPy and Matplotlib
(plus Typer/Rich for the command line).

## Command line

```bash
mcengine price --model gbm --product european --type call --K 100 --n 100000
mcengine price --model heston --product barrier --type down-and-out-call --K 100 --B 90 --n 1000000 --method qmc
mcengine price --model gbm --product american --type put --S0 36 --K 40 --r 0.06 --method lsm
mcengine validate            # the full validation table (results/validation.md)
mcengine figures             # every figure into figures/
mcengine benchmark           # throughput and efficiency (results/benchmark.md)
mcengine calibrate --csv chain.csv --spot 100 --rate 0.02
```

`mcengine price` prints the estimate, its standard error and 95 % confidence interval, the
estimator's diagnostics and, whenever one exists, an independent reference price with the
error in standard errors. `mcengine --help` and `mcengine <command> --help` list all
options.

## Python API

```python
from mcengine import GBM, EuropeanOption, price_mc, price_analytic
from mcengine.models import Heston
from mcengine.products.barrier import BarrierOption
from mcengine.engines.fourier import gil_pelaez_price

model = GBM(s0=100.0, r=0.05, sigma=0.2)
call = EuropeanOption(strike=100.0, maturity=1.0)

mc = price_mc(model, call, n_paths=100_000, method="cv", seed=42)
bs = price_analytic(model, call)
print(mc, bs, mc.error_in_se(bs.price))

heston = Heston(s0=100.0, r=0.03, v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
qmc = price_mc(heston, call, n_paths=16 * 2**13, n_steps=50, method="qmc", seed=1)
print(qmc, gil_pelaez_price(heston, 100.0, 1.0))
```

Every result is a frozen [`PricingResult`](api/core.md) with `price`, `std_error`,
`ci_low`, `ci_high`, `n_paths`, `n_steps`, `method`, `elapsed_s` and `diagnostics`.

## Reproducing every number

```bash
mcengine validate && mcengine figures && mcengine benchmark
python scripts/render_readme.py   # refreshes the README and the docs tables
```

## Tests and quality gates

```bash
ruff check . && ruff format --check . && mypy --strict src
pytest --cov                     # fast suite (coverage >= 90 %)
pytest -m ""                     # including slow tests and pytest-benchmark
```
