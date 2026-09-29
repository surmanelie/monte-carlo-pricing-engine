# Interactive dashboard

The Streamlit dashboard lets you choose the model, product, parameters and estimator. For
each choice it shows:

- the Monte Carlo price with its 95 % confidence interval;
- the independent reference price (when one exists) and the error in standard errors;
- a convergence plot (estimates with confidence intervals for increasing path counts);
- sample paths with strike and barrier levels;
- the implied-volatility smile of the model (Fourier pricing + implied-vol inversion);
- the P&L distribution of a discrete delta hedge (rebalancing frequency and costs
  adjustable).

Heavy computations are cached with `st.cache_data`, keyed on every input.

## Run locally

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
2. Click **Create app**, choose the repository
   `surmanelie/monte-carlo-pricing-engine`, branch `main`, and main file
   `app/streamlit_app.py`.
3. Streamlit Cloud installs the dependencies listed in `app/requirements.txt`, which pins
   the package at the `v1.0.0` tag with the `[app]` extra.
4. Under **Advanced settings**, choose Python 3.12. No secrets are needed.
