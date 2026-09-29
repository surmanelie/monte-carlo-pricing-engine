# Discrete delta hedging

A trader sells one call at the Black-Scholes price $V_0 = \mathrm{BS}(S_0, \sigma_h)$ and
holds $\Delta_i = \partial_S\mathrm{BS}(S_{t_i}, T - t_i, \sigma_h)$ shares, rebalanced at
$N$ equally spaced dates. The cash account accrues interest at $r$ and receives the
dividend yield on the shares. Proportional transaction costs $c|\Delta_i - \Delta_{i-1}|S_{t_i}$
are paid on every trade, including the initial purchase and the final liquidation. The
discounted P&L is

$$
\mathrm{PnL} = e^{-rT}\Big(B_T + \Delta_{N-1}S_T(1-c) - (S_T - K)^+\Big).
$$

**Correct model (GBM, $\sigma_h = \sigma$, no costs).** The P&L is a pure discretisation
error. Its mean is close to zero and its standard deviation decays like $N^{-1/2}$
(Boyle & Emanuel, 1980; Bertsimas, Kogan & Lo, 2000): over each interval the hedge
leaves a gamma exposure $\tfrac12\Gamma S^2(\Delta W^2 - \Delta t)$ whose variance is
$O(\Delta t^2)$, summed over $N$ intervals.

**Misspecified model (Heston).** Hedging with the Black-Scholes delta at the at-the-money
implied volatility leaves a volatility-risk component. The hedging error then plateaus
instead of vanishing as $N\to\infty$.

**Transaction costs.** Costs grow with the rebalancing frequency (roughly like $\sqrt N$
for proportional costs; Leland, 1985). The mean P&L therefore decreases with $N$ while
the standard deviation still shrinks.

All frequencies are evaluated on the **same** simulated paths (a fine grid whose step
count is a multiple of every $N$), so the comparison across $N$ uses common random numbers.
