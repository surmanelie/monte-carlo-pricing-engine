| Model | Product | Method | Paths | Steps | MC price | 95 % CI | Reference | Ref. method | Error (SE) | In 99 % CI | Pass |
|---|---|---|--:|--:|--:|---|--:|---|--:|:-:|:-:|
| GBM | European call K=100 T=1 | `mc-plain` | 100,000 | 1 | 10.5628 | [10.4708, 10.6548] | 10.4506 | Black-Scholes | +2.39 | yes | ✅ |
| GBM | European call K=100 T=1 | `mc-antithetic` | 100,000 | 1 | 10.4604 | [10.3959, 10.5250] | 10.4506 | Black-Scholes | +0.30 | yes | ✅ |
| GBM | European call K=100 T=1 | `mc-cv` | 100,000 | 1 | 10.4743 | [10.4395, 10.5092] | 10.4506 | Black-Scholes | +1.34 | yes | ✅ |
| GBM | European call K=120 T=1 | `mc-plain` | 100,000 | 1 | 3.2346 | [3.1810, 3.2882] | 3.2475 | Black-Scholes | -0.47 | yes | ✅ |
| GBM | European call K=120 T=1 | `mc-antithetic` | 100,000 | 1 | 3.2300 | [3.1798, 3.2802] | 3.2475 | Black-Scholes | -0.68 | yes | ✅ |
| GBM | European call K=120 T=1 | `mc-cv` | 100,000 | 1 | 3.1934 | [3.1584, 3.2284] | 3.2475 | Black-Scholes | -3.03 | no | ✅ |
| GBM | European put K=100 T=1 | `mc-plain` | 100,000 | 1 | 5.6168 | [5.5629, 5.6707] | 5.5735 | Black-Scholes | +1.57 | yes | ✅ |
| GBM | European put K=100 T=1 | `mc-antithetic` | 100,000 | 1 | 5.5664 | [5.5254, 5.6074] | 5.5735 | Black-Scholes | -0.34 | yes | ✅ |
| GBM | European put K=100 T=1 | `mc-cv` | 100,000 | 1 | 5.5844 | [5.5497, 5.6192] | 5.5735 | Black-Scholes | +0.61 | yes | ✅ |
| GBM | Digital call K=100 T=1 | `mc-plain` | 100,000 | 1 | 0.5339 | [0.5310, 0.5368] | 0.5323 | e^{-rT} N(d2) | +1.05 | yes | ✅ |
| GBM | Digital call K=100 T=1 | `mc-cv` | 100,000 | 1 | 0.5333 | [0.5315, 0.5352] | 0.5323 | e^{-rT} N(d2) | +1.07 | yes | ✅ |

11/11 rows pass (|error| < 4 SE); 10/11 references lie inside the 99% confidence interval (about 0.1 misses expected by chance for unbiased estimators).
