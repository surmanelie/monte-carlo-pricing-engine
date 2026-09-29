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
| GBM | Asian geome. call K=100 T=1 m=12 | `mc-plain` | 100,000 | 12 | 5.9115 | [5.8605, 5.9625] | 5.9402 | Kemna-Vorst | -1.10 | yes | ✅ |
| GBM | Asian geome. call K=100 T=1 m=12 | `mc-antithetic` | 100,000 | 12 | 5.9135 | [5.8781, 5.9490] | 5.9402 | Kemna-Vorst | -1.47 | yes | ✅ |
| GBM | Asian arith. call K=100 T=1 m=12 | `mc-plain` | 100,000 | 12 | 6.1446 | [6.0919, 6.1972] | 6.1560 | Recursive convolution | -0.43 | yes | ✅ |
| GBM | Asian arith. call K=100 T=1 m=12 | `mc-antithetic` | 100,000 | 12 | 6.1492 | [6.1126, 6.1857] | 6.1560 | Recursive convolution | -0.37 | yes | ✅ |
| GBM | Asian arith. call K=100 T=1 m=12 | `mc-cv` | 100,000 | 12 | 6.1563 | [6.1548, 6.1578] | 6.1560 | Recursive convolution | +0.38 | yes | ✅ |
| GBM | Asian arith. put K=100 T=1 m=12 | `mc-cv` | 100,000 | 12 | 3.5345 | [3.5336, 3.5354] | 3.5345 | Recursive convolution | +0.04 | yes | ✅ |
| GBM | Asian arith. call K=100 T=1 m=52 | `mc-cv` | 100,000 | 52 | 5.8544 | [5.8530, 5.8557] | 5.8539 | Recursive convolution | +0.71 | yes | ✅ |
| GBM | down-and-out call K=100 H=90 T=1 | `mc-plain` | 100,000 | 50 | 8.6471 | [8.5579, 8.7363] | 8.6655 | Reiner-Rubinstein | -0.40 | yes | ✅ |
| GBM | down-and-out put K=100 H=90 T=1 | `mc-plain` | 100,000 | 50 | 0.1547 | [0.1499, 0.1595] | 0.1512 | Reiner-Rubinstein | +1.43 | yes | ✅ |
| GBM | down-and-in call K=100 H=90 T=1 | `mc-plain` | 100,000 | 50 | 1.7820 | [1.7479, 1.8161] | 1.7851 | Reiner-Rubinstein | -0.18 | yes | ✅ |
| GBM | down-and-in put K=100 H=90 T=1 | `mc-plain` | 100,000 | 50 | 5.3827 | [5.3290, 5.4365] | 5.4223 | Reiner-Rubinstein | -1.44 | yes | ✅ |
| GBM | up-and-out call K=100 H=120 T=1 | `mc-plain` | 100,000 | 50 | 1.1798 | [1.1613, 1.1984] | 1.1761 | Reiner-Rubinstein | +0.40 | yes | ✅ |
| GBM | up-and-out put K=100 H=120 T=1 | `mc-plain` | 100,000 | 50 | 5.3772 | [5.3237, 5.4307] | 5.3601 | Reiner-Rubinstein | +0.63 | yes | ✅ |
| GBM | up-and-in call K=100 H=120 T=1 | `mc-plain` | 100,000 | 50 | 9.2409 | [9.1476, 9.3343] | 9.2745 | Reiner-Rubinstein | -0.70 | yes | ✅ |
| GBM | up-and-in put K=100 H=120 T=1 | `mc-plain` | 100,000 | 50 | 0.2150 | [0.2060, 0.2241] | 0.2134 | Reiner-Rubinstein | +0.36 | yes | ✅ |

26/26 rows pass (|error| < 4 SE); 25/26 references lie inside the 99% confidence interval (about 0.3 misses expected by chance for unbiased estimators).
