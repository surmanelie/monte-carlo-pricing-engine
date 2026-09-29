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
| GBM S0=36 σ=0.2 | American put K=40 T=1 (50 dates) | `lsm` | 100,000 | 50 | 4.4732 | [4.4552, 4.4911] | 4.4778 | CRR Bermudan tree (N=5000, BBS-Richardson) | -0.51 | yes | ✅ |
| GBM S0=36 σ=0.2 | American put K=40 T=2 (100 dates) | `lsm` | 100,000 | 100 | 4.8238 | [4.8021, 4.8456] | 4.8402 | CRR Bermudan tree (N=10000, BBS-Richardson) | -1.47 | yes | ✅ |
| GBM S0=36 σ=0.4 | American put K=40 T=1 (50 dates) | `lsm` | 100,000 | 50 | 7.0942 | [7.0569, 7.1315] | 7.1013 | CRR Bermudan tree (N=5000, BBS-Richardson) | -0.37 | yes | ✅ |
| GBM S0=36 σ=0.4 | American put K=40 T=2 (100 dates) | `lsm` | 100,000 | 100 | 8.5025 | [8.4583, 8.5466] | 8.5068 | CRR Bermudan tree (N=10000, BBS-Richardson) | -0.19 | yes | ✅ |
| GBM S0=40 σ=0.2 | American put K=40 T=1 (50 dates) | `lsm` | 100,000 | 50 | 2.3063 | [2.2893, 2.3232] | 2.3141 | CRR Bermudan tree (N=5000, BBS-Richardson) | -0.90 | yes | ✅ |
| GBM S0=40 σ=0.2 | American put K=40 T=2 (100 dates) | `lsm` | 100,000 | 100 | 2.8943 | [2.8735, 2.9151] | 2.8846 | CRR Bermudan tree (N=10000, BBS-Richardson) | +0.92 | yes | ✅ |
| GBM S0=40 σ=0.4 | American put K=40 T=1 (50 dates) | `lsm` | 100,000 | 50 | 5.2793 | [5.2440, 5.3146] | 5.3120 | CRR Bermudan tree (N=5000, BBS-Richardson) | -1.82 | yes | ✅ |
| GBM S0=40 σ=0.4 | American put K=40 T=2 (100 dates) | `lsm` | 100,000 | 100 | 6.9135 | [6.8707, 6.9562] | 6.9171 | CRR Bermudan tree (N=10000, BBS-Richardson) | -0.17 | yes | ✅ |
| GBM S0=44 σ=0.2 | American put K=40 T=1 (50 dates) | `lsm` | 100,000 | 50 | 1.1181 | [1.1052, 1.1311] | 1.1099 | CRR Bermudan tree (N=5000, BBS-Richardson) | +1.25 | yes | ✅ |
| GBM S0=44 σ=0.2 | American put K=40 T=2 (100 dates) | `lsm` | 100,000 | 100 | 1.6714 | [1.6545, 1.6884] | 1.6898 | CRR Bermudan tree (N=10000, BBS-Richardson) | -2.12 | yes | ✅ |
| GBM S0=44 σ=0.4 | American put K=40 T=1 (50 dates) | `lsm` | 100,000 | 50 | 3.9597 | [3.9276, 3.9918] | 3.9477 | CRR Bermudan tree (N=5000, BBS-Richardson) | +0.74 | yes | ✅ |
| GBM S0=44 σ=0.4 | American put K=40 T=2 (100 dates) | `lsm` | 100,000 | 100 | 5.6122 | [5.5720, 5.6524] | 5.6412 | CRR Bermudan tree (N=10000, BBS-Richardson) | -1.42 | yes | ✅ |
| GBM S0=40 σ=0.2 | American call K=40 T=1 (50 dates) | `lsm` | 100,000 | 50 | 4.4212 | [4.3840, 4.4585] | 4.3958 | Black-Scholes (no early exercise) | +1.34 | yes | ✅ |
| Heston QE (Δt = 1/50) | European call K=90 T=1 | `mc-plain` | 100,000 | 50 | 15.7310 | [15.6489, 15.8132] | 15.7717 | Gil-Pelaez (little trap) | -0.97 | yes | ✅ |
| Heston QE (Δt = 1/50) | European call K=100 T=1 | `mc-plain` | 100,000 | 50 | 8.9515 | [8.8870, 9.0161] | 8.9294 | Gil-Pelaez (little trap) | +0.67 | yes | ✅ |
| Heston QE (Δt = 1/50) | European call K=110 T=1 | `mc-plain` | 100,000 | 50 | 3.9796 | [3.9358, 4.0234] | 3.9785 | Gil-Pelaez (little trap) | +0.05 | yes | ✅ |
| Heston QE (Δt = 1/50) | European call K=100 T=1 | `mc-antithetic` | 100,000 | 50 | 8.9599 | [8.9170, 9.0029] | 8.9294 | Gil-Pelaez (little trap) | +1.39 | yes | ✅ |
| Heston QE (Δt = 1/50) | European put K=100 T=1 | `mc-cv` | 100,000 | 50 | 5.9467 | [5.9120, 5.9815] | 5.9740 | Gil-Pelaez (little trap) | -1.54 | yes | ✅ |
| Heston Euler FT (Δt = 1/200) | European call K=100 T=1 | `mc-plain` | 100,000 | 200 | 8.9819 | [8.9174, 9.0465] | 8.9294 | Gil-Pelaez (little trap) | +1.59 | yes | ✅ |
| Merton | European call K=80 T=1 | `mc-plain` | 100,000 | 1 | 26.0874 | [25.9430, 26.2318] | 25.9555 | Merton series | +1.79 | yes | ✅ |
| Merton | European call K=100 T=1 | `mc-plain` | 100,000 | 1 | 12.7965 | [12.6826, 12.9105] | 12.7613 | Merton series | +0.61 | yes | ✅ |
| Merton | European call K=120 T=1 | `mc-plain` | 100,000 | 1 | 5.0374 | [4.9624, 5.1125] | 5.0906 | Merton series | -1.39 | yes | ✅ |
| Merton | European call K=100 T=1 | `mc-cv` | 100,000 | 1 | 12.7585 | [12.7113, 12.8057] | 12.7613 | Merton series | -0.12 | yes | ✅ |
| Merton | European put K=100 T=1 | `mc-antithetic` | 100,000 | 1 | 7.9331 | [7.8736, 7.9925] | 7.8842 | Merton series | +1.61 | yes | ✅ |
| GBM | European call K=100 T=1: delta | `mc-bump` | 100,000 | 1 | 0.6369 | [0.6334, 0.6405] | 0.6368 | Black-Scholes delta | +0.05 | yes | ✅ |
| GBM | European call K=100 T=1: delta | `mc-pathwise` | 100,000 | 1 | 0.6363 | [0.6327, 0.6398] | 0.6368 | Black-Scholes delta | -0.32 | yes | ✅ |
| GBM | European call K=100 T=1: delta | `mc-lr` | 100,000 | 1 | 0.6434 | [0.6342, 0.6525] | 0.6368 | Black-Scholes delta | +1.40 | yes | ✅ |
| GBM | European call K=100 T=1: gamma | `mc-bump` | 100,000 | 1 | 0.0190 | [0.0183, 0.0196] | 0.0188 | Black-Scholes gamma | +0.61 | yes | ✅ |
| GBM | European call K=100 T=1: gamma | `mc-pathwise` | 100,000 | 1 | 0.0187 | [0.0185, 0.0189] | 0.0188 | Black-Scholes gamma | -0.57 | yes | ✅ |
| GBM | European call K=100 T=1: gamma | `mc-lr` | 100,000 | 1 | 0.0189 | [0.0180, 0.0197] | 0.0188 | Black-Scholes gamma | +0.21 | yes | ✅ |
| GBM | European call K=100 T=1: vega | `mc-bump` | 100,000 | 1 | 37.4369 | [36.9689, 37.9049] | 37.5240 | Black-Scholes vega | -0.36 | yes | ✅ |
| GBM | European call K=100 T=1: vega | `mc-pathwise` | 100,000 | 1 | 37.7906 | [37.3173, 38.2639] | 37.5240 | Black-Scholes vega | +1.10 | yes | ✅ |
| GBM | European call K=100 T=1: vega | `mc-lr` | 100,000 | 1 | 37.8512 | [36.1463, 39.5561] | 37.5240 | Black-Scholes vega | +0.38 | yes | ✅ |
| GBM | Digital call K=100 T=1: delta | `mc-bump` | 100,000 | 1 | 0.0195 | [0.0189, 0.0201] | 0.0188 | Black-Scholes delta | +2.43 | yes | ✅ |
| GBM | Digital call K=100 T=1: delta | `mc-lr` | 100,000 | 1 | 0.0187 | [0.0185, 0.0188] | 0.0188 | Black-Scholes delta | -1.14 | yes | ✅ |
| GBM | Digital call K=100 T=1: gamma | `mc-lr` | 100,000 | 1 | -0.0003 | [-0.0003, -0.0003] | -0.0003 | Black-Scholes gamma | +0.85 | yes | ✅ |
| GBM | Digital call K=100 T=1: vega | `mc-lr` | 100,000 | 1 | -0.6371 | [-0.6654, -0.6087] | -0.6567 | Black-Scholes vega | +1.36 | yes | ✅ |

63/63 rows pass (|error| < 4 SE); 62/63 references lie inside the 99% confidence interval (about 0.6 misses expected by chance for unbiased estimators).
