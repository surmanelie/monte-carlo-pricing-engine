Machine: python 3.12.14, platform Windows-11-10.0.26200-SP0, processor Intel64 Family 6 Model 154 Stepping 4, GenuineIntel, cpu_count 12, numpy 2.5.3, scipy 1.18.1, numba 0.67.0, mcengine 0.5.0

### Throughput

| Model | Engine | Backend | Steps | Paths | Time (s) | Paths / s | Path-steps / s |
|---|---|---|--:|--:|--:|--:|--:|
| GBM (exact) | `mc-plain` | numpy | 1 | 1,000,000 | 0.065 | 15,355,251 | 15,355,251 |
| GBM (exact) | `mc-plain` | numpy | 252 | 100,000 | 1.036 | 96,564 | 24,334,211 |
| Heston QE | `mc-plain` | numpy | 252 | 100,000 | 2.708 | 36,932 | 9,306,968 |
| Heston QE | `mc-plain` | numba | 252 | 100,000 | 1.627 | 61,464 | 15,488,900 |
| Heston Euler FT | `mc-plain` | numpy | 252 | 100,000 | 2.901 | 34,474 | 8,687,423 |
| Merton (exact) | `mc-plain` | numpy | 252 | 100,000 | 3.209 | 31,160 | 7,852,250 |
| GBM, American put | `lsm` | numpy | 50 | 100,000 | 1.340 | 74,643 | 3,732,127 |
| GBM, American put | `lsm` | numba | 50 | 100,000 | 0.890 | 112,352 | 5,617,599 |

### Efficiency at a fixed path budget

| Problem | Method | Paths | Std error | Time (s) | Variance reduction | Efficiency gain (var x time) |
|---|---|--:|--:|--:|--:|--:|
| European call K=140 | `mc-plain` | 1,048,576 | 4.14e-03 | 0.078 | 1.0 | 1.0 |
| European call K=140 | `mc-antithetic` | 1,048,576 | 4.07e-03 | 0.048 | 1.0 | 1.7 |
| European call K=140 | `mc-cv` | 1,048,576 | 3.54e-03 | 0.118 | 1.4 | 0.9 |
| European call K=140 | `qmc-sobol-bb` | 1,048,576 | 2.67e-05 | 0.102 | 24,043.0 | 18,401.4 |
| European call K=140 | `mc-is` | 1,048,576 | 8.54e-04 | 0.084 | 23.5 | 22.1 |
| Arithmetic Asian call K=100, m=12 | `mc-plain` | 1,048,576 | 8.31e-03 | 0.492 | 1.0 | 1.0 |
| Arithmetic Asian call K=100, m=12 | `mc-antithetic` | 1,048,576 | 5.75e-03 | 0.441 | 2.1 | 2.3 |
| Arithmetic Asian call K=100, m=12 | `mc-cv` | 1,048,576 | 2.32e-04 | 1.137 | 1,286.9 | 556.3 |
| Arithmetic Asian call K=100, m=12 | `qmc-sobol-bb` | 1,048,576 | 1.15e-04 | 1.713 | 5,236.4 | 1,502.4 |
| Arithmetic Asian call K=100, m=12 | `mc-is` | 1,048,576 | 9.80e-03 | 0.920 | 0.7 | 0.4 |
