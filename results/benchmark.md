Machine: python 3.12.14, platform Windows-11-10.0.26200-SP0, processor Intel64 Family 6 Model 154 Stepping 4, GenuineIntel, cpu_count 12, numpy 2.5.3, scipy 1.18.1, numba 0.67.0, mcengine 1.0.0

### Throughput

| Model | Engine | Backend | Steps | Paths | Time (s) | Paths / s | Path-steps / s |
|---|---|---|--:|--:|--:|--:|--:|
| GBM (exact) | `mc-plain` | numpy | 1 | 1,000,000 | 0.096 | 10,412,480 | 10,412,480 |
| GBM (exact) | `mc-plain` | numpy | 252 | 100,000 | 1.417 | 70,549 | 17,778,315 |
| Heston QE | `mc-plain` | numpy | 252 | 100,000 | 5.453 | 18,338 | 4,621,172 |
| Heston QE | `mc-plain` | numba | 252 | 100,000 | 2.075 | 48,189 | 12,143,654 |
| Heston Euler FT | `mc-plain` | numpy | 252 | 100,000 | 3.590 | 27,853 | 7,019,064 |
| Merton (exact) | `mc-plain` | numpy | 252 | 100,000 | 5.240 | 19,085 | 4,809,371 |
| GBM, American put | `lsm` | numpy | 50 | 100,000 | 1.717 | 58,237 | 2,911,833 |
| GBM, American put | `lsm` | numba | 50 | 100,000 | 1.572 | 63,625 | 3,181,256 |

### Efficiency at a fixed path budget

| Problem | Method | Paths | Std error | Time (s) | Variance reduction | Efficiency gain (var x time) |
|---|---|--:|--:|--:|--:|--:|
| European call K=140 | `mc-plain` | 1,048,576 | 4.14e-03 | 0.121 | 1.0 | 1.0 |
| European call K=140 | `mc-antithetic` | 1,048,576 | 4.07e-03 | 0.075 | 1.0 | 1.7 |
| European call K=140 | `mc-cv` | 1,048,576 | 3.54e-03 | 0.177 | 1.4 | 0.9 |
| European call K=140 | `qmc-sobol-bb` | 1,048,576 | 2.67e-05 | 0.183 | 24,043.0 | 15,861.6 |
| European call K=140 | `mc-is` | 1,048,576 | 8.54e-04 | 0.197 | 23.5 | 14.5 |
| Arithmetic Asian call K=100, m=12 | `mc-plain` | 1,048,576 | 8.31e-03 | 1.337 | 1.0 | 1.0 |
| Arithmetic Asian call K=100, m=12 | `mc-antithetic` | 1,048,576 | 5.75e-03 | 1.239 | 2.1 | 2.3 |
| Arithmetic Asian call K=100, m=12 | `mc-cv` | 1,048,576 | 2.32e-04 | 1.892 | 1,286.9 | 909.5 |
| Arithmetic Asian call K=100, m=12 | `qmc-sobol-bb` | 1,048,576 | 1.15e-04 | 2.758 | 5,236.4 | 2,539.1 |
| Arithmetic Asian call K=100, m=12 | `mc-is` | 1,048,576 | 9.80e-03 | 1.753 | 0.7 | 0.5 |
