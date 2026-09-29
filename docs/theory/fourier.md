# Fourier pricing and implied volatility

## Gil-Pelaez inversion

For a model with characteristic function $\varphi(u) = \mathbb E[e^{iu\ln S_T}]$, the two
risk-neutral probabilities of the Heston (1993) formula combine into a single integral.
Using $\varphi(-i) = S_0e^{(r-q)T}$ and $k = \ln K$:

$$
C = \tfrac12\big(S_0e^{-qT} - Ke^{-rT}\big) + \frac{e^{-rT}}{\pi}\int_0^\infty
\mathrm{Re}\Big[\frac{e^{-iuk}\big(\varphi(u-i) - K\varphi(u)\big)}{iu}\Big]\,du
$$

(Gil-Pelaez, 1951). Two implementations are provided: adaptive QUADPACK integration (the
validation reference), and a fixed Gauss-Legendre rule on the domain truncated where
$|\varphi| < 10^{-14}$, vectorised over strikes (used by calibration). Puts follow from
put-call parity.

## Carr-Madan FFT

The damped call $e^{\alpha k}C(k)$ is square-integrable for $\alpha > 0$, and its Fourier
transform is

$$
\psi(v) = \frac{e^{-rT}\varphi(v - (\alpha+1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha+1)v}
$$

(Carr & Madan, 1999). It is inverted with one FFT on a log-strike grid (Simpson weights)
and interpolated at the requested strikes with a cubic spline. The damping $\alpha$ is
configurable. Too small an $\alpha$ requires a finer integration grid, and $\alpha + 1$
must keep $\mathbb E[S_T^{\alpha+1}]$ finite.

## Implied volatility

The Black-Scholes implied volatility inverts $\mathrm{BS}(\sigma) = V$:

1. prices outside the no-arbitrage bounds
   $\max(S_0e^{-qT} - Ke^{-rT}, 0) < C < S_0e^{-qT}$ return `nan` (or raise);
2. vectorised Newton steps $\sigma \leftarrow \sigma - (\mathrm{BS}(\sigma) - V)/\mathcal V(\sigma)$
   start from the Manaster & Koehler (1982) guess;
3. entries where Newton fails (tiny vega) are solved by Brent's method (Brent, 1973).

The accuracy in volatility is the price accuracy divided by vega. Far from the money the
volatility is only as precise as the input price allows.
