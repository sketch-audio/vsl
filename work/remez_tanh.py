"""tanh(x) for vsl::tanh: both precisions.

Standard: the sigmoid form

    tanh(x) ~ s / sqrt(1 + s^2),   s = x * R(x^2) ~ sinh(x)

has the asymptote in the algebra (|y| < 1, monotonic). An error dR in R moves
y by x dR (1 + s^2)^(-3/2) ~ x sech^3(x) dR, so R is a Remez fit of
sinh(x) / x in v = x^2, weighted by x sech^3(x) / tanh(x): minimax in y's
*relative* error, i.e. its ULP error near 0. Degree 9 (five terms). The
input is clamped to +/-9.2: tanh rounds to 1 in f32 past ~9.01, and 9.2
lands on exactly 1 fused or not.

Fast: the [7/6] Pade approximant, the truncation of Lambert's continued
fraction tanh x = x / (1 + x^2 / (3 + x^2 / (5 + ...))). An odd rational
can't level off, so it runs past 1 near x ~ 5; the input is clamped at the
L that minimizes the max absolute error, the rational on [0, L] and its
value at L beyond.

Plain Chebyshev explorations of tanh (clamped polynomial, Pade on [-4, 4],
an earlier sinh proxy) are in cheb_tanh.py.
"""

from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np

from remez import remez

SIGMOID_RANGE = 10.0  # fit s on [0, 10]; tanh(10) = 1 - 4e-9
STANDARD_CLAMP = np.float32(9.2)
PADE_DEGREE = 7       # [7/6]


def sinh_over_x(v):
    x = np.sqrt(np.asarray(v, dtype=np.float64))
    safe = np.where(x == 0, 1.0, x)
    return np.where(x == 0, 1.0, np.sinh(safe) / safe)


def relative_weight(v):
    x = np.sqrt(np.asarray(v, dtype=np.float64))
    safe = np.where(x == 0, 1.0, x)
    return np.where(x == 0, 1.0, safe / np.cosh(safe) ** 3 / np.tanh(safe))


def lambert_pade(degree):
    """Even polynomials P, Q in v = x^2 with tanh x ~ x P(v) / Q(v), from
    Lambert's continued fraction cut after the term `degree` (odd)."""
    # Bottom up: K = (2k + 1) + v / K as a ratio of polynomials num / den.
    num, den = [Fraction(degree)], [Fraction(1)]
    for k in range(degree - 2, 0, -2):
        # (2k + 1) + v den / num  ->  ((2k + 1) num + v den) / num
        new = [Fraction(k) * c for c in num] + [Fraction(0)]
        for i, c in enumerate(den):
            new[i + 1] += c
        num, den = new, num
    # tanh x = x / K = x den / num; normalize so both start at 1.
    def trim(c):
        while c[-1] == 0:
            c = c[:-1]
        return [k / den[0] for k in c]
    return trim(den), trim(num)


def best_pade_clamp(p, q):
    xs = np.linspace(0, 6, 60001)
    y = xs * np.polynomial.polynomial.polyval(xs * xs, p) / np.polynomial.polynomial.polyval(xs * xs, q)
    err = np.abs(y - np.tanh(xs))
    limits = np.round(np.arange(3.0, 6.0, 0.01), 2)
    best = [max(err[xs <= L].max(), abs(1 - y[np.searchsorted(xs, L)])) for L in limits]
    return float(limits[int(np.argmin(best))])


def horner32(c, v):
    c32 = np.asarray(c, dtype=np.float32)
    val = c32[-1]
    for k in reversed(c32[:-1]):
        val = val * v + k
    return val


# --- Standard ---------------------------------------------------------------

r, E, history = remez(sinh_over_x, relative_weight, 0.0, SIGMOID_RANGE**2, 5, grid_size=40001)
print(f"Standard: sigmoid R, degree 9, {len(history)} Remez iterations, level {abs(E):.3e}")
for k, c in enumerate(r):
    print(f"  v^{k}: {c: .15e}")

# --- Fast -------------------------------------------------------------------

p, q = lambert_pade(2 * PADE_DEGREE - 1)
p_f, q_f = [float(c) for c in p], [float(c) for c in q]
pade_clamp = best_pade_clamp(p_f, q_f)
print(f"\nFast: Pade [{PADE_DEGREE}/{PADE_DEGREE - 1}], input clamp {pade_clamp}")
print("  P:", " ".join(str(c) for c in p))
print("  Q:", " ".join(str(c) for c in q))

# --- f32 error --------------------------------------------------------------

xs = np.linspace(0, 12, 400_001, dtype=np.float32)
true = np.tanh(xs.astype(np.float64))

xc = np.minimum(xs, STANDARD_CLAMP)
s = xc * horner32(r, xc * xc)
y_std = (s / np.sqrt(s * s + np.float32(1))).astype(np.float64)

xc = np.minimum(xs, np.float32(pade_clamp))
y_fast = (xc * horner32(p_f, xc * xc) / horner32(q_f, xc * xc)).astype(np.float64)

ulp = np.spacing(true.astype(np.float32)).astype(np.float64)
nz = xs > 0
ulp_std = np.abs(y_std - true)[nz] / ulp[nz]
abs_fast = np.abs(y_fast - true)
print(f"\nf32 Standard: max {ulp_std.max():.2f} ULP, {20 * np.log10(np.abs(y_std - true).max()):.1f} dB")
print(f"f32 Fast:     max abs {abs_fast.max():.3e} ({20 * np.log10(abs_fast.max()):.1f} dB), "
      f"saturates at {y_fast[-1]:.6f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
a1.plot(xs[nz], ulp_std, ".", color="#1f77b4", ms=1)
a1.set_title(f"Standard: f32 ULP (max = {ulp_std.max():.2f})"); a1.set_xlabel("x")
a2.semilogy(xs, abs_fast + 1e-12, color="#d62728", lw=1)
a2.set_title(f"Fast: f32 abs error (max = {abs_fast.max():.2e})"); a2.set_xlabel("x")
for a in (a1, a2):
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("work/out/remez_tanh.png", dpi=130)
print("Saved work/out/remez_tanh.png")


def _f(c):
    return f"X({float(np.float32(c)):.9e}f)"


print("\n// === C++ Standard ===")
print(f"auto r = {_f(r[-1])};")
for c in reversed(r[:-1]):
    print(f"r = mul_add(r, u, {_f(c)});")
print(f"\n// === C++ Fast (clamp {pade_clamp}) ===")
print(f"auto p = {_f(p_f[-1])};")
for c in reversed(p_f[:-1]):
    print(f"p = mul_add(p, x2, {_f(c)});")
print(f"auto q = {_f(q_f[-1])};")
for c in reversed(q_f[:-1]):
    print(f"q = mul_add(q, x2, {_f(c)});")
