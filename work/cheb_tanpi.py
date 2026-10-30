"""tan(pi*x) Chebyshev approximation on [-1/2, 1/2] (one full period).

tan(pi*x) has period 1, a zero at x=0 and POLES at x=+/-1/2. A pure polynomial
can't represent the poles, so -- exactly as in cheb_tan.py -- we use a rational
form. Factor:

    tan(pi*x) = x * g(x) / (1/4 - x^2)

where g(x) = tan(pi*x) * (1/4 - x^2) / x is smooth and EVEN on [-1/2, 1/2]:
  - g(0)      = pi/4          (from tan(pi*x)/x -> pi, times 1/4)
  - g(+/-1/2) = 2/pi

Fit g as a Chebyshev series in t = 2x on [-1, 1], truncate to a polynomial
Q(x^2), then evaluate

    tan(pi*x) ~ x * Q(x^2) / (1/4 - x^2).

The scaled form is what makes this nicer than the unscaled cheb_tan.py: the
poles sit at x = +/- 1/2, which is EXACT in f32. So the denominator factors
(1/2 - x)(1/2 + x) are Sterbenz-exact near the poles -- no pi/2 major/minor
split is needed, and the wrap into range is the cheap x - round(x).
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

PI = np.pi
HALF = 0.5
QUARTER = 0.25  # (1/2)^2


def g(x):
    """g(x) = tan(pi*x) * (1/4 - x^2) / x.  Smooth, even on [-1/2, 1/2]."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    eps = 1e-3

    near0 = np.abs(x) < eps
    nearpole = np.abs(np.abs(x) - HALF) < eps
    mid = ~(near0 | nearpole)

    xm = x[mid]
    out[mid] = np.tan(PI * xm) * (QUARTER - xm * xm) / xm

    # Near 0: tan(pi*x)/x = pi*(1 + (pi x)^2/3 + 2(pi x)^4/15 + 17(pi x)^6/315)
    x0 = x[near0]
    y2 = (PI * x0) ** 2
    tan_over_x = PI * (1 + y2 / 3 + 2 * y2**2 / 15 + 17 * y2**3 / 315)
    out[near0] = tan_over_x * (QUARTER - x0 * x0)

    # Near x = s/2 (s = +/-1): u = x - s/2. Then
    #   tan(pi*(s/2 + u)) = -cot(pi*u)
    #   1/4 - x^2         = -u * (s + u)
    # so g = -cot(pi*u) * (-u(s+u)) / x = cos(pi*u)*(s+u) / (pi*sinc(pi*u)*x),
    # where sin(pi*u) = pi*u*sinc, sinc = 1 - (pi u)^2/6 + (pi u)^4/120 - ...
    xp = x[nearpole]
    s = np.sign(xp)
    u = xp - s * HALF
    pu2 = (PI * u) ** 2
    cos_pu = 1 - pu2 / 2 + pu2**2 / 24 - pu2**3 / 720
    sinc_pu = 1 - pu2 / 6 + pu2**2 / 120 - pu2**3 / 5040
    out[nearpole] = cos_pu * (s + u) / (PI * sinc_pu * xp)

    return out


# Fit g on [-1/2, 1/2] via Chebyshev interpolation in t = 2x. DEGREE=8 (5 even
# coeffs) matches the sibling cheb_tan.py cost and already beats it; DEGREE=10
# (6 coeffs) trims the near-pole max from ~4.0 to ~3.2 ULP for one extra fma.
DEGREE = int(sys.argv[1]) if len(sys.argv) > 1 else 8  # Precision::Fast: 4
cheb = Chebyshev.interpolate(lambda t: g(HALF * t), DEGREE, domain=[-1, 1])

print("Chebyshev coefficients a_n of g((1/2)*t) = sum a_n T_n(t):")
for n, c in enumerate(cheb.coef):
    tag = "  (odd, ~0)" if n % 2 else ""
    print(f"  a_{n:2d} = {c: .6e}{tag}")

# Convert T_n(t) series to power series in t, then substitute t = 2x.
poly_t = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
coef_x = np.array([c * (2.0**k) for k, c in enumerate(poly_t.coef)])
Q_coef = coef_x[::2]  # Q(x^2) = sum Q_coef[k] * x^(2k)

print("\nQ(x^2) so that g(x) ~ Q(x^2):")
for k, c in enumerate(Q_coef):
    print(f"  x^{2*k:2d}: {c: .15e}")


def approx_tanpi(x):
    """Evaluate tan(pi*x) ~ x * Q(x^2) / (1/4 - x^2)."""
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    val = Q_coef[-1]
    for c in reversed(Q_coef[:-1]):
        val = val * x2 + c
    return x * val / (QUARTER - x2)


# Measure error inside (-1/2, 1/2), avoiding a small neighborhood of the poles
# where tan itself is huge / overflow-adjacent.
BOUND = HALF - 1e-3 / PI  # tan(pi*BOUND) ~ +/- 1000
xs = np.linspace(-BOUND, BOUND, 200_001)
approx = approx_tanpi(xs)
true = np.tan(PI * xs)
abs_err = np.abs(approx - true)
rel_err = np.abs((approx - true) / true)
valid_rel = np.abs(true) > 1e-6

print(f"\nOn [-{BOUND:.5f}, {BOUND:.5f}] (|tan(pi*x)| <= ~1000):")
print(f"  Max absolute error: {abs_err.max():.3e}")
print(f"  Max relative error: {rel_err[valid_rel].max():.3e}")


# ---------------------------------------------------------------------------
# f32 evaluation + ULP error. Factor (1/4 - x^2) = (1/2 - x)(1/2 + x); 1/2 is
# exact in f32 and each subtraction is Sterbenz-exact near the corresponding
# pole, so no pi/2 major/minor split is needed.
# ---------------------------------------------------------------------------


def approx_tanpi_f32(x):
    x = x.astype(np.float32)
    Q32 = Q_coef.astype(np.float32)
    half = np.float32(HALF)
    x2 = x * x
    val = Q32[-1]
    for c in reversed(Q32[:-1]):
        val = val * x2 + c
    lo = half - x  # ~ 1/2 - x (Sterbenz-exact near +1/2)
    hi = half + x  # ~ 1/2 + x (Sterbenz-exact near -1/2)
    return x * val / (lo * hi)


xs32 = np.linspace(-BOUND, BOUND, 200_001, dtype=np.float32)
true64 = np.tan(PI * xs32.astype(np.float64))
approx32 = approx_tanpi_f32(xs32).astype(np.float64)

ulp = np.abs(np.spacing(true64.astype(np.float32))).astype(np.float64)
ulp_err = np.abs(approx32 - true64) / np.where(ulp > 0, ulp, np.nan)
valid = np.isfinite(ulp_err) & (np.abs(true64) > 1e-6)

n_bins = 200
bin_edges = np.linspace(-BOUND, BOUND, n_bins + 1)
bin_idx = np.clip(np.searchsorted(bin_edges, xs32, side="right") - 1, 0, n_bins - 1)
sums = np.bincount(bin_idx, weights=np.where(valid, ulp_err, 0.0), minlength=n_bins)
counts = np.bincount(bin_idx, weights=valid.astype(np.float64), minlength=n_bins)
bin_mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

overall_mean = ulp_err[valid].mean()
overall_max = ulp_err[valid].max()
print(f"\nf32 ULP error on {BOUND:.4f}-bounded domain: mean = {overall_mean:.2f}, max = {overall_max:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xs32[valid], ulp_err[valid], ".", color="#cccccc", ms=1, label="pointwise |ULP error|")
ax.plot(bin_centers, bin_mean, color="#1f77b4", lw=1.8, label="binned mean")
ax.axhline(overall_mean, color="#d62728", ls="--", lw=1, label=f"overall mean = {overall_mean:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("|error| in f32 ULPs of tan(pi*x)")
ax.set_title(f"tan(pi*x) Chebyshev approximation, f32 ULP error (max = {overall_max:.0f})")
ax.set_xlim(-HALF, HALF)
ax.set_ylim(0, np.nanpercentile(ulp_err[valid], 99.5) * 1.5)
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("work/out/cheb_tanpi_ulp.png", dpi=130)
print("Saved work/out/cheb_tanpi_ulp.png")


def _f(c):
    return f"{float(np.float32(c)):.9e}f"


print("\n// === C++ ===")
print(f"// tan(pi*x) on [-1/2, 1/2].  Max f32 ULP ~{overall_max:.2f}.")
print("inline float approx_tanpi(float x) {")
print("    const float x2 = x * x;")
print(f"    float q = {_f(Q_coef[-1])};")
for c in reversed(Q_coef[:-1]):
    print(f"    q = q * x2 + {_f(c)};")
print("    const float lo = 0.5f - x;  // Sterbenz-exact near +1/2")
print("    const float hi = 0.5f + x;  // Sterbenz-exact near -1/2")
print("    return x * q / (lo * hi);")
print("}")

plt.show()
