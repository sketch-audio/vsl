"""cos(2*pi*x) Chebyshev approximation on [-0.5, 0.5] (one full period).

cos(2*pi*x) has zeros at x = +/- 1/4 -- exactly representable in f32, so no
major/minor split needed. Factor:
    cos(2*pi*x) = (x^2 - 1/16) * Q(x^2),
where f(x) = cos(2*pi*x) / (x^2 - 1/16) is smooth and EVEN:
    f(0)        = -16            (cos(0)=1, / (-1/16))
    f(+/-1/4)   = -4*pi
Fit f via Chebyshev in t = 2x in [-1, 1], convert to monomials in x^2,
recover cos(2*pi*x) by the factored form.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

PI = np.pi
TWO_PI = 2 * PI
HALF = 0.5
QUARTER = 0.25
SIXTEENTH = 1.0 / 16.0  # (1/4)^2


def f(x):
    """f(x) = cos(2*pi*x) / (x^2 - 1/16).  Smooth, even on [-1/2, 1/2]."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    eps = 1e-3
    nearq = np.abs(np.abs(x) - QUARTER) < eps
    mid = ~nearq

    xm = x[mid]
    out[mid] = np.cos(TWO_PI * xm) / (xm * xm - SIXTEENTH)

    # Near x = s/4 (s = +/-1): u = x - s/4.
    # cos(2*pi*(s/4 + u)) = cos(s*pi/2 + 2*pi*u) = -s * sin(2*pi*u)
    # x^2 - 1/16 = u*(s/2 + u)
    # f = -s * sin(2*pi*u) / (u * (s/2 + u))
    #   = -s * 2*pi * sinc(2*pi*u) / (s/2 + u)
    xp = x[nearq]
    s = np.sign(xp)
    u = xp - s * QUARTER
    sinc_term = TWO_PI * (1 - (TWO_PI * u)**2 / 6 + (TWO_PI * u)**4 / 120)
    out[nearq] = -s * sinc_term / (s * HALF + u)

    return out


DEGREE = 12
cheb = Chebyshev.interpolate(lambda t: f(HALF * t), DEGREE, domain=[-1, 1])
print("Chebyshev coefficients of f((1/2)*t):")
for n, c in enumerate(cheb.coef):
    tag = "  (odd, ~0)" if n % 2 else ""
    print(f"  a_{n:2d} = {c: .6e}{tag}")

poly_t = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
coef_x = np.array([c * (2.0**k) for k, c in enumerate(poly_t.coef)])
Q_coef = coef_x[::2]

print(f"\nQ(x^2) such that f(x) ~ Q(x^2):")
for k, c in enumerate(Q_coef):
    print(f"  x^{2*k:2d}: {c: .15e}")


def approx_cos2pi(x):
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    val = Q_coef[-1]
    for c in reversed(Q_coef[:-1]):
        val = val * x2 + c
    return (x2 - SIXTEENTH) * val


xs = np.linspace(-HALF, HALF, 200_001)
approx = approx_cos2pi(xs)
true = np.cos(TWO_PI * xs)
abs_err = np.abs(approx - true)
mask = np.abs(true) > 1e-6
rel_err = np.abs((approx[mask] - true[mask]) / true[mask])
print(f"\nf64 max abs err: {abs_err.max():.3e}, max rel err: {rel_err.max():.3e}")


def approx_cos2pi_f32(x):
    x = x.astype(np.float32)
    Q32 = Q_coef.astype(np.float32)
    q = np.float32(QUARTER)
    x2 = x * x
    val = Q32[-1]
    for c in reversed(Q32[:-1]):
        val = val * x2 + c
    # Factor (x^2 - 1/16) = (x - 1/4)(x + 1/4); 1/4 is exact in f32 and the
    # subtraction is Sterbenz-exact near x = +/- 1/4, avoiding the cancellation
    # that (x^2 - 1/16) would suffer (x^2 has already rounded).
    lo = x - q
    hi = x + q
    return (lo * hi) * val


xs32 = np.linspace(-HALF, HALF, 200_001, dtype=np.float32)
true64 = np.cos(TWO_PI * xs32.astype(np.float64))
approx32 = approx_cos2pi_f32(xs32).astype(np.float64)

ulp = np.abs(np.spacing(true64.astype(np.float32))).astype(np.float64)
ulp_err = np.abs(approx32 - true64) / np.where(ulp > 0, ulp, np.nan)
valid = np.isfinite(ulp_err) & (np.abs(true64) > 1e-6)

n_bins = 200
bin_edges = np.linspace(-HALF, HALF, n_bins + 1)
bin_idx = np.clip(np.searchsorted(bin_edges, xs32, side="right") - 1, 0, n_bins - 1)
sums = np.bincount(bin_idx, weights=np.where(valid, ulp_err, 0.0), minlength=n_bins)
counts = np.bincount(bin_idx, weights=valid.astype(np.float64), minlength=n_bins)
bin_mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

overall_mean = ulp_err[valid].mean()
overall_max = ulp_err[valid].max()
print(f"f32 ULP: mean = {overall_mean:.2f}, max = {overall_max:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xs32[valid], ulp_err[valid], ".", color="#cccccc", ms=1, label="pointwise")
ax.plot(bin_centers, bin_mean, color="#1f77b4", lw=1.8, label="binned mean")
ax.axhline(overall_mean, color="#d62728", ls="--", lw=1, label=f"mean = {overall_mean:.2f}")
ax.set_xlabel("x"); ax.set_ylabel("|error| in f32 ULPs of cos(2*pi*x)")
ax.set_title(f"cos(2*pi*x) f32 ULP on [-1/2, 1/2] (max = {overall_max:.2f})")
ax.set_xlim(-HALF, HALF); ax.set_ylim(0, np.nanpercentile(ulp_err[valid], 99.5) * 1.5)
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("work/out/cheb_cos2pi_ulp.png", dpi=130)
print("Saved work/out/cheb_cos2pi_ulp.png")


def _f(c):
    return f"{float(np.float32(c)):.9e}f"


print("\n// === C++ ===")
print(f"// cos(2*pi*x) on [-1/2, 1/2].  Max f32 ULP ~{overall_max:.2f}.")
print("inline float approx_cos2pi(float x) {")
print("    const float x2 = x * x;")
print(f"    float q = {_f(Q_coef[-1])};")
for c in reversed(Q_coef[:-1]):
    print(f"    q = q * x2 + {_f(c)};")
print("    const float lo = x - 0.25f;  // Sterbenz-exact near +/- 1/4")
print("    const float hi = x + 0.25f;")
print("    return (lo * hi) * q;")
print("}")
