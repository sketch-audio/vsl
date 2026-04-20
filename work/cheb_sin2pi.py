"""sin(2*pi*x) Chebyshev approximation on [-0.5, 0.5] (one full period).

sin(2*pi*x) has zeros at x = 0, +/- 0.5 -- all exactly representable in f32,
so no major/minor split is needed. Factor them out:
    sin(2*pi*x) = x * (x^2 - 1/4) * P(x^2),
where f(x) = sin(2*pi*x) / (x * (x^2 - 1/4)) is smooth and EVEN:
    f(0)      = -8*pi           (from sin(2*pi*x)/x -> 2*pi, times 1/(-1/4))
    f(+/-1/2) = -4*pi

Fit f via Chebyshev in t = 2x in [-1, 1], convert to monomials in x^2,
recover sin(2*pi*x) by the factored form.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

PI = np.pi
TWO_PI = 2 * PI
HALF = 0.5
QUARTER = 0.25  # (1/2)^2


def f(x):
    """f(x) = sin(2*pi*x) / (x * (x^2 - 1/4)).  Smooth, even on [-1/2, 1/2]."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    eps = 1e-3
    near0 = np.abs(x) < eps
    nearhalf = np.abs(np.abs(x) - HALF) < eps
    mid = ~(near0 | nearhalf)

    xm = x[mid]
    out[mid] = np.sin(TWO_PI * xm) / (xm * (xm * xm - QUARTER))

    # Near 0: sin(2*pi*x)/x = 2*pi - (2*pi)^3 x^2/6 + ...
    x0 = x[near0]
    s_over_x = TWO_PI - (TWO_PI**3) * x0**2 / 6 + (TWO_PI**5) * x0**4 / 120
    out[near0] = s_over_x / (x0 * x0 - QUARTER)

    # Near x = s/2 (s = +/-1): u = x - s/2.
    # sin(2*pi*(s/2 + u)) = sin(s*pi + 2*pi*u) = -sin(2*pi*u)  [for s=+/-1]
    # x^2 - 1/4 = u*(s + u)
    # x = s/2 + u
    # so f(x) = -sin(2*pi*u) / ((s/2 + u) * u * (s + u))
    #       = -(2*pi * sinc(2*pi*u/pi ...)) ...  simpler: compute directly.
    xp = x[nearhalf]
    s = np.sign(xp)
    u = xp - s * HALF
    # sin(2*pi*u) / u = 2*pi * (1 - (2*pi*u)^2/6 + ...)
    sinc_term = TWO_PI * (1 - (TWO_PI * u)**2 / 6 + (TWO_PI * u)**4 / 120)
    out[nearhalf] = -sinc_term / ((s * HALF + u) * (s + u))

    return out


DEGREE = 12
cheb = Chebyshev.interpolate(lambda t: f(HALF * t), DEGREE, domain=[-1, 1])
print("Chebyshev coefficients of f((1/2)*t):")
for n, c in enumerate(cheb.coef):
    tag = "  (odd, ~0)" if n % 2 else ""
    print(f"  a_{n:2d} = {c: .6e}{tag}")

poly_t = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
# t = x / (1/2) = 2x, so t^k = (2x)^k => coef_in_x[k] = poly_t[k] * 2^k
coef_x = np.array([c * (2.0**k) for k, c in enumerate(poly_t.coef)])
P_coef = coef_x[::2]  # even powers only

print(f"\nP(x^2) such that f(x) ~ P(x^2):")
for k, c in enumerate(P_coef):
    print(f"  x^{2*k:2d}: {c: .15e}")


def approx_sin2pi(x):
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    val = P_coef[-1]
    for c in reversed(P_coef[:-1]):
        val = val * x2 + c
    return x * (x2 - QUARTER) * val


xs = np.linspace(-HALF, HALF, 200_001)
approx = approx_sin2pi(xs)
true = np.sin(TWO_PI * xs)
abs_err = np.abs(approx - true)
mask = np.abs(true) > 1e-6
rel_err = np.abs((approx[mask] - true[mask]) / true[mask])
print(f"\nf64 max abs err: {abs_err.max():.3e}, max rel err: {rel_err.max():.3e}")


def approx_sin2pi_f32(x):
    x = x.astype(np.float32)
    P32 = P_coef.astype(np.float32)
    half = np.float32(HALF)
    x2 = x * x
    val = P32[-1]
    for c in reversed(P32[:-1]):
        val = val * x2 + c
    # Factor (x^2 - 1/4) = (x - 1/2)(x + 1/2); 1/2 is exact in f32 and the
    # subtraction is Sterbenz-exact near x = +/- 1/2, avoiding the cancellation
    # that x^2 - 0.25 would suffer (since x^2 itself has already rounded).
    lo = x - half
    hi = x + half
    return x * (lo * hi) * val


xs32 = np.linspace(-HALF, HALF, 200_001, dtype=np.float32)
true64 = np.sin(TWO_PI * xs32.astype(np.float64))
approx32 = approx_sin2pi_f32(xs32).astype(np.float64)

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
ax.set_xlabel("x"); ax.set_ylabel("|error| in f32 ULPs of sin(2*pi*x)")
ax.set_title(f"sin(2*pi*x) f32 ULP on [-1/2, 1/2] (max = {overall_max:.2f})")
ax.set_xlim(-HALF, HALF); ax.set_ylim(0, np.nanpercentile(ulp_err[valid], 99.5) * 1.5)
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("work/out/cheb_sin2pi_ulp.png", dpi=130)
print("Saved work/out/cheb_sin2pi_ulp.png")


def _f(c):
    return f"{float(np.float32(c)):.9e}f"


print("\n// === C++ ===")
print(f"// sin(2*pi*x) on [-1/2, 1/2].  Max f32 ULP ~{overall_max:.2f}.")
print("inline float approx_sin2pi(float x) {")
print("    const float x2 = x * x;")
print(f"    float p = {_f(P_coef[-1])};")
for c in reversed(P_coef[:-1]):
    print(f"    p = p * x2 + {_f(c)};")
print("    const float lo = x - 0.5f;  // Sterbenz-exact near +/- 1/2")
print("    const float hi = x + 0.5f;")
print("    return x * (lo * hi) * p;")
print("}")
