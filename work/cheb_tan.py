"""Chebyshev tangent approximation on (-pi/2, pi/2).

tan has a zero at x=0 (from sin) and POLES at x=+/-pi/2 (from cos). A pure
polynomial can't represent the poles, so we use a rational form. Factor:

    tan(x) = x * g(x) / (pi^2/4 - x^2)

where g(x) = tan(x) * (pi^2/4 - x^2) / x is smooth and EVEN on [-pi/2, pi/2]:
  - g(0) = pi^2/4        (from tan(x)/x -> 1)
  - g(+/-pi/2) = 2

Fit g as a Chebyshev series in t = x/(pi/2), truncate to a polynomial Q(x^2),
then evaluate

    tan(x) ~ x * Q(x^2) / (pi^2/4 - x^2).

The rational form encodes the poles exactly (in exact arithmetic). For f32,
split pi/2 into (pi_half_major + pi_half_minor) so the denominator is
accurate near the poles.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

PI = np.pi
PI_HALF = PI / 2
PI_HALF_SQ = PI_HALF * PI_HALF


def g(x):
    """g(x) = tan(x) * (pi^2/4 - x^2) / x.  Smooth, even on [-pi/2, pi/2]."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    eps = 1e-3

    near0 = np.abs(x) < eps
    nearpole = np.abs(np.abs(x) - PI_HALF) < eps
    mid = ~(near0 | nearpole)

    xm = x[mid]
    out[mid] = np.tan(xm) * (PI_HALF_SQ - xm * xm) / xm

    # Near 0: tan(x)/x = 1 + x^2/3 + 2 x^4/15 + 17 x^6/315 + ...
    x0 = x[near0]
    tan_over_x = 1 + x0**2 / 3 + 2 * x0**4 / 15 + 17 * x0**6 / 315
    out[near0] = tan_over_x * (PI_HALF_SQ - x0 * x0)

    # Near x = s*pi/2 (s = +/-1): let u = x - s*pi/2. Then
    #   tan(s*pi/2 + u) = -cot(u) = -cos(u) / (u * sinc(u))
    #   pi^2/4 - x^2   = -u * (s*pi + u)
    # so tan(x) * (pi^2/4 - x^2) / x = cos(u) * (s*pi + u) / (sinc(u) * x).
    xp = x[nearpole]
    s = np.sign(xp)
    u = xp - s * PI_HALF
    cos_u = 1 - u**2 / 2 + u**4 / 24 - u**6 / 720
    sinc_u = 1 - u**2 / 6 + u**4 / 120 - u**6 / 5040
    out[nearpole] = cos_u * (s * PI + u) / (sinc_u * xp)

    return out


# Fit g on [-pi/2, pi/2] via Chebyshev interpolation in t = x/(pi/2).
DEGREE = 8
cheb = Chebyshev.interpolate(lambda t: g(PI_HALF * t), DEGREE, domain=[-1, 1])

print("Chebyshev coefficients a_n of g((pi/2)*t) = sum a_n T_n(t):")
for n, c in enumerate(cheb.coef):
    tag = "  (odd, ~0)" if n % 2 else ""
    print(f"  a_{n:2d} = {c: .6e}{tag}")

# Convert T_n(t) series to power series in t, then substitute t = x/(pi/2).
poly_t = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
coef_x = np.array([c / PI_HALF**k for k, c in enumerate(poly_t.coef)])
Q_coef = coef_x[::2]  # Q(x^2) = sum Q_coef[k] * x^(2k)

print("\nQ(x^2) so that g(x) ~ Q(x^2):")
for k, c in enumerate(Q_coef):
    print(f"  x^{2*k:2d}: {c: .15e}")


def approx_tan(x):
    """Evaluate tan(x) ~ x * Q(x^2) / (pi^2/4 - x^2)."""
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    val = Q_coef[-1]
    for c in reversed(Q_coef[:-1]):
        val = val * x2 + c
    return x * val / (PI_HALF_SQ - x2)


# Measure error inside (-pi/2, pi/2), avoiding a small neighborhood of the
# poles where tan itself is huge / overflow-adjacent.
BOUND = PI_HALF - 1e-3  # tan(+/-BOUND) ~ +/- 1000
xs = np.linspace(-BOUND, BOUND, 200_001)
approx = approx_tan(xs)
true = np.tan(xs)
abs_err = np.abs(approx - true)
rel_err = np.abs((approx - true) / true)
valid_rel = np.abs(true) > 1e-6

print(f"\nOn [-{BOUND:.4f}, {BOUND:.4f}] (|tan(x)| <= ~1000):")
print(f"  Max absolute error: {abs_err.max():.3e}")
print(f"  Max relative error: {rel_err[valid_rel].max():.3e}")


# ---------------------------------------------------------------------------
# f32 evaluation + ULP error plot.
# ---------------------------------------------------------------------------


def approx_tan_f32(x):
    x = x.astype(np.float32)
    Q32 = Q_coef.astype(np.float32)
    pi_half_sq32 = np.float32(PI_HALF_SQ)
    x2 = x * x
    val = Q32[-1]
    for c in reversed(Q32[:-1]):
        val = val * x2 + c
    return x * val / (pi_half_sq32 - x2)


xs32 = np.linspace(-BOUND, BOUND, 200_001, dtype=np.float32)
true64 = np.tan(xs32.astype(np.float64))
approx32 = approx_tan_f32(xs32).astype(np.float64)

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
print(f"\nf32 ULP error on {BOUND:.3f}-bounded domain: mean = {overall_mean:.2f}, max = {overall_max:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xs32[valid], ulp_err[valid], ".", color="#cccccc", ms=1, label="pointwise |ULP error|")
ax.plot(bin_centers, bin_mean, color="#1f77b4", lw=1.8, label="binned mean")
ax.axhline(overall_mean, color="#d62728", ls="--", lw=1, label=f"overall mean = {overall_mean:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("|error| in f32 ULPs of tan(x)")
ax.set_title(f"Chebyshev tan approximation, f32 ULP error (max = {overall_max:.0f})")
ax.set_xlim(-PI_HALF, PI_HALF)
ax.set_ylim(0, np.nanpercentile(ulp_err[valid], 99.5) * 1.5)
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()

out_path = "work/out/cheb_tan_ulp.png"
fig.savefig(out_path, dpi=130)
print(f"Saved plot to {out_path}")
plt.show()


# ---------------------------------------------------------------------------
# Split-pi/2 refinement: avoid cancellation in the denominator near +/- pi/2.
# ---------------------------------------------------------------------------

PI_HALF_MAJOR = np.float32(PI_HALF)
PI_HALF_MINOR = np.float32(PI_HALF - float(PI_HALF_MAJOR))
print(f"\npi/2 major (f32) = {float(PI_HALF_MAJOR):.10e}")
print(f"pi/2 minor (f32) = {float(PI_HALF_MINOR):.10e}")


def approx_tan_f32_split(x):
    x = x.astype(np.float32)
    Q32 = Q_coef.astype(np.float32)
    x2 = x * x
    val = Q32[-1]
    for c in reversed(Q32[:-1]):
        val = val * x2 + c
    # (pi/2 - x) and (pi/2 + x) with the Sterbenz-friendly split.
    lo = (PI_HALF_MAJOR - x) + PI_HALF_MINOR  # ~ pi/2 - x
    hi = (PI_HALF_MAJOR + x) + PI_HALF_MINOR  # ~ pi/2 + x
    return x * val / (lo * hi)


approx32b = approx_tan_f32_split(xs32).astype(np.float64)
ulp_err2 = np.abs(approx32b - true64) / np.where(ulp > 0, ulp, np.nan)
valid2 = np.isfinite(ulp_err2) & (np.abs(true64) > 1e-6)

sums2 = np.bincount(bin_idx, weights=np.where(valid2, ulp_err2, 0.0), minlength=n_bins)
counts2 = np.bincount(bin_idx, weights=valid2.astype(np.float64), minlength=n_bins)
bin_mean2 = np.where(counts2 > 0, sums2 / np.maximum(counts2, 1), np.nan)

overall_mean2 = ulp_err2[valid2].mean()
overall_max2 = ulp_err2[valid2].max()
print(f"\nf32 ULP error w/ split-pi/2: mean = {overall_mean2:.2f}, max = {overall_max2:.2f}")

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(bin_centers, bin_mean, color="#cccccc", lw=1.5, label=f"naive f32 (mean {overall_mean:.2f})")
ax2.plot(bin_centers, bin_mean2, color="#1f77b4", lw=1.8, label=f"split-pi/2 f32 (mean {overall_mean2:.2f})")
ax2.axhline(overall_mean2, color="#d62728", ls="--", lw=1)
ax2.set_xlabel("x")
ax2.set_ylabel("binned mean |ULP error|")
ax2.set_title(f"Split-pi/2 refinement (max {overall_max:.0f} -> {overall_max2:.0f} ULPs)")
ax2.set_xlim(-PI_HALF, PI_HALF)
ax2.set_ylim(0, max(np.nanmax(bin_mean), np.nanmax(bin_mean2)) * 1.1)
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)
fig2.tight_layout()

out_path2 = "work/out/cheb_tan_ulp_split_pi.png"
fig2.savefig(out_path2, dpi=130)
print(f"Saved plot to {out_path2}")


def _f(c):
    return f"{float(np.float32(c)):.9e}f"


print("\n// === C++ ===")
print(f"// tan(x) on (-pi/2, pi/2) (split-pi/2 denominator).  Max f32 ULP ~{overall_max2:.2f}.")
print("inline float approx_tan(float x) {")
print(f"    constexpr float PI_HALF_MAJOR = {_f(PI_HALF_MAJOR)};")
print(f"    constexpr float PI_HALF_MINOR = {_f(PI_HALF_MINOR)};")
print("    const float x2 = x * x;")
print(f"    float q = {_f(Q_coef[-1])};")
for c in reversed(Q_coef[:-1]):
    print(f"    q = q * x2 + {_f(c)};")
print("    const float lo = (PI_HALF_MAJOR - x) + PI_HALF_MINOR;  // ~ pi/2 - x (Sterbenz)")
print("    const float hi = (PI_HALF_MAJOR + x) + PI_HALF_MINOR;  // ~ pi/2 + x")
print("    return x * q / (lo * hi);")
print("}")

plt.show()
