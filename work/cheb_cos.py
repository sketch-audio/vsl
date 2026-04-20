"""Chebyshev cosine approximation on [-pi, pi], mirroring cheb_sin.py.

Recipe:
  1. cos(x) has zeros at x = +/- pi/2 inside [-pi, pi]. Factor them out:
         cos(x) = (x^2 - (pi/2)^2) * f(x)
     where f(x) = cos(x) / (x^2 - pi^2/4) is smooth and even on [-pi, pi].
  2. Expand f as a truncated Chebyshev series in t = x/pi on [-1, 1].
  3. Convert to a polynomial P(x^2), then recover
         cos(x) ~ (x^2 - pi^2/4) * P(x^2).
  4. For f32 precision, split pi/2 into (pi_half_major + pi_half_minor) to
     avoid catastrophic cancellation at the zeros.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

PI = np.pi
PI_HALF = PI / 2
PI_QUARTER_SQ = PI_HALF * PI_HALF  # (pi/2)^2


def f(x):
    """f(x) = cos(x) / (x^2 - pi^2/4). Smooth, even on [-pi, pi]. Uses Taylor
    expansions near x = +/- pi/2 to dodge 0/0.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    eps = 1e-3

    nearph = np.abs(np.abs(x) - PI_HALF) < eps
    mid = ~nearph

    xm = x[mid]
    out[mid] = np.cos(xm) / (xm * xm - PI_QUARTER_SQ)

    # Near x = s*pi/2 (s = +/-1): let u = x - s*pi/2. Then
    #   cos(x) = -s * sin(u)          (check: cos(pi/2+u) = -sin(u),
    #                                  cos(-pi/2+u) = sin(u))
    #   x^2 - pi^2/4 = u * (s*pi + u) (full pi, not pi/2; see derivation)
    # so f(x) = -s * sinc(u) / (s*pi + u).
    xp = x[nearph]
    s = np.sign(xp)
    u = xp - s * PI_HALF
    sinc_u = 1 - u**2 / 6 + u**4 / 120 - u**6 / 5040
    out[nearph] = -s * sinc_u / (s * PI + u)

    return out


# Fit f on [-pi, pi] via Chebyshev interpolation in t = x/pi.
# f is even, so odd-index coefficients come out ~0. Six nonzero even
# coefficients (a_0, a_2, ..., a_10) yield a degree-12 cos polynomial.
DEGREE = 10
cheb = Chebyshev.interpolate(lambda t: f(PI * t), DEGREE, domain=[-1, 1])

print("Chebyshev coefficients a_n of f(pi*t) = sum a_n T_n(t):")
for n, c in enumerate(cheb.coef):
    tag = "  (odd, ~0)" if n % 2 else ""
    print(f"  a_{n:2d} = {c: .6e}{tag}")

poly_t = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
coef_x = np.array([c / PI**k for k, c in enumerate(poly_t.coef)])
P_coef = coef_x[::2]  # P(x^2) = sum P_coef[k] * x^(2k)

print("\nP(x^2) so that f(x) ~ P(x^2):")
for k, c in enumerate(P_coef):
    print(f"  x^{2*k:2d}: {c: .15e}")

# Expand cos(x) ~ (x^2 - pi^2/4) * P(x^2) into monomials.
# Coefficient of x^(2k) is P_coef[k-1] - (pi^2/4) * P_coef[k]
# (with P_coef[-1] := 0 and P_coef[n] := 0 for k = n+1).
n = len(P_coef)
cos_coefs = np.zeros(n + 1)
for k in range(n + 1):
    lo = P_coef[k - 1] if k >= 1 else 0.0
    hi = -PI_QUARTER_SQ * P_coef[k] if k < n else 0.0
    cos_coefs[k] = lo + hi

print("\ncos(x) ~ sum c_k * x^(2k):")
for k, c in enumerate(cos_coefs):
    print(f"  x^{2*k:2d}: {c: .15e}")


def approx_cos(x):
    """Evaluate cos(x) ~ (x^2 - pi^2/4) * P(x^2) via Horner in x^2."""
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    val = P_coef[-1]
    for c in reversed(P_coef[:-1]):
        val = val * x2 + c
    return (x2 - PI_QUARTER_SQ) * val


xs = np.linspace(-PI, PI, 200_001)
approx = approx_cos(xs)
true = np.cos(xs)
abs_err = np.abs(approx - true)

mask = np.abs(true) > 1e-6
rel_err = np.abs((approx[mask] - true[mask]) / true[mask])

print(f"\nMax absolute error on [-pi, pi]:            {abs_err.max():.3e}")
print(f"Max relative error (where |cos| > 1e-6):    {rel_err.max():.3e}")
print(f"Argmax abs error at x = {xs[np.argmax(abs_err)]: .6f}")


# ---------------------------------------------------------------------------
# f32 evaluation + ULP error plot.
# ---------------------------------------------------------------------------


def approx_cos_f32(x):
    x = x.astype(np.float32)
    P32 = P_coef.astype(np.float32)
    pi_half_sq32 = np.float32(PI_QUARTER_SQ)
    x2 = x * x
    val = P32[-1]
    for c in reversed(P32[:-1]):
        val = val * x2 + c
    return (x2 - pi_half_sq32) * val


xs32 = np.linspace(-PI, PI, 200_001, dtype=np.float32)
true64 = np.cos(xs32.astype(np.float64))
approx32 = approx_cos_f32(xs32).astype(np.float64)

ulp = np.abs(np.spacing(true64.astype(np.float32))).astype(np.float64)
ulp_err = np.abs(approx32 - true64) / np.where(ulp > 0, ulp, np.nan)
valid = np.isfinite(ulp_err) & (np.abs(true64) > 1e-6)

n_bins = 200
bin_edges = np.linspace(-PI, PI, n_bins + 1)
bin_idx = np.clip(np.searchsorted(bin_edges, xs32, side="right") - 1, 0, n_bins - 1)
sums = np.bincount(bin_idx, weights=np.where(valid, ulp_err, 0.0), minlength=n_bins)
counts = np.bincount(bin_idx, weights=valid.astype(np.float64), minlength=n_bins)
bin_mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

overall_mean = ulp_err[valid].mean()
overall_max = ulp_err[valid].max()
print(f"\nf32 ULP error (|cos(x)| > 1e-6): mean = {overall_mean:.2f}, max = {overall_max:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xs32[valid], ulp_err[valid], ".", color="#cccccc", ms=1, label="pointwise |ULP error|")
ax.plot(bin_centers, bin_mean, color="#1f77b4", lw=1.8, label="binned mean")
ax.axhline(overall_mean, color="#d62728", ls="--", lw=1, label=f"overall mean = {overall_mean:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("|error| in f32 ULPs of cos(x)")
ax.set_title(f"Chebyshev cos approximation, f32 ULP error (max = {overall_max:.0f})")
ax.set_xlim(-PI, PI)
ax.set_ylim(0, np.nanpercentile(ulp_err[valid], 99.5) * 1.5)
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()

out_path = "work/out/cheb_cos_ulp.png"
fig.savefig(out_path, dpi=130)
print(f"Saved plot to {out_path}")
plt.show()


# ---------------------------------------------------------------------------
# Split-pi/2 refinement: avoid cancellation near x = +/- pi/2.
# ---------------------------------------------------------------------------

PI_HALF_MAJOR = np.float32(PI_HALF)
PI_HALF_MINOR = np.float32(PI_HALF - float(PI_HALF_MAJOR))
print(f"\npi/2 major (f32) = {float(PI_HALF_MAJOR):.10e}")
print(f"pi/2 minor (f32) = {float(PI_HALF_MINOR):.10e}")
print(f"major + minor - true pi/2 = {float(PI_HALF_MAJOR) + float(PI_HALF_MINOR) - PI_HALF:.3e}")


def approx_cos_f32_split(x):
    x = x.astype(np.float32)
    P32 = P_coef.astype(np.float32)
    x2 = x * x
    val = P32[-1]
    for c in reversed(P32[:-1]):
        val = val * x2 + c
    lo = (x - PI_HALF_MAJOR) - PI_HALF_MINOR  # ~ x - pi/2
    hi = (x + PI_HALF_MAJOR) + PI_HALF_MINOR  # ~ x + pi/2
    return lo * hi * val


approx32b = approx_cos_f32_split(xs32).astype(np.float64)
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
ax2.set_xlim(-PI, PI)
ax2.set_ylim(0, max(np.nanmax(bin_mean), np.nanmax(bin_mean2)) * 1.1)
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)
fig2.tight_layout()

out_path2 = "work/out/cheb_cos_ulp_split_pi.png"
fig2.savefig(out_path2, dpi=130)
print(f"Saved plot to {out_path2}")


def _f(c):
    return f"{float(np.float32(c)):.9e}f"


print("\n// === C++ ===")
print(f"// cos(x) on [-pi, pi] (split-pi/2).  Max f32 ULP ~{overall_max2:.2f}.")
print("inline float approx_cos(float x) {")
print(f"    constexpr float PI_HALF_MAJOR = {_f(PI_HALF_MAJOR)};")
print(f"    constexpr float PI_HALF_MINOR = {_f(PI_HALF_MINOR)};")
print("    const float x2 = x * x;")
print(f"    float p = {_f(P_coef[-1])};")
for c in reversed(P_coef[:-1]):
    print(f"    p = p * x2 + {_f(c)};")
print("    const float lo = (x - PI_HALF_MAJOR) - PI_HALF_MINOR;  // ~ x - pi/2 (Sterbenz)")
print("    const float hi = (x + PI_HALF_MAJOR) + PI_HALF_MINOR;  // ~ x + pi/2")
print("    return lo * hi * p;")
print("}")

plt.show()
