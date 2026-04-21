"""asin(x) Chebyshev approximation.

asin has branch-point singularities at x = +/- 1 (vertical tangents), so a
direct Chebyshev fit on [-1, 1] converges slowly. Standard fix: fit on
[-1/2, 1/2] where asin is analytic, and use the half-angle identity for the
tail:

    |x| > 1/2:  asin(|x|) = pi/2 - 2*asin(sqrt((1 - |x|)/2))
    (then asin(x) = sign(x) * asin(|x|))

Note sqrt((1-|x|)/2) in [0, 1/2] for |x| in [1/2, 1], so we recurse into the
same polynomial. The sqrt is what smooths the branch point -- asin(sqrt(u))
is analytic in u on [0, 1].

Factor out the origin zero: asin(x)/x is smooth and EVEN on [-1/2, 1/2] with
value 1 at x = 0 and 2*asin(1/2) = pi/3 at x = +/-1/2.

  asin(x) ~ x * P(x^2)   for |x| <= 1/2

The pi/2 constant in the tail gets the major/minor split for f32 accuracy.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

PI = np.pi
PI_HALF = PI / 2
HALF = 0.5


def g(x):
    """g(x) = asin(x) / x.  Smooth and even on [-1/2, 1/2]."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    out[~small] = np.arcsin(x[~small]) / x[~small]
    # Taylor: asin(x)/x = 1 + x^2/6 + 3 x^4/40 + 15 x^6/336 + ...
    xs = x[small]
    out[small] = 1 + xs**2/6 + 3*xs**4/40 + 15*xs**6/336
    return out


# Fit g on [-1/2, 1/2] via Chebyshev interpolation in t = 2x.
# Bernstein rho ~ 3.73 (branch points at +/- 1 land at t = +/- 2), so
# DEGREE=12 gives ~23 bits -- right at the f32 edge. Adequate for f32 ULP
# in the low single digits.
DEGREE = 8
cheb = Chebyshev.interpolate(lambda t: g(HALF * t), DEGREE, domain=[-1, 1])
print("Chebyshev coefficients of asin((1/2)*t)/((1/2)*t):")
for n, c in enumerate(cheb.coef):
    tag = "  (odd, ~0)" if n % 2 else ""
    print(f"  a_{n:2d} = {c: .6e}{tag}")

poly_t = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
# t = x / (1/2) = 2x
coef_x = np.array([c * (2.0**k) for k, c in enumerate(poly_t.coef)])
P_coef = coef_x[::2]

print(f"\nP(x^2) such that asin(x) ~ x * P(x^2) for |x| <= 1/2:")
for k, c in enumerate(P_coef):
    print(f"  x^{2*k:2d}: {c: .15e}")


def approx_asin_core(x):
    """Polynomial part, valid for |x| <= 1/2."""
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    val = P_coef[-1]
    for c in reversed(P_coef[:-1]):
        val = val * x2 + c
    return x * val


def approx_asin(x):
    """Extend to [-1, 1] via asin(|x|) = pi/2 - 2*asin(sqrt((1-|x|)/2))."""
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    # For |x| > 1/2: inner = sqrt((1 - |x|) / 2), which lives in [0, 1/2].
    inner = np.sqrt(np.clip((1.0 - ax) / 2.0, 0.0, None))
    core_tail = approx_asin_core(inner)
    tail_magnitude = PI_HALF - 2.0 * core_tail
    core_direct = approx_asin_core(x)
    return np.where(ax <= HALF, core_direct, np.sign(x) * tail_magnitude)


xs = np.linspace(-1.0, 1.0, 200_001)
approx = approx_asin(xs)
true = np.arcsin(xs)
abs_err = np.abs(approx - true)
mask = np.abs(true) > 1e-6
rel_err = np.abs((approx[mask] - true[mask]) / true[mask])
print(f"\nf64 on [-1, 1]: max abs err = {abs_err.max():.3e}, max rel err = {rel_err.max():.3e}")


PI_HALF_MAJOR = np.float32(PI_HALF)
PI_HALF_MINOR = np.float32(PI_HALF - float(PI_HALF_MAJOR))
print(f"\npi/2 major (f32) = {float(PI_HALF_MAJOR):.10e}")
print(f"pi/2 minor (f32) = {float(PI_HALF_MINOR):.10e}")


def approx_asin_core_f32(x):
    x = x.astype(np.float32)
    P32 = P_coef.astype(np.float32)
    x2 = x * x
    val = P32[-1]
    for c in reversed(P32[:-1]):
        val = val * x2 + c
    return x * val


def approx_asin_f32(x):
    x = x.astype(np.float32)
    ax = np.abs(x)
    inner = np.sqrt(np.clip((np.float32(1.0) - ax) / np.float32(2.0), np.float32(0.0), None))
    core_tail = approx_asin_core_f32(inner)
    # pi/2 - 2*core_tail, with major/minor split: (PI_HALF_MAJOR - 2*core) + PI_HALF_MINOR
    tail_mag = (PI_HALF_MAJOR - np.float32(2.0) * core_tail) + PI_HALF_MINOR
    core_direct = approx_asin_core_f32(x)
    s = np.sign(x).astype(np.float32)
    return np.where(ax <= np.float32(HALF), core_direct, s * tail_mag)


xs32 = np.linspace(-1.0, 1.0, 200_001, dtype=np.float32)
true64 = np.arcsin(xs32.astype(np.float64))
approx32 = approx_asin_f32(xs32).astype(np.float64)

ulp = np.abs(np.spacing(true64.astype(np.float32))).astype(np.float64)
ulp_err = np.abs(approx32 - true64) / np.where(ulp > 0, ulp, np.nan)
valid = np.isfinite(ulp_err) & (np.abs(true64) > 1e-6)

n_bins = 200
bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
bin_idx = np.clip(np.searchsorted(bin_edges, xs32, side="right") - 1, 0, n_bins - 1)
sums = np.bincount(bin_idx, weights=np.where(valid, ulp_err, 0.0), minlength=n_bins)
counts = np.bincount(bin_idx, weights=valid.astype(np.float64), minlength=n_bins)
bin_mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

overall_mean = ulp_err[valid].mean()
overall_max = ulp_err[valid].max()
print(f"f32 ULP on [-1, 1]: mean = {overall_mean:.2f}, max = {overall_max:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xs32[valid], ulp_err[valid], ".", color="#cccccc", ms=1, label="pointwise")
ax.plot(bin_centers, bin_mean, color="#1f77b4", lw=1.8, label="binned mean")
ax.axhline(overall_mean, color="#d62728", ls="--", lw=1, label=f"mean = {overall_mean:.2f}")
ax.axvline(-0.5, color="#999", lw=0.5); ax.axvline(0.5, color="#999", lw=0.5)
ax.set_xlabel("x"); ax.set_ylabel("|error| in f32 ULPs of asin(x)")
ax.set_title(f"asin f32 ULP on [-1, 1] (max = {overall_max:.2f})")
ax.set_xlim(-1.0, 1.0); ax.set_ylim(0, np.nanpercentile(ulp_err[valid], 99.5) * 1.5)
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("work/out/cheb_asin_ulp.png", dpi=130)
print("Saved work/out/cheb_asin_ulp.png")


def _f(c):
    return f"{float(np.float32(c)):.9e}f"


print("\n// === C++ ===")
print(f"// asin(x) on [-1, 1].  Max f32 ULP ~{overall_max:.2f}.")
print("inline float approx_asin(float x) {")
print(f"    constexpr float PI_HALF_MAJOR = {_f(PI_HALF_MAJOR)};")
print(f"    constexpr float PI_HALF_MINOR = {_f(PI_HALF_MINOR)};")
print("    const float ax = x < 0.0f ? -x : x;")
print("    const bool big = ax > 0.5f;")
print("    const float inner = std::sqrt((1.0f - ax) * 0.5f);")
print("    const float r = big ? inner : x;")
print("    const float r2 = r * r;")
print(f"    float p = {_f(P_coef[-1])};")
for c in reversed(P_coef[:-1]):
    print(f"    p = p * r2 + {_f(c)};")
print("    const float core = r * p;")
print("    const float sign = x < 0.0f ? -1.0f : 1.0f;")
print("    const float tail = sign * ((PI_HALF_MAJOR - 2.0f * core) + PI_HALF_MINOR);")
print("    return big ? tail : core;")
print("}")
