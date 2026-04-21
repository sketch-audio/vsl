"""atan(x) Chebyshev approximation with two-step range reduction.

atan is odd, bounded, and analytic on R (nearest poles at +/- i). On [-1, 1]
the Bernstein ellipse rho ~ 2.41, so ~22 terms are needed for f32. To keep
DEGREE at 12 we reduce twice so the polynomial only sees |x| <= tan(pi/8):

  1. If |x| > 1:            atan(x) = sign(x)*pi/2 - atan(1/x)
                            (brings 1/x into [0, 1])
  2. If result > tan(pi/8): atan(y) = pi/4 + atan((y - 1)/(y + 1))
                            (maps [tan(pi/8), 1] to [-tan(pi/8), 0],
                            using tan(A - B) identity with B = pi/4)

After both reductions the argument is in [-tan(pi/8), tan(pi/8)] ~
[-0.4142, 0.4142]. On that interval the singularity at +/- i maps to
t = +/- i*cot(pi/8) ~ +/- 2.414i, giving Bernstein rho ~ 5.03, i.e.
DEGREE = 12 buys ~28 bits -- comfortably past f32.

Factor out the zero at origin: atan(x)/x is smooth and EVEN, so

  atan(x) ~ x * P(x^2)   for |x| <= tan(pi/8)

Both pi/2 and pi/4 constants are split into major/minor f32 halves to keep
the combine step accurate when the polynomial sums to something close to the
constant.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

PI = np.pi
PI_HALF = PI / 2
PI_QUARTER = PI / 4
TAN_PI_8 = np.tan(PI / 8)  # sqrt(2) - 1 ~ 0.4142135624


def g(x):
    """g(x) = atan(x) / x.  Smooth and even; fit on [-tan(pi/8), tan(pi/8)]."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    out[~small] = np.arctan(x[~small]) / x[~small]
    # Taylor: atan(x)/x = 1 - x^2/3 + x^4/5 - x^6/7 + ...
    xs = x[small]
    out[small] = 1 - xs**2/3 + xs**4/5 - xs**6/7
    return out


DEGREE = 8
cheb = Chebyshev.interpolate(lambda t: g(TAN_PI_8 * t), DEGREE, domain=[-1, 1])
print(f"Chebyshev coefficients of atan(K*t)/(K*t), K=tan(pi/8):")
for n, c in enumerate(cheb.coef):
    tag = "  (odd, ~0)" if n % 2 else ""
    print(f"  a_{n:2d} = {c: .6e}{tag}")

poly_t = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
# t = x / K, so coef_in_x[k] = poly_t.coef[k] * (1/K)^k
coef_x = np.array([c * (1.0 / TAN_PI_8)**k for k, c in enumerate(poly_t.coef)])
P_coef = coef_x[::2]

print(f"\nP(x^2) such that atan(x) ~ x * P(x^2) for |x| <= tan(pi/8):")
for k, c in enumerate(P_coef):
    print(f"  x^{2*k:2d}: {c: .15e}")


def approx_atan_core(x):
    """Polynomial part, valid for |x| <= tan(pi/8)."""
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    val = P_coef[-1]
    for c in reversed(P_coef[:-1]):
        val = val * x2 + c
    return x * val


def approx_atan(x):
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    inv = ax > 1.0
    ax = np.where(inv, 1.0 / np.where(ax > 0, ax, 1.0), ax)  # ax in [0, 1]
    big = ax > TAN_PI_8
    r = np.where(big, (ax - 1.0) / (ax + 1.0), ax)  # |r| <= tan(pi/8)
    val = approx_atan_core(r)
    val = np.where(big, val + PI_QUARTER, val)
    val = np.where(inv, PI_HALF - val, val)
    return sign * val


# f64 accuracy on [-1, 1]
xs = np.linspace(-1, 1, 200_001)
approx = approx_atan(xs)
true = np.arctan(xs)
abs_err = np.abs(approx - true)
mask = np.abs(true) > 1e-6
rel_err = np.abs((approx[mask] - true[mask]) / true[mask])
print(f"\nf64 on [-1, 1]: max abs err = {abs_err.max():.3e}, max rel err = {rel_err.max():.3e}")

# f64 accuracy on a wider range showing the tail works
xs2 = np.linspace(-10, 10, 200_001)
err2 = np.abs(approx_atan(xs2) - np.arctan(xs2))
print(f"f64 on [-10, 10]: max abs err = {err2.max():.3e}")


PI_HALF_MAJOR = np.float32(PI_HALF)
PI_HALF_MINOR = np.float32(PI_HALF - float(PI_HALF_MAJOR))
PI_QUARTER_MAJOR = np.float32(PI_QUARTER)
PI_QUARTER_MINOR = np.float32(PI_QUARTER - float(PI_QUARTER_MAJOR))
print(f"\npi/2   major (f32) = {float(PI_HALF_MAJOR):.10e}")
print(f"pi/2   minor (f32) = {float(PI_HALF_MINOR):.10e}")
print(f"pi/4   major (f32) = {float(PI_QUARTER_MAJOR):.10e}")
print(f"pi/4   minor (f32) = {float(PI_QUARTER_MINOR):.10e}")


def approx_atan_core_f32(x):
    x = x.astype(np.float32)
    P32 = P_coef.astype(np.float32)
    x2 = x * x
    val = P32[-1]
    for c in reversed(P32[:-1]):
        val = val * x2 + c
    return x * val


def approx_atan_f32(x):
    x = x.astype(np.float32)
    tan_pi_8 = np.float32(TAN_PI_8)
    sign = np.sign(x).astype(np.float32)
    ax = np.abs(x)
    inv = ax > np.float32(1.0)
    # Avoid 1/0 where ax is 0.
    ax_safe = np.where(ax > 0, ax, np.float32(1.0))
    ax = np.where(inv, np.float32(1.0) / ax_safe, ax).astype(np.float32)
    big = ax > tan_pi_8
    r = np.where(big, (ax - np.float32(1.0)) / (ax + np.float32(1.0)), ax).astype(np.float32)
    val = approx_atan_core_f32(r)
    # Add pi/4 with split-constant trick when big.
    val = np.where(big, (PI_QUARTER_MAJOR + val) + PI_QUARTER_MINOR, val).astype(np.float32)
    # pi/2 - val with split-constant when inv.
    val = np.where(inv, (PI_HALF_MAJOR - val) + PI_HALF_MINOR, val).astype(np.float32)
    return sign * val


# Evaluate ULP on [-10, 10].
DOMAIN = 10.0
xs32 = np.linspace(-DOMAIN, DOMAIN, 200_001, dtype=np.float32)
true64 = np.arctan(xs32.astype(np.float64))
approx32 = approx_atan_f32(xs32).astype(np.float64)

ulp = np.abs(np.spacing(true64.astype(np.float32))).astype(np.float64)
ulp_err = np.abs(approx32 - true64) / np.where(ulp > 0, ulp, np.nan)
valid = np.isfinite(ulp_err) & (np.abs(true64) > 1e-6)

n_bins = 200
bin_edges = np.linspace(-DOMAIN, DOMAIN, n_bins + 1)
bin_idx = np.clip(np.searchsorted(bin_edges, xs32, side="right") - 1, 0, n_bins - 1)
sums = np.bincount(bin_idx, weights=np.where(valid, ulp_err, 0.0), minlength=n_bins)
counts = np.bincount(bin_idx, weights=valid.astype(np.float64), minlength=n_bins)
bin_mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

overall_mean = ulp_err[valid].mean()
overall_max = ulp_err[valid].max()
print(f"f32 ULP on [-{DOMAIN}, {DOMAIN}]: mean = {overall_mean:.2f}, max = {overall_max:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xs32[valid], ulp_err[valid], ".", color="#cccccc", ms=1, label="pointwise")
ax.plot(bin_centers, bin_mean, color="#1f77b4", lw=1.8, label="binned mean")
ax.axhline(overall_mean, color="#d62728", ls="--", lw=1, label=f"mean = {overall_mean:.2f}")
ax.axvline(-1, color="#999", lw=0.5); ax.axvline(1, color="#999", lw=0.5)
ax.axvline(-TAN_PI_8, color="#bbb", lw=0.5, ls=":"); ax.axvline(TAN_PI_8, color="#bbb", lw=0.5, ls=":")
ax.set_xlabel("x"); ax.set_ylabel("|error| in f32 ULPs of atan(x)")
ax.set_title(f"atan f32 ULP on [-{DOMAIN}, {DOMAIN}] (max = {overall_max:.2f})")
ax.set_xlim(-DOMAIN, DOMAIN); ax.set_ylim(0, np.nanpercentile(ulp_err[valid], 99.5) * 1.5)
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("work/out/cheb_atan_ulp.png", dpi=130)
print("Saved work/out/cheb_atan_ulp.png")


def _f(c):
    return f"{float(np.float32(c)):.9e}f"


print("\n// === C++ ===")
print(f"// atan(x) on R.  Max f32 ULP on [-10, 10] ~{overall_max:.2f}.")
print("inline float approx_atan(float x) {")
print(f"    constexpr float PI_HALF_MAJOR = {_f(PI_HALF_MAJOR)};")
print(f"    constexpr float PI_HALF_MINOR = {_f(PI_HALF_MINOR)};")
print(f"    constexpr float PI_QUARTER_MAJOR = {_f(PI_QUARTER_MAJOR)};")
print(f"    constexpr float PI_QUARTER_MINOR = {_f(PI_QUARTER_MINOR)};")
print(f"    constexpr float TAN_PI_8 = {_f(TAN_PI_8)};  // sqrt(2) - 1")
print("    const float sign = x < 0.0f ? -1.0f : 1.0f;")
print("    float ax = x < 0.0f ? -x : x;")
print("    const bool inv = ax > 1.0f;")
print("    if (inv) ax = 1.0f / ax;  // now ax in [0, 1]")
print("    const bool big = ax > TAN_PI_8;")
print("    const float r = big ? (ax - 1.0f) / (ax + 1.0f) : ax;")
print("    const float r2 = r * r;")
print(f"    float p = {_f(P_coef[-1])};")
for c in reversed(P_coef[:-1]):
    print(f"    p = p * r2 + {_f(c)};")
print("    float val = r * p;")
print("    if (big) val = (PI_QUARTER_MAJOR + val) + PI_QUARTER_MINOR;")
print("    if (inv) val = (PI_HALF_MAJOR - val) + PI_HALF_MINOR;")
print("    return sign * val;")
print("}")
