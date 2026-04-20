"""log2(x) Chebyshev approximation on [1, 2].

log2 has a zero at x = 1, so factor it out for good relative error there:
    log2(x) = (x - 1) * g(x - 1)
where g(u) = log2(1 + u) / u is smooth on [0, 1]:
    g(0) = 1/ln(2),   g(1) = 1.
Fit g as a Chebyshev series in t = 2u - 1 on [-1, 1], convert to monomials
in u, then reconstruct log2(x) = (x - 1) * P(x - 1).

Intended for range-reduced log2 where the caller decomposes x = 2^e * m with
m in [1, 2) and computes log2(x) = e + log2(m).
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

LN2 = np.log(2)


def g(u):
    """g(u) = log2(1 + u) / u.  Smooth on [0, 1]; g(0) = 1/ln(2)."""
    u = np.asarray(u, dtype=np.float64)
    out = np.empty_like(u)
    small = np.abs(u) < 1e-3
    big = ~small
    out[big] = np.log2(1.0 + u[big]) / u[big]
    # Taylor: log2(1+u)/u = (1/ln2) * (1 - u/2 + u^2/3 - u^3/4 + u^4/5 - ...)
    us = u[small]
    out[small] = (1.0 - us/2 + us**2/3 - us**3/4 + us**4/5 - us**5/6) / LN2
    return out


# Fit g on [0, 1] via Chebyshev interpolation in t = 2u - 1.
A, B = 0.0, 1.0
MID, HALF = 0.5, 0.5
DEGREE = 8
cheb = Chebyshev.interpolate(lambda t: g(MID + HALF * t), DEGREE, domain=[-1, 1])
print("Chebyshev coefficients of g((t+1)/2):")
for n, c in enumerate(cheb.coef):
    print(f"  a_{n:2d} = {c: .6e}")

poly_t = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
u_as_t = Polynomial([-MID / HALF, 1.0 / HALF])  # t = (u - MID)/HALF
result = Polynomial([poly_t.coef[-1]])
for c in reversed(poly_t.coef[:-1]):
    result = result * u_as_t + c
P_coef = result.coef

print(f"\nP(u) such that log2(1+u) ~ u * P(u), u in [0, 1]:")
for k, c in enumerate(P_coef):
    print(f"  u^{k:2d}: {c: .15e}")


def approx_log2(x):
    x = np.asarray(x, dtype=np.float64)
    u = x - 1.0
    val = P_coef[-1]
    for c in reversed(P_coef[:-1]):
        val = val * u + c
    return u * val


xs = np.linspace(1.0, 2.0, 200_001)
approx = approx_log2(xs)
true = np.log2(xs)
abs_err = np.abs(approx - true)
mask = np.abs(true) > 1e-6
rel_err = np.abs((approx[mask] - true[mask]) / true[mask])
print(f"\nf64 max abs err: {abs_err.max():.3e}, max rel err (|log2|>1e-6): {rel_err.max():.3e}")


def approx_log2_f32(x):
    x = x.astype(np.float32)
    P32 = P_coef.astype(np.float32)
    u = x - np.float32(1.0)
    val = P32[-1]
    for c in reversed(P32[:-1]):
        val = val * u + c
    return u * val


xs32 = np.linspace(1.0, 2.0, 200_001, dtype=np.float32)
true64 = np.log2(xs32.astype(np.float64))
approx32 = approx_log2_f32(xs32).astype(np.float64)

ulp = np.abs(np.spacing(true64.astype(np.float32))).astype(np.float64)
ulp_err = np.abs(approx32 - true64) / np.where(ulp > 0, ulp, np.nan)
valid = np.isfinite(ulp_err) & (np.abs(true64) > 1e-6)

n_bins = 100
bin_edges = np.linspace(1.0, 2.0, n_bins + 1)
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
ax.set_xlabel("x"); ax.set_ylabel("|error| in f32 ULPs of log2(x)")
ax.set_title(f"log2 f32 ULP on [1, 2] (max = {overall_max:.2f})")
ax.set_xlim(1.0, 2.0); ax.set_ylim(bottom=0)
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("work/out/cheb_log2_ulp.png", dpi=130)
print("Saved work/out/cheb_log2_ulp.png")


def _f(c):
    return f"{float(np.float32(c)):.9e}f"


print("\n// === C++ ===")
print(f"// log2(x) on [1, 2] (post range-reduction).  Max f32 ULP ~{overall_max:.2f}.")
print("inline float approx_log2_m(float x) {")
print("    const float u = x - 1.0f;")
print(f"    float p = {_f(P_coef[-1])};")
for c in reversed(P_coef[:-1]):
    print(f"    p = p * u + {_f(c)};")
print("    return u * p;")
print("}")
