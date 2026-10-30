"""exp2(x) = 2^x Chebyshev approximation on [0, 1].

exp2 is smooth, positive, entire. No zeros, no poles -- direct Chebyshev fit.
Substitute t = 2x - 1 so t in [-1, 1]. Convergence is exponential.

  exp2(x) ~ c_0 + c_1*x + c_2*x^2 + ... + c_N*x^N.

Intended for range-reduced exp2 where the caller handles x = k + r with
r in [0, 1) and multiplies the polynomial by 2^k via ldexp/bit ops.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

from remez import remez

A, B = 0.0, 1.0
MID, HALF = 0.5 * (A + B), 0.5 * (B - A)

DEGREE = int(sys.argv[1]) if len(sys.argv) > 1 else 6  # Precision::Fast: 5
cheb = Chebyshev.interpolate(lambda t: np.exp2(MID + HALF * t), DEGREE, domain=[-1, 1])

print("Chebyshev coefficients a_n:")
for n, c in enumerate(cheb.coef):
    print(f"  a_{n:2d} = {c: .6e}")

# Convert T_n(t) series to power series in t, then compose t = (x - MID)/HALF
# into a polynomial in x via Horner on the linear map.
poly_t = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
x_as_t = Polynomial([-MID / HALF, 1.0 / HALF])  # t = (x - MID)/HALF
result = Polynomial([poly_t.coef[-1]])
for c in reversed(poly_t.coef[:-1]):
    result = result * x_as_t + c
coefs = result.coef

# 2^0 must be exactly 1 (db_to_gain(0) = 1, pow(x, 0) = 1). If c_0 doesn't round to 1 in f32,
# pin it: 2^x = 1 + x Q(x), Q a Remez fit of (2^x - 1) / x weighted by x / 2^x (2^x's relative
# error; the small floor keeps the weight nonzero at 0).
if np.float32(coefs[0]) != 1:
    def quotient(x):
        safe = np.where(x == 0, 1.0, x)
        return np.where(x == 0, np.log(2), np.expm1(safe * np.log(2)) / safe)
    q, _, _ = remez(quotient, lambda x: (x + 1e-3) / np.exp2(x), A, B, DEGREE, grid_size=40001)
    coefs = np.concatenate([[1.0], q])
    print("\nc_0 pinned to 1 (Remez fit of (2^x - 1) / x)")

print(f"\nMonomial coefficients c_k of exp2(x) ~ sum c_k * x^k on [{A}, {B}]:")
for k, c in enumerate(coefs):
    print(f"  x^{k:2d}: {c: .15e}")


def approx_exp2(x):
    x = np.asarray(x, dtype=np.float64)
    val = coefs[-1]
    for c in reversed(coefs[:-1]):
        val = val * x + c
    return val


xs = np.linspace(A, B, 200_001)
approx = approx_exp2(xs)
true = np.exp2(xs)
abs_err = np.abs(approx - true)
rel_err = np.abs((approx - true) / true)
print(f"\nf64 max abs err: {abs_err.max():.3e}, max rel err: {rel_err.max():.3e}")


def approx_exp2_f32(x):
    x = x.astype(np.float32)
    c32 = coefs.astype(np.float32)
    val = c32[-1]
    for c in reversed(c32[:-1]):
        val = val * x + c
    return val


xs32 = np.linspace(A, B, 200_001, dtype=np.float32)
true64 = np.exp2(xs32.astype(np.float64))
approx32 = approx_exp2_f32(xs32).astype(np.float64)

ulp = np.abs(np.spacing(true64.astype(np.float32))).astype(np.float64)
ulp_err = np.abs(approx32 - true64) / np.where(ulp > 0, ulp, np.nan)
valid = np.isfinite(ulp_err)

n_bins = 100
bin_edges = np.linspace(A, B, n_bins + 1)
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
ax.set_xlabel("x"); ax.set_ylabel("|error| in f32 ULPs")
ax.set_title(f"exp2 f32 ULP (max = {overall_max:.2f})")
ax.set_xlim(A, B); ax.set_ylim(bottom=0)
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("work/out/cheb_exp2_ulp.png", dpi=130)
print("Saved work/out/cheb_exp2_ulp.png")


def _f(c):
    return f"{float(np.float32(c)):.9e}f"


print("\n// === C++ ===")
print(f"// 2^x on [0, 1] (post range-reduction).  Max f32 ULP ~{overall_max:.2f}.")
print("inline float approx_exp2_frac(float x) {")
print(f"    float v = {_f(coefs[-1])};")
for c in reversed(coefs[:-1]):
    print(f"    v = v * x + {_f(c)};")
print("    return v;")
print("}")
