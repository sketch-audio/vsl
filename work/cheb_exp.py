"""exp(x) Chebyshev approximation on [-ln(2)/2, ln(2)/2].

Intended for range-reduced exp: x = n * ln(2) + r, r in [-ln(2)/2, ln(2)/2],
then exp(x) = 2^n * exp(r). Note: achieving r in [-ln(2)/2, ln(2)/2] requires
n = round(x/ln(2)); using floor (as in your snippet) yields r in [0, ln(2)],
in which case just refit with A=0, B=ln(2) below.

exp is entire and smooth. Direct Chebyshev fit, substitute t = x/(ln(2)/2).

  exp(x) ~ c_0 + c_1*x + c_2*x^2 + ... + c_N*x^N.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

LN2 = np.log(2)
A, B = -LN2 / 2, LN2 / 2
MID, HALF = 0.5 * (A + B), 0.5 * (B - A)

DEGREE = 6
cheb = Chebyshev.interpolate(lambda t: np.exp(MID + HALF * t), DEGREE, domain=[-1, 1])
print("Chebyshev coefficients a_n:")
for n, c in enumerate(cheb.coef):
    print(f"  a_{n:2d} = {c: .6e}")

poly_t = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
x_as_t = Polynomial([-MID / HALF, 1.0 / HALF])
result = Polynomial([poly_t.coef[-1]])
for c in reversed(poly_t.coef[:-1]):
    result = result * x_as_t + c
coefs = result.coef

print(f"\nMonomial coefficients of exp(x) ~ sum c_k x^k on [{A:.4f}, {B:.4f}]:")
for k, c in enumerate(coefs):
    print(f"  x^{k:2d}: {c: .15e}")


def approx_exp(x):
    x = np.asarray(x, dtype=np.float64)
    val = coefs[-1]
    for c in reversed(coefs[:-1]):
        val = val * x + c
    return val


xs = np.linspace(A, B, 200_001)
approx = approx_exp(xs)
true = np.exp(xs)
abs_err = np.abs(approx - true)
rel_err = np.abs((approx - true) / true)
print(f"\nf64 max abs err: {abs_err.max():.3e}, max rel err: {rel_err.max():.3e}")


def approx_exp_f32(x):
    x = x.astype(np.float32)
    c32 = coefs.astype(np.float32)
    val = c32[-1]
    for c in reversed(c32[:-1]):
        val = val * x + c
    return val


xs32 = np.linspace(A, B, 200_001, dtype=np.float32)
true64 = np.exp(xs32.astype(np.float64))
approx32 = approx_exp_f32(xs32).astype(np.float64)

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
ax.set_xlabel("x"); ax.set_ylabel("|error| in f32 ULPs of exp(x)")
ax.set_title(f"exp f32 ULP on [-ln2/2, ln2/2] (max = {overall_max:.2f})")
ax.set_xlim(A, B); ax.set_ylim(bottom=0)
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("work/out/cheb_exp_ulp.png", dpi=130)
print("Saved work/out/cheb_exp_ulp.png")


def _f(c):
    return f"{float(np.float32(c)):.9e}f"


print("\n// === C++ ===")
print(f"// exp(x) on [-ln(2)/2, ln(2)/2] (post range-reduction).  Max f32 ULP ~{overall_max:.2f}.")
print("inline float approx_exp_r(float x) {")
print(f"    float v = {_f(coefs[-1])};")
for c in reversed(coefs[:-1]):
    print(f"    v = v * x + {_f(c)};")
print("    return v;")
print("}")
