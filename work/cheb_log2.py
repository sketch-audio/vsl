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

import sys

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


# ===========================================================================
# [0.75, 1.5) reduction -- zero of log2 at x=1 is interior to the range.
# ===========================================================================
# Reduction: from m in [1, 2), if m >= 1.5 halve to [0.75, 1.0) and add 1 to e.
# After reduction m_adj in [0.75, 1.5), u = m_adj - 1 in [-0.25, 0.5).
# Factor: log2(m_adj) = u * G(u), G(u) = log2(1+u)/u, smooth on [-0.25, 0.5).
# Fit G via Chebyshev in t = (u - 0.125) / 0.375 on [-1, 1].

A2, B2 = -0.25, 0.5
MID2 = (A2 + B2) / 2    # 0.125
HALF2 = (B2 - A2) / 2   # 0.375

DEGREE2 = int(sys.argv[1]) if len(sys.argv) > 1 else 8  # Precision::Fast: 6
cheb2 = Chebyshev.interpolate(lambda t: g(MID2 + HALF2 * t), DEGREE2, domain=[-1, 1])
print("\n\n=== [0.75, 1.5) reduction ===")
print("Chebyshev coefficients of G(0.125 + 0.375*t):")
for n, c in enumerate(cheb2.coef):
    print(f"  a_{n:2d} = {c: .6e}")

poly_t2 = cheb2.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
# Substitute t = (u - MID2) / HALF2 to get polynomial in u via Horner composition.
u_as_t2 = Polynomial([-MID2 / HALF2, 1.0 / HALF2])
result2 = Polynomial([poly_t2.coef[-1]])
for c in reversed(poly_t2.coef[:-1]):
    result2 = result2 * u_as_t2 + c
P2_coef = result2.coef

print(f"\nP(u) such that log2(1+u) ~ u * P(u), u in [-0.25, 0.5):")
for k, c in enumerate(P2_coef):
    print(f"  u^{k:2d}: {c: .15e}")


def approx_log2_75(m):
    """Evaluate log2(m) for m in [0.75, 1.5).  m_adj comes in already reduced."""
    m = np.asarray(m, dtype=np.float64)
    u = m - 1.0
    val = P2_coef[-1]
    for c in reversed(P2_coef[:-1]):
        val = val * u + c
    return u * val


xs2 = np.linspace(0.75, 1.5, 200_001)
approx2 = approx_log2_75(xs2)
true2 = np.log2(xs2)
abs_err2 = np.abs(approx2 - true2)
mask2 = np.abs(true2) > 1e-6
rel_err2 = np.abs((approx2[mask2] - true2[mask2]) / true2[mask2])
print(f"\nf64 max abs err on [0.75, 1.5): {abs_err2.max():.3e}, max rel err: {rel_err2.max():.3e}")


def approx_log2_75_f32(m):
    m = m.astype(np.float32)
    P32 = P2_coef.astype(np.float32)
    u = m - np.float32(1.0)
    val = P32[-1]
    for c in reversed(P32[:-1]):
        val = val * u + c
    return u * val


xs2_32 = np.linspace(0.75, 1.5, 200_001, dtype=np.float32)
true2_64 = np.log2(xs2_32.astype(np.float64))
approx2_32 = approx_log2_75_f32(xs2_32).astype(np.float64)

ulp2 = np.abs(np.spacing(true2_64.astype(np.float32))).astype(np.float64)
ulp_err2 = np.abs(approx2_32 - true2_64) / np.where(ulp2 > 0, ulp2, np.nan)
valid2 = np.isfinite(ulp_err2) & (np.abs(true2_64) > 1e-6)

n_bins2 = 100
bin_edges2 = np.linspace(0.75, 1.5, n_bins2 + 1)
bin_idx2 = np.clip(np.searchsorted(bin_edges2, xs2_32, side="right") - 1, 0, n_bins2 - 1)
sums2 = np.bincount(bin_idx2, weights=np.where(valid2, ulp_err2, 0.0), minlength=n_bins2)
counts2 = np.bincount(bin_idx2, weights=valid2.astype(np.float64), minlength=n_bins2)
bin_mean2 = np.where(counts2 > 0, sums2 / np.maximum(counts2, 1), np.nan)
bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])

overall_mean2 = ulp_err2[valid2].mean()
overall_max2 = ulp_err2[valid2].max()
print(f"f32 ULP on [0.75, 1.5): mean = {overall_mean2:.2f}, max = {overall_max2:.2f}")

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(xs2_32[valid2], ulp_err2[valid2], ".", color="#cccccc", ms=1, label="pointwise")
ax2.plot(bin_centers2, bin_mean2, color="#1f77b4", lw=1.8, label="binned mean")
ax2.axhline(overall_mean2, color="#d62728", ls="--", lw=1, label=f"mean = {overall_mean2:.2f}")
ax2.axvline(1.0, color="#999", lw=0.8, ls="--", label="x=1 (zero)")
ax2.set_xlabel("m"); ax2.set_ylabel("|error| in f32 ULPs of log2(m)")
ax2.set_title(f"log2 f32 ULP on [0.75, 1.5) (max = {overall_max2:.2f})")
ax2.set_xlim(0.75, 1.5); ax2.set_ylim(bottom=0)
ax2.legend(); ax2.grid(alpha=0.3); fig2.tight_layout()
fig2.savefig("work/out/cheb_log2_75_ulp.png", dpi=130)
print("Saved work/out/cheb_log2_75_ulp.png")

print("\n// === C++ [0.75, 1.5) reduction] ===")
print(f"// log2(x) on R.  Mantissa reduced to [0.75, 1.5).  Max f32 ULP ~{overall_max2:.2f}.")
print("template<typename X>")
print("force_inline constexpr auto log2(X x) -> X {")
print("    using S = scalar_t<X>;")
print("    constexpr auto exp_bias = ieee_exp_bias_v<S>;")
print("    constexpr auto sig_bits = ieee_sig_bits_v<S>;")
print("    constexpr auto one = decltype(sig_bits){1};")
print("    constexpr auto sig_mask = (one << sig_bits) - one;")
print("    const auto bits = reinterpret_as_int(x);")
print("    const auto val  = unsigned_to_signed(bits >> sig_bits) - unsigned_to_signed(exp_bias);")
print("    const auto m    = reinterpret_as_float((exp_bias << sig_bits) | (bits & sig_mask));")
print("    // Fold [1, 2) -> [0.75, 1.5): if m >= 1.5, halve m and add 1 to exponent.")
print("    const auto fold = m >= S(1.5f);")
print("    const auto m_adj = select(fold, m * S(0.5f), m);")
print("    const auto e_adj = select(fold, signed_to_float(val) + S(1.f), signed_to_float(val));")
print("    const auto u = m_adj - S(1.f);")
print(f"    auto p = X({_f(P2_coef[-1])});")
for c in reversed(P2_coef[:-1]):
    print(f"    p = p * u + S({_f(c)});")
print("    return e_adj + u * p;")
print("}")
