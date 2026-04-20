"""tanh(x) Chebyshev approximation on [-L, L] with L = 1.

tanh is odd, analytic on R with nearest poles at +/- i*pi/2. Factor out the
zero at origin: tanh(x)/x is smooth and EVEN, with value 1 at x=0 and
tanh(1) ~ 0.7616 at the endpoints.

  tanh(x) ~ x * P(x^2)   for |x| <= 1

Chebyshev Bernstein-ellipse convergence in scaled t = x/L has
rho = sqrt(1 + (pi/(2L))^2) + pi/(2L). At L=1, rho ~ 3.43, so a modest
DEGREE = 12 suffices for f32. Widening L beyond ~1 slows convergence fast
(L=2 needs ~20 terms; L=3 needs ~32, and the monomial-basis Horner path
starts losing precision as coefficients span many orders of magnitude).

Outside [-1, 1] the caller should range-reduce via tanh(x) = 1 - 2/(exp(2x)+1)
-- that is outside the scope of this Chebyshev demo.

No major/minor trick applies -- no pi constants in the evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

L = 1.0


def g(x):
    """g(x) = tanh(x) / x.  Smooth and even."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    out[~small] = np.tanh(x[~small]) / x[~small]
    # Taylor: tanh(x)/x = 1 - x^2/3 + 2 x^4/15 - 17 x^6/315 + ...
    xs = x[small]
    out[small] = 1 - xs**2/3 + 2*xs**4/15 - 17*xs**6/315
    return out


DEGREE = 12
cheb = Chebyshev.interpolate(lambda t: g(L * t), DEGREE, domain=[-1, 1])
print(f"Chebyshev coefficients of g(L*t), L={L}:")
for n, c in enumerate(cheb.coef):
    tag = "  (odd, ~0)" if n % 2 else ""
    print(f"  a_{n:2d} = {c: .6e}{tag}")

poly_t = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
# t = x/L, so coef_in_x[k] = poly_t.coef[k] * (1/L)^k
coef_x = np.array([c * (1.0 / L)**k for k, c in enumerate(poly_t.coef)])
P_coef = coef_x[::2]

print(f"\nP(x^2) such that tanh(x) ~ x * P(x^2):")
for k, c in enumerate(P_coef):
    print(f"  x^{2*k:2d}: {c: .15e}")


def approx_tanh(x):
    x = np.asarray(x, dtype=np.float64)
    xc = np.clip(x, -L, L)
    x2 = xc * xc
    val = P_coef[-1]
    for c in reversed(P_coef[:-1]):
        val = val * x2 + c
    result = xc * val
    # Saturate outside [-L, L].
    return np.where(x > L, 1.0, np.where(x < -L, -1.0, result))


xs = np.linspace(-L, L, 200_001)
approx = approx_tanh(xs)
true = np.tanh(xs)
abs_err = np.abs(approx - true)
mask = np.abs(true) > 1e-6
rel_err = np.abs((approx[mask] - true[mask]) / true[mask])
print(f"\nf64 max abs err on [-{L}, {L}]: {abs_err.max():.3e}, max rel err: {rel_err.max():.3e}")


def approx_tanh_f32(x):
    x = x.astype(np.float32)
    P32 = P_coef.astype(np.float32)
    Lf = np.float32(L)
    xc = np.clip(x, -Lf, Lf)
    x2 = xc * xc
    val = P32[-1]
    for c in reversed(P32[:-1]):
        val = val * x2 + c
    result = xc * val
    return np.where(x > Lf, np.float32(1.0), np.where(x < -Lf, np.float32(-1.0), result))


xs32 = np.linspace(-L, L, 200_001, dtype=np.float32)
true64 = np.tanh(xs32.astype(np.float64))
approx32 = approx_tanh_f32(xs32).astype(np.float64)

ulp = np.abs(np.spacing(true64.astype(np.float32))).astype(np.float64)
ulp_err = np.abs(approx32 - true64) / np.where(ulp > 0, ulp, np.nan)
valid = np.isfinite(ulp_err) & (np.abs(true64) > 1e-6)

n_bins = 200
bin_edges = np.linspace(-L, L, n_bins + 1)
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
ax.set_xlabel("x"); ax.set_ylabel("|error| in f32 ULPs of tanh(x)")
ax.set_title(f"tanh f32 ULP on [-{L}, {L}] (max = {overall_max:.2f})")
ax.set_xlim(-L, L); ax.set_ylim(0, np.nanpercentile(ulp_err[valid], 99.5) * 1.5)
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("work/out/cheb_tanh_ulp.png", dpi=130)
print("Saved work/out/cheb_tanh_ulp.png")


def _f(c):
    return f"{float(np.float32(c)):.9e}f"


print("\n// === C++ ===")
print(f"// tanh(x) on [-1, 1] (saturates outside).  Max f32 ULP ~{overall_max:.2f}.")
print("inline float approx_tanh(float x) {")
print("    const float xc = x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);")
print("    const float x2 = xc * xc;")
print(f"    float p = {_f(P_coef[-1])};")
for c in reversed(P_coef[:-1]):
    print(f"    p = p * x2 + {_f(c)};")
print("    return xc * p;")
print("}")
