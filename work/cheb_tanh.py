"""tanh(x) Chebyshev approximation on [-L, L] with L = 1.

Exploration only: vsl::tanh's coefficients (both precisions) come from remez_tanh.py.

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


# ===========================================================================
# Padé rational approximant on [-L_RAT, L_RAT].
# tanh(x) = x * N(x^2) / D(x^2) exploiting odd symmetry and zero at origin.
# G(u) = tanh(x)/x is smooth and even, fit as rational N(u)/D(u) in u = x^2.
# Taylor coefficients of G(u) via the identity:
#   G(u) = sum_{k=0}^inf  c_k * u^k
# where c_k = (-1)^k * 4^(k+1) * (4^(k+1) - 1) * B_{2k+2} / (2k+2)!
# and B_{2n} are Bernoulli numbers.  We use mpmath for high precision.
# ===========================================================================

print("\n\n=== Padé rational approximant for tanh on [-4, 4] ===")

import mpmath


def pade(an, m):
    """Padé approximant [m / (len(an)-1-m)] of the series sum an[k]*x^k.
    Returns (num_coef, den_coef) as ascending-power numpy arrays with den[0]=1.
    Equivalent to the removed scipy.misc.pade.
    """
    p = m
    q = len(an) - 1 - m
    a = np.array(an, dtype=np.float64)
    # Build q×q system for denominator coefficients d_1..d_q.
    # Row i (0-indexed): sum_{j=1}^{q} a[p+i+1-j]*d_j = -a[p+i+1]  for i=0..q-1
    A_mat = np.zeros((q, q))
    b_vec = np.zeros(q)
    for i in range(q):
        b_vec[i] = -a[p + i + 1] if (p + i + 1) < len(a) else 0.0
        for j in range(q):
            idx = p + i - j
            A_mat[i, j] = a[idx] if 0 <= idx < len(a) else 0.0
    d_coef = np.linalg.solve(A_mat, b_vec) if q > 0 else np.array([])
    den = np.concatenate([[1.0], d_coef])
    # Numerator: N = A * D, truncated to degree p.
    num = np.zeros(p + 1)
    for k in range(p + 1):
        num[k] = sum(a[k - j] * den[j] for j in range(min(k + 1, q + 1)) if (k - j) >= 0)
    return num, den

mpmath.mp.dps = 50  # 50 decimal digits

L_RAT = 4.0

# Taylor coefficients of G(u) = tanh(sqrt(u))/sqrt(u)  (even, so a poly in u).
# c_k such that G(u) = sum c_k * u^k.
# Taylor series converges for |u| < (pi/2)^2 ~ 2.47, but Padé extends beyond.
N_TERMS = 14  # gives [6/6] Padé at most

def g_taylor_coef(k):
    """c_k of G(u) = tanh(x)/x = sum c_k * (x^2)^k.
    c_k = 4^(k+1) * (4^(k+1)-1) * B_{2(k+1)} / (2(k+1))!
    B_{2n} is the signed Bernoulli number; it already provides the alternating sign.
    """
    n = k + 1  # Bernoulli index: B_{2n}
    four_n = mpmath.power(4, n)
    bern = mpmath.bernoulli(2 * n)
    fac = mpmath.factorial(2 * n)
    return four_n * (four_n - 1) * bern / fac

g_coef_mp = [g_taylor_coef(k) for k in range(N_TERMS)]
print("Taylor coefficients of G(u) = tanh(x)/x in u=x^2:")
for k, c in enumerate(g_coef_mp):
    print(f"  c_{k} = {float(c): .15e}")

# Convert to float64 numpy array for scipy.misc.pade.
g_coef_f64 = np.array([float(c) for c in g_coef_mp], dtype=np.float64)

print()
for (p_deg, q_deg) in [(3, 3), (4, 4), (5, 5), (6, 6)]:
    n_coef = p_deg + q_deg + 1
    if n_coef > N_TERMS:
        continue
    num_c, den_c = pade(g_coef_f64[:n_coef], p_deg)
    num = np.polynomial.Polynomial(num_c)
    den = np.polynomial.Polynomial(den_c)

    for test_L in [2.0, 3.0, 4.0]:
        def approx_tanh_pade_f64(x, num=num, den=den, tL=test_L):
            x = np.asarray(x, dtype=np.float64)
            u = x * x
            g_val = num(u) / den(u)
            result = x * g_val
            return np.where(x > tL, 1.0, np.where(x < -tL, -1.0, result))

        xs_dense = np.linspace(-test_L, test_L, 400_001)
        approx_f64 = approx_tanh_pade_f64(xs_dense)
        true_f64 = np.tanh(xs_dense)
        abs_err_f64 = np.abs(approx_f64 - true_f64)
        print(f"[{p_deg}/{q_deg}] on [-{test_L},{test_L}]: max abs err = {abs_err_f64.max():.3e}")

# Pick [5/5] as the target for f32 evaluation on [-3, 3].
p_deg_best, q_deg_best = 5, 5
L_RAT = 3.0
num_coef, den_coef = pade(g_coef_f64[:p_deg_best + q_deg_best + 1], p_deg_best)
# num_coef/den_coef are ascending-power arrays; den_coef[0] == 1.

# Verify D(0) = 1 (scipy.misc.pade normalises den[0]=1).
print(f"\n[{p_deg_best}/{q_deg_best}] num coeffs (asc): {num_coef}")
print(f"[{p_deg_best}/{q_deg_best}] den coeffs (asc): {den_coef}")


def approx_tanh_pade_f32(x, n_coef=num_coef, d_coef=den_coef):
    x = x.astype(np.float32)
    n32 = n_coef.astype(np.float32)
    d32 = d_coef.astype(np.float32)
    u = x * x
    # Horner for numerator and denominator.
    n_val = n32[-1]
    for c in reversed(n32[:-1]):
        n_val = n_val * u + c
    d_val = d32[-1]
    for c in reversed(d32[:-1]):
        d_val = d_val * u + c
    result = x * (n_val / d_val)
    Lf = np.float32(L_RAT)
    return np.where(x > Lf, np.float32(1.0), np.where(x < -Lf, np.float32(-1.0), result))


xs32_rat = np.linspace(-L_RAT, L_RAT, 400_001, dtype=np.float32)
true64_rat = np.tanh(xs32_rat.astype(np.float64))
approx32_rat = approx_tanh_pade_f32(xs32_rat).astype(np.float64)

ulp_rat = np.abs(np.spacing(true64_rat.astype(np.float32))).astype(np.float64)
ulp_err_rat = np.abs(approx32_rat - true64_rat) / np.where(ulp_rat > 0, ulp_rat, np.nan)
valid_rat = np.isfinite(ulp_err_rat) & (np.abs(true64_rat) > 1e-6)

n_bins_rat = 200
bin_edges_rat = np.linspace(-L_RAT, L_RAT, n_bins_rat + 1)
bin_idx_rat = np.clip(np.searchsorted(bin_edges_rat, xs32_rat, side="right") - 1, 0, n_bins_rat - 1)
sums_rat = np.bincount(bin_idx_rat, weights=np.where(valid_rat, ulp_err_rat, 0.0), minlength=n_bins_rat)
counts_rat = np.bincount(bin_idx_rat, weights=valid_rat.astype(np.float64), minlength=n_bins_rat)
bin_mean_rat = np.where(counts_rat > 0, sums_rat / np.maximum(counts_rat, 1), np.nan)
bin_centers_rat = 0.5 * (bin_edges_rat[:-1] + bin_edges_rat[1:])

overall_mean_rat = ulp_err_rat[valid_rat].mean()
overall_max_rat = ulp_err_rat[valid_rat].max()
print(f"\nf32 ULP [{p_deg_best}/{q_deg_best}] Padé on [-{L_RAT},{L_RAT}]: mean = {overall_mean_rat:.2f}, max = {overall_max_rat:.2f}")

fig_rat, ax_rat = plt.subplots(figsize=(10, 5))
ax_rat.plot(xs32_rat[valid_rat], ulp_err_rat[valid_rat], ".", color="#cccccc", ms=1, label="pointwise")
ax_rat.plot(bin_centers_rat, bin_mean_rat, color="#1f77b4", lw=1.8, label="binned mean")
ax_rat.axhline(overall_mean_rat, color="#d62728", ls="--", lw=1, label=f"mean = {overall_mean_rat:.2f}")
ax_rat.axvline(-1, color="#999", lw=0.8, ls="--"); ax_rat.axvline(1, color="#999", lw=0.8, ls="--")
ax_rat.set_xlabel("x"); ax_rat.set_ylabel("|error| in f32 ULPs of tanh(x)")
ax_rat.set_title(f"tanh Padé [{p_deg_best}/{q_deg_best}] f32 ULP on [-{L_RAT},{L_RAT}] (max = {overall_max_rat:.2f})")
ax_rat.set_xlim(-L_RAT, L_RAT); ax_rat.set_ylim(bottom=0)
ax_rat.legend(); ax_rat.grid(alpha=0.3); fig_rat.tight_layout()
fig_rat.savefig("work/out/cheb_tanh_pade_ulp.png", dpi=130)
print("Saved work/out/cheb_tanh_pade_ulp.png")


print(f"\n// === C++ Padé [{p_deg_best}/{q_deg_best}] rational tanh ===")
print(f"// tanh(x) via [{p_deg_best}/{q_deg_best}] Padé in u=x^2.  Accurate on [-{L_RAT},{L_RAT}].")
print(f"// Max f32 ULP ~{overall_max_rat:.2f}.")
print("inline float approx_tanh_pade(float x) {")
print(f"    if (x > {float(L_RAT)}f) return  1.0f;")
print(f"    if (x < {float(-L_RAT)}f) return -1.0f;")
print("    const float u = x * x;")
print(f"    float n = {_f(num_coef[-1])};")
for c in reversed(num_coef[:-1]):
    print(f"    n = n * u + {_f(c)};")
print(f"    float d = {_f(den_coef[-1])};")
for c in reversed(den_coef[:-1]):
    print(f"    d = d * u + {_f(c)};")
print("    return x * (n / d);")
print("}")


# ===========================================================================
# sinh-proxy: tanh(x) = p(x) / sqrt(p(x)^2 + 1)
# Identity: sinh(x)/cosh(x) = sinh(x)/sqrt(sinh(x)^2+1) = tanh(x).
# p(x) is an odd polynomial approximating sinh(x).  Any odd poly gives a
# result bounded in (-1,1) with automatic smooth saturation -- no domain
# guard needed.  rsqrt is a single SIMD instruction (e.g. _mm_rsqrt_ps).
#
# We write p(x) = x * R(x^2) (odd symmetry), fit R to sinh(x)/x on [0,L]
# via Chebyshev, then refine the coefficients with scipy to directly minimize
# the max tanh approximation error on the same domain.
# ===========================================================================

print("\n\n=== sinh-proxy: tanh(x) = p(x)/sqrt(p(x)^2+1) ===")

from scipy.optimize import minimize as sp_minimize

L_SP = 4.0   # evaluation domain; saturation is automatic beyond this
N_SP = 5     # number of terms in R, giving p(x) = x*(r0 + r1*x^2 + ... + r4*x^8)
             # i.e. p has degree 2*N_SP - 1 = 9


def sinh_over_x(x):
    """sinh(x)/x, smooth everywhere; value 1 at x=0."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-6
    out[~small] = np.sinh(x[~small]) / x[~small]
    xs2 = x[small]**2
    out[small] = 1 + xs2/6 + xs2**2/120 + xs2**3/5040
    return out


# Initial coefficients: Chebyshev fit of R(u)=sinh(sqrt(u))/sqrt(u) on [0, L^2].
# Map u in [0,L^2] to t in [-1,1]: t = 2u/L^2 - 1.
L2 = L_SP**2
cheb_sinh = Chebyshev.interpolate(
    lambda t: sinh_over_x(L_SP * np.sqrt(0.5 * (t + 1))),
    2 * N_SP, domain=[-1, 1]
)
poly_sinh_t = cheb_sinh.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
u_as_t_sinh = Polynomial([-1.0, 2.0 / L2])   # t = 2u/L^2 - 1
composed = Polynomial([poly_sinh_t.coef[-1]])
for c in reversed(poly_sinh_t.coef[:-1]):
    composed = composed * u_as_t_sinh + c
R0_coef = composed.coef[:N_SP]  # first N_SP ascending-power coefficients in u

print(f"Initial R coefficients from Chebyshev fit to sinh(x)/x on [0,{L_SP}]:")
for k, c in enumerate(R0_coef):
    print(f"  r_{k} = {c: .10e}")


def tanh_proxy(x, r_coef):
    """tanh ≈ p(x)/sqrt(p(x)^2+1), p(x) = x * R(x^2)."""
    u = x * x
    R_val = r_coef[-1]
    for c in reversed(r_coef[:-1]):
        R_val = R_val * u + c
    px = x * R_val
    return px / np.sqrt(px * px + 1.0)


# Dense grid on [0, L_SP] (odd symmetry so only positive half needed).
xs_sp = np.linspace(0.0, L_SP, 200_001)
true_sp = np.tanh(xs_sp)

def max_abs_err(r_coef):
    return np.max(np.abs(tanh_proxy(xs_sp, r_coef) - true_sp))

err0 = max_abs_err(R0_coef)
print(f"Initial (Chebyshev) f64 max abs err on [-{L_SP},{L_SP}]: {err0:.3e}")

# Refine with Nelder-Mead to directly minimise the tanh approximation error.
result = sp_minimize(max_abs_err, R0_coef, method="Nelder-Mead",
                     options={"maxiter": 200_000, "xatol": 1e-14, "fatol": 1e-14})
R_coef = result.x
print(f"Refined               f64 max abs err on [-{L_SP},{L_SP}]: {result.fun:.3e}")
print("Refined R coefficients:")
for k, c in enumerate(R_coef):
    print(f"  r_{k} = {c: .10e}")


# f32 ULP evaluation on [-L_SP, L_SP].
def tanh_proxy_f32(x, r_coef=R_coef):
    x = x.astype(np.float32)
    r32 = r_coef.astype(np.float32)
    u = x * x
    R_val = r32[-1]
    for c in reversed(r32[:-1]):
        R_val = R_val * u + c
    px = x * R_val
    return (px / np.sqrt(px * px + np.float32(1.0))).astype(np.float64)


xs32_sp = np.linspace(-L_SP, L_SP, 400_001, dtype=np.float32)
true64_sp = np.tanh(xs32_sp.astype(np.float64))
approx32_sp = tanh_proxy_f32(xs32_sp)

ulp_sp = np.abs(np.spacing(true64_sp.astype(np.float32))).astype(np.float64)
ulp_err_sp = np.abs(approx32_sp - true64_sp) / np.where(ulp_sp > 0, ulp_sp, np.nan)
valid_sp = np.isfinite(ulp_err_sp) & (np.abs(true64_sp) > 1e-6)

n_bins_sp = 200
bin_edges_sp = np.linspace(-L_SP, L_SP, n_bins_sp + 1)
bin_idx_sp = np.clip(np.searchsorted(bin_edges_sp, xs32_sp, side="right") - 1, 0, n_bins_sp - 1)
sums_sp = np.bincount(bin_idx_sp, weights=np.where(valid_sp, ulp_err_sp, 0.0), minlength=n_bins_sp)
counts_sp = np.bincount(bin_idx_sp, weights=valid_sp.astype(np.float64), minlength=n_bins_sp)
bin_mean_sp = np.where(counts_sp > 0, sums_sp / np.maximum(counts_sp, 1), np.nan)
bin_centers_sp = 0.5 * (bin_edges_sp[:-1] + bin_edges_sp[1:])

overall_mean_sp = ulp_err_sp[valid_sp].mean()
overall_max_sp = ulp_err_sp[valid_sp].max()
print(f"f32 ULP sinh-proxy degree-{2*N_SP-1} on [-{L_SP},{L_SP}]: mean={overall_mean_sp:.2f}, max={overall_max_sp:.2f}")

fig_sp, ax_sp = plt.subplots(figsize=(10, 5))
ax_sp.plot(xs32_sp[valid_sp], ulp_err_sp[valid_sp], ".", color="#cccccc", ms=1, label="pointwise")
ax_sp.plot(bin_centers_sp, bin_mean_sp, color="#1f77b4", lw=1.8, label="binned mean")
ax_sp.axhline(overall_mean_sp, color="#d62728", ls="--", lw=1, label=f"mean={overall_mean_sp:.2f}")
ax_sp.axvline(-1, color="#999", lw=0.8, ls="--"); ax_sp.axvline(1, color="#999", lw=0.8, ls="--")
ax_sp.set_xlabel("x"); ax_sp.set_ylabel("|error| in f32 ULPs of tanh(x)")
ax_sp.set_title(f"tanh sinh-proxy degree-{2*N_SP-1} f32 ULP on [-{L_SP},{L_SP}] (max={overall_max_sp:.2f})")
ax_sp.set_xlim(-L_SP, L_SP); ax_sp.set_ylim(bottom=0)
ax_sp.legend(); ax_sp.grid(alpha=0.3); fig_sp.tight_layout()
fig_sp.savefig("work/out/cheb_tanh_sinhproxy_ulp.png", dpi=130)
print("Saved work/out/cheb_tanh_sinhproxy_ulp.png")


print(f"\n// === C++ sinh-proxy degree-{2*N_SP-1} ===")
print(f"// tanh(x) = p(x)/sqrt(p(x)^2+1), p(x)=x*R(x^2). No domain guard needed.")
print(f"// Max f32 ULP ~{overall_max_sp:.2f} on [-{L_SP},{L_SP}]; saturates smoothly beyond.")
print("inline float approx_tanh_sinh_proxy(float x) {")
print("    const float u = x * x;")
print(f"    float r = {_f(R_coef[-1])};")
for c in reversed(R_coef[:-1]):
    print(f"    r = r * u + {_f(c)};")
print("    const float p = x * r;")
print("    return p * rsqrtf(p * p + 1.0f);")
print("}")
