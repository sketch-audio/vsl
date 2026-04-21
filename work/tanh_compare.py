"""Visual comparison of tanh approximations on [-10, 10]."""

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev, Polynomial

# ---------------------------------------------------------------------------
# Re-derive all approximations (coefficients match cheb_tanh.py outputs).
# ---------------------------------------------------------------------------

# --- 1. Chebyshev polynomial on [-1, 1], saturates outside ---
L_poly = 1.0
DEGREE_poly = 12

def g_poly(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    out[~small] = np.tanh(x[~small]) / x[~small]
    xs = x[small]
    out[small] = 1 - xs**2/3 + 2*xs**4/15 - 17*xs**6/315
    return out

cheb_p = Chebyshev.interpolate(lambda t: g_poly(L_poly * t), DEGREE_poly, domain=[-1, 1])
poly_t = cheb_p.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
coef_x = np.array([c * (1.0/L_poly)**k for k, c in enumerate(poly_t.coef)])
P_coef = coef_x[::2]

def tanh_poly(x):
    x = np.asarray(x, dtype=np.float64)
    xc = np.clip(x, -L_poly, L_poly)
    x2 = xc * xc
    val = P_coef[-1]
    for c in reversed(P_coef[:-1]):
        val = val * x2 + c
    return np.where(x > L_poly, 1.0, np.where(x < -L_poly, -1.0, xc * val))


# --- 2. Padé [5/5] on [-3, 3], saturates outside ---
import mpmath
mpmath.mp.dps = 50

def g_taylor_coef(k):
    n = k + 1
    four_n = mpmath.power(4, n)
    bern = mpmath.bernoulli(2 * n)
    fac = mpmath.factorial(2 * n)
    return four_n * (four_n - 1) * bern / fac

N_TERMS = 13
g_coef_f64 = np.array([float(g_taylor_coef(k)) for k in range(N_TERMS)])

def pade(an, m):
    p, q = m, len(an) - 1 - m
    a = np.array(an, dtype=np.float64)
    A_mat = np.zeros((q, q))
    b_vec = np.zeros(q)
    for i in range(q):
        b_vec[i] = -a[p + i + 1] if (p + i + 1) < len(a) else 0.0
        for j in range(q):
            idx = p + i - j
            A_mat[i, j] = a[idx] if 0 <= idx < len(a) else 0.0
    d_coef = np.linalg.solve(A_mat, b_vec) if q > 0 else np.array([])
    den = np.concatenate([[1.0], d_coef])
    num = np.zeros(p + 1)
    for k in range(p + 1):
        num[k] = sum(a[k-j] * den[j] for j in range(min(k+1, q+1)) if (k-j) >= 0)
    return num, den

L_pade = 3.0
num_coef, den_coef = pade(g_coef_f64[:11], 5)

def tanh_pade(x):
    x = np.asarray(x, dtype=np.float64)
    u = x * x
    n_val = num_coef[-1]
    for c in reversed(num_coef[:-1]):
        n_val = n_val * u + c
    d_val = den_coef[-1]
    for c in reversed(den_coef[:-1]):
        d_val = d_val * u + c
    result = x * (n_val / d_val)
    return np.where(x > L_pade, 1.0, np.where(x < -L_pade, -1.0, result))


# --- 3. sinh-proxy degree-9: p(x)/sqrt(p(x)^2+1), p(x)=x*R(x^2) ---
from scipy.optimize import minimize as sp_minimize

L_SP = 4.0

def sinh_over_x(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-6
    out[~small] = np.sinh(x[~small]) / x[~small]
    xs2 = x[small]**2
    out[small] = 1 + xs2/6 + xs2**2/120 + xs2**3/5040
    return out

import math
R0 = np.array([1.0/math.factorial(2*k+1) for k in range(5)])

xs_opt = np.linspace(0.0, L_SP, 50001)

def tanh_from_R(x, r):
    u = x * x
    R_val = r[-1]
    for c in reversed(r[:-1]):
        R_val = R_val * u + c
    px = x * R_val
    return px / np.sqrt(px * px + 1.0)

result = sp_minimize(lambda r: np.max(np.abs(tanh_from_R(xs_opt, r) - np.tanh(xs_opt))),
                     R0, method="Nelder-Mead",
                     options={"maxiter": 300_000, "xatol": 1e-15, "fatol": 1e-15})
R_coef = result.x

def tanh_sinh_proxy(x):
    return tanh_from_R(np.asarray(x, dtype=np.float64), R_coef)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
xs = np.linspace(-10, 10, 10_001)
true = np.tanh(xs)

fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1.2]})

# Top: all curves
ax = axes[0]
ax.plot(xs, true,              color="black",    lw=2.0,  label="tanh (true)")
ax.plot(xs, tanh_poly(xs),     color="#1f77b4",  lw=1.5,  ls="--", label=f"Chebyshev poly L={L_poly} (saturates)")
ax.plot(xs, tanh_pade(xs),     color="#2ca02c",  lw=1.5,  ls="-.", label=f"Padé [5/5] L={L_pade} (saturates)")
ax.plot(xs, tanh_sinh_proxy(xs), color="#d62728", lw=1.5, ls=":",  label="sinh-proxy deg-9 (auto-sat)")

ax.axvline(-L_poly, color="#1f77b4", lw=0.6, alpha=0.5)
ax.axvline( L_poly, color="#1f77b4", lw=0.6, alpha=0.5)
ax.axvline(-L_pade, color="#2ca02c", lw=0.6, alpha=0.5)
ax.axvline( L_pade, color="#2ca02c", lw=0.6, alpha=0.5)
ax.axvline(-L_SP,   color="#d62728", lw=0.6, alpha=0.5)
ax.axvline( L_SP,   color="#d62728", lw=0.6, alpha=0.5)

ax.set_ylabel("tanh(x)")
ax.set_title("tanh approximation comparison on [-10, 10]")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
ax.set_ylim(-1.3, 1.3)

# Bottom: absolute error (log scale)
ax2 = axes[1]
eps = 1e-16
ax2.semilogy(xs, np.abs(tanh_poly(xs) - true) + eps,
             color="#1f77b4", lw=1.2, ls="--", label="Chebyshev poly")
ax2.semilogy(xs, np.abs(tanh_pade(xs) - true) + eps,
             color="#2ca02c", lw=1.2, ls="-.", label="Padé [5/5]")
ax2.semilogy(xs, np.abs(tanh_sinh_proxy(xs) - true) + eps,
             color="#d62728", lw=1.2, ls=":",  label="sinh-proxy deg-9")
ax2.axhline(1.2e-7, color="#999", lw=0.8, ls="--", label="f32 ULP @ 1.0")

ax2.set_xlabel("x"); ax2.set_ylabel("|error|")
ax2.legend(loc="upper right", fontsize=8); ax2.grid(alpha=0.3, which="both")
ax2.set_ylim(1e-17, 10)

plt.tight_layout()
plt.savefig("work/out/tanh_compare.png", dpi=150)
print("Saved work/out/tanh_compare.png")
