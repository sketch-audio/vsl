"""2^f - 1 Chebyshev approximation on [-1/2, 1/2], for vsl::expm1.

expm1 reduces x = n ln(2) + r with n = round(x / ln(2)), so f = r / ln(2) is
in [-1/2, 1/2], and rebuilds

    e^x - 1 = 2^n (2^f - 1) + (2^n - 1)

where 2^n - 1 is exact. The -1 must never be formed from 2^f itself, so fit
the quotient instead:

    2^f - 1 = f * Q(f),    Q(f) = (2^f - 1) / f,   Q(0) = ln(2)

With n = 0 the result is f * Q(f) alone, and small x keeps its relative
precision. Q is smooth and positive; fit it as a Chebyshev series in
t = 2f on [-1, 1] and convert to monomials in f.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

LN2 = np.log(2)
A, B = -0.5, 0.5


def q(f):
    """(2^f - 1) / f, ln(2) at f = 0."""
    f = np.asarray(f, dtype=np.float64)
    safe = np.where(f == 0, 1.0, f)
    return np.where(f == 0, LN2, np.expm1(safe * LN2) / safe)


DEGREE = 5
cheb = Chebyshev.interpolate(q, DEGREE, domain=[A, B])
print("Chebyshev coefficients a_n:")
for n, c in enumerate(cheb.coef):
    print(f"  a_{n:2d} = {c: .6e}")

coefs = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1]).coef
print(f"\nQ(f) such that 2^f - 1 ~ f * Q(f), f in [{A}, {B}]:")
for k, c in enumerate(coefs):
    print(f"  f^{k:2d}: {c: .15e}")


def approx_f32(f):
    f = f.astype(np.float32)
    c32 = coefs.astype(np.float32)
    val = c32[-1]
    for c in reversed(c32[:-1]):
        val = val * f + c
    return f * val


xs32 = np.linspace(A, B, 200_001, dtype=np.float32)
true64 = np.expm1(xs32.astype(np.float64) * LN2)
approx32 = approx_f32(xs32).astype(np.float64)
ulp = np.abs(np.spacing(true64.astype(np.float32))).astype(np.float64)
ulp_err = np.abs(approx32 - true64) / np.where(ulp > 0, ulp, np.nan)
valid = np.isfinite(ulp_err)
overall_max = ulp_err[valid].max()
print(f"\nf32 ULP of 2^f - 1: mean = {ulp_err[valid].mean():.2f}, max = {overall_max:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xs32[valid], ulp_err[valid], ".", color="#cccccc", ms=1)
ax.set_xlabel("f"); ax.set_ylabel("|error| in f32 ULPs")
ax.set_title(f"2^f - 1 = f Q(f), f32 ULP (max = {overall_max:.2f})")
ax.set_xlim(A, B); ax.set_ylim(bottom=0); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("work/out/cheb_expm1_ulp.png", dpi=130)
print("Saved work/out/cheb_expm1_ulp.png")


def _f(c):
    return f"X({float(np.float32(c)):.9e}f)"


print("\n// === C++ ===")
print(f"// 2^f - 1 = f Q(f), f in [-1/2, 1/2].  Max f32 ULP ~{overall_max:.2f}.")
print(f"auto q = {_f(coefs[-1])};")
for c in reversed(coefs[:-1]):
    print(f"q = mul_add(q, f, {_f(c)});")
