"""cos(2*pi*x) Chebyshev approximation on [-0.5, 0.5] (one full period).

cos(2*pi*x) has zeros at x = +/- 1/4 -- exactly representable in f32, so no
major/minor split needed. Factor:
    cos(2*pi*x) = (x^2 - 1/16) * Q(x^2),
where f(x) = cos(2*pi*x) / (x^2 - 1/16) is smooth and EVEN:
    f(0)        = -16            (cos(0)=1, / (-1/16))
    f(+/-1/4)   = -4*pi
Fit f via Chebyshev in t = 2x in [-1, 1], convert to monomials in x^2,
recover cos(2*pi*x) by the factored form.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

PI = np.pi
TWO_PI = 2 * PI
HALF = 0.5
QUARTER = 0.25
SIXTEENTH = 1.0 / 16.0  # (1/4)^2


def f(x):
    """f(x) = cos(2*pi*x) / (x^2 - 1/16).  Smooth, even on [-1/2, 1/2]."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    eps = 1e-3
    nearq = np.abs(np.abs(x) - QUARTER) < eps
    mid = ~nearq

    xm = x[mid]
    out[mid] = np.cos(TWO_PI * xm) / (xm * xm - SIXTEENTH)

    # Near x = s/4 (s = +/-1): u = x - s/4.
    # cos(2*pi*(s/4 + u)) = cos(s*pi/2 + 2*pi*u) = -s * sin(2*pi*u)
    # x^2 - 1/16 = u*(s/2 + u)
    # f = -s * sin(2*pi*u) / (u * (s/2 + u))
    #   = -s * 2*pi * sinc(2*pi*u) / (s/2 + u)
    xp = x[nearq]
    s = np.sign(xp)
    u = xp - s * QUARTER
    sinc_term = TWO_PI * (1 - (TWO_PI * u)**2 / 6 + (TWO_PI * u)**4 / 120)
    out[nearq] = -s * sinc_term / (s * HALF + u)

    return out


DEGREE = 12
cheb = Chebyshev.interpolate(lambda t: f(HALF * t), DEGREE, domain=[-1, 1])
print("Chebyshev coefficients of f((1/2)*t):")
for n, c in enumerate(cheb.coef):
    tag = "  (odd, ~0)" if n % 2 else ""
    print(f"  a_{n:2d} = {c: .6e}{tag}")

poly_t = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
coef_x = np.array([c * (2.0**k) for k, c in enumerate(poly_t.coef)])
Q_coef = coef_x[::2]

print(f"\nQ(x^2) such that f(x) ~ Q(x^2):")
for k, c in enumerate(Q_coef):
    print(f"  x^{2*k:2d}: {c: .15e}")


def approx_cos2pi(x):
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    val = Q_coef[-1]
    for c in reversed(Q_coef[:-1]):
        val = val * x2 + c
    return (x2 - SIXTEENTH) * val


xs = np.linspace(-HALF, HALF, 200_001)
approx = approx_cos2pi(xs)
true = np.cos(TWO_PI * xs)
abs_err = np.abs(approx - true)
mask = np.abs(true) > 1e-6
rel_err = np.abs((approx[mask] - true[mask]) / true[mask])
print(f"\nf64 max abs err: {abs_err.max():.3e}, max rel err: {rel_err.max():.3e}")


def approx_cos2pi_f32(x):
    x = x.astype(np.float32)
    Q32 = Q_coef.astype(np.float32)
    q = np.float32(QUARTER)
    x2 = x * x
    val = Q32[-1]
    for c in reversed(Q32[:-1]):
        val = val * x2 + c
    # Factor (x^2 - 1/16) = (x - 1/4)(x + 1/4); 1/4 is exact in f32 and the
    # subtraction is Sterbenz-exact near x = +/- 1/4, avoiding the cancellation
    # that (x^2 - 1/16) would suffer (x^2 has already rounded).
    lo = x - q
    hi = x + q
    return (lo * hi) * val


xs32 = np.linspace(-HALF, HALF, 200_001, dtype=np.float32)
true64 = np.cos(TWO_PI * xs32.astype(np.float64))
approx32 = approx_cos2pi_f32(xs32).astype(np.float64)

ulp = np.abs(np.spacing(true64.astype(np.float32))).astype(np.float64)
ulp_err = np.abs(approx32 - true64) / np.where(ulp > 0, ulp, np.nan)
valid = np.isfinite(ulp_err) & (np.abs(true64) > 1e-6)

n_bins = 200
bin_edges = np.linspace(-HALF, HALF, n_bins + 1)
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
ax.set_xlabel("x"); ax.set_ylabel("|error| in f32 ULPs of cos(2*pi*x)")
ax.set_title(f"cos(2*pi*x) f32 ULP on [-1/2, 1/2] (max = {overall_max:.2f})")
ax.set_xlim(-HALF, HALF); ax.set_ylim(0, np.nanpercentile(ulp_err[valid], 99.5) * 1.5)
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("work/out/cheb_cos2pi_ulp.png", dpi=130)
print("Saved work/out/cheb_cos2pi_ulp.png")


def _f(c):
    return f"{float(np.float32(c)):.9e}f"


print("\n// === C++ ===")
print(f"// cos(2*pi*x) on [-1/2, 1/2].  Max f32 ULP ~{overall_max:.2f}.")
print("inline float approx_cos2pi(float x) {")
print("    const float x2 = x * x;")
print(f"    float q = {_f(Q_coef[-1])};")
for c in reversed(Q_coef[:-1]):
    print(f"    q = q * x2 + {_f(c)};")
print("    const float lo = x - 0.25f;  // Sterbenz-exact near +/- 1/4")
print("    const float hi = x + 0.25f;")
print("    return (lo * hi) * q;")
print("}")


# ---------------------------------------------------------------------------
# [0, 1] domain (native) — reuses Q_coef; Horner variable v = (x-0.5)^2
# ---------------------------------------------------------------------------
# Substituting u = x-0.5 maps [0,1] -> [-0.5,0.5]. Then:
#   cos(2*pi*x) = cos(2*pi*(u+0.5)) = -cos(2*pi*u)
#              ~= -(u^2 - 1/16) * Q(u^2)  =  -(v - 1/16) * Q(v),  v=(x-0.5)^2
# Factor (v - 1/16) = (x-0.5)^2 - 0.25^2 = (x-0.75)(x-0.25).
# Both 0.25 and 0.75 are exact in f32, so each difference is Sterbenz-exact
# near the corresponding zero.  No fold needed.


def approx_cos2pi_01(x):
    x = np.asarray(x, dtype=np.float64)
    u = x - 0.5
    v = u * u
    val = Q_coef[-1]
    for c in reversed(Q_coef[:-1]):
        val = val * v + c
    return -(x - 0.75) * (x - 0.25) * val


xs01 = np.linspace(0.0, 1.0, 200_001)
approx01 = approx_cos2pi_01(xs01)
true01 = np.cos(TWO_PI * xs01)
abs_err01 = np.abs(approx01 - true01)
mask01 = np.abs(true01) > 1e-6
rel_err01 = np.abs((approx01[mask01] - true01[mask01]) / true01[mask01])
print(f"\n[0,1] f64 max abs err: {abs_err01.max():.3e}, max rel err: {rel_err01.max():.3e}")


def approx_cos2pi_01_f32(x):
    x = x.astype(np.float32)
    Q32 = Q_coef.astype(np.float32)
    u = x - np.float32(0.5)
    v = u * u
    val = Q32[-1]
    for c in reversed(Q32[:-1]):
        val = val * v + c
    lo = x - np.float32(0.25)   # Sterbenz-exact near x=0.25
    hi = x - np.float32(0.75)   # Sterbenz-exact near x=0.75
    return -(lo * hi) * val


xs01_32 = np.linspace(0.0, 1.0, 200_001, dtype=np.float32)
true01_64 = np.cos(TWO_PI * xs01_32.astype(np.float64))
approx01_32 = approx_cos2pi_01_f32(xs01_32).astype(np.float64)

ulp01 = np.abs(np.spacing(true01_64.astype(np.float32))).astype(np.float64)
ulp_err01 = np.abs(approx01_32 - true01_64) / np.where(ulp01 > 0, ulp01, np.nan)
valid01 = np.isfinite(ulp_err01) & (np.abs(true01_64) > 1e-6)

overall_mean01 = ulp_err01[valid01].mean()
overall_max01 = ulp_err01[valid01].max()
print(f"[0,1] f32 ULP: mean = {overall_mean01:.2f}, max = {overall_max01:.2f}")

print("\n// === C++ [0,1] domain ===")
print(f"// cos(2*pi*x) on [0, 1].  Max f32 ULP ~{overall_max01:.2f}.")
print("inline float approx_cos2pi_01(float x) {")
print("    const float u = x - 0.5f;")
print("    const float v = u * u;  // (x-0.5)^2")
print(f"    float q = {_f(Q_coef[-1])};")
for c in reversed(Q_coef[:-1]):
    print(f"    q = q * v + {_f(c)};")
print("    const float lo = x - 0.25f;  // Sterbenz-exact near x=0.25")
print("    const float hi = x - 0.75f;  // Sterbenz-exact near x=0.75")
print("    return -(lo * hi) * q;")
print("}")


# ---------------------------------------------------------------------------
# Oscillator SINAD
# ---------------------------------------------------------------------------
_FFT_N = 131072
_K = 997
_phase = (np.arange(_FFT_N, dtype=np.float64) * _K / _FFT_N) % 1.0
_phase_f32 = _phase.astype(np.float32)

_y_cheb = approx_cos2pi_01_f32(_phase_f32).astype(np.float64)
_y_libm = np.cos(TWO_PI * _phase).astype(np.float32).astype(np.float64)


def _sinad(y):
    Y = np.fft.rfft(y)
    pwr = np.abs(Y) ** 2
    noise = pwr.copy(); noise[_K] = 0.0
    return 10.0 * np.log10(pwr[_K] / max(noise.sum(), 1e-300)), Y, pwr


_snr_cheb, _Y_cheb, _pwr_cheb = _sinad(_y_cheb)
_snr_libm, _, _pwr_libm = _sinad(_y_libm)
print(f"\nOscillator SINAD ({_FFT_N}-pt FFT, {_K} cycles):")
print(f"  Chebyshev f32 : {_snr_cheb:.1f} dB")
print(f"  libm f32      : {_snr_libm:.1f} dB")

_db_cheb = 10.0 * np.log10(_pwr_cheb / _pwr_cheb[_K] + 1e-300)
_db_libm = 10.0 * np.log10(_pwr_libm / _pwr_libm[_K] + 1e-300)
_freq = np.arange(len(_Y_cheb)) / _FFT_N

fig_snr, ax_snr = plt.subplots(figsize=(12, 5))
ax_snr.plot(_freq[1:], _db_cheb[1:], lw=0.6, color="#1f77b4",
            label=f"Chebyshev f32 (SINAD {_snr_cheb:.1f} dB)")
ax_snr.plot(_freq[1:], _db_libm[1:], lw=0.6, color="#d62728", alpha=0.7,
            label=f"libm f32 (SINAD {_snr_libm:.1f} dB)")
ax_snr.set_xlabel("Normalized frequency")
ax_snr.set_ylabel("dB (relative to fundamental)")
ax_snr.set_title(f"cos(2πx) oscillator SINAD — {_FFT_N}-pt FFT, {_K} cycles")
ax_snr.set_ylim(-180, 5)
ax_snr.set_xlim(0, 0.5)
ax_snr.legend()
ax_snr.grid(alpha=0.3)
fig_snr.tight_layout()
fig_snr.savefig("work/out/cheb_cos2pi_snr.png", dpi=130)
print("Saved work/out/cheb_cos2pi_snr.png")


