"""Chebyshev sine approximation, reproducing
https://mooooo.ooo/chebyshev-sine-approximation/

Recipe:
  1. sin(x) has known zeros at x = 0, ±pi, so write
         sin(x) = x * (x^2 - pi^2) * f(x)
     where f(x) = sin(x) / (x*(x^2 - pi^2)) is smooth and even on [-pi, pi].
  2. Expand f as a truncated Chebyshev series using t = x/pi in [-1, 1].
  3. Convert that series to a polynomial P(x^2), then recover
         sin(x) ~ x * (x^2 - pi^2) * P(x^2).

The article scales sin by s(x) = (8/(3*pi))*x - (8/(3*pi^3))*x^3 instead.
That's just -(8/(3*pi^3))*x*(x^2 - pi^2), so it differs from the factoring
above only by a constant absorbed into the polynomial: same final answer.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

PI = np.pi


def f(x):
    """f(x) = sin(x) / (x * (x^2 - pi^2)). Smooth and even on [-pi, pi].

    Uses Taylor expansions near x = 0 and x = ±pi to dodge 0/0.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    eps = 1e-3

    near0 = np.abs(x) < eps
    nearpi = np.abs(np.abs(x) - PI) < eps
    mid = ~(near0 | nearpi)

    xm = x[mid]
    out[mid] = np.sin(xm) / (xm * (xm * xm - PI * PI))

    # Near 0: sin(x)/x = 1 - x^2/6 + x^4/120 - ...
    x0 = x[near0]
    sinc0 = 1 - x0**2 / 6 + x0**4 / 120 - x0**6 / 5040
    out[near0] = sinc0 / (x0 * x0 - PI * PI)

    # Near x = s*pi (s = +/-1): let u = x - s*pi. Then sin(x) = -sin(u),
    # x^2 - pi^2 = u*(2*s*pi + u), x = s*pi + u.
    xp = x[nearpi]
    s = np.sign(xp)
    u = xp - s * PI
    sinc_u = 1 - u**2 / 6 + u**4 / 120 - u**6 / 5040
    out[nearpi] = -sinc_u / ((s * PI + u) * (2 * s * PI + u))

    return out


# Fit f on [-pi, pi] via Chebyshev interpolation in t = x/pi.
# f is even, so odd-index coefficients come out ~0. Six nonzero even
# coefficients (a_0, a_2, ..., a_10) yield a degree-11 sin polynomial,
# matching the article.
DEGREE = 10
cheb = Chebyshev.interpolate(lambda t: f(PI * t), DEGREE, domain=[-1, 1])

print("Chebyshev coefficients a_n of f(pi*t) = sum a_n T_n(t):")
for n, c in enumerate(cheb.coef):
    tag = "  (odd, ~0)" if n % 2 else ""
    print(f"  a_{n:2d} = {c: .6e}{tag}")

# Convert T_n(t) series to power series in t, then substitute t = x/pi.
poly_t = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
coef_x = np.array([c / PI**k for k, c in enumerate(poly_t.coef)])

# Keep even powers only: P(x^2) = sum P_coef[k] * x^(2k).
P_coef = coef_x[::2]

print("\nP(x^2) so that f(x) ~ P(x^2):")
for k, c in enumerate(P_coef):
    print(f"  x^{2*k:2d}: {c: .15e}")

# Expand sin(x) ~ x * (x^2 - pi^2) * P(x^2) into monomials for display.
# Coefficient of x^(2k+1) is P_coef[k-1] - pi^2 * P_coef[k] (with P_coef[-1]:=0).
n = len(P_coef)
sin_coefs = np.zeros(n + 1)
for k in range(n + 1):
    lo = P_coef[k - 1] if k >= 1 else 0.0
    hi = -PI * PI * P_coef[k] if k < n else 0.0
    sin_coefs[k] = lo + hi

print("\nsin(x) ~ sum c_k * x^(2k+1):")
for k, c in enumerate(sin_coefs):
    print(f"  x^{2*k+1:2d}: {c: .15e}")


def approx_sin(x):
    """Evaluate sin(x) ~ x * (x^2 - pi^2) * P(x^2) via Horner in x^2."""
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    val = P_coef[-1]
    for c in reversed(P_coef[:-1]):
        val = val * x2 + c
    return x * (x2 - PI * PI) * val


xs = np.linspace(-PI, PI, 200_001)
approx = approx_sin(xs)
true = np.sin(xs)
abs_err = np.abs(approx - true)

mask = np.abs(true) > 1e-6
rel_err = np.abs((approx[mask] - true[mask]) / true[mask])

print(f"\nMax absolute error on [-pi, pi]:            {abs_err.max():.3e}")
print(f"Max relative error (where |sin| > 1e-6):    {rel_err.max():.3e}")
print(f"Argmax abs error at x = {xs[np.argmax(abs_err)]: .6f}")


# ULP error plot: evaluate the approximation in f32 (matching the article's
# target precision) and compare against f64 sin as truth.
def approx_sin_f32(x):
    x = x.astype(np.float32)
    P32 = P_coef.astype(np.float32)
    pi32 = np.float32(PI)
    x2 = x * x
    val = P32[-1]
    for c in reversed(P32[:-1]):
        val = val * x2 + c
    return x * (x2 - pi32 * pi32) * val


xs32 = np.linspace(-PI, PI, 200_001, dtype=np.float32)
true64 = np.sin(xs32.astype(np.float64))
approx32 = approx_sin_f32(xs32).astype(np.float64)

# 1 ULP near y in f32 is the (unsigned) spacing between adjacent f32 values.
# np.spacing returns a signed value, so take |.|. Skip samples where the true
# value is denormal/zero -- ULP is meaningless there.
ulp = np.abs(np.spacing(true64.astype(np.float32))).astype(np.float64)
ulp_err = np.abs(approx32 - true64) / np.where(ulp > 0, ulp, np.nan)

valid = np.isfinite(ulp_err) & (np.abs(true64) > 1e-6)

# Binned mean: average |ULP error| in equal-width bins across [-pi, pi].
n_bins = 200
bin_edges = np.linspace(-PI, PI, n_bins + 1)
bin_idx = np.clip(np.searchsorted(bin_edges, xs32, side="right") - 1, 0, n_bins - 1)
weights = np.where(valid, ulp_err, 0.0)
counts = np.bincount(bin_idx, weights=valid.astype(np.float64), minlength=n_bins)
sums = np.bincount(bin_idx, weights=weights, minlength=n_bins)
bin_mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

overall_mean = ulp_err[valid].mean()
overall_max = ulp_err[valid].max()
print(f"\nf32 ULP error (|sin(x)| > 1e-6): mean = {overall_mean:.2f}, max = {overall_max:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xs32[valid], ulp_err[valid], ".", color="#cccccc", ms=1, label="pointwise |ULP error|")
ax.plot(bin_centers, bin_mean, color="#1f77b4", lw=1.8, label="binned mean")
ax.axhline(overall_mean, color="#d62728", ls="--", lw=1, label=f"overall mean = {overall_mean:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("|error| in f32 ULPs of sin(x)")
ax.set_title(f"Chebyshev sin approximation, f32 ULP error (max = {overall_max:.0f})")
ax.set_xlim(-PI, PI)
# Clip y-axis to a readable range; spikes near zero crossings of sin blow up.
ax.set_ylim(0, np.nanpercentile(ulp_err[valid], 99.5) * 1.5)
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()

out_path = "work/out/cheb_sin_ulp.png"
fig.savefig(out_path, dpi=130)
print(f"Saved plot to {out_path}")
plt.show()


# ---------------------------------------------------------------------------
# Optional refinement: split pi into two f32 values to kill the cancellation
# in (x^2 - pi^2) near x = +/- pi.
#
# Let pi_major = f32(pi) (the nearest f32 to true pi) and
#     pi_minor = f32(true_pi - pi_major)    (the tiny leftover)
# so pi_major + pi_minor approximates true pi much better than pi_major alone.
#
# Then factor sin(x) ~ ((x - pi_major) - pi_minor) * ((x + pi_major) + pi_minor)
#                     * x * P(x^2),
# evaluated in that order. When x is near +pi the subtraction (x - pi_major)
# is (near-)exact by Sterbenz's lemma, and then subtracting the tiny pi_minor
# preserves precision that a direct (x^2 - pi_f32^2) would lose to cancellation.
# ---------------------------------------------------------------------------

PI_MAJOR = np.float32(np.pi)  # ~3.1415927
PI_MINOR = np.float32(np.pi - float(PI_MAJOR))  # ~-8.742278e-08
print(f"\npi_major (f32) = {float(PI_MAJOR):.10e}")
print(f"pi_minor (f32) = {float(PI_MINOR):.10e}")
print(f"pi_major + pi_minor - true pi = {float(PI_MAJOR) + float(PI_MINOR) - np.pi:.3e}")


def approx_sin_f32_split_pi(x):
    x = x.astype(np.float32)
    P32 = P_coef.astype(np.float32)
    x2 = x * x
    val = P32[-1]
    for c in reversed(P32[:-1]):
        val = val * x2 + c
    lo = (x - PI_MAJOR) - PI_MINOR
    hi = (x + PI_MAJOR) + PI_MINOR
    return lo * hi * x * val


approx32b = approx_sin_f32_split_pi(xs32).astype(np.float64)
ulp_err2 = np.abs(approx32b - true64) / np.where(ulp > 0, ulp, np.nan)
valid2 = np.isfinite(ulp_err2) & (np.abs(true64) > 1e-6)

sums2 = np.bincount(bin_idx, weights=np.where(valid2, ulp_err2, 0.0), minlength=n_bins)
counts2 = np.bincount(bin_idx, weights=valid2.astype(np.float64), minlength=n_bins)
bin_mean2 = np.where(counts2 > 0, sums2 / np.maximum(counts2, 1), np.nan)

overall_mean2 = ulp_err2[valid2].mean()
overall_max2 = ulp_err2[valid2].max()
print(f"\nf32 ULP error w/ split-pi: mean = {overall_mean2:.2f}, max = {overall_max2:.2f}")

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(bin_centers, bin_mean, color="#cccccc", lw=1.5, label=f"naive f32 (mean {overall_mean:.2f})")
ax2.plot(bin_centers, bin_mean2, color="#1f77b4", lw=1.8, label=f"split-pi f32 (mean {overall_mean2:.2f})")
ax2.axhline(overall_mean2, color="#d62728", ls="--", lw=1)
ax2.set_xlabel("x")
ax2.set_ylabel("binned mean |ULP error|")
ax2.set_title(f"Split-pi refinement (max {overall_max:.0f} -> {overall_max2:.0f} ULPs)")
ax2.set_xlim(-PI, PI)
ax2.set_ylim(0, max(np.nanmax(bin_mean), np.nanmax(bin_mean2)) * 1.1)
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)
fig2.tight_layout()

out_path2 = "work/out/cheb_sin_ulp_split_pi.png"
fig2.savefig(out_path2, dpi=130)
print(f"Saved plot to {out_path2}")


def _f(c):
    return f"{float(np.float32(c)):.9e}f"


print("\n// === C++ ===")
print(f"// sin(x) on [-pi, pi] (split-pi).  Max f32 ULP ~{overall_max2:.2f}.")
print("inline float approx_sin(float x) {")
print(f"    constexpr float PI_MAJOR = {_f(PI_MAJOR)};")
print(f"    constexpr float PI_MINOR = {_f(PI_MINOR)};")
print("    const float x2 = x * x;")
print(f"    float p = {_f(P_coef[-1])};")
for c in reversed(P_coef[:-1]):
    print(f"    p = p * x2 + {_f(c)};")
print("    const float lo = (x - PI_MAJOR) - PI_MINOR;  // ~ x - pi (Sterbenz)")
print("    const float hi = (x + PI_MAJOR) + PI_MINOR;  // ~ x + pi")
print("    return lo * hi * x * p;")
print("}")


# ---------------------------------------------------------------------------
# Oscillator SINAD
# ---------------------------------------------------------------------------
_FFT_N = 131072
_K = 997
_phase_f64 = 2 * PI * ((np.arange(_FFT_N, dtype=np.float64) * _K / _FFT_N) % 1.0 - 0.5)
_phase_f32 = _phase_f64.astype(np.float32)

_y_cheb = approx_sin_f32_split_pi(_phase_f32).astype(np.float64)
_y_libm = np.sin(_phase_f64).astype(np.float32).astype(np.float64)


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
ax_snr.set_title(f"sin(x) oscillator SINAD — {_FFT_N}-pt FFT, {_K} cycles")
ax_snr.set_ylim(-180, 5)
ax_snr.set_xlim(0, 0.5)
ax_snr.legend()
ax_snr.grid(alpha=0.3)
fig_snr.tight_layout()
fig_snr.savefig("work/out/cheb_sin_snr.png", dpi=130)
print("Saved work/out/cheb_sin_snr.png")

plt.show()
