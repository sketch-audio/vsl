"""acos(x) Chebyshev approximation on [-1, 1].

The identity  acos(x) = pi/2 - asin(x)  suffers catastrophic cancellation
near x = 1 (acos(1) = 0, but pi/2 and asin(1) = pi/2 nearly cancel in f32,
giving ~81 ULP error there).

Fix: use the same half-angle reduction as asin, but directed toward acos:
  |x| <= 0.5 (core):  acos(x) = pi/2 - asin(x)  = pi/2 - x * P(x^2)
                       [no cancellation: acos ~= pi/2, x*P ~ 0]
  |x| >  0.5 (tail):  inner = sqrt((1 - |x|) / 2)  in [0, 0.5]
                       acos(|x|) = 2 * asin(inner)  = 2 * inner * P(inner^2)
                       [direct evaluation, no cancellation near x = 1]
  Reflect:             acos(-x) = pi - acos(x)

Both pi/2 and pi get major/minor f32 splits to avoid adding small corrections
to large f32 constants.

P(x^2) is identical to the one in cheb_asin.py:
  asin(x) ~ x * P(x^2)  for |x| <= 1/2.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

PI = np.pi
PI_HALF = PI / 2
HALF = 0.5


def g(x):
    """g(x) = asin(x) / x.  Smooth and even on [-1/2, 1/2]."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    out[~small] = np.arcsin(x[~small]) / x[~small]
    xs = x[small]
    # Taylor: asin(x)/x = 1 + x^2/6 + 3x^4/40 + 15x^6/336 + ...
    out[small] = 1 + xs**2 / 6 + 3 * xs**4 / 40 + 15 * xs**6 / 336
    return out


DEGREE = 8
cheb = Chebyshev.interpolate(lambda t: g(HALF * t), DEGREE, domain=[-1, 1])
print("Chebyshev coefficients of asin((1/2)*t)/((1/2)*t):")
for n, c in enumerate(cheb.coef):
    tag = "  (odd, ~0)" if n % 2 else ""
    print(f"  a_{n:2d} = {c: .6e}{tag}")

poly_t = cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1])
# t = 2x, so t^k = (2x)^k => multiply by 2^k
coef_x = np.array([c * (2.0**k) for k, c in enumerate(poly_t.coef)])
P_coef = coef_x[::2]  # even powers of x only

print(f"\nP(x^2) such that asin(x) ~ x * P(x^2) for |x| <= 1/2:")
for k, c in enumerate(P_coef):
    print(f"  x^{2*k:2d}: {c: .15e}")


def approx_asin_core(x):
    """Polynomial approximation asin(x) ~ x * P(x^2), valid for |x| <= 1/2."""
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    val = P_coef[-1]
    for c in reversed(P_coef[:-1]):
        val = val * x2 + c
    return x * val


def approx_acos(x):
    """acos(x) on [-1, 1] via half-angle reduction."""
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    inner = np.sqrt(np.clip((1.0 - ax) / 2.0, 0.0, None))
    # big branch:   acos(|x|) = 2 * asin(inner),  inner in [0, 0.5]
    # small branch: acos(|x|) = pi/2 - asin(|x|), |x| in [0, 0.5]
    # Both use ax so the final reflection applies uniformly.
    asin_tail = approx_asin_core(inner)
    asin_mid = approx_asin_core(ax)
    pos_result = np.where(ax > HALF, 2.0 * asin_tail, PI_HALF - asin_mid)
    # reflect: acos(-|x|) = pi - acos(|x|)
    return np.where(x < 0.0, PI - pos_result, pos_result)


xs = np.linspace(-1.0, 1.0, 200_001)
approx = approx_acos(xs)
true = np.arccos(xs)
abs_err = np.abs(approx - true)
mask = np.abs(true) > 1e-6
rel_err = np.abs((approx[mask] - true[mask]) / true[mask])
print(f"\nf64 on [-1, 1]: max abs err = {abs_err.max():.3e}, max rel err = {rel_err.max():.3e}")


# ---------------------------------------------------------------------------
# f32 evaluation.
# ---------------------------------------------------------------------------

PI_HALF_MAJOR = np.float32(PI_HALF)
PI_HALF_MINOR = np.float32(PI_HALF - float(PI_HALF_MAJOR))
PI_MAJOR = np.float32(PI)
PI_MINOR = np.float32(PI - float(PI_MAJOR))

print(f"\npi/2 major (f32) = {float(PI_HALF_MAJOR):.10e}")
print(f"pi/2 minor (f32) = {float(PI_HALF_MINOR):.10e}")
print(f"pi   major (f32) = {float(PI_MAJOR):.10e}")
print(f"pi   minor (f32) = {float(PI_MINOR):.10e}")


def approx_asin_core_f32(x):
    x = x.astype(np.float32)
    P32 = P_coef.astype(np.float32)
    x2 = x * x
    val = P32[-1]
    for c in reversed(P32[:-1]):
        val = val * x2 + c
    return x * val


def approx_acos_f32(x):
    x = x.astype(np.float32)
    ax = np.abs(x)
    inner = np.sqrt(np.clip((np.float32(1.0) - ax) / np.float32(2.0), np.float32(0.0), None))
    big = ax > np.float32(HALF)
    # Both branches take ax (not signed x) so reflection applies uniformly.
    r = np.where(big, inner, ax)
    asin_val = approx_asin_core_f32(r)

    # big:   2 * asin(inner)
    tail_pos = np.float32(2.0) * asin_val
    # small: pi/2 - asin(|x|), with split
    small_val = (PI_HALF_MAJOR - asin_val) + PI_HALF_MINOR

    pos_result = np.where(big, tail_pos, small_val)
    # reflect: acos(-|x|) = pi - acos(|x|), with split
    neg_result = (PI_MAJOR - pos_result) + PI_MINOR
    return np.where(x < np.float32(0.0), neg_result, pos_result)


xs32 = np.linspace(-1.0, 1.0, 200_001, dtype=np.float32)
true64 = np.arccos(xs32.astype(np.float64))
approx32 = approx_acos_f32(xs32).astype(np.float64)

ulp = np.abs(np.spacing(true64.astype(np.float32))).astype(np.float64)
ulp_err = np.abs(approx32 - true64) / np.where(ulp > 0, ulp, np.nan)
valid = np.isfinite(ulp_err) & (np.abs(true64) > 1e-6)

n_bins = 200
bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
bin_idx = np.clip(np.searchsorted(bin_edges, xs32, side="right") - 1, 0, n_bins - 1)
sums = np.bincount(bin_idx, weights=np.where(valid, ulp_err, 0.0), minlength=n_bins)
counts = np.bincount(bin_idx, weights=valid.astype(np.float64), minlength=n_bins)
bin_mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

overall_mean = ulp_err[valid].mean()
overall_max = ulp_err[valid].max()
print(f"f32 ULP on [-1, 1]: mean = {overall_mean:.2f}, max = {overall_max:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xs32[valid], ulp_err[valid], ".", color="#cccccc", ms=1, label="pointwise")
ax.plot(bin_centers, bin_mean, color="#1f77b4", lw=1.8, label="binned mean")
ax.axhline(overall_mean, color="#d62728", ls="--", lw=1, label=f"mean = {overall_mean:.2f}")
ax.axvline(-0.5, color="#999", lw=0.5)
ax.axvline(0.5, color="#999", lw=0.5)
ax.set_xlabel("x")
ax.set_ylabel("|error| in f32 ULPs of acos(x)")
ax.set_title(f"acos f32 ULP on [-1, 1] (max = {overall_max:.2f})")
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(0, np.nanpercentile(ulp_err[valid], 99.5) * 1.5)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("work/out/cheb_acos_ulp.png", dpi=130)
print("Saved work/out/cheb_acos_ulp.png")


# ---------------------------------------------------------------------------
# C++ emission.
# ---------------------------------------------------------------------------

def _f(c):
    return f"{float(np.float32(c)):.9e}f"


print("\n// === C++ ===")
print(f"// acos(x) on [-1, 1].  Max f32 ULP ~{overall_max:.2f}.")
print("inline float approx_acos(float x) {")
print(f"    constexpr float PI_HALF_MAJOR = {_f(PI_HALF_MAJOR)};")
print(f"    constexpr float PI_HALF_MINOR = {_f(PI_HALF_MINOR)};")
print(f"    constexpr float PI_MAJOR      = {_f(PI_MAJOR)};")
print(f"    constexpr float PI_MINOR      = {_f(PI_MINOR)};")
print("    const float ax = x < 0.0f ? -x : x;")
print("    const bool big = ax > 0.5f;")
print("    const float inner = std::sqrt((1.0f - ax) * 0.5f);")
print("    const float r = big ? inner : ax;  // always non-negative")
print("    const float r2 = r * r;")
print(f"    float p = {_f(P_coef[-1])};")
for c in reversed(P_coef[:-1]):
    print(f"    p = p * r2 + {_f(c)};")
print("    const float asin_val = r * p;")
print("    // big: acos(|x|) = 2 * asin(inner)  [no cancellation near x=1]")
print("    // small: acos(x) = pi/2 - asin(x)   [no cancellation: acos ~ pi/2]")
print("    const float pos = big ? 2.0f * asin_val")
print("                          : (PI_HALF_MAJOR - asin_val) + PI_HALF_MINOR;")
print("    // reflect: acos(-x) = pi - acos(x)")
print("    return x < 0.0f ? (PI_MAJOR - pos) + PI_MINOR : pos;")
print("}")


# ---------------------------------------------------------------------------
# Oscillator SINAD — phase signal via approx_acos used as an inverse.
# Instead of generating a waveform directly, measure the round-trip
# cos -> approx_acos quality: feed libm cos(theta) into our acos and compare
# the reconstructed theta against the original.  That tests ULP in context.
#
# Alternatively: use approx_acos to build a piecewise waveform from a
# sawtooth phase.  For a direct comparison with the other scripts, generate
# a cosine oscillator signal via: y = approx_acos(2*phase - 1) is NOT a
# sinusoid.  Better: generate y = cos(2*pi*phase) where phase sweeps [0,1],
# using approx_acos (via the identity cos(theta) is the input) is circular.
#
# Practical SINAD test: use approx_acos as a lookup table.  Feed uniformly
# spaced phase values in [0,1], map to angle via phase * pi, then evaluate
# approx_acos(cos(phase * pi)) — should recover phase * pi.  Compare to
# the same chain through libm.  Spectral purity of the recovered angle
# sequence gives the SINAD.
# ---------------------------------------------------------------------------

_FFT_N = 131072
_K = 997

# Phase theta in [0, pi]: theta_k = pi * k / FFT_N for k in [0, FFT_N-1]
# y_k = approx_acos(cos(theta_k)) should equal theta_k.
# The error is the reconstruction noise.  But this is just testing acos(cos(x))=x,
# not a waveform oscillator.
#
# More useful: synthesise a cosine wave via cos(2*pi*k*_K/_FFT_N) using
# approx_acos in the inverse direction isn't natural.  Instead, since the
# other scripts test the function directly as an oscillator generator, we
# show the round-trip residual spectrum: feed x = cos(theta) through
# approx_acos, subtract ground-truth theta, and take the FFT of that error
# sequence.  This reveals harmonic distortion introduced by acos error.

_theta = np.linspace(0.0, PI, _FFT_N, endpoint=False, dtype=np.float64)
_cos_theta = np.cos(_theta).astype(np.float32)

_recovered_f32 = approx_acos_f32(_cos_theta).astype(np.float64)
_recovered_libm = np.arccos(_cos_theta.astype(np.float64))

_err_cheb = _recovered_f32 - _theta
_err_libm = _recovered_libm - _theta


def _sinad_err(err):
    """SINAD of a residual signal (signal power from DC-removed version)."""
    Y = np.fft.rfft(err)
    pwr = np.abs(Y) ** 2
    # Total power of residual; DC is index 0
    noise = pwr.copy()
    noise[0] = 0.0
    signal_pwr = pwr[0]  # DC is the mean offset (bias)
    total_ac = noise.sum()
    if total_ac < 1e-300:
        return np.inf, Y, pwr
    return 10.0 * np.log10(signal_pwr / total_ac) if signal_pwr > 0 else -np.inf, Y, pwr


# For oscillator SINAD, synthesize cos via identity: use approx_acos as
# the forward mapping from a linear phase ramp (same as other scripts).
# acos maps [-1,1] -> [0, pi].  A cosine oscillator samples from [-1,1].
# Reconstruct the angle and treat it as the waveform to measure spectral purity.
# This is effectively testing acos purity on a full-cycle sweep.

# Generate a periodic angle ramp at frequency _K in _FFT_N samples:
# phase_angle[n] = pi * (n * _K / _FFT_N) mod pi
_phase_angle = (np.arange(_FFT_N, dtype=np.float64) * _K / _FFT_N % 1.0) * PI
_cos_in = np.cos(_phase_angle).astype(np.float32)

_acos_cheb = approx_acos_f32(_cos_in).astype(np.float64)
_acos_libm = np.arccos(_cos_in.astype(np.float64))


def _sinad_waveform(y):
    Y = np.fft.rfft(y)
    pwr = np.abs(Y) ** 2
    # Find dominant bin (should be _K for a pure tone)
    peak = np.argmax(pwr[1:]) + 1
    noise = pwr.copy()
    noise[peak] = 0.0
    return 10.0 * np.log10(pwr[peak] / max(noise.sum(), 1e-300)), Y, pwr, peak


_snr_cheb, _Y_cheb, _pwr_cheb, _peak_cheb = _sinad_waveform(_acos_cheb)
_snr_libm, _Y_libm, _pwr_libm, _peak_libm = _sinad_waveform(_acos_libm)

print(f"\nOscillator SINAD via acos(cos(phase)) ({_FFT_N}-pt FFT, {_K} cycles):")
print(f"  Chebyshev f32 : {_snr_cheb:.1f} dB  (peak bin {_peak_cheb})")
print(f"  libm f32      : {_snr_libm:.1f} dB  (peak bin {_peak_libm})")

_db_cheb = 10.0 * np.log10(_pwr_cheb / _pwr_cheb[_peak_cheb] + 1e-300)
_db_libm = 10.0 * np.log10(_pwr_libm / _pwr_libm[_peak_libm] + 1e-300)
_freq = np.arange(len(_Y_cheb)) / _FFT_N

fig_snr, ax_snr = plt.subplots(figsize=(12, 5))
ax_snr.plot(_freq[1:], _db_cheb[1:], lw=0.6, color="#1f77b4",
            label=f"Chebyshev f32 (SINAD {_snr_cheb:.1f} dB)")
ax_snr.plot(_freq[1:], _db_libm[1:], lw=0.6, color="#d62728", alpha=0.7,
            label=f"libm f64 (SINAD {_snr_libm:.1f} dB)")
ax_snr.set_xlabel("Normalized frequency")
ax_snr.set_ylabel("dB (relative to fundamental)")
ax_snr.set_title(f"acos(cos(phase)) SINAD — {_FFT_N}-pt FFT, {_K} cycles")
ax_snr.set_ylim(-180, 5)
ax_snr.set_xlim(0, 0.5)
ax_snr.legend()
ax_snr.grid(alpha=0.3)
fig_snr.tight_layout()
fig_snr.savefig("work/out/cheb_acos_snr.png", dpi=130)
print("Saved work/out/cheb_acos_snr.png")

plt.show()
