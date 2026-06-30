#pragma once

#include <algorithm> // min, max, clamp
#include <cassert>
#include <cmath>
#include <numbers>

#include "vsl_core.hpp"
#include "vsl_logic.hpp"

namespace vsl {

///
template<Floating_point X>
force_inline constexpr auto abs(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::abs(x);
    }
    else {
        return std::abs(x);
    }
}

///
template<Floating_point X>
force_inline constexpr auto trunc(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::trunc(x);
    }
    else {
        return std::trunc(x);
    }
}

///
template<Floating_point X>
force_inline constexpr auto floor(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::floor(x);
    }
    else {
        return std::floor(x);
    }
}

///
template<Floating_point X>
force_inline constexpr auto ceil(X x) -> X 
{
    if constexpr (is_vector_v<X>) {
        return X::ceil(x);
    }
    else {
        return std::ceil(x);
    }
}

///
template<Floating_point X>
force_inline constexpr auto round(X x) -> X 
{
    if constexpr (is_vector_v<X>) {
        return X::round(x);
    }
    else {
        return std::round(x);
    }
}

// MARK: - sqrt

///
template<Floating_point X>
force_inline constexpr auto sqrt(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::sqrt(x);
    }
    else {
        return std::sqrt(x);
    }
}

///
template<Floating_point X>
force_inline constexpr auto rsqrt(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::rsqrt(x);
    }
    else {
        return 1 / std::sqrt(x);
    }
}

// MARK: - fmod, wrap

///
template<Floating_point X>
force_inline constexpr auto fmod(X x, X y) -> X
{
    if constexpr (is_vector_v<X>) {
        return x - trunc(x / y) * y;
    }
    else {
        return std::fmod(x, y);
    }
}

/// Normalized wrap to [0, 1]
template<Floating_point X>
force_inline constexpr auto wrap(X x) -> X
{
    return x - floor(x);
}

/// Symmetric wrap to [-k, k]
template<Floating_point X, Floating_point S>
force_inline constexpr auto wrap(X x, S k) -> X
{
    x += k;
    const auto two_k = 2 * k;
    const auto m = x - two_k * trunc(x / two_k);
    return select(x >= 0, m, m + two_k) - k;
}

/// Arbitrary wrap to [a, b]
template<Floating_point X, Floating_point S>
force_inline constexpr auto wrap(X x, S a, S b) -> X
{
    const auto rect = x - a;
    const auto range = b - a;
    return range * wrap(rect / range) + a;
}

// MARK: - Trig Functions

///
template<bool needs_wrap, Floating_point X>
force_inline constexpr auto cos(X x) -> X
{
    using S = scalar_t<X>;

    if constexpr (needs_wrap) {
        constexpr auto pi = std::numbers::pi_v<S>;
        x = wrap(x, pi);
    }

    // See work/cheb_cos.py
    constexpr auto pi_half_major = S(1.570796371e+00f);
    constexpr auto pi_half_minor = S(-4.371138829e-08f);
    const auto x2 = x * x;
    auto p = X(1.772737512e-09f);
    p = p * x2 + S(-2.677252269e-07f);
    p = p * x2 + S(2.412236245e-05f);
    p = p * x2 + S(-1.329319435e-03f);
    p = p * x2 + S(3.838663921e-02f);
    p = p * x2 + S(-4.052847326e-01f);
    const auto lo = (x - pi_half_major) - pi_half_minor;  // ~ x - pi/2 (Sterbenz)
    const auto hi = (x + pi_half_major) + pi_half_minor;  // ~ x + pi/2
    return lo * hi * p;
}

///
template<bool needs_wrap, Floating_point X>
force_inline constexpr auto sin(X x) -> X
{
    using S = scalar_t<X>;

    if constexpr (needs_wrap) {
        constexpr auto pi = std::numbers::pi_v<S>;
        x = wrap(x, pi);
    }

    // See work/cheb_sin.py
    constexpr auto pi_major = S(3.141592741e+00f);
    constexpr auto pi_minor = S(-8.742277657e-08f);
    const auto x2 = x * x;
    auto p = X(1.345148160e-10f);
    p = p * x2 + S(-2.335733917e-08f);
    p = p * x2 + S(2.522634304e-06f);
    p = p * x2 + S(-1.735062688e-04f);
    p = p * x2 + S(6.620881148e-03f);
    p = p * x2 + S(-1.013211831e-01f);
    const auto lo = (x - pi_major) - pi_minor;  // ~ x - pi (Sterbenz)
    const auto hi = (x + pi_major) + pi_minor;  // ~ x + pi
    return lo * hi * x * p;
}

///
template<bool needs_wrap, Floating_point X>
force_inline constexpr auto tan(X x) -> X
{
    using S = scalar_t<X>;

    if constexpr (needs_wrap) {
        constexpr auto half_pi = std::numbers::pi_v<S> / 2;
        x = wrap(x, half_pi);
    }

    // See work/cheb_tan.py
    constexpr auto pi_half_major = S(1.570796371e+00f);
    constexpr auto pi_half_minor = S(-4.371138829e-08f);
    const auto x2 = x * x;
    auto q = X(-9.648306332e-06f);
    q = q * x2 + S(-1.677224500e-04f);
    q = q * x2 + S(-4.349521827e-03f);
    q = q * x2 + S(-1.775324047e-01f);
    q = q * x2 + S(2.467401028e+00f);
    const auto lo = (pi_half_major - x) + pi_half_minor;  // ~ pi/2 - x (Sterbenz)
    const auto hi = (pi_half_major + x) + pi_half_minor;  // ~ pi/2 + x
    return x * q / (lo * hi);
}

///
template<bool needs_wrap, Floating_point X>
force_inline constexpr auto cos_2pi(X x) -> X
{
    using S = scalar_t<X>;

    if constexpr (needs_wrap) {
        x = wrap(x);
    }

    x = select(x < 0.5f, x, 1 - x); // Approximation on -0.5...0.5 makes us exact at 0.

    // See work/cheb_cos2pi.py
    const auto x2 = x * x;
    auto q = X(-1.483706594e+00f);
    q = q * x2 + S(7.732257843e+00f);
    q = q * x2 + S(-2.592922974e+01f);
    q = q * x2 + S(5.862275696e+01f);
    q = q * x2 + S(-8.179283142e+01f);
    q = q * x2 + S(5.982733917e+01f);
    q = q * x2 + S(-1.600000000e+01f);
    const auto lo = x - 0.25f;
    const auto hi = x + 0.25f;
    return (lo * hi) * q;
}

///
template<bool needs_wrap, Floating_point X>
force_inline constexpr auto sin_2pi(X x) -> X
{
    using S = scalar_t<X>;

    if constexpr (needs_wrap) {
        x = wrap(x);
    }

    // Approximation on 0...1, see work/cheb_sin2pi.py
    const auto u = x - 0.5f;
    const auto v = u * u;
    auto p = X(-6.151015759e-01f);
    p = p * v + S(3.622930765e+00f);
    p = p * v + S(-1.417941380e+01f);
    p = p * v + S(3.851269913e+01f);
    p = p * v + S(-6.707761383e+01f);
    p = p * v + S(6.483584595e+01f);
    p = p * v + S(-2.513274193e+01f);
    return -u * x * (x - 1.f) * p;
}

// MARK: - Inverse Trig Functions

///
template<Floating_point X>
force_inline constexpr auto acos(X x) -> X
{
    using S = scalar_t<X>;

    // See work/cheb_acos.py
    constexpr auto pi_half_major = S(1.570796371e+00f);
    constexpr auto pi_half_minor = S(-4.371138829e-08f);
    constexpr auto pi_major = S(3.141592741e+00f);
    constexpr auto pi_minor = S(-8.742277657e-08f);
    const auto ax = select(x < 0.f, -x, x);
    const auto big = ax > 0.5f;
    const auto inner = sqrt((1.f - ax) * 0.5f);
    const auto r = select(big, inner, ax);  // always non-negative
    const auto r2 = r * r;
    auto p = X(4.784576222e-02f);
    p = p * r2 + S(4.104170576e-02f);
    p = p * r2 + S(7.526044548e-02f);
    p = p * r2 + S(1.666617095e-01f);
    p = p * r2 + S(1.000000000e+00f);
    const auto asin_val = r * p;
    // big: acos(|x|) = 2 * asin(inner)  [no cancellation near x=1]
    // small: acos(x) = pi/2 - asin(x)   [no cancellation: acos ~ pi/2]
    const auto pos = select(big, 2.f * asin_val, (pi_half_major - asin_val) + pi_half_minor);
    // reflect: acos(-x) = pi - acos(x)
    return select(x < 0.f, (pi_major - pos) + pi_minor, pos);
}

///
template<Floating_point X>
force_inline constexpr auto asin(X x) -> X
{
    using S = scalar_t<X>;

    // See work/cheb_asin.py
    constexpr auto pi_half_major = S(1.570796371e+00f);
    constexpr auto pi_half_minor = S(-4.371138829e-08f);
    const auto ax = select(x < 0.f, -x, x);
    const auto big = ax > 0.5f;
    const auto inner = sqrt((1.f - ax) * 0.5f);
    const auto r = select(big, inner, x);
    const auto r2 = r * r;
    auto p = X(4.784576222e-02f);
    p = p * r2 + S(4.104170576e-02f);
    p = p * r2 + S(7.526044548e-02f);
    p = p * r2 + S(1.666617095e-01f);
    p = p * r2 + S(1.000000000e+00f);
    const auto core = r * p;
    const auto sign = select(x < 0.f, X(-1.f), X(1.f));
    const auto tail = sign * ((pi_half_major - 2.f * core) + pi_half_minor);
    return select(big, tail, core);
}

///
template<Floating_point X>
force_inline constexpr auto atan(X x) -> X
{
    using S = scalar_t<X>;

    constexpr auto pi_half_major = S(1.570796371e+00f);
    constexpr auto pi_half_minor = S(-4.371138829e-08f);
    constexpr auto pi_quarter_major = S(7.853981853e-01f);
    constexpr auto pi_quarter_minor = S(-2.185569414e-08f);
    constexpr auto tan_pi_8 = S(4.142135680e-01f);  // sqrt(2) - 1
    const auto sign = select(x < 0.f, X(-1), X(1));
    auto ax = select(x < 0.f, -x, x);
    const auto inv = ax > 1.f;
    ax = select(inv, 1 / ax, ax);
    const auto big = ax > tan_pi_8;
    const auto r = select(big, (ax - 1) / (ax + 1), ax);
    const auto r2 = r * r;
    auto p = X(8.243747801e-02f);
    p = p * r2 + S(-1.394065171e-01f);
    p = p * r2 + S(1.998402327e-01f);
    p = p * r2 + S(-3.333312869e-01f);
    p = p * r2 + S(1.000000000e+00f);
    auto val = r * p;
    val = select(big, (pi_quarter_major + val) + pi_quarter_minor, val);
    val = select(inv, (pi_half_major - val) + pi_half_minor, val);
    return sign * val;
}

// MARK: - exp2, log2

///
template<Floating_point X>
force_inline constexpr auto exp2(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto exp_bias = ieee_exp_bias_v<S>;
    constexpr auto sig_bits = ieee_sig_bits_v<S>;

    const auto int_part = float_to_signed(floor(x));
    const auto dec_part = x - signed_to_float(int_part);
    const auto val = (signed_to_unsigned(int_part) + exp_bias) << sig_bits;
    const auto int_pow = reinterpret_as_float(val);

    // See work/cheb_exp2.py
    auto v = X(2.186578495e-04f);
    v = v * dec_part + S(1.239133184e-03f);
    v = v * dec_part + S(9.684186429e-03f);
    v = v * dec_part + S(5.548062921e-02f);
    v = v * dec_part + S(2.402304560e-01f);
    v = v * dec_part + S(6.931469440e-01f);
    v = v * dec_part + S(1.000000000e+00f);
    return select(x < S(-126.f), X{}, int_pow * v);
}

///
template<Floating_point X>
force_inline constexpr auto log2(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto exp_bias = ieee_exp_bias_v<S>;
    constexpr auto sig_bits = ieee_sig_bits_v<S>;
    constexpr auto one = decltype(sig_bits){1};
    constexpr auto sig_mask = (one << sig_bits) - one;
    
    const auto bits = reinterpret_as_int(x);
    const auto val = unsigned_to_signed(bits >> sig_bits) - unsigned_to_signed(exp_bias);
    const auto m = reinterpret_as_float((exp_bias << sig_bits) | (bits & sig_mask));

    // See: https://innovation.ebayinc.com/stories/fast-approximate-logarithms-part-i-the-basics/
    // Fold [1, 2) -> [0.75, 1.5): if m >= 1.5, halve m and add 1 to exponent.
    // Places the zero of log2 (at x=1) in the interior of the reduced range,
    // eliminating cancellation for x near 1. See work/cheb_log2.py.
    const auto fold  = m >= S(1.5f);
    const auto m_adj = select(fold, m * S(0.5f), m);
    const auto e_adj = select(fold, signed_to_float(val) + S(1.f), signed_to_float(val));

    const auto u = m_adj - S(1.f);
    auto p = X(7.639224082e-02f);
    p = p * u + S(-1.725005209e-01f);
    p = p * u + S(2.160132378e-01f);
    p = p * u + S(-2.428877503e-01f);
    p = p * u + S(2.882365286e-01f);
    p = p * u + S(-3.605715632e-01f);
    p = p * u + S(4.809007943e-01f);
    p = p * u + S(-7.213485241e-01f);
    p = p * u + S(1.442695022e+00f);
    return e_adj + u * p;
}

// MARK: - exp, log, pow, etc.

///
template<Floating_point X>
force_inline constexpr auto exp(X x) -> X
{
    using S = scalar_t<X>;

    constexpr auto ln_2 = std::numbers::ln2_v<S>;
    constexpr auto ln2_hi = S(6.93145752e-01f); // 0x3f317200: 8 trailing zero mantissa bits
    constexpr auto ln2_lo = S(1.42860677e-06f); // ln(2) - ln2_hi
    const auto n = floor(x / ln_2);
    const auto r = (x - n * ln2_hi) - n * ln2_lo;

    constexpr auto exp_bias = ieee_exp_bias_v<S>;
    constexpr auto sig_bits = ieee_sig_bits_v<S>;
    const auto arg = signed_to_unsigned(float_to_signed(n));
    const auto bits = (arg + exp_bias) << sig_bits;
    const auto two_pow = reinterpret_as_float(bits);

    // exp(r) = exp2(r * log2e), r*log2e in [0, 1) -- reuses the Chebyshev-fitted
    // exp2 polynomial instead of the Taylor coefficients that blow up at r = ln2.
    constexpr auto log2e = S(std::numbers::log2e_v<double>);
    const auto t = r * log2e;
    auto v = X(2.186578495e-04f);
    v = v * t + S(1.239133184e-03f);
    v = v * t + S(9.684186429e-03f);
    v = v * t + S(5.548062921e-02f);
    v = v * t + S(2.402304560e-01f);
    v = v * t + S(6.931469440e-01f);
    v = v * t + S(1.000000000e+00f);

    return select(x < S(-87.34f), X{}, two_pow * v);
}

///
template<Floating_point X>
force_inline constexpr auto log(X x) -> X
{
    using S = scalar_t<X>;
    return std::numbers::ln2_v<S> * log2(x);
}

///
template<Floating_point X>
force_inline constexpr auto log10(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto log10_2 = std::numbers::ln2_v<S> / std::numbers::ln10_v<S>;
    return log10_2 * log2(x);
}

///
template<Floating_point X>
force_inline constexpr auto logB(X b, X x) -> X
{
    return log2(x) / log2(b);
}

// Domain: x >= 0 !!!
template<Floating_point X>
force_inline constexpr auto pow(X x, X y) -> X
{
    //assert(all(x >= 0) && "vsl::pow requires x >= 0!");
    return select(x == 0, X{}, exp2(log2(x) * y)); // Revisit this.
}

// MARK: - Hyperbolic Trig Functions

///
template<Floating_point X>
force_inline constexpr auto cosh(X x) -> X
{
    const auto b = exp(x);
    const auto br = 1 / (2 * b);
    const auto bh = b / 2;
    return bh + br;
}

///
template<Floating_point X>
force_inline constexpr auto sinh(X x) -> X
{
    const auto b = exp(x);
    const auto br = 1 / (2 * b);
    const auto bh = b / 2;
    return bh - br;
}

///
template<Floating_point X>
force_inline constexpr auto tanh(X x) -> X
{
    using S = scalar_t<X>;

    const auto u = x * x;
    auto r = X(3.292487918e-06f);
    r = r * u + S(1.962047536e-04f);
    r = r * u + S(8.333356120e-03f);
    r = r * u + S(1.666701734e-01f);
    r = r * u + S(9.999960065e-01f);
    const auto p = x * r;
    return p * rsqrt(p * p + 1);
}

// MARK: - Other

///
template<Floating_point X, Floating_point C>
force_inline constexpr auto min(X a, C b) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::min(a, X(b));
    }
    else {
        return std::min(a, X(b));
    }
}

///
template<Floating_point X, Floating_point C>
force_inline constexpr auto max(X a, C b) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::max(a, X(b));
    }
    else {
        return std::max(a, X(b));
    }
}

///
template<Floating_point X, Floating_point C>
force_inline constexpr auto clamp(X x, C a, C b) -> X
{
    if constexpr (is_vector_v<X>) {
        return min(max(x, a), b);
    }
    else {
        return std::clamp(x, X(a), X(b));
    }
}

///
template<Floating_point X>
force_inline constexpr auto sign(X x) -> X
{
    return select(x > 0, X(1), select(x < 0, X(-1), X(0)));
}

} // namespace vsl