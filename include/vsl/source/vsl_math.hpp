#pragma once

#include <cassert>
#include <cmath>
#include <concepts> // std::floating_point
#include <cstdint>  // std::int64_t
#include <numbers>
#include <type_traits> // std::is_constant_evaluated

#include "vsl_core.hpp"
#include "vsl_logic.hpp"

// -ffast-math implies -ffinite-math-only, where any infinity is UB, so overflow saturates to finite
// values instead. MSVC's /fp:fast doesn't make infinities UB, so this stays off there.
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#define VSL_FINITE_MATH 1
#endif

namespace vsl {

// constexpr helpers.
namespace detail {

template<std::floating_point F>
constexpr auto two_pow(int32_t n) -> F
{
    auto r = F{1};
    for (auto i = int32_t{}; i < n; ++i) { r *= F{2}; }
    for (auto i = int32_t{}; i > n; --i) { r /= F{2}; }
    return r;
}

template<std::floating_point F>
constexpr auto abs(F x) -> F
{
    return x < F{0} ? -x : x;
}

template<std::floating_point F>
constexpr auto trunc(F x) -> F
{
    using I = signed_t<F>;
    if (x != x) { return x; } // NaN
    constexpr auto big = two_pow<F>(std::numeric_limits<F>::digits);
    if (abs(x) >= big) { return x; }
    return static_cast<F>(static_cast<I>(x));
}

template<std::floating_point F>
constexpr auto floor(F x) -> F
{
    const auto t = trunc(x);
    return t > x ? t - F{1} : t;
}

template<std::floating_point F>
constexpr auto ceil(F x) -> F
{
    const auto t = trunc(x);
    return t < x ? t + F{1} : t;
}

template<std::floating_point F>
constexpr auto round(F x) -> F
{
    const auto t = trunc(x);
    const auto d = x - t; // fractional part, carries the sign of x
    if (d >= F{0.5}) { return t + F{1}; }
    if (d <= F{-0.5}) { return t - F{1}; }
    return t;
}

template<std::floating_point F>
constexpr auto sqrt(F x) -> F
{
#if VSL_FINITE_MATH
    if (x <= F{0}) { return F{0}; }
#else
    constexpr auto inf = std::numeric_limits<F>::infinity();
    if (x != x || x == inf || x == F{0}) { return x; } // NaN, +inf, +/-0
    if (x < F{0}) { return std::numeric_limits<F>::quiet_NaN(); }
#endif

    // Reduce x = m * 4^k with m in [1, 4), so sqrt(x) = sqrt(m) * 2^k.
    auto k = int32_t{};
    auto m = x;
    while (m >= F{4}) { m /= F{4}; ++k; }
    while (m < F{1}) { m *= F{4}; --k; }

    // Newton-Raphson on [1, 4) from a fixed seed converges in a few steps.
    auto g = F{1.5};
    for (auto i = int32_t{}; i < 12; ++i) {
        const auto next = F{0.5} * (g + m / g);
        if (next == g) { break; }
        g = next;
    }
    return g * two_pow<F>(k);
}

/// +inf, or the largest finite value under VSL_FINITE_MATH.
template<std::floating_point F>
constexpr auto overflow() -> F
{
#if VSL_FINITE_MATH
    return std::numeric_limits<F>::max();
#else
    return std::numeric_limits<F>::infinity();
#endif
}

} // namespace detail

///
template<Floating_point X>
force_inline constexpr auto abs(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::abs(x);
    }
    else {
        if (std::is_constant_evaluated()) { return detail::abs(x); }
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
        if (std::is_constant_evaluated()) { return detail::trunc(x); }
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
        if (std::is_constant_evaluated()) { return detail::floor(x); }
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
        if (std::is_constant_evaluated()) { return detail::ceil(x); }
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
        if (std::is_constant_evaluated()) { return detail::round(x); }
        return std::round(x);
    }
}


/// On the SSE4.2 path this is just plain a * b + c. Would love to revisit later but it was becoming a whole thing.
template<Floating_point X>
force_inline constexpr auto fma(X a, X b, X c) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::fma(a, b, c);
    }
    else {
        if (std::is_constant_evaluated()) { return a * b + c; }
#if VSL_SSE4
        return a * b + c;
#else
        return std::fma(a, b, c);
#endif
    }
}

// MARK: - min, max, clamp

/// If a is NaN, returns b on every backend. b must not be NaN.
template<Floating_point X, Floating_point C>
force_inline constexpr auto min(X a, C b) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::min(a, X(b));
    }
    else {
        return a < X(b) ? a : X(b);
    }
}

/// If a is NaN, returns b on every backend. b must not be NaN.
template<Floating_point X, Floating_point C>
force_inline constexpr auto max(X a, C b) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::max(a, X(b));
    }
    else {
        return a > X(b) ? a : X(b);
    }
}

/// If x is NaN, returns a on every backend. pre: a <= b, neither NaN
template<Floating_point X, Floating_point C>
force_inline constexpr auto clamp(X x, C a, C b) -> X
{
    return min(max(x, a), b);
}

// MARK: - sqrt

/// pre: x >= 0
template<Floating_point X>
force_inline constexpr auto sqrt(X x) -> X
{
    assert(all(x >= X{0}) && "vsl::sqrt: x >= 0");
    if constexpr (is_vector_v<X>) {
        return X::sqrt(x);
    }
    else {
        if (std::is_constant_evaluated()) { return detail::sqrt(x); }
        return std::sqrt(x);
    }
}

/// pre: x > 0
template<Floating_point X>
force_inline constexpr auto rsqrt(X x) -> X
{
    assert(all(x > X{0}) && "vsl::rsqrt: x > 0");
    if constexpr (is_vector_v<X>) {
        return X::rsqrt(x);
    }
    else {
        if (std::is_constant_evaluated()) { return X{1} / detail::sqrt(x); }
        return 1 / std::sqrt(x);
    }
}

// MARK: - fmod, wrap

/// pre: y != 0
template<Floating_point X>
force_inline constexpr auto fmod(X x, X y) -> X
{
    assert(all(y != X{0}) && "vsl::fmod: y != 0");
    if constexpr (is_vector_v<X>) {
        return x - trunc(x / y) * y;
    }
    else {
        if (std::is_constant_evaluated()) { return x - detail::trunc(x / y) * y; }
        return std::fmod(x, y);
    }
}

/// Normalized wrap to [0, 1]
template<Floating_point X>
force_inline constexpr auto wrap(X x) -> X
{
    return x - floor(x);
}

/// Symmetric wrap to [-k, k]. pre: k != 0
template<Floating_point X, Floating_point S>
force_inline constexpr auto wrap(X x, S k) -> X
{
    assert(all(k != S{0}) && "vsl::wrap: k != 0");
    x += k;
    const auto two_k = 2 * k;
    const auto m = x - two_k * trunc(x / two_k);
    return select(x >= 0, m, m + two_k) - k;
}

/// Arbitrary wrap to [a, b]. pre: a != b
template<Floating_point X, Floating_point S>
force_inline constexpr auto wrap(X x, S a, S b) -> X
{
    assert(all(a != b) && "vsl::wrap: a != b");
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

    [[maybe_unused]] constexpr auto pi = std::numbers::pi_v<S>;
    if constexpr (needs_wrap) {
        x = wrap(x, pi);
    }
    else {
        assert(all(x >= X{-pi}) && all(x <= X{pi}) && "vsl::cos: x in [-pi, pi] without wrap");
    }

    // See work/cheb_cos.py
    constexpr auto pi_half_major = S(1.570796371e+00f);
    constexpr auto pi_half_minor = S(-4.371138829e-08f);
    const auto x2 = x * x;
    auto p = X(1.772737512e-09f);
    p = fma(p, x2, X(-2.677252269e-07f));
    p = fma(p, x2, X(2.412236245e-05f));
    p = fma(p, x2, X(-1.329319435e-03f));
    p = fma(p, x2, X(3.838663921e-02f));
    p = fma(p, x2, X(-4.052847326e-01f));
    const auto lo = (x - pi_half_major) - pi_half_minor;  // ~ x - pi/2 (Sterbenz)
    const auto hi = (x + pi_half_major) + pi_half_minor;  // ~ x + pi/2
    return lo * hi * p;
}

///
template<bool needs_wrap, Floating_point X>
force_inline constexpr auto sin(X x) -> X
{
    using S = scalar_t<X>;

    [[maybe_unused]] constexpr auto pi = std::numbers::pi_v<S>;
    if constexpr (needs_wrap) {
        x = wrap(x, pi);
    }
    else {
        assert(all(x >= X{-pi}) && all(x <= X{pi}) && "vsl::sin: x in [-pi, pi] without wrap");
    }

    // See work/cheb_sin.py
    constexpr auto pi_major = S(3.141592741e+00f);
    constexpr auto pi_minor = S(-8.742277657e-08f);
    const auto x2 = x * x;
    auto p = X(1.345148160e-10f);
    p = fma(p, x2, X(-2.335733917e-08f));
    p = fma(p, x2, X(2.522634304e-06f));
    p = fma(p, x2, X(-1.735062688e-04f));
    p = fma(p, x2, X(6.620881148e-03f));
    p = fma(p, x2, X(-1.013211831e-01f));
    const auto lo = (x - pi_major) - pi_minor;  // ~ x - pi (Sterbenz)
    const auto hi = (x + pi_major) + pi_minor;  // ~ x + pi
    return x * (lo * hi * p);
}

///
template<bool needs_wrap, Floating_point X>
force_inline constexpr auto tan(X x) -> X
{
    using S = scalar_t<X>;

    [[maybe_unused]] constexpr auto half_pi = std::numbers::pi_v<S> / 2;
    if constexpr (needs_wrap) {
        x = wrap(x, half_pi);
    }
    else {
        assert(all(x >= X{-half_pi}) && all(x <= X{half_pi}) && "vsl::tan: x in [-pi/2, pi/2] without wrap");
    }

    // See work/cheb_tan.py
    constexpr auto pi_half_major = S(1.570796371e+00f);
    constexpr auto pi_half_minor = S(-4.371138829e-08f);
    const auto x2 = x * x;
    auto q = X(-9.648306332e-06f);
    q = fma(q, x2, X(-1.677224500e-04f));
    q = fma(q, x2, X(-4.349521827e-03f));
    q = fma(q, x2, X(-1.775324047e-01f));
    q = fma(q, x2, X(2.467401028e+00f));
    const auto lo = (pi_half_major - x) + pi_half_minor;  // ~ pi/2 - x (Sterbenz)
    const auto hi = (pi_half_major + x) + pi_half_minor;  // ~ pi/2 + x
    return x * (q / (lo * hi));
}

///
template<bool needs_wrap, Floating_point X>
force_inline constexpr auto cos_2pi(X x) -> X
{
    if constexpr (needs_wrap) {
        x = wrap(x);
    }
    else {
        assert(all(x >= X{0}) && all(x <= X{1}) && "vsl::cos_2pi: x in [0, 1] without wrap");
    }

    x = select(x < 0.5f, x, 1 - x); // Approximation on -0.5...0.5 makes us exact at 0.

    // See work/cheb_cos2pi.py
    const auto x2 = x * x;
    auto q = X(-1.483706594e+00f);
    q = fma(q, x2, X(7.732257843e+00f));
    q = fma(q, x2, X(-2.592922974e+01f));
    q = fma(q, x2, X(5.862275696e+01f));
    q = fma(q, x2, X(-8.179283142e+01f));
    q = fma(q, x2, X(5.982733917e+01f));
    q = fma(q, x2, X(-1.600000000e+01f));
    const auto lo = x - 0.25f;
    const auto hi = x + 0.25f;
    return (lo * hi) * q;
}

///
template<bool needs_wrap, Floating_point X>
force_inline constexpr auto sin_2pi(X x) -> X
{
    if constexpr (needs_wrap) {
        x = wrap(x);
    }
    else {
        assert(all(x >= X{0}) && all(x <= X{1}) && "vsl::sin_2pi: x in [0, 1] without wrap");
    }

    // Approximation on 0...1, see work/cheb_sin2pi.py
    const auto u = x - 0.5f;
    const auto v = u * u;
    auto p = X(-6.151015759e-01f);
    p = fma(p, v, X(3.622930765e+00f));
    p = fma(p, v, X(-1.417941380e+01f));
    p = fma(p, v, X(3.851269913e+01f));
    p = fma(p, v, X(-6.707761383e+01f));
    p = fma(p, v, X(6.483584595e+01f));
    p = fma(p, v, X(-2.513274193e+01f));
    return -u * x * (x - 1.f) * p;
}

///
template<bool needs_wrap, Floating_point X>
force_inline constexpr auto tan_pi(X x) -> X
{
    if constexpr (needs_wrap) {
        x -= round(x);  // period 1 -> [-1/2, 1/2]
    }
    else {
        assert(all(x >= X{-0.5f}) && all(x <= X{0.5f}) && "vsl::tan_pi: x in [-1/2, 1/2] without wrap");
    }

    const auto x2 = x * x;
    auto q = X(-2.914071456e-02f);
    q = fma(q, x2, X(-5.132636800e-02f));
    q = fma(q, x2, X(-1.348624676e-01f));
    q = fma(q, x2, X(-5.577344894e-01f));
    q = fma(q, x2, X(7.853981853e-01f));
    const auto lo = 0.5f - x;
    const auto hi = 0.5f + x;
    return x * (q / (lo * hi));
}

// MARK: - Inverse Trig Functions

/// pre: x in [-1, 1]
template<Floating_point X>
force_inline constexpr auto acos(X x) -> X
{
    assert(all(x >= X{-1}) && all(x <= X{1}) && "vsl::acos: x in [-1, 1]");
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
    p = fma(p, r2, X(4.104170576e-02f));
    p = fma(p, r2, X(7.526044548e-02f));
    p = fma(p, r2, X(1.666617095e-01f));
    p = fma(p, r2, X(1.000000000e+00f));
    const auto asin_val = r * p;
    // big: acos(|x|) = 2 * asin(inner)  [no cancellation near x=1]
    // small: acos(x) = pi/2 - asin(x)   [no cancellation: acos ~ pi/2]
    const auto pos = select(big, 2.f * asin_val, (pi_half_major - asin_val) + pi_half_minor);
    // reflect: acos(-x) = pi - acos(x)
    return select(x < 0.f, (pi_major - pos) + pi_minor, pos);
}

/// pre: x in [-1, 1]
template<Floating_point X>
force_inline constexpr auto asin(X x) -> X
{
    assert(all(x >= X{-1}) && all(x <= X{1}) && "vsl::asin: x in [-1, 1]");
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
    p = fma(p, r2, X(4.104170576e-02f));
    p = fma(p, r2, X(7.526044548e-02f));
    p = fma(p, r2, X(1.666617095e-01f));
    p = fma(p, r2, X(1.000000000e+00f));
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
    p = fma(p, r2, X(-1.394065171e-01f));
    p = fma(p, r2, X(1.998402327e-01f));
    p = fma(p, r2, X(-3.333312869e-01f));
    p = fma(p, r2, X(1.000000000e+00f));
    auto val = r * p;
    val = select(big, (pi_quarter_major + val) + pi_quarter_minor, val);
    val = select(inv, (pi_half_major - val) + pi_half_minor, val);
    return sign * val;
}

// MARK: - exp2, log2

namespace detail {

// Folds through a small correction term. We need this for pow.
template<Floating_point X>
force_inline constexpr auto exp2(X x, X corr) -> X
{
    using S = scalar_t<X>;
    constexpr auto exp_bias = ieee_exp_bias_v<S>;
    constexpr auto sig_bits = ieee_sig_bits_v<S>;

    // bias + 1 builds the all-ones exponent (inf), -bias the zero exponent (0, no subnormals).
#if VSL_FINITE_MATH
    constexpr auto x_hi = S(exp_bias + 1) * S(1 - 0x1p-20);
#else
    constexpr auto x_hi = S(exp_bias + 1);
#endif
    x = vsl::clamp(x, -S(exp_bias), x_hi);
    const auto fl = vsl::floor(x);
    const auto dec_part = (x - fl) + corr; // fl is already integral; no round-trip
    const auto int_part = vsl::float_to_signed(fl);
    const auto val = (vsl::signed_to_unsigned(int_part) + exp_bias) << sig_bits;
    const auto int_pow = vsl::reinterpret_as_float(val);

    // See work/cheb_exp2.py
    auto v = X(2.186578495e-04f);
    v = fma(v, dec_part, X(1.239133184e-03f));
    v = fma(v, dec_part, X(9.684186429e-03f));
    v = fma(v, dec_part, X(5.548062921e-02f));
    v = fma(v, dec_part, X(2.402304560e-01f));
    v = fma(v, dec_part, X(6.931469440e-01f));
    v = fma(v, dec_part, X(1.000000000e+00f));
    return int_pow * v;
}

} // namespace detail

///
template<Floating_point X>
force_inline constexpr auto exp2(X x) -> X
{
    return detail::exp2(x, X{});
}

/// log2(0) = -inf, or -150 (one below the smallest subnormal) under VSL_FINITE_MATH.
/// Subnormal x is only correct under DAZ, where it compares equal to 0. pre: x >= 0
template<Floating_point X>
force_inline constexpr auto log2(X x) -> X
{
    assert(all(x >= X{0}) && "vsl::log2: x >= 0");
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
    p = fma(p, u, X(-1.725005209e-01f));
    p = fma(p, u, X(2.160132378e-01f));
    p = fma(p, u, X(-2.428877503e-01f));
    p = fma(p, u, X(2.882365286e-01f));
    p = fma(p, u, X(-3.605715632e-01f));
    p = fma(p, u, X(4.809007943e-01f));
    p = fma(p, u, X(-7.213485241e-01f));
    p = fma(p, u, X(1.442695022e+00f));
#if VSL_FINITE_MATH
    constexpr auto log2_zero = -S(exp_bias + sig_bits);
#else
    constexpr auto log2_zero = -std::numeric_limits<S>::infinity();
#endif
    return select(x == X{0}, X(log2_zero), e_adj + u * p);
}

// MARK: - exp, log, pow, etc.

///
template<Floating_point X>
force_inline constexpr auto exp(X x) -> X
{
    using S = scalar_t<X>;

    constexpr auto ln_2 = std::numbers::ln2_v<S>;
    constexpr auto log2e = S(std::numbers::log2e_v<double>);
    constexpr auto ln2_hi = S(6.93145752e-01f); // 0x3f317200: 8 trailing zero mantissa bits
    constexpr auto ln2_lo = S(1.42860677e-06f); // ln(2) - ln2_hi
    constexpr auto exp_bias = ieee_exp_bias_v<S>;
    constexpr auto sig_bits = ieee_sig_bits_v<S>;

    // n = bias + 1 builds the all-ones exponent (inf), n = -bias the zero exponent (0, no subnormals).
    constexpr auto x_lo = -(S(exp_bias) - S(0.5f)) * ln_2;
#if VSL_FINITE_MATH
    constexpr auto x_hi = S(exp_bias + 1) * S(1 - 0x1p-20) * ln_2;
#else
    constexpr auto x_hi = (S(exp_bias) + S(1.5f)) * ln_2;
#endif
    x = clamp(x, x_lo, x_hi);
    const auto n = floor(x * log2e);
    const auto r = (x - n * ln2_hi) - n * ln2_lo;

    const auto arg = signed_to_unsigned(float_to_signed(n));
    const auto bits = (arg + exp_bias) << sig_bits;
    const auto two_pow = reinterpret_as_float(bits);

    // exp(r) = exp2(r * log2e), r*log2e in [0, 1) -- reuses the Chebyshev-fitted
    // exp2 polynomial instead of the Taylor coefficients that blow up at r = ln2.
    const auto t = r * log2e;
    auto v = X(2.186578495e-04f);
    v = fma(v, t, X(1.239133184e-03f));
    v = fma(v, t, X(9.684186429e-03f));
    v = fma(v, t, X(5.548062921e-02f));
    v = fma(v, t, X(2.402304560e-01f));
    v = fma(v, t, X(6.931469440e-01f));
    v = fma(v, t, X(1.000000000e+00f));
    return two_pow * v;
}

/// pre: x > 0
template<Floating_point X>
force_inline constexpr auto log(X x) -> X
{
    using S = scalar_t<X>;
    return std::numbers::ln2_v<S> * log2(x);
}

/// pre: x > 0
template<Floating_point X>
force_inline constexpr auto log10(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto log10_2 = std::numbers::ln2_v<S> / std::numbers::ln10_v<S>;
    return log10_2 * log2(x);
}

/// pre: b > 0, x > 0
template<Floating_point X>
force_inline constexpr auto logB(X b, X x) -> X
{
    return log2(x) / log2(b);
}

/// pre: x >= 0
template<Floating_point X>
force_inline constexpr auto pow(X x, X y) -> X
{
    assert(all(x >= X{0}) && "vsl::pow: x >= 0");
    using S = scalar_t<X>;
    constexpr auto huge = detail::overflow<S>();
    if constexpr (is_vector_v<X>) {
        // FMA plus new split impl gets us down to ~5 ULP.
        const auto is_zero = x == X{0};
        const auto l = log2(select(is_zero, X{1}, x)); // avoid log2(0) on zero lanes
        const auto hi = y * l;
        const auto lo = fma(y, l, -hi);
        const auto zero_pow = select(y < X{0}, X(huge), select(y == X{0}, X{1}, X{}));
        return select(is_zero, zero_pow, detail::exp2(hi, lo));
    }
    else {
        // Widen to double, then round back to float. This gets us down to ~1 ULP.
        using W = double;
        if (x == X{0}) { return y < X{0} ? huge : (y == X{0} ? X{1} : X{0}); }
        const auto l = log2(static_cast<W>(x));
        return static_cast<X>(exp2(l * static_cast<W>(y)));
    }
}

// MARK: - Hyperbolic Trig Functions

namespace detail {

// Under VSL_FINITE_MATH, keeps exp(-|x|) normal so 1 / (2 b) stays finite.
template<Floating_point X>
force_inline constexpr auto hyperbolic_arg(X x) -> X
{
#if VSL_FINITE_MATH
    using S = scalar_t<X>;
    constexpr auto lim = S(ieee_exp_bias_v<S> - 2) * std::numbers::ln2_v<S>;
    return vsl::clamp(x, -lim, lim);
#else
    return x;
#endif
}

} // namespace detail

///
template<Floating_point X>
force_inline constexpr auto cosh(X x) -> X
{
    const auto b = exp(detail::hyperbolic_arg(x));
    const auto br = 1 / (2 * b);
    const auto bh = b / 2;
    return bh + br;
}

/// Taylor below |x| = 1, where b/2 - 1/(2b) cancels.
template<Floating_point X>
force_inline constexpr auto sinh(X x) -> X
{
    const auto b = exp(detail::hyperbolic_arg(x));
    const auto br = 1 / (2 * b);
    const auto bh = b / 2;

    const auto u = x * x;
    auto p = X(2.755731922e-06f);
    p = fma(p, u, X(1.984126984e-04f));
    p = fma(p, u, X(8.333333333e-03f));
    p = fma(p, u, X(1.666666667e-01f));
    const auto small = fma(x * u, p, x);
    return select(abs(x) < 1.f, small, bh - br);
}

/// tanh rounds to 1 in float past ~9.01, so the clamp is free and s * s can't overflow.
/// 9.2 lands on exactly 1, fused or not; 9.1 and 9.25 don't.
template<Floating_point X>
force_inline constexpr auto tanh(X x) -> X
{
    const auto xc = clamp(x, -9.2f, 9.2f);
    const auto u = xc * xc;
    auto r = X(3.567437808e-06f);
    r = fma(r, u, X(1.918139418e-04f));
    r = fma(r, u, X(8.349913590e-03f));
    r = fma(r, u, X(1.666557794e-01f));
    r = fma(r, u, X(1.000000829e+00f));
    const auto s = xc * r;
    return s / sqrt(fma(s, s, X(1)));
}

// MARK: - Other

template<Floating_point X, Floating_point C>
force_inline constexpr auto smooth_max(X a, X b, C s) -> X
{
    return X(0.5f) * (a + b + vsl::sqrt((a - b) * (a - b) + X(s)));
}

///
template<Floating_point X>
force_inline constexpr auto sign(X x) -> X
{
    return select(x > 0, X(1), select(x < 0, X(-1), X(0)));
}

} // namespace vsl