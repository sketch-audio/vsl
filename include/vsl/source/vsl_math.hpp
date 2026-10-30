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

// Split constants such as (x - pi_major) - pi_minor only work if the compiler keeps the order, which
// -ffast-math doesn't. Clang lets the header opt out of reassociation alone (float_control(precise)
// would also make std::fma and friends errno-setting library calls); elsewhere this is a no-op.
// Callers' own code keeps whatever flags they chose.
#if defined(__clang__)
#pragma float_control(push)
#pragma clang fp reassociate(off)
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
constexpr auto rint(F x) -> F
{
    const auto t = trunc(x);
    const auto d = x - t;
    const auto odd = detail::trunc(t * F{0.5}) != t * F{0.5};
    if (d > F{0.5} || (d == F{0.5} && odd)) { return t + F{1}; }
    if (d < F{-0.5} || (d == F{-0.5} && odd)) { return t - F{1}; }
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


/// Round to nearest, ties to even (std::nearbyint in the default rounding mode). One instruction
/// on both backends, where round's ties-away costs several on SSE.
template<Floating_point X>
force_inline constexpr auto rint(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::rint(x);
    }
    else {
        if (std::is_constant_evaluated()) { return detail::rint(x); }
        return std::nearbyint(x);
    }
}

/// a * b + c, fused where the target has it (VSL_FMA), otherwise multiply then add.
/// Not guaranteed to round once.
template<Floating_point X>
force_inline constexpr auto mul_add(X a, X b, X c) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::mul_add(a, b, c);
    }
    else {
        if (std::is_constant_evaluated()) { return a * b + c; }
#if VSL_NEON
        return std::fma(a, b, c);
#elif VSL_FMA
        if constexpr (std::same_as<X, float>) {
            return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
        }
        else {
            return _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)));
        }
#else
        return a * b + c;
#endif
    }
}

template<Floating_point X>
[[deprecated("use vsl::mul_add, which doesn't promise a single rounding")]]
force_inline constexpr auto fma(X a, X b, X c) -> X
{
    return mul_add(a, b, c);
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
    return X{1} / sqrt(x);
}

/// The hardware estimate: about 8 bits on NEON, 12 on SSE (exact for double on SSE). pre: x > 0
template<Floating_point X>
force_inline auto rsqrt_estimate(X x) -> X
{
    assert(all(x > X{0}) && "vsl::rsqrt_estimate: x > 0");
    if constexpr (is_vector_v<X>) {
        return X::rsqrt_estimate(x);
    }
    else {
        return vector_t<X>::rsqrt_estimate(vector_t<X>{x})[0];
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

// MARK: - Precision

/// Accuracy tier. Fast only differs from Standard where it's measurably cheaper and stays inaudible;
/// both tiers give the same results for 0, the clamps and overflow.
enum class Precision { Standard, Fast };

// MARK: - Range reduction

/// x - rint(x): exact for every float, in [-1/2, 1/2].
template<Floating_point X>
force_inline constexpr auto reduce_turns(X x) -> X
{
    return x - rint(x);
}

///
template<Floating_point X>
struct Reduced_pi {
    X r; // in [-pi/2, pi/2]
    X n; // integral
};

namespace detail {

// x - k pi with pi in three parts (Cody-Waite). The first two products are exact while |k| < 2^12.
template<Floating_point X>
force_inline constexpr auto sub_pi_multiple(X x, X k) -> X
{
    using S = scalar_t<X>;
    auto r = mul_add(k, X(S(-3.140625f)), x);
    r = mul_add(k, X(S(-9.67502594e-04f)), r);
    return mul_add(k, X(S(-1.50995803e-07f)), r);
}

// (-1)^n y for integral n: n's low bit into y's sign bit.
template<Floating_point X>
force_inline constexpr auto flip_if_odd(X y, X n) -> X
{
    if constexpr (is_vector_v<X>) {
        constexpr auto shift = sizeof(scalar_t<X>) * 8 - 1;
        const auto odd = signed_to_unsigned(float_to_signed(n)) << shift;
        return reinterpret_as_float(reinterpret_as_int(y) ^ odd);
    }
    else {
        if (std::is_constant_evaluated()) { return static_cast<int64_t>(n) % 2 ? -y : y; }
        // Through lane 0 of a vector: integer registers would cost two moves on the critical path.
        using V = vector_t<X>;
        return flip_if_odd(V{y}, V{n})[0];
    }
}

} // namespace detail

namespace detail {

// Past ~1e4 the reduction loses r entirely; the clamp keeps the kernels bounded there.
template<Floating_point X>
force_inline constexpr auto clamp_half_pi(X r) -> X
{
    using S = scalar_t<X>;
    constexpr auto half_pi = std::numbers::pi_v<S> / 2;
    return clamp(r, -half_pi, half_pi);
}

} // namespace detail

/// x = n pi + r. r keeps its relative accuracy near every multiple of pi for |x| up to ~1e2,
/// and its absolute accuracy (~3e-7) to ~1e4.
template<Floating_point X>
force_inline constexpr auto reduce_pi(X x) -> Reduced_pi<X>
{
    using S = scalar_t<X>;
    const auto n = rint(x * S(std::numbers::inv_pi_v<double>));
    return {detail::clamp_half_pi(detail::sub_pi_multiple(x, n)), n};
}

// MARK: - Trig kernels

/// No range reduction: for callers who already have the argument in range.
namespace detail {

// sin x for x in [-pi, pi]. See work/cheb_sin.py
template<Precision P, Floating_point X>
force_inline constexpr auto sin_kernel(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto pi_major = S(3.141592741e+00f);
    constexpr auto pi_minor = S(-8.742277657e-08f);
    const auto x2 = x * x;
    auto p = X(1.345148160e-10f);
    p = mul_add(p, x2, X(-2.335733917e-08f));
    p = mul_add(p, x2, X(2.522634304e-06f));
    p = mul_add(p, x2, X(-1.735062688e-04f));
    p = mul_add(p, x2, X(6.620881148e-03f));
    p = mul_add(p, x2, X(-1.013211831e-01f));
    const auto lo = (x - pi_major) - pi_minor;  // ~ x - pi (Sterbenz)
    const auto hi = (x + pi_major) + pi_minor;  // ~ x + pi
    return x * (lo * hi * p);
}

// Past ~2e7 the reduction loses n, and r leaves [-pi, pi]. Tested alongside the reduction, so
// it's off the critical path, unlike clamping r.
template<Floating_point X>
force_inline constexpr auto radians_in_range(X x)
{
    using S = scalar_t<X>;
    return abs(x) < S(16777216.f);
}

// tan r for r in [-pi/2, pi/2], given the pole distances lo = pi/2 - r and hi = pi/2 + r,
// which have to be more accurate than r itself. See work/cheb_tan.py
template<Precision P, Floating_point X>
force_inline constexpr auto tan_reduced(X r, X lo, X hi) -> X
{
    const auto r2 = r * r;
    auto q = X(-9.648306332e-06f);
    q = mul_add(q, r2, X(-1.677224500e-04f));
    q = mul_add(q, r2, X(-4.349521827e-03f));
    q = mul_add(q, r2, X(-1.775324047e-01f));
    q = mul_add(q, r2, X(2.467401028e+00f));
    return r * (q / (lo * hi));
}

} // namespace detail

namespace kernel {

/// pre: x in [-pi, pi]
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto cos(X x) -> X
{
    using S = scalar_t<X>;
    [[maybe_unused]] constexpr auto pi = std::numbers::pi_v<S>;
    assert(all(x >= X{-pi}) && all(x <= X{pi}) && "vsl::kernel::cos: x in [-pi, pi]");

    // See work/cheb_cos.py
    constexpr auto pi_half_major = S(1.570796371e+00f);
    constexpr auto pi_half_minor = S(-4.371138829e-08f);
    const auto x2 = x * x;
    auto p = X(1.772737512e-09f);
    p = mul_add(p, x2, X(-2.677252269e-07f));
    p = mul_add(p, x2, X(2.412236245e-05f));
    p = mul_add(p, x2, X(-1.329319435e-03f));
    p = mul_add(p, x2, X(3.838663921e-02f));
    p = mul_add(p, x2, X(-4.052847326e-01f));
    const auto lo = (x - pi_half_major) - pi_half_minor;  // ~ x - pi/2 (Sterbenz)
    const auto hi = (x + pi_half_major) + pi_half_minor;  // ~ x + pi/2
    return lo * hi * p;
}

/// pre: x in [-pi, pi]
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto sin(X x) -> X
{
    using S = scalar_t<X>;
    [[maybe_unused]] constexpr auto pi = std::numbers::pi_v<S>;
    assert(all(x >= X{-pi}) && all(x <= X{pi}) && "vsl::kernel::sin: x in [-pi, pi]");
    return detail::sin_kernel<P>(x);
}

/// pre: x in [-pi/2, pi/2]
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto tan(X x) -> X
{
    using S = scalar_t<X>;
    [[maybe_unused]] constexpr auto half_pi = std::numbers::pi_v<S> / 2;
    assert(all(x >= X{-half_pi}) && all(x <= X{half_pi}) && "vsl::kernel::tan: x in [-pi/2, pi/2]");

    constexpr auto pi_half_major = S(1.570796371e+00f);
    constexpr auto pi_half_minor = S(-4.371138829e-08f);
    const auto lo = (pi_half_major - x) + pi_half_minor;  // ~ pi/2 - x (Sterbenz)
    const auto hi = (pi_half_major + x) + pi_half_minor;  // ~ pi/2 + x
    return detail::tan_reduced<P>(x, lo, hi);
}

} // namespace kernel

namespace detail {

// sin(2 pi r), r in [-1/2, 1/2]. See work/cheb_sin2pi.py (Fast: degree 10), sign flipped.
template<Precision P, Floating_point X>
force_inline constexpr auto sin_2pi_centred(X r) -> X
{
    const auto r2 = r * r;
    auto q = X{};
    if constexpr (P == Precision::Fast) {
        q = X(-3.199693680e+00f);
        q = mul_add(q, r2, X(1.407349300e+01f));
        q = mul_add(q, r2, X(-3.850109863e+01f));
        q = mul_add(q, r2, X(6.707709503e+01f));
        q = mul_add(q, r2, X(-6.483583832e+01f));
        q = mul_add(q, r2, X(2.513274193e+01f));
    }
    else {
        q = X(6.151015759e-01f);
        q = mul_add(q, r2, X(-3.622930765e+00f));
        q = mul_add(q, r2, X(1.417941380e+01f));
        q = mul_add(q, r2, X(-3.851269913e+01f));
        q = mul_add(q, r2, X(6.707761383e+01f));
        q = mul_add(q, r2, X(-6.483584595e+01f));
        q = mul_add(q, r2, X(2.513274193e+01f));
    }
    return r * ((0.5f - r) * (0.5f + r) * q); // r last: subnormal r keeps its bits
}

// cos(2 pi a), a in [0, 1/2]. See work/cheb_cos2pi.py
template<Precision P, Floating_point X>
force_inline constexpr auto cos_2pi_half(X a) -> X
{
    const auto a2 = a * a;
    auto q = X(-1.483706594e+00f);
    q = mul_add(q, a2, X(7.732257843e+00f));
    q = mul_add(q, a2, X(-2.592922974e+01f));
    q = mul_add(q, a2, X(5.862275696e+01f));
    q = mul_add(q, a2, X(-8.179283142e+01f));
    q = mul_add(q, a2, X(5.982733917e+01f));
    q = mul_add(q, a2, X(-1.600000000e+01f));
    return (a - 0.25f) * (a + 0.25f) * q;
}

// tan(pi r), r in [-1/2, 1/2]. See work/cheb_tanpi.py (Fast: degree 4)
template<Precision P, Floating_point X>
force_inline constexpr auto tan_pi_centred(X r) -> X
{
    const auto r2 = r * r;
    auto q = X{};
    if constexpr (P == Precision::Fast) {
        q = X(-1.531784832e-01f);
        q = mul_add(q, r2, X(-5.565539598e-01f));
        q = mul_add(q, r2, X(7.853981853e-01f));
    }
    else {
        q = X(-2.914071456e-02f);
        q = mul_add(q, r2, X(-5.132636800e-02f));
        q = mul_add(q, r2, X(-1.348624676e-01f));
        q = mul_add(q, r2, X(-5.577344894e-01f));
        q = mul_add(q, r2, X(7.853981853e-01f));
    }
    return r * (q / ((0.5f - r) * (0.5f + r)));
}

} // namespace detail

namespace kernel {

/// sin(2 pi x), exact at 0, 1/2 and 1. pre: x in [0, 1]
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto sin_2pi(X x) -> X
{
    assert(all(x >= X{0}) && all(x <= X{1}) && "vsl::kernel::sin_2pi: x in [0, 1]");

    // See work/cheb_sin2pi.py (Fast: degree 10)
    const auto u = x - 0.5f;
    const auto v = u * u;
    auto p = X{};
    if constexpr (P == Precision::Fast) {
        p = X(3.199693680e+00f);
        p = mul_add(p, v, X(-1.407349300e+01f));
        p = mul_add(p, v, X(3.850109863e+01f));
        p = mul_add(p, v, X(-6.707709503e+01f));
        p = mul_add(p, v, X(6.483583832e+01f));
        p = mul_add(p, v, X(-2.513274193e+01f));
    }
    else {
        p = X(-6.151015759e-01f);
        p = mul_add(p, v, X(3.622930765e+00f));
        p = mul_add(p, v, X(-1.417941380e+01f));
        p = mul_add(p, v, X(3.851269913e+01f));
        p = mul_add(p, v, X(-6.707761383e+01f));
        p = mul_add(p, v, X(6.483584595e+01f));
        p = mul_add(p, v, X(-2.513274193e+01f));
    }
    return -u * x * (x - 1.f) * p;
}

/// cos(2 pi x). pre: x in [0, 1]
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto cos_2pi(X x) -> X
{
    assert(all(x >= X{0}) && all(x <= X{1}) && "vsl::kernel::cos_2pi: x in [0, 1]");
    return detail::cos_2pi_half<P>(select(x < 0.5f, x, 1 - x)); // 1 - x is exact here
}

/// tan(pi x). Fast: 0.07 cents up to 20 kHz at 48 kHz (x = fc / fs).
/// Centred, unlike the other turns kernels: fc / fs needs no fold, which would cost ~30%.
/// pre: x in [-1/2, 1/2]
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto tan_pi(X x) -> X
{
    assert(all(x >= X{-0.5f}) && all(x <= X{0.5f}) && "vsl::kernel::tan_pi: x in [-1/2, 1/2]");
    return detail::tan_pi_centred<P>(x);
}

} // namespace kernel

// MARK: - Trig Functions

/// Every float; relative accuracy holds near the zeros for |x| up to ~1e2. 0 for |x| >= 2^24.
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto sin(X x) -> X
{
    using S = scalar_t<X>;
    const auto n = rint(x * S(std::numbers::inv_pi_v<double>));
    const auto y = detail::flip_if_odd(detail::sin_kernel<P>(detail::sub_pi_multiple(x, n)), n);
    return select(detail::radians_in_range(x), y, X{});
}

/// Every float; relative accuracy holds near the zeros for |x| up to ~1e2. 0 for |x| >= 2^24.
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto cos(X x) -> X
{
    // x = (n + 1/2) pi + r puts the zeros at r = 0, and cos x = (-1)^(n + 1) sin r.
    using S = scalar_t<X>;
    const auto k = floor(x * S(std::numbers::inv_pi_v<double>)) + S(0.5f);
    const auto y = detail::flip_if_odd(detail::sin_kernel<P>(detail::sub_pi_multiple(x, k)), k + S(0.5f));
    return select(detail::radians_in_range(x), y, X{});
}

/// Every float; relative accuracy holds near the zeros and poles for |x| up to ~1e2. 0 for |x| >= 2^24.
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto tan(X x) -> X
{
    // The pole distances come from x directly: r rounds away the last bits of pi/2 - r.
    using S = scalar_t<X>;
    const auto n = rint(x * S(std::numbers::inv_pi_v<double>));
    const auto r = detail::sub_pi_multiple(x, n);
    const auto lo = detail::sub_pi_multiple(x, n + S(0.5f));
    const auto hi = detail::sub_pi_multiple(x, n - S(0.5f));
    return select(detail::radians_in_range(x), detail::tan_reduced<P>(r, -lo, hi), X{});
}

/// sin(2 pi x) for every float.
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto sin_2pi(X x) -> X
{
    return detail::sin_2pi_centred<P>(reduce_turns(x));
}

/// cos(2 pi x) for every float.
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto cos_2pi(X x) -> X
{
    return detail::cos_2pi_half<P>(abs(reduce_turns(x)));
}

///
template<Floating_point X>
struct Sin_cos {
    X sin;
    X cos;
};

/// sin(2 pi x) and cos(2 pi x) for every float, with one reduction.
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto sin_cos_2pi(X x) -> Sin_cos<X>
{
    const auto r = reduce_turns(x);
    return {detail::sin_2pi_centred<P>(r), detail::cos_2pi_half<P>(abs(r))};
}

/// tan(pi x) for every float. Fast: 0.07 cents up to 20 kHz at 48 kHz (x = fc / fs).
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto tan_pi(X x) -> X
{
    return detail::tan_pi_centred<P>(reduce_turns(x));
}

// MARK: - Inverse Trig Functions

/// pre: x in [-1, 1]
template<Precision P = Precision::Standard, Floating_point X>
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
    p = mul_add(p, r2, X(4.104170576e-02f));
    p = mul_add(p, r2, X(7.526044548e-02f));
    p = mul_add(p, r2, X(1.666617095e-01f));
    p = mul_add(p, r2, X(1.000000000e+00f));
    const auto asin_val = r * p;
    // big: acos(|x|) = 2 * asin(inner)  [no cancellation near x=1]
    // small: acos(x) = pi/2 - asin(x)   [no cancellation: acos ~ pi/2]
    const auto pos = select(big, 2.f * asin_val, (pi_half_major - asin_val) + pi_half_minor);
    // reflect: acos(-x) = pi - acos(x)
    return select(x < 0.f, (pi_major - pos) + pi_minor, pos);
}

/// pre: x in [-1, 1]
template<Precision P = Precision::Standard, Floating_point X>
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
    p = mul_add(p, r2, X(4.104170576e-02f));
    p = mul_add(p, r2, X(7.526044548e-02f));
    p = mul_add(p, r2, X(1.666617095e-01f));
    p = mul_add(p, r2, X(1.000000000e+00f));
    const auto core = r * p;
    const auto sign = select(x < 0.f, X(-1.f), X(1.f));
    const auto tail = sign * ((pi_half_major - 2.f * core) + pi_half_minor);
    return select(big, tail, core);
}

///
template<Precision P = Precision::Standard, Floating_point X>
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
    p = mul_add(p, r2, X(-1.394065171e-01f));
    p = mul_add(p, r2, X(1.998402327e-01f));
    p = mul_add(p, r2, X(-3.333312869e-01f));
    p = mul_add(p, r2, X(1.000000000e+00f));
    auto val = r * p;
    val = select(big, (pi_quarter_major + val) + pi_quarter_minor, val);
    val = select(inv, (pi_half_major - val) + pi_half_minor, val);
    return sign * val;
}

// MARK: - exp2, log2

namespace detail {

// 2^f, f in [0, 1], exactly 1 at 0. See work/cheb_exp2.py (Fast: degree 5)
template<Precision P, Floating_point X>
force_inline constexpr auto exp2_unit(X f) -> X
{
    auto v = X{};
    if constexpr (P == Precision::Fast) {
        v = X(1.866662991e-03f);
        v = mul_add(v, f, X(9.018208832e-03f));
        v = mul_add(v, f, X(5.579887331e-02f));
        v = mul_add(v, f, X(2.401648164e-01f));
        v = mul_add(v, f, X(6.931512952e-01f));
        v = mul_add(v, f, X(1.000000000e+00f));
    }
    else {
        v = X(2.186578495e-04f);
        v = mul_add(v, f, X(1.239133184e-03f));
        v = mul_add(v, f, X(9.684186429e-03f));
        v = mul_add(v, f, X(5.548062921e-02f));
        v = mul_add(v, f, X(2.402304560e-01f));
        v = mul_add(v, f, X(6.931469440e-01f));
        v = mul_add(v, f, X(1.000000000e+00f));
    }
    return v;
}

// Folds through a small correction term. We need this for pow.
template<Precision P, Floating_point X>
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
    return int_pow * exp2_unit<P>(dec_part);
}

} // namespace detail

///
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto exp2(X x) -> X
{
    return detail::exp2<P>(x, X{});
}

/// log2(0) = -inf, or -150 (one below the smallest subnormal) under VSL_FINITE_MATH.
/// Subnormal x is treated as 0, as under DAZ. pre: x >= 0
template<Precision P = Precision::Standard, Floating_point X>
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
    // eliminating cancellation for x near 1. See work/cheb_log2.py (Fast: degree 6).
    const auto fold  = m >= S(1.5f);
    const auto m_adj = select(fold, m * S(0.5f), m);
#if VSL_FINITE_MATH
    constexpr auto log2_zero = -S(exp_bias + sig_bits);
#else
    constexpr auto log2_zero = -std::numeric_limits<S>::infinity();
#endif
    // Zero and subnormals by their exponent bits: no float compare, so no move between register files,
    // and it's off the polynomial's critical path. x = 0 gives u = 0.
    constexpr auto exp_mask = (one << ieee_exp_bits_v<S>) - one;
    const auto is_zero = ((bits >> sig_bits) & exp_mask) == decltype(bits){};
    auto e_adj = select(fold, signed_to_float(val) + S(1.f), signed_to_float(val));
    e_adj = select(is_zero, X(log2_zero), e_adj);

    const auto u = m_adj - S(1.f);
    auto p = X{};
    if constexpr (P == Precision::Fast) {
        p = X(1.172318384e-01f);
        p = mul_add(p, u, X(-2.406200022e-01f));
        p = mul_add(p, u, X(2.981740236e-01f));
        p = mul_add(p, u, X(-3.617303967e-01f));
        p = mul_add(p, u, X(4.806416035e-01f));
        p = mul_add(p, u, X(-7.213248014e-01f));
        p = mul_add(p, u, X(1.442696214e+00f));
    }
    else {
        p = X(7.639224082e-02f);
        p = mul_add(p, u, X(-1.725005209e-01f));
        p = mul_add(p, u, X(2.160132378e-01f));
        p = mul_add(p, u, X(-2.428877503e-01f));
        p = mul_add(p, u, X(2.882365286e-01f));
        p = mul_add(p, u, X(-3.605715632e-01f));
        p = mul_add(p, u, X(4.809007943e-01f));
        p = mul_add(p, u, X(-7.213485241e-01f));
        p = mul_add(p, u, X(1.442695022e+00f));
    }
    return mul_add(u, p, e_adj);
}

// MARK: - exp, log, pow, etc.

///
template<Precision P = Precision::Standard, Floating_point X>
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
    return two_pow * detail::exp2_unit<P>(r * log2e);
}

/// pre: x > 0
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto log(X x) -> X
{
    using S = scalar_t<X>;
    return std::numbers::ln2_v<S> * log2<P>(x);
}

/// pre: x > 0
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto log10(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto log10_2 = std::numbers::ln2_v<S> / std::numbers::ln10_v<S>;
    return log10_2 * log2<P>(x);
}

/// e^x - 1, accurate near 0 (the one-pole coefficient 1 - e^-w at low cutoffs). -1 below ~-18,
/// inf on overflow.
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto expm1(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto ln_2 = std::numbers::ln2_v<S>;
    constexpr auto log2e = S(std::numbers::log2e_v<double>);
    constexpr auto ln2_hi = S(6.93145752e-01f); // 0x3f317200: 8 trailing zero mantissa bits
    constexpr auto ln2_lo = S(1.42860677e-06f); // ln(2) - ln2_hi
    constexpr auto exp_bias = ieee_exp_bias_v<S>;
    constexpr auto sig_bits = ieee_sig_bits_v<S>;

    // Below -26 ln 2 the result is -1 in float. 2^(n - 1) is built rather than 2^n, and doubled at
    // the end, so n = bias + 1 reaches inf without wrapping the exponent.
#if VSL_FINITE_MATH
    constexpr auto x_hi = S(exp_bias + 1) * S(1 - 0x1p-20) * ln_2;
#else
    constexpr auto x_hi = (S(exp_bias) + S(1.5f)) * ln_2;
#endif
    x = clamp(x, S(-26) * ln_2, x_hi);
    const auto n = rint(x * log2e);
    const auto r = mul_add(n, X(-ln2_lo), mul_add(n, X(-ln2_hi), x));
    const auto f = r * log2e; // in ~[-1/2, 1/2]

    // 2^f - 1 = f Q(f), see work/cheb_expm1.py. With n = 0 this is f Q(f) alone, so small x keeps
    // its relative precision.
    auto q = X(1.545316336e-04f);
    q = mul_add(q, f, X(1.339086331e-03f));
    q = mul_add(q, f, X(9.618083015e-03f));
    q = mul_add(q, f, X(5.550356954e-02f));
    q = mul_add(q, f, X(2.402265072e-01f));
    q = mul_add(q, f, X(6.931471825e-01f));
    const auto bits = (signed_to_unsigned(float_to_signed(n) - 1) + exp_bias) << sig_bits;
    const auto half_pow = reinterpret_as_float(bits);
    return 2 * mul_add(half_pow, f * q, half_pow - S(0.5f)); // 2^n - 1 = 2 (2^(n - 1) - 1/2), exact
}

/// 10^(dB / 20): exactly 1 at 0 dB, 0 at -inf dB.
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto db_to_gain(X db) -> X
{
    using S = scalar_t<X>;
    constexpr auto k = S(0.16609640474436813); // log2(10) / 20
    return exp2<P>(db * k);
}

/// 20 log10(gain): exactly 0 dB at 1, -inf dB at 0. pre: gain >= 0
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto gain_to_db(X gain) -> X
{
    using S = scalar_t<X>;
    constexpr auto k = S(6.0205999132796239); // 20 log10(2)
    return k * log2<P>(gain);
}

/// pre: b > 0, x > 0
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto logB(X b, X x) -> X
{
    return log2<P>(x) / log2<P>(b);
}

/// pre: x >= 0
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto pow(X x, X y) -> X
{
    assert(all(x >= X{0}) && "vsl::pow: x >= 0");
    using S = scalar_t<X>;
    constexpr auto huge = detail::overflow<S>();
    if constexpr (is_vector_v<X>) {
        // lo is y * l's rounding error when mul_add is fused, and 0 otherwise.
        const auto is_zero = x == X{0};
        const auto l = log2<P>(select(is_zero, X{1}, x)); // avoid log2(0) on zero lanes
        const auto hi = y * l;
        const auto lo = mul_add(y, l, -hi);
        const auto zero_pow = select(y < X{0}, X(huge), select(y == X{0}, X{1}, X{}));
        return select(is_zero, zero_pow, detail::exp2<P>(hi, lo));
    }
    else {
        // Widen to double, then round back to float. This gets us down to ~1 ULP.
        using W = double;
        if (x == X{0}) { return y < X{0} ? huge : (y == X{0} ? X{1} : X{0}); }
        const auto l = log2<P>(static_cast<W>(x));
        return static_cast<X>(exp2<P>(l * static_cast<W>(y)));
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
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto cosh(X x) -> X
{
    const auto b = exp<P>(detail::hyperbolic_arg(x));
    const auto br = 1 / (2 * b);
    const auto bh = b / 2;
    return bh + br;
}

/// Taylor below |x| = 1, where b/2 - 1/(2b) cancels.
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto sinh(X x) -> X
{
    const auto b = exp<P>(detail::hyperbolic_arg(x));
    const auto br = 1 / (2 * b);
    const auto bh = b / 2;

    const auto u = x * x;
    auto p = X(2.755731922e-06f);
    p = mul_add(p, u, X(1.984126984e-04f));
    p = mul_add(p, u, X(8.333333333e-03f));
    p = mul_add(p, u, X(1.666666667e-01f));
    const auto small = mul_add(x * u, p, x);
    return select(abs(x) < 1.f, small, bh - br);
}

/// Standard: -120 dB, saturating at exactly 1. Fast: Pade [7/6], -83 dB, saturating at ~0.9999.
/// See work/remez_tanh.py.
template<Precision P = Precision::Standard, Floating_point X>
force_inline constexpr auto tanh(X x) -> X
{
    if constexpr (P == Precision::Fast) {
        // Past 4.79 the rational climbs through 1; the clamp also keeps P and Q from overflowing.
        const auto xc = clamp(x, -4.79f, 4.79f);
        const auto x2 = xc * xc;
        auto p = X(7.400007235e-06f);
        p = mul_add(p, x2, X(2.797202906e-03f));
        p = mul_add(p, x2, X(1.282051355e-01f));
        p = mul_add(p, x2, X(1.000000000e+00f));
        auto q = X(2.072002098e-04f);
        q = mul_add(q, x2, X(2.331002243e-02f));
        q = mul_add(q, x2, X(4.615384638e-01f));
        q = mul_add(q, x2, X(1.000000000e+00f));
        return xc * p / q;
    }
    else {
        // tanh rounds to 1 in float past ~9.01, so the clamp is free and s * s can't overflow.
        // 9.2 lands on exactly 1, fused or not; 9.1 and 9.25 don't.
        const auto xc = clamp(x, -9.2f, 9.2f);
        const auto u = xc * xc;
        auto r = X(3.567437716e-06f);
        r = mul_add(r, u, X(1.918139460e-04f));
        r = mul_add(r, u, X(8.349913172e-03f));
        r = mul_add(r, u, X(1.666557789e-01f));
        r = mul_add(r, u, X(1.000000834e+00f));
        const auto s = xc * r;
        return s / sqrt(mul_add(s, s, X(1)));
    }
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

#if defined(__clang__)
#pragma float_control(pop)
#endif
