#pragma once

#include <algorithm> // min, max, clamp
#include <cassert>
#include <cmath>
#include <numbers>

#include "vsl_core.h"
#include "vsl_logic.h"

namespace vsl {

template<typename X>
force_inline constexpr auto abs(X x) -> X
{
    return select(x >= 0, x, -x);
}

static_assert(abs(-1) == 1);
static_assert(abs(0) == 0);
static_assert(abs(2) == 2);
static_assert(abs(-1.f) == 1.f);
static_assert(abs(0.f) == 0.f);
static_assert(abs(2.f) == 2.f);
static_assert(abs(-1.0) == 1.0);
static_assert(abs(0.0) == 0.0);
static_assert(abs(2.0) == 2.0);

///
template<typename X>
force_inline constexpr auto trunc(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::trunc(x);
    }
    else {
        using S = scalar_t<X>;
        constexpr auto sig_bits = ieee_sig_bits_v<S>;
        constexpr auto one = decltype(sig_bits){1};
        constexpr auto thresh = (one << sig_bits); // Values >= thresh are integers. For a float, this is 8388608.f
        using I = signed_t<X>;
        return (abs(x) >= thresh) ? x : static_cast<X>(static_cast<I>(x));
    }
}

static_assert(trunc(0.f) == 0.f);
static_assert(trunc(3.621f) == 3.f);
static_assert(trunc(5.f) == 5.f);
static_assert(trunc(-1.2f) == -1.f);
static_assert(trunc(1e20f) == 1e20f);
static_assert(trunc(0.0) == 0.0);
static_assert(trunc(3.621) == 3.0);
static_assert(trunc(5.0) == 5.0);
static_assert(trunc(-1.2) == -1.0);
static_assert(trunc(1e20) == 1e20);

///
template<typename X>
force_inline constexpr auto floor(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::floor(x);
    }
    else {
        const auto t = trunc(x);
        return select(x >= 0, t, select(x == t, t, t - 1));
    }
}

static_assert(floor(2.35f) == 2.f);
static_assert(floor(-2.3) == -3.0);
static_assert(floor(5.0) == 5.0);
static_assert(floor(-1.f) == -1.f);

///
template<typename X>
force_inline constexpr auto ceil(X x) -> X 
{
    if constexpr (is_vector_v<X>) {
        return X::ceil(x);
    }
    else {
        const auto f = floor(x);
        return select(x == f, f, f + 1);
    }
}

static_assert(ceil(0.f) == 0.f);
static_assert(ceil(3.25f) == 4.f);
static_assert(ceil(-2.1) == -2.0);
static_assert(ceil(5.f) == 5.f);

///
template<typename X>
force_inline constexpr auto round(X x) -> X 
{
    if constexpr (is_vector_v<X>) {
        return X::round(x);
    }
    else {
        return select(x >= 0, floor(x + 0.5f), ceil(x - 0.5f));
    }
}

static_assert(round(1.5f) == 2.f);
static_assert(round(1.49f) == 1.f);
static_assert(round(-1.5f) == -2.f);
static_assert(round(-2.9f) == -3.f);
static_assert(round(1.5) == 2.0);
static_assert(round(1.49) == 1.0);
static_assert(round(-1.5) == -2.0);
static_assert(round(-2.9) == -3.0);

// MARK: - fmod, wrap

///
template<typename X>
force_inline constexpr auto fmod(X x, X y) -> X
{
    return x - trunc(x / y) * y;
}

static_assert(abs_equal(fmod(5.3, 2.0), 1.3));
static_assert(abs_equal(fmod(18.5f, 4.2f), 1.7f, 1e-6f));

///
template<typename X>
force_inline constexpr auto wrap(X x) -> X
{
    return x - floor(x);
}

static_assert(abs_equal(wrap(1.1), 0.1));
static_assert(abs_equal(wrap(2.3f), 0.3f));

///
template<typename X, typename B>
force_inline constexpr auto wrap(X x, B a, B b) -> X
{
    const auto rect = x - a;
    const auto range = b - a;
    return range * wrap(rect / range) + a; // or fmod(rect, range) + a;
}

static_assert([]() {
    // -pi should wrap to pi on 0..<2 * pi
    constexpr auto pi = std::numbers::pi_v<float>;
    return abs_equal(wrap(-pi, 0.f, 2 * pi), pi);
}());

static_assert([]() {
    // 2 * pi should wrap to 0 on -pi..<pi
    constexpr auto pi = std::numbers::pi_v<float>;
    return abs_equal(wrap(2 * pi, -pi, pi), 0.f);
}());

static_assert([]() {
    // 5 * pi should wrap to pi on 0..<2 * pi
    constexpr auto pi = std::numbers::pi_v<float>;
    return abs_equal(wrap(5 * pi, 0.f, 2 * pi), pi);
}());

// MARK: - Trig Functions

namespace impl {
// Fast wrap, symmetric around zero.
template<typename X, typename S>
force_inline constexpr auto wrap_sym(X x, S k) -> X
{
    // See: https://github.com/Chowdhury-DSP/math_approx/blob/main/include/math_approx/src/trig_approx.hpp
    const auto two_k = 2 * k;
    x += k;
    const auto m = x - two_k * trunc(x * (1 / two_k));
    return select(x >= 0, m, m + two_k) - k;
}
} // namespace impl

///
template<typename X>
force_inline constexpr auto cos(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto pi = std::numbers::pi_v<S>;
    x = impl::wrap_sym(x, pi);

    // Computed with CVXPY. See work/cos_approx.py.
    const auto c0 = S(9.99999990e-01);
    const auto c2 = S(-4.99999924e-01);
    const auto c4 = S(4.16665522e-02);
    const auto c6 = S(-1.38881637e-03);
    const auto c8 = S(2.47784807e-05);
    const auto c10 = S(-2.71696877e-07);
    const auto c12 = S(1.75939307e-09);

    const auto x2 = x * x;
    return c0 + x2 * (c2 + x2 * (c4 + x2 * (c6 + x2 * (c8 + x2 * (c10 + x2 * c12)))));
}

static_assert(abs_equal(cos(0.f), 1.f));

template<typename X>
force_inline constexpr auto cos_v2(X x) -> X
{
    // https://mooooo.ooo/chebyshev-sine-approximation/
    using S = scalar_t<X>;
    constexpr auto pi = std::numbers::pi_v<S>;
    x = impl::wrap_sym(x, pi);

    // [-4.11217365e-09 -1.43526532e-16  6.29230644e-07  2.43059030e-15
    //  -5.68236022e-05 -1.31467246e-14  3.13209043e-03  2.45952282e-14
    //  -9.04463450e-02 -1.06985581e-14  9.54929659e-01]
    const auto c0 = S(9.54929659e-01);
    const auto c2 = S(-9.04463450e-02);
    const auto c4 = S(3.13209043e-03);
    const auto c6 = S(-5.68236022e-05);
    const auto c8 = S(6.29230644e-07);
    const auto c10 = S(-4.11217365e-09);

    const auto x2 = x * x;
    const auto k = S(-0.42441318157); // -4 / (3 * pi)
    const auto scale = k * x2 + pi / 3;
    const auto poly = c0 + x2 * (c2 + x2 * (c4 + x2 * (c6 + x2 * (c8 + x2 * c10))));

    return scale * poly;
}

///
template<typename X>
force_inline constexpr auto sin(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto pi = std::numbers::pi_v<S>;
    x = impl::wrap_sym(x, pi);

    // Coefficients computed with CVXPY. See work/sin_approx.py.
    const auto c1 = S(9.99999699e-01);
    const auto c3 = S(-1.66665244e-01);
    const auto c5 = S(8.33211548e-03);
    const auto c7 = S(-1.98004157e-04);
    const auto c9 = S(2.69047940e-06);
    const auto c11 = S(-2.00016290e-08);

    const auto x2 = x * x;
    return x * (c1 + x2 * (c3 + x2 * (c5 + x2 * (c7 + x2 * (c9 + x2 * c11)))));
}

static_assert(abs_equal(sin(0.f), 0.f));

// Domain: [0, 1] !!!
template<typename X>
force_inline constexpr auto cos_2pi(X x) -> X
{
    using S = scalar_t<X>;

    //assert(all(x >= 0) && all(x <= 1) && "cos_2pi requires 0 <= x <= 1");
    x = select(x < 0.5f, x, x - 1);

    // [ 4.62331203e+01 -1.02599716e-12 -8.26473806e+01  4.56383727e-13
    //   6.46984880e+01 -6.09849071e-14 -1.97310048e+01  2.04709892e-15
    //   9.99931447e-01]
    const auto c0 = S(9.99931447e-01);
    const auto c2 = S(-1.97310048e+01);
    const auto c4 = S(6.46984880e+01);
    const auto c6 = S(-8.26473806e+01);
    const auto c8 = S(4.62331203e+01);

    const auto x2 = x * x;
    return c0 + x2 * (c2 + x2 * (c4 + x2 * (c6 + x2 * c8)));
}

// Domain: [0, 1] !!!
template<typename X>
force_inline constexpr auto sin_2pi(X x) -> X
{
    using S = scalar_t<X>;

    //assert(all(x >= 0) && all(x <= 1) && "sin_2pi requires 0 <= x <= 1");
    x = select(x < 0.5f, x, x - 1);

    // [ 3.18403159e+01  6.09903478e-09 -7.39799267e+01 -3.32049873e-09
    //   8.12847877e+01  5.54247386e-10 -4.13272028e+01 -2.62944449e-11
    //   6.28306165e+00  1.26896905e-16]
    const auto c1 = S(6.28306165e+00);
    const auto c3 = S(-4.13272028e+01);
    const auto c5 = S(8.12847877e+01);
    const auto c7 = S(-7.39799267e+01);
    const auto c9 = S(3.18403159e+01);

    const auto x2 = x * x;

    return x * (c1 + x2 * (c3 + x2 * (c5 + x2 * (c7 + x2 * c9))));
}

///
template<typename X>
force_inline constexpr auto tan(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto half_pi = std::numbers::pi_v<S> / 2;
    x = impl::wrap_sym(x, half_pi);

    // Padé approximant. See work/tan_approx.py
    constexpr auto a1 = S(1);
    constexpr auto a3 = S(-1.37254883e-01);
    constexpr auto a5 = S(3.92156623e-03);
    constexpr auto a7 = S(-2.87293887e-05);
    constexpr auto a9 = S(2.90195033e-08);

    constexpr auto b0 = S(1);
    constexpr auto b2 = S(-4.70588217e-01);
    constexpr auto b4 = S(2.74509718e-02);
    constexpr auto b6 = S(-4.02211733e-04);
    constexpr auto b8 = S(1.30587986e-06);

    const auto x2 = x * x;

    const auto numer = x * (a1 + x2 * (a3 + x2 * (a5 + x2 * (a7 + x2 * a9))));
    const auto denom = b0 + x2 * (b2 + x2 * (b4 + x2 * (b6 + x2 * b8)));

    return numer / denom;
}

static_assert(abs_equal(tan(0.f), 0.f));

// MARK: - Inverse Trig Functions

///
template<typename X>
force_inline constexpr auto asin(X x) -> X
{
    using S = scalar_t<X>;

    // Padé approximant. See work/asin_approx.py
    constexpr auto a1 = S(1);
    constexpr auto a3 = S(-1.27636876);
    constexpr auto a5 = S(0.3993034);
    constexpr auto a7 = S(-0.01937842);
    constexpr auto b0 = S(1);
    constexpr auto b2 = S(-1.44303543);
    constexpr auto b4 = S(0.56480931);
    constexpr auto b6 = S(-0.04992851);

    const auto x2 = x * x;
    
    const auto numer = x * (a1 + x2 * (a3 + x2 * (a5 + x2 * a7)));
    const auto denom = b0 + x2 * (b2 + x2 * (b4 + x2 * b6));
    
    return numer / denom;
}

static_assert(abs_equal(asin(0.f), 0.f));

///
template<typename X>
force_inline constexpr auto acos(X x) -> X
{
    using S = scalar_t<X>;
    return X(std::numbers::pi_v<S> / 2) - asin(x);
}

static_assert(abs_equal(acos(0.f), std::numbers::pi_v<float> / 2));

///
template<typename X>
force_inline constexpr auto atan(X x) -> X
{
    using S = scalar_t<X>;

    // Padé approximant. See work/atan_approx.py
    constexpr auto a1 = S(1);
    constexpr auto a3 = S(1.28205128);
    constexpr auto a5 = S(0.3958042);
    constexpr auto a7 = S(0.01704962);
    constexpr auto b0 = S(1);
    constexpr auto b2 = S(1.61538462);
    constexpr auto b4 = S(0.73426573);
    constexpr auto b6 = S(0.08158508);

    const auto x2 = x * x;
    
    const auto numer = x * (a1 + x2 * (a3 + x2 * (a5 + x2 * a7)));
    const auto denom = b0 + x2 * (b2 + x2 * (b4 + x2 * b6));
    
    return numer / denom;
}

static_assert(abs_equal(atan(0.f), 0.f));

// MARK: - exp2, log2

///
template<typename X>
force_inline constexpr auto exp2(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto exp_bias = ieee_exp_bias_v<S>;
    constexpr auto sig_bits = ieee_sig_bits_v<S>;

    const auto int_part = float_to_signed(floor(x));
    const auto dec_part = x - signed_to_float(int_part);
    const auto val = (signed_to_unsigned(int_part) + exp_bias) << sig_bits;
    const auto int_pow = reinterpret_as_float(val);

    // Minimax on 0...1
    // [0.00187758 0.00898934 0.05582632 0.24015362 0.69315307 0.99999993]
    const auto c0 = S(0.99999993);
    const auto c1 = S(0.69315307);
    const auto c2 = S(0.24015362);
    const auto c3 = S(0.05582632);
    const auto c4 = S(0.00898934);
    const auto c5 = S(0.00187758);
    const auto dec_pow = c0 + dec_part * (c1 + dec_part * (c2 + dec_part * (c3 + dec_part * (c4 + dec_part * c5))));

    return int_pow * dec_pow;
}

static_assert(abs_equal(exp2(-3.f), 1 / 8.f));
static_assert(abs_equal(exp2(-1.f), 0.5f));
static_assert(abs_equal(exp2(0.f), 1.f, 1e-6f));
static_assert(abs_equal(exp2(1.f), 2.f, 1e-6f));
static_assert(abs_equal(exp2(2.f), 4.f, 1e-6f));
static_assert(abs_equal(exp2(3.5f), 11.313708f, 1e-5f));

static_assert(abs_equal(exp2(-3.0), 1 / 8.0));
static_assert(abs_equal(exp2(-1.0), 0.5));
static_assert(abs_equal(exp2(0.0), 1.0));
static_assert(abs_equal(exp2(1.0), 2.0, 1e-6));
static_assert(abs_equal(exp2(2.0), 4.0, 1e-6));
static_assert(abs_equal(exp2(3.5), 11.313708, 1e-5));

///
template<typename X>
force_inline constexpr auto log2(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto exp_bias = ieee_exp_bias_v<S>;
    constexpr auto sig_bits = ieee_sig_bits_v<S>;
    constexpr auto one = decltype(sig_bits){1};
    constexpr auto sig_mask = (one << sig_bits) - one;
    
    const auto bits = reinterpret_as_int(x);
    const auto val = unsigned_to_signed(bits >> sig_bits) - unsigned_to_signed(exp_bias);
    const auto int_part = signed_to_float(val);
    const auto m = reinterpret_as_float((exp_bias << sig_bits) | (bits & sig_mask));

    // Minimax approximation on [1, 2]
    const auto c1 = S(1.44269469);
    const auto c2 = S(-0.72130752);
    const auto c3 = S(0.48003618);
    const auto c4 = S(-0.35326465);
    const auto c5 = S(0.25569301);
    const auto c6 = S(-0.1549388);
    const auto c7 = S(0.06333335);
    const auto c8 = S(-0.01224645);

    const auto xm = m - 1; // polynomial is in x-1
    const auto dec_part = xm * (c1 + xm * (c2 + xm * (c3 + xm * (c4 + xm * (c5 + xm * (c6 + xm * (c7 + xm * c8)))))));

    return int_part + dec_part;
}

static_assert(abs_equal(log2(0.1f), -3.321928f));
static_assert(abs_equal(log2(0.5f), -1.f));
static_assert(abs_equal(log2(1.f), 0.f));
static_assert(abs_equal(log2(2.f), 1.f));
static_assert(abs_equal(log2(8.f), 3.f));
static_assert(abs_equal(log2(69.f), 6.1085244f));

static_assert(abs_equal(log2(0.1), -3.321928));
static_assert(abs_equal(log2(0.5), -1.0));
static_assert(abs_equal(log2(1.0), 0.0));
static_assert(abs_equal(log2(2.0), 1.0));
static_assert(abs_equal(log2(8.0), 3.0));
static_assert(abs_equal(log2(69.0), 6.1085244));

// MARK: - exp, log, pow, etc.

///
template<typename X>
force_inline constexpr auto exp(X x) -> X
{
    using S = scalar_t<X>;

    constexpr auto ln_2 = std::numbers::ln2_v<S>;
    const auto n = floor(x * (1 / ln_2));
    const auto r = x - n * ln_2;

    constexpr auto exp_bias = ieee_exp_bias_v<S>;
    constexpr auto sig_bits = ieee_sig_bits_v<S>;
    const auto arg = signed_to_unsigned(float_to_signed(n));
    const auto bits = (arg + exp_bias) << sig_bits;
    const auto two_pow = reinterpret_as_float(bits);

    // Going to 6th order gives us better results on our sinh/cosh tests.
    // [0.0019568  0.00777467 0.04192972 0.16660428 0.50000694 0.99999972
    // 1.        ]
    const auto c0 = S(1);
    const auto c1 = S(0.99999972);
    const auto c2 = S(0.50000694);
    const auto c3 = S(0.16660428);
    const auto c4 = S(0.04192972);
    const auto c5 = S(0.00777467);
    const auto c6 = S(0.0019568);
    const auto r_pow = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * c6)))));

    return two_pow * r_pow;
}

///
template<typename X>
force_inline constexpr auto log(X x) -> X
{
    using S = scalar_t<X>;
    return std::numbers::ln2_v<S> * log2(x);
}

///
template<typename X>
force_inline constexpr auto log10(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto log10_2 = std::numbers::ln2_v<S> / std::numbers::ln10_v<S>;
    return log10_2 * log2(x);
}

///
template<typename X>
force_inline constexpr auto logB(X b, X x) -> X
{
    return log2(x) / log2(b);
}

// Domain: x >= 0 !!!
template<typename X>
force_inline constexpr auto pow(X x, X y) -> X
{
    //assert(all(x >= 0) && "vsl::pow requires x >= 0!");
    return select(x == 0, X{}, exp2(log2(x) * y)); // Revisit this.
}

static_assert(abs_equal(pow(2.f, 1.5f), 2.828427f, 1e-6f));
static_assert(rel_equal(pow(10.f, -48 / 20.f), 0.00398f));
static_assert(rel_equal(pow(10.f, 36 / 20.f), 63.09573f));

///
template<typename X>
force_inline constexpr auto sqrt(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::sqrt(x);
    }
    else {
        return std::sqrt(x);
    }
}

// MARK: - Hyperbolic Trig Functions

///
template<typename X>
force_inline constexpr auto cosh(X x) -> X
{
    const auto b = exp(x);
    const auto br = 1 / (2 * b);
    const auto bh = b / 2;
    return bh + br;
}

///
template<typename X>
force_inline constexpr auto sinh(X x) -> X
{
    const auto b = exp(x);
    const auto br = 1 / (2 * b);
    const auto bh = b / 2;
    return bh - br;
}

///
template<typename X>
force_inline constexpr auto tanh(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto a1 = S(1);
    constexpr auto a3 = S(5 / 39.0);
    constexpr auto a5 = S(2 / 715.0);
    constexpr auto a7 = S(1 / 135135.0);
    constexpr auto b0 = S(1);
    constexpr auto b2 = S(6 / 13.0);
    constexpr auto b4 = S(10 / 429.0);
    constexpr auto b6 = S(4 / 19305.0);

    const auto x2 = x * x;

    const auto numer = x * (a1 + x2 * (a3 + x2 * (a5 + x2 * a7)));
    const auto denom = b0 + x2 * (b2 + x2 * (b4 + x2 * b6));

    return numer / denom;
}

// MARK: - Other

///
template<typename X, typename C>
force_inline constexpr auto min(X a, C b) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::min(a, X(b));
    }
    else {
        return select(a < b, a, X(b));
    }
}

static_assert(min(-2, 5) == -2);

///
template<typename X, typename C>
force_inline constexpr auto max(X a, C b) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::max(a, X(b));
    }
    else {
        return select(a > b, a, X(b));
    }
}

static_assert(max(-2, 5) == 5);

///
template<typename X, typename C>
force_inline constexpr auto clamp(X x, C a, C b) -> X
{
    return min(max(x, a), b);
}

static_assert(clamp(-2, 0, 1) == 0);
static_assert(clamp(5, 0, 1) == 1);

///
template<typename X>
force_inline constexpr auto sign(X x) -> X
{
    return select(x > 0, X(1), select(x < 0, X(-1), X(0)));
}

static_assert(sign(2.f) == 1);
static_assert(sign(-5) == -1);
static_assert(sign(0.0) == 0);


} // namespace vsl