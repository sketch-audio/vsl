#pragma once

#include <algorithm> // min, max, clamp
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
    using S = scalar_t<X>;
    constexpr auto sig_bits = ieee_sig_bits_v<S>;
    constexpr auto one = decltype(sig_bits){1};
    constexpr auto thresh = (one << sig_bits); // Values >= thresh are integers. For a float, this is 8388608.f

    // These casts are OK because sig_bits < the number of bits in our integer counterpart.
    if constexpr (is_vector_v<X>) {
        return select(abs(x) >= thresh, x, signed_to_float(float_to_signed(x)));
    }
    else {
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
    const auto t = trunc(x);
    return select(x >= 0, t, select(x == t, t, t - 1));
}

static_assert(floor(2.35f) == 2.f);
static_assert(floor(-2.3) == -3.0);
static_assert(floor(5.0) == 5.0);
static_assert(floor(-1.f) == -1.f);

///
template<typename X>
force_inline constexpr auto ceil(X x) -> X 
{
    const auto f = floor(x);
    return select(x == f, f, f + 1);
}

static_assert(ceil(0.f) == 0.f);
static_assert(ceil(3.25f) == 4.f);
static_assert(ceil(-2.1) == -2.0);
static_assert(ceil(5.f) == 5.f);

///
template<typename X>
force_inline constexpr auto round(X x) -> X 
{
    return vsl::select(x >= 0, floor(x + 0.5f), ceil(x - 0.5f));
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

    // We lose some absolute accuracy here for large x (as opposed to round), but this helps us stay competitive with std::exp2 on performance.
    const auto int_part = float_to_signed(floor(x));
    const auto dec_part = x - signed_to_float(int_part);

    // In C++20 we have two's complement.
    const auto val = (int_part + exp_bias) << sig_bits;
    const auto int_pow = reinterpret_as_float(val);
    
    // minimax approximation of exp2 on -0.5...0.5
    // computed with cvxpy
    // const auto c0 = S(1.00000007);
    // const auto c1 = S(0.69314697);
    // const auto c2 = S(0.2402212);
    // const auto c3 = S(0.05550713);
    // const auto c4 = S(0.00967553);
    // const auto c5 = S(0.00132767);
    // const auto dec_pow = c0 + dec_part * (c1 + dec_part * (c2 + dec_part * (c3 + dec_part * (c4 + dec_part * c5))));

    //[1.53454060e-04 1.33997989e-03 9.61848889e-03 5.55032909e-02
    // 2.40226469e-01 6.93147206e-01 1.00000000e+00]
    const auto c0 = S(1);
    const auto c1 = S(6.93147206e-01);
    const auto c2 = S(2.40226469e-01);
    const auto c3 = S(5.55032909e-02);
    const auto c4 = S(9.61848889e-03);
    const auto c5 = S(1.33997989e-03);
    const auto c6 = S(1.53454060e-04);
    const auto dec_pow = c0 + dec_part * (c1 + dec_part * (c2 + dec_part * (c3 + dec_part * (c4 + dec_part * (c5 + dec_part * c6)))));

    // [1.51959080e-05 1.54650965e-04 1.33339126e-03 9.61804085e-03
    //  5.55041042e-02 2.40226511e-01 6.93147181e-01 1.00000000e+00]
    // const auto c0 = S(1);
    // const auto c1 = S(6.93147181e-01);
    // const auto c2 = S(2.40226511e-01);
    // const auto c3 = S(5.55041042e-02);
    // const auto c4 = S(9.61804085e-03);
    // const auto c5 = S(1.33339126e-03);
    // const auto c6 = S(1.54650965e-04);
    // const auto c7 = S(1.51959080e-05);
    // const auto dec_pow = c0 + dec_part * (c1 + dec_part * (c2 + dec_part * (c3 + dec_part * (c4 + dec_part * (c5 + dec_part * (c6 + dec_part * c7))))));

    return int_pow * dec_pow;
}

static_assert(abs_equal(exp2(-3.f), 1 / 8.f));
static_assert(abs_equal(exp2(-1.f), 0.5f));
static_assert(abs_equal(exp2(0.f), 1.f, 1e-6f));
static_assert(abs_equal(exp2(1.f), 2.f, 1e-6f));
static_assert(abs_equal(exp2(2.f), 4.f, 1e-6f));
static_assert(abs_equal(exp2(3.5f), 11.313708f, 1e-6f));

static_assert(abs_equal(exp2(-3.0), 1 / 8.0));
static_assert(abs_equal(exp2(-1.0), 0.5));
static_assert(abs_equal(exp2(0.0), 1.0));
static_assert(abs_equal(exp2(1.0), 2.0, 1e-6));
static_assert(abs_equal(exp2(2.0), 4.0, 1e-6));
static_assert(abs_equal(exp2(3.5), 11.313708, 1e-6));

///
template<typename X>
force_inline constexpr auto log2(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto exp_bias = ieee_exp_bias_v<S>;
    constexpr auto sig_bits = ieee_sig_bits_v<S>;
    constexpr auto one = decltype(sig_bits){1};
    constexpr auto sig_mask = (one << sig_bits) - one;
    
    // In C++ 20 we have two's complement.
    const auto bits = reinterpret_as_int(x);
    const auto val = unsigned_to_signed(bits >> sig_bits) - unsigned_to_signed(exp_bias);
    const auto int_part = signed_to_float(val);
    const auto m = reinterpret_as_float((exp_bias << sig_bits) | (bits & sig_mask));

    // Compute quadratic approximation of log2 on 1...2.
    // see: https://tech.ebayinc.com/engineering/fast-approximate-logarithms-part-i-the-basics/
    // const auto dec_part = X(-1 / 3.0) * m * m + 2 * m - X(5 / 3.0);
    
    // minimax approximation
    // computed using cvxpy
    // (log2 on 1...2)
    // const auto c1 = S(1.44268127);
    // const auto c2 = S(-0.72039364);
    // const auto c3 = S(0.46899335);
    // const auto c4 = S(-0.30262538);
    // const auto c5 = S(0.1456237);
    // const auto c6 = S(-0.03428757);

    const auto c1 = S(1.44269469);
    const auto c2 = S(-0.72130752);
    const auto c3 = S(0.48003618);
    const auto c4 = S(-0.35326465);
    const auto c5 = S(0.25569301);
    const auto c6 = S(-0.1549388);
    const auto c7 = S(0.06333335);
    const auto c8 = S(-0.01224645);

    const auto xm = m - 1; // polynomial is in x-1
    // const auto dec_part = xm * (c1 + xm * (c2 + xm * (c3 + xm * (c4 + xm * (c5 + xm * c6)))));
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
    constexpr auto ln_2_inv = 1 / ln_2;
    const auto n = round(x * ln_2_inv);
    const auto r = x - n * ln_2;

    // exp(x) on [-ln(2)/2, ln(2)/2]
    // [0.00138365 0.00837473 0.04166823 0.16666421 0.49999992 1.00000004
    //  1.        ]
    // const auto c0 = S(1);
    // const auto c1 = S(1.00000004);
    // const auto c2 = S(0.49999992);
    // const auto c3 = S(0.16666421);
    // const auto c4 = S(0.04166823);
    // const auto c5 = S(0.00837473);
    // const auto c6 = S(0.00138365);
    //const auto r_pow = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * c6)))));

    // [1.97673489e-04 1.39444012e-03 8.33355487e-03 4.16662843e-02
    //  1.66666653e-01 5.00000008e-01 1.00000000e+00 1.00000000e+00]
    const auto c0 = S(1);
    const auto c1 = S(1);
    const auto c2 = S(5.00000008e-01);
    const auto c3 = S(1.66666653e-01);
    const auto c4 = S(4.16662843e-02);
    const auto c5 = S(8.33355487e-03);
    const auto c6 = S(1.39444012e-03);
    const auto c7 = S(1.97673489e-04);
    const auto r_pow = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * (c6 + r * c7))))));

    return exp2(n) * r_pow;
    //return exp2(std::numbers::log2e_v<S> *x);
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

///
template<typename X>
force_inline constexpr auto pow(X x, X y) -> X
{
    return vsl::select(x <= 0, X{}, exp2(log2(x) * y)); // Revisit this.
}

static_assert(abs_equal(pow(2.f, 1.5f), 2.828427f, 1e-6f));
static_assert(rel_equal(pow(10.f, -48 / 20.f), 0.00398f));
static_assert(rel_equal(pow(10.f, 36 / 20.f), 63.09573f));

// MARK: - Hyperbolic Trig Functions

///
template<typename X>
force_inline constexpr auto cosh(X x) -> X
{
    const auto b = exp(x);
    const auto br = 1 / (2 * b);
    const auto bh = b / 2;
    return bh + br;
    // using S = scalar_t<X>;
    // constexpr auto a0 = S(1);
    // constexpr auto a2 = S(3665 / 7788.0);
    // constexpr auto a4 = S(711 / 25960.0);
    // constexpr auto a6 = S(301 / 808396.0);
    // constexpr auto b0 = S(1);
    // constexpr auto b2 = S(-229 / 7788.0);
    // constexpr auto b4 = S(1 / 2360.0);
    // constexpr auto b6 = S(-1 / 309067.0);

    // const auto x2 = x * x;

    // const auto numer = a0 + x2 * (a2 + x2 * (a4 + x2 * a6));
    // const auto denom = b0 + x2 * (b2 + x2 * (b4 + x2 * b6));

    // return numer / denom;
}

///
template<typename X>
force_inline constexpr auto sinh(X x) -> X
{
    const auto b = exp(x);
    const auto br = 1 / (2 * b);
    const auto bh = b / 2;
    return bh - br;
    // using S = scalar_t<X>;
    // constexpr auto a1 = S(1);
    // constexpr auto a3 = S(29593 / 207636.0);
    // constexpr auto a5 = S(1911 / 416747.0);
    // constexpr auto a7 = S(13 / 312254.0);
    // constexpr auto b0 = S(1);
    // constexpr auto b2 = S(-1671 / 69212.0);
    // constexpr auto b4 = S(97 / 351384.0);
    // constexpr auto b6 = S(-1 / 626945.0);

    // const auto x2 = x * x;

    // const auto numer = x * (a1 + x2 * (a3 + x2 * (a5 + x2 * a7)));
    // const auto denom = b0 + x2 * (b2 + x2 * (b4 + x2 * b6));

    // return numer / denom;
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
    return select(a < b, a, X(b));
}

static_assert(min(-2, 5) == -2);

///
template<typename X, typename C>
force_inline constexpr auto max(X a, C b) -> X
{
    return select(a > b, a, X(b));
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