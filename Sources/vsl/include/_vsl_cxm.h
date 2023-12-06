#ifndef _vsl_cxm_h
#define _vsl_cxm_h

#include <bit>
#include <numbers>

#include "_vsl_core.h"
#include "_vsl_utils.h" // select, etc.

namespace vsl {
namespace cxm {

/**
 * @brief abs
 *
 * @tparam X
 * @param x
 * @return X
 */
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

/**
 * @brief trunc
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto trunc(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto sig_bits = ieee_sig_bits_v<S>;
    constexpr auto one = decltype(sig_bits){1};
    constexpr auto thresh = (one << sig_bits); // Values >= thresh are integers. For a float, this is 8388608.f

    // These casts are OK because sig_bits < the number of bits in our integer counterpart.
    if constexpr (is_vector_v<X>) {
        return select(cxm::abs(x) >= thresh, x, signed_to_float(float_to_signed(x)));
    }
    else {
        using I = int_t<X>;
        return (cxm::abs(x) >= thresh) ? x : static_cast<X>(static_cast<I>(x));
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

/**
 * @brief floor
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto floor(X x) -> X
{
    const auto t = cxm::trunc(x);
    return select(x >= 0, t, select(x == t, t, t - 1));
}

static_assert(floor(2.35f) == 2.f);
static_assert(floor(-2.3) == -3.0);
static_assert(floor(5.0) == 5.0);
static_assert(floor(-1.f) == -1.f);

/**
 * @brief ceil
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto ceil(X x) -> X 
{
    const auto f = cxm::floor(x);
    return select(x == f, f, f + 1);
}

static_assert(ceil(0.f) == 0.f);
static_assert(ceil(3.25f) == 4.f);
static_assert(ceil(-2.1) == -2.0);
static_assert(ceil(5.f) == 5.f);

/**
 * @brief round
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto round(X x) -> X 
{
    return cxm::floor(x + 0.5f);
}

static_assert(round(1.5) == 2.0);
static_assert(round(1.49f) == 1.f);
static_assert(round(-1.5f) == -1.f);
static_assert(round(-2.9) == -3.0);

/**
 * @brief trunc a floating-point type to a same-sized integral type
 *
 * @tparam X
 * @param x
 * @return int_t<X>
 */
template<FloatingPoint X>
force_inline constexpr auto trunc_to_int(X x) -> int_t<X>
{
    return float_to_signed(cxm::trunc(x));
}

/**
 * @brief floor a floating-point type to a same-sized integral type
 *
 * @tparam X
 * @param x
 * @return int_t<X>
 */
template<FloatingPoint X>
force_inline constexpr auto floor_to_int(X x) -> int_t<X>
{
    return float_to_signed(cxm::floor(x));
}

/**
 * @brief ceil a floating-point type to a same-sized integral type
 *
 * @tparam X
 * @param x
 * @return int_t<X>
 */
template<FloatingPoint X>
force_inline constexpr auto ceil_to_int(X x) -> int_t<X>
{
    return float_to_signed(cxm::ceil(x));
}

/**
 * @brief round a floating-point type to a same-sized integral type
 *
 * @tparam X
 * @param x
 * @return int_t<X>
 */
template<FloatingPoint X>
force_inline constexpr auto round_to_int(X x) -> int_t<X>
{
    return float_to_signed(cxm::round(x));
}

/**
 * @brief reinterpret an integral type as the same-sized floating-point type
 *
 * @tparam X
 * @param x
 * @return auto (a floating-point type)
 */
template<Unsigned X>
force_inline constexpr auto reinterpret_as_float(X x)
{
    using F = unsigned_counterpart_t<X>;
    return std::bit_cast<F>(x);
}

/**
 * @brief reinterpret a floating-point type as the same-sized (unsigned) integral type
 *
 * @tparam X
 * @param x
 * @return uint_t<X>
 */
template<FloatingPoint X>
force_inline constexpr auto reinterpret_as_int(X x) -> uint_t<X>
{
    using I = unsigned_counterpart_t<X>;
    return std::bit_cast<I>(x);
}

/**
 * @brief about_equal
 *
 * @tparam X
 * @param a
 * @param b
 * @param thr
 * @return auto (bool in scalar case, mask_t in vector case)
 */
template<typename X>
force_inline constexpr auto about_equal(X a, X b, X tol = 1e-6f)
{
    return cxm::abs(a - b) < tol;
}

static_assert(about_equal(1.f, 1 + 1e-7f));
static_assert(about_equal(1.f, 1 + 1e-5f) == false);
static_assert(about_equal(1.0, 1 + 1e-7));
static_assert(about_equal(1.0, 1 + 1e-5) == false);

/**
 * @brief fmod
 *
 * @tparam X
 * @param x
 * @param y
 * @return X
 */
template<typename X>
force_inline constexpr auto fmod(X x, X y) -> X
{
    return x - cxm::trunc(x / y) * y;
}

static_assert(about_equal(fmod(5.3, 2.0), 1.3));
static_assert(about_equal(fmod(18.5f, 4.2f), 1.7f));

/**
 * @brief wrap (0..<1)
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto wrap(X x) -> X
{
    return x - cxm::floor(x);
}

static_assert(about_equal(wrap(1.1), 0.1));
static_assert(about_equal(wrap(2.3f), 0.3f));

/**
 * @brief wrap (a..<b)
 *
 * @tparam X
 * @tparam B
 * @param x
 * @param a
 * @param b
 * @return X
 */
template<typename X, typename B>
force_inline constexpr auto wrap(X x, B a, B b) -> X
{
    const auto rect = x - a;
    const auto range = b - a;
    return range * cxm::wrap(rect / range) + a; // or cxm::fmod(rect, range) + a;
}

static_assert([]() {
    // -pi should wrap to pi on 0..<2 * pi
    constexpr auto pi = std::numbers::pi_v<float>;
    return about_equal(wrap(-pi, 0.f, 2 * pi), pi);
}());

static_assert([]() {
    // 2 * pi should wrap to 0 on -pi..<pi
    constexpr auto pi = std::numbers::pi_v<float>;
    return about_equal(wrap(2 * pi, -pi, pi), 0.f);
}());

static_assert([]() {
    // 5 * pi should wrap to pi on 0..<2 * pi
    constexpr auto pi = std::numbers::pi_v<float>;
    return about_equal(wrap(5 * pi, 0.f, 2 * pi), pi);
}());

/**
 * @brief cos (approx on -pi...pi)
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto cos(X x) -> X
{
    constexpr auto c0 = X(9.99999991e-01);
    constexpr auto c2 = X(-4.99999934e-01);
    constexpr auto c4 = X(4.16665646e-02);
    constexpr auto c6 = X(-1.38882256e-03);
    constexpr auto c8 = X(2.47799311e-05);
    constexpr auto c10 = X(-2.71853320e-07);
    constexpr auto c12 = X(1.76564052e-09);

    const auto x2 = x * x;

    return c0 + x2 * (c2 + x2 * (c4 + x2 * (c6 + x2 * (c8 + x2 * (c10 + x2 * c12)))));
}

static_assert(about_equal(cos(0.f), 1.f));

/**
 * @brief sin (approx on -pi...pi)
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto sin(X x) -> X
{
    constexpr auto c1 = X(9.99999737e-01);
    constexpr auto c3 = X(-1.66665387e-01);
    constexpr auto c5 = X(8.33221031e-03);
    constexpr auto c7 = X(-1.98027220e-04);
    constexpr auto c9 = X(2.69284266e-06);
    constexpr auto c11 = X(-2.00882849e-08);

    const auto x2 = x * x;

    return x * (c1 + x2 * (c3 + x2 * (c5 + x2 * (c7 + x2 * (c9 + x2 * c11)))));
}

static_assert(about_equal(sin(0.f), 0.f));

/**
 * @brief tan (approx on -pi/2...pi/2)
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto tan(X x) -> X
{
    constexpr auto a1 = X(1);
    constexpr auto a3 = X(-5 / 39.0);
    constexpr auto a5 = X(2 / 715.0);
    constexpr auto a7 = X(-1 / 135135.0);
    constexpr auto b0 = X(1);
    constexpr auto b2 = X(-6 / 13.0);
    constexpr auto b4 = X(10 / 429.0);
    constexpr auto b6 = X(-4 / 19305.0);

    const auto x2 = x * x;

    const auto numer = x * (a1 + x2 * (a3 + x2 * (a5 + x2 * a7)));
    const auto denom = b0 + x2 * (b2 + x2 * (b4 + x2 * b6));

    return numer / denom;
}

static_assert(about_equal(tan(0.f), 0.f));

/**
 * @brief cosh (approx)
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto cosh(X x) -> X
{
    constexpr auto a0 = X(1);
    constexpr auto a2 = X(3665 / 7788.0);
    constexpr auto a4 = X(711 / 25960.0);
    constexpr auto a6 = X(301 / 808396.0);
    constexpr auto b0 = X(1);
    constexpr auto b2 = X(-229 / 7788.0);
    constexpr auto b4 = X(1 / 2360.0);
    constexpr auto b6 = X(-1 / 309067.0);

    const auto x2 = x * x;

    const auto numer = a0 + x2 * (a2 + x2 * (a4 + x2 * a6));
    const auto denom = b0 + x2 * (b2 + x2 * (b4 + x2 * b6));

    return numer / denom;
}

/**
 * @brief sinh (approx)
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto sinh(X x) -> X
{
    constexpr auto a1 = X(1);
    constexpr auto a3 = X(29593 / 207636.0);
    constexpr auto a5 = X(1911 / 416747.0);
    constexpr auto a7 = X(13 / 312254.0);
    constexpr auto b0 = X(1);
    constexpr auto b2 = X(-1671 / 69212.0);
    constexpr auto b4 = X(97 / 351384.0);
    constexpr auto b6 = X(-1 / 626945.0);

    const auto x2 = x * x;

    const auto numer = x * (a1 + x2 * (a3 + x2 * (a5 + x2 * a7)));
    const auto denom = b0 + x2 * (b2 + x2 * (b4 + x2 * b6));

    return numer / denom;
}

/**
 * @brief tanh (approx)
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto tanh(X x) -> X
{
    constexpr auto a1 = X(1);
    constexpr auto a3 = X(5 / 39.0);
    constexpr auto a5 = X(2 / 715.0);
    constexpr auto a7 = X(1 / 135135.0);
    constexpr auto b0 = X(1);
    constexpr auto b2 = X(6 / 13.0);
    constexpr auto b4 = X(10 / 429.0);
    constexpr auto b6 = X(4 / 19305.0);

    const auto x2 = x * x;

    const auto numer = x * (a1 + x2 * (a3 + x2 * (a5 + x2 * a7)));
    const auto denom = b0 + x2 * (b2 + x2 * (b4 + x2 * b6));

    return numer / denom;
}

/**
 * @brief exp2 (approx)
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto exp2(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto exp_bias = ieee_exp_bias_v<S>;
    constexpr auto sig_bits = ieee_sig_bits_v<S>;

    const auto int_part = cxm::round_to_int(x);
    const auto dec_part = x - signed_to_float(int_part);
    
    // In C++20 we have two's complement.
    const auto val = (signed_to_unsigned(int_part) + exp_bias) << sig_bits;
    const auto int_pow = cxm::reinterpret_as_float(val);
    
    // minimax approximation of exp2 on -0.5...0.5
    // computed with cvxpy
    constexpr auto c0 = X(1.00000007);
    constexpr auto c1 = X(0.69314697);
    constexpr auto c2 = X(0.2402212);
    constexpr auto c3 = X(0.05550713);
    constexpr auto c4 = X(0.00967553);
    constexpr auto c5 = X(0.00132767);
    
    const auto dec_pow = c0 + dec_part * (c1 + dec_part * (c2 + dec_part * (c3 + dec_part * (c4 + dec_part * c5))));

    return int_pow * dec_pow;
}

static_assert(about_equal(exp2(-3.f), 1 / 8.f));
static_assert(about_equal(exp2(-1.f), 0.5f));
static_assert(about_equal(exp2(0.f), 1.f));
static_assert(about_equal(exp2(1.f), 2.f));
static_assert(about_equal(exp2(2.f), 4.f));
static_assert(about_equal(exp2(3.5f), 11.313708f));

static_assert(about_equal(exp2(-3.0), 1 / 8.0));
static_assert(about_equal(exp2(-1.0), 0.5));
static_assert(about_equal(exp2(0.0), 1.0));
static_assert(about_equal(exp2(1.0), 2.0));
static_assert(about_equal(exp2(2.0), 4.0));
static_assert(about_equal(exp2(3.5), 11.313708));

/**
 * @brief log2 (approx)
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto log2(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto exp_bias = ieee_exp_bias_v<S>;
    constexpr auto sig_bits = ieee_sig_bits_v<S>;
    constexpr auto one = decltype(sig_bits){1};
    constexpr auto sig_mask = (one << sig_bits) - one;
    
    // In C++ 20 we have two's complement.
    const auto bits = cxm::reinterpret_as_int(x);
    const auto val = unsigned_to_signed((bits >> sig_bits) - exp_bias);
    const auto int_part = signed_to_float(val);
    const auto m = cxm::reinterpret_as_float((exp_bias << sig_bits) | (bits & sig_mask));

    // Compute quadratic approximation of log2 on 1...2.
    // see: https://tech.ebayinc.com/engineering/fast-approximate-logarithms-part-i-the-basics/
    // const auto dec_part = X(-1 / 3.0) * m * m + 2 * m - X(5 / 3.0);
    
    // minimax approximation
    // computed using cvxpy
    constexpr auto c1 = X(1.44268127);
    constexpr auto c2 = X(-0.72039364);
    constexpr auto c3 = X(0.46899335);
    constexpr auto c4 = X(-0.30262538);
    constexpr auto c5 = X(0.1456237);
    constexpr auto c6 = X(-0.03428757);

    const auto xm = m - 1; // polynomial is in x-1
    const auto dec_part = xm * (c1 + xm * (c2 + xm * (c3 + xm * (c4 + xm * (c5 + xm * c6)))));

    return int_part + dec_part;
}

static_assert(about_equal(log2(0.1f), -3.321928f, 1e-5f));
static_assert(about_equal(log2(0.5f), -1.f));
static_assert(about_equal(log2(1.f), 0.f));
static_assert(about_equal(log2(2.f), 1.f));
static_assert(about_equal(log2(8.f), 3.f));
static_assert(about_equal(log2(69.f), 6.108524f, 1e-5f));

static_assert(about_equal(log2(0.1), -3.321928, 1e-5));
static_assert(about_equal(log2(0.5), -1.0));
static_assert(about_equal(log2(1.0), 0.0));
static_assert(about_equal(log2(2.0), 1.0));
static_assert(about_equal(log2(8.0), 3.0));
static_assert(about_equal(log2(69.0), 6.108524, 1e-5));

/**
 * @brief exp (approx)
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto exp(X x) -> X
{
    using S = scalar_t<X>;
    return cxm::exp2(std::numbers::log2e_v<S> *x);
}

/**
 * @brief log (approx)
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto log(X x) -> X
{
    using S = scalar_t<X>;
    return std::numbers::ln2_v<S> *cxm::log2(x);
}

/**
 * @brief log10 (approx)
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto log10(X x) -> X
{
    using S = scalar_t<X>;
    constexpr auto log10_2 = std::numbers::ln2_v<S> / std::numbers::ln10_v<S>;
    return log10_2 * cxm::log2(x);
}

/**
 * @brief logB (approx)
 *
 * @tparam X
 * @param b
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto logB(X b, X x) -> X
{
    return cxm::log2(x) / cxm::log2(b);
}

/**
 * @brief pow (approx)
 *
 * @tparam X
 * @param x
 * @param y
 * @return X
 */
template<typename X>
force_inline constexpr auto pow(X x, X y) -> X
{
    return cxm::exp2(cxm::log2(x) * y);
}

static_assert(about_equal(pow(2.f, 1.5f), 2.828427f));

/**
 * @brief min
 *
 * @tparam X
 * @tparam C
 * @param a
 * @param b
 * @return X
 */
template<typename X, typename C>
force_inline constexpr auto min(X a, C b) -> X
{
    return select(a < b, a, X(b));
}

static_assert(min(-2, 5) == -2);

/**
 * @brief max
 *
 * @tparam X
 * @tparam C
 * @param a
 * @param b
 * @return X
 */
template<typename X, typename C>
force_inline constexpr auto max(X a, C b) -> X
{
    return select(a > b, a, X(b));
}

static_assert(max(-2, 5) == 5);

/**
 * @brief clamp
 *
 * @tparam X
 * @tparam C
 * @param x
 * @param a
 * @param b
 * @return X
 */
template<typename X, typename C>
force_inline constexpr auto clamp(X x, C a, C b) -> X
{
    return cxm::min(cxm::max(x, a), b);
}

static_assert(clamp(-2, 0, 1) == 0);
static_assert(clamp(5, 0, 1) == 1);

/**
 * @brief sign (maybe not the fastest, but hey)
 *
 * @tparam X
 * @param x
 * @return X
 */
template<typename X>
force_inline constexpr auto sign(X x) -> X
{
    return select(x > 0, X(1), select(x < 0, X(-1), X(0)));
}

static_assert(sign(2.f) == 1);
static_assert(sign(-5) == -1);
static_assert(sign(0.0) == 0);

} // namespace cxm
} // namespace vsl

#endif /* _vsl_cxm_h */
