#ifndef _vsl_utils_h
#define _vsl_utils_h

#include <bit>

#include <simd/simd.h>

#include "_vsl_core.h"

namespace vsl {

// MARK: - Logic

/// A simd-compatible ternary operator.
template<typename C, typename X>
force_inline constexpr auto select(C cond, X val_t, X val_f) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::bitselect(val_f, val_t, cond);
    }
    else {
        return cond ? val_t : val_f;
    }
}

static_assert(select(true, 1.f, 0.f) == 1.f);
static_assert(select(false, 1.f, 0.f) == 0.f);

/// Any
template<typename C>
force_inline constexpr auto any(C cond) -> bool
{
    if constexpr (is_vector_v<C>) {
        return simd::any(cond);
    }
    else {
        return cond;
    }
}

static_assert(any(true));
static_assert(!any(false));

/// All
template<typename C>
force_inline constexpr auto all(C cond) -> bool
{
    if constexpr (is_vector_v<C>) {
        return simd::all(cond);
    }
    else {
        return cond;
    }
}

static_assert(all(true));
static_assert(!all(false));

// MARK: - Casts

///
template<Unsigned X>
force_inline constexpr auto unsigned_to_signed(X x)
{
    if constexpr (is_vector_v<X>) {
        using C = signed_eq_t<scalar_t<X>>;
        return simd::convert<C>(x);
    }
    else {
        using C = signed_eq_t<X>;
        return static_cast<C>(x);
    }
}

///
template<Signed X>
force_inline constexpr auto signed_to_unsigned(X x)
{
    if constexpr (is_vector_v<X>) {
        using C = unsigned_eq_t<scalar_t<X>>;
        return simd::convert<C>(x);
    }
    else {
        using C = unsigned_eq_t<X>;
        return static_cast<C>(x);
    }
}

///
template<FloatingPoint X>
force_inline constexpr auto float_to_signed(X x)
{
    if constexpr (is_vector_v<X>) {
        using C = signed_counterpart_t<scalar_t<X>>;
        return simd::convert<C>(x);
    }
    else {
        using C = signed_counterpart_t<X>;
        return static_cast<C>(x);
    }
}

///
template<Signed X>
force_inline constexpr auto signed_to_float(X x)
{
    if constexpr (is_vector_v<X>) {
        using C = signed_counterpart_t<scalar_t<X>>;
        return simd::convert<C>(x);
    }
    else {
        using C = signed_counterpart_t<X>;
        return static_cast<C>(x);
    }
}

///
template<FloatingPoint X>
force_inline constexpr auto float_to_unsigned(X x)
{
    if constexpr (is_vector_v<X>) {
        using C = unsigned_counterpart_t<scalar_t<X>>;
        return simd::convert<C>(x);
    }
    else {
        using C = unsigned_counterpart_t<X>;
        return static_cast<C>(x);
    }
}

///
template<Unsigned X>
force_inline constexpr auto unsigned_to_float(X x)
{
    if constexpr (is_vector_v<X>) {
        using C = unsigned_counterpart_t<scalar_t<X>>;
        return simd::convert<C>(x);
    }
    else {
        using C = unsigned_counterpart_t<X>;
        return static_cast<C>(x);
    }
}

///
template<Unsigned X>
force_inline constexpr auto reinterpret_as_float(X x)
{
    using F = unsigned_counterpart_t<X>;
    return std::bit_cast<F>(x);
}

///
template<FloatingPoint X>
force_inline constexpr auto reinterpret_as_int(X x) -> uint_t<X>
{
    using I = unsigned_counterpart_t<X>;
    return std::bit_cast<I>(x);
}

// MARK: - Equality

/// Are two numbers
template<typename X>
force_inline constexpr auto abs_equal(X a, X b, X tol = 1e-7f)
{
    return (a - b) < tol & (b - a) < tol; // We don't have `abs` yet.
}

static_assert(abs_equal(1.f, 1 + 1e-8f));
static_assert(!abs_equal(1.f, 1 + 1e-5f));
static_assert(abs_equal(1.0, 1 + 1e-8));
static_assert(!abs_equal(1.0, 1 + 1e-5));

///
template<typename X>
force_inline constexpr auto rel_equal(X a, X b, X tol = 1e-3f)
{
    return ((a - b) / b) < tol & ((b - a) / b) < tol;
}

static_assert(rel_equal(1.f, 1 + 1e-4f));
static_assert(!rel_equal(1.f, 1 + 1e-2f));
static_assert(rel_equal(100.1f, 100.f));
static_assert(!rel_equal(101.f, 100.f));

///
template<Scalar X>
force_inline constexpr auto bit_equal(X a, X b, uint_t<X> tol = 5) -> bool
{
    constexpr auto exp_bits = ieee_exp_bits_v<X>;
    constexpr auto exp_bias = ieee_exp_bias_v<X>;
    constexpr auto sig_bits = ieee_sig_bits_v<X>;
    constexpr auto one = decltype(sig_bits){1};
    constexpr auto sig_mask = (one << sig_bits) - one;
    
    const auto a_bits = reinterpret_as_int(a);
    const auto b_bits = reinterpret_as_int(b);
    
    const auto a_exp = (a_bits >> sig_bits) - exp_bias;
    const auto b_exp = (b_bits >> sig_bits) - exp_bias;
    
    if (a_exp != b_exp) {
        return false;
    }
    
    const auto a_sig = a_bits & sig_mask;
    const auto b_sig = b_bits & sig_mask;
    
    const auto matching_bits = std::countl_zero((a_sig ^ b_sig) << (exp_bits + 1));
    
    return matching_bits >= tol;
}

static_assert(bit_equal(1.f, 1.01f, 6));
static_assert(bit_equal(1.f, 1.1f, 3));
static_assert(!bit_equal(200.f, 201.f, 7));
static_assert(!bit_equal(200.f, 202.f, 6));

// MARK: - Masks

///
template<typename X>
force_inline auto mask_to_bool(X x)
{
    if constexpr (is_vector_v<X>) {
        return x;
    }
    else {
        return x == -1;
    }
}

///
template<typename X>
force_inline auto bool_to_mask(X x)
{
    if constexpr (is_vector_v<X>) {
        return x;
    }
    else {
        return x ? -1 : 0;
    }
}

///
template<typename X>
force_inline auto mask_for_member(size_t i) -> mask_t<X>
{
    if constexpr (is_vector_v<X>) {
        auto mask = mask_t<X>(0);
        mask[i % num_members_v<X>] = -1;
        return mask;
    } else {
        return -1;
    }
}

// MARK: - Members

///
template<typename X>
force_inline auto get_member(X x, size_t i) -> scalar_t<X>
{
    if constexpr (is_vector_v<X>) {
        return x[i % num_members_v<X>];
    }
    else {
        return x;
    }
}

// MARK: - Conversions

///
template<typename X>
force_inline auto reduce_add(X x) -> scalar_t<X>
{
    if constexpr (is_vector_v<X>) {
        return simd::reduce_add(x);
    }
    else {
        return x;
    }
}



} // namespace vsl

#endif /* _vsl_utils_h */
