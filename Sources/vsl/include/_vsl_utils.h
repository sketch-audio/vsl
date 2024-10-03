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

/// About equal
template<typename X>
force_inline constexpr auto about_equal(X a, X b, X tol = 1e-6f)
{
    return (a - b) < tol & (b - a) < tol; // We don't have `abs` yet.
}

static_assert(about_equal(1.f, 1 + 1e-7f));
static_assert(!about_equal(1.f, 1 + 1e-5f));
static_assert(about_equal(1.0, 1 + 1e-7));
static_assert(!about_equal(1.0, 1 + 1e-5));


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
template<FloatingPoint X>
force_inline auto mask_for_lane(size_t i) -> mask_t<X>
{
    static_assert(is_vector_v<X>, "Can't use mask_for_lane in scalar code.");
    auto mask = mask_t<X>(0);
    mask[i % num_members_v<X>] = -1;
    return mask;
}

} // namespace vsl

#endif /* _vsl_utils_h */
