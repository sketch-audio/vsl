#ifndef _vsl_utils_h
#define _vsl_utils_h

#include "_vsl_core.h"

namespace vsl {

template<typename X>
force_inline constexpr auto elements_equal(X a, X b) -> bool
{
    if constexpr (is_vector_v<X>) {
        using I = __int128_t; // !!!
        const auto result = std::bit_cast<I>(a == b); // Not yet constexpr sadly.
        return result == I(-1);
    }
    else {
        return a == b;
    }
}

static_assert(elements_equal(1.f, 1.f));
static_assert(elements_equal(1.f, 2.f) == false);

/**
 * @brief select (a simd-compatible ternary operator)
 *
 * @tparam C
 * @tparam X
 * @param cond
 * @param val_t
 * @param val_f
 * @return X
 */
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

/**
 * @brief 
 * 
 * @tparam X 
 * @param x
 * @return auto
 */
template<Unsigned X>
force_inline constexpr auto unsigned_to_signed(X x)
{
    if constexpr (is_vector_v<X>) {
        using C = us_cvt_t<scalar_t<X>>;
        return simd::convert<C>(x);
    }
    else {
        using C = us_cvt_t<X>;
        return static_cast<C>(x);
    }
}

/**
 * @brief 
 * 
 * @tparam X 
 * @param x 
 * @return auto 
 */
template<Signed X>
force_inline constexpr auto signed_to_unsigned(X x)
{
    if constexpr (is_vector_v<X>) {
        using C = su_cvt_t<scalar_t<X>>;
        return simd::convert<C>(x);
    }
    else {
        using C = su_cvt_t<X>;
        return static_cast<C>(x);
    }
}

/**
 * @brief 
 * 
 * @tparam X 
 * @param x 
 * @return auto 
 */
template<FloatingPoint X>
force_inline constexpr auto float_to_signed(X x)
{
    if constexpr (is_vector_v<X>) {
        using C = counterpart_t<scalar_t<X>>;
        return simd::convert<C>(x);
    }
    else {
        using C = counterpart_t<X>;
        return static_cast<C>(x);
    }
}

/**
 * @brief 
 * 
 * @tparam X 
 * @param x 
 * @return auto 
 */
template<Signed X>
force_inline constexpr auto signed_to_float(X x)
{
    if constexpr (is_vector_v<X>) {
        using C = counterpart_t<scalar_t<X>>;
        return simd::convert<C>(x);
    }
    else {
        using C = counterpart_t<X>;
        return static_cast<C>(x);
    }
}

/**
 * @brief 
 * 
 * @tparam X 
 * @param x 
 * @return auto 
 */
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

/**
 * @brief 
 * 
 * @tparam X 
 * @param x 
 * @return auto 
 */
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

/**
 
 */
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

/**
 
 */
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

/**
 
 */
template<FloatingPoint X>
force_inline auto mask_for_lane(size_t i) -> mask_t<X>
{
    static_assert(is_vector_v<X>, "Can't use mask_for_lane in scalar code.");
    auto mask = mask_t<X>(0);
    mask[i] = -1;
    return mask;
}

} // namespace vsl

#endif /* _vsl_utils_h */
