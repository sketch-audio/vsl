#ifndef _vsl_utils_h
#define _vsl_utils_h

#include "_vsl_core.h"

namespace vsl {

template<typename X>
force_inline constexpr auto elements_equal(X a, X b) -> bool
{
    if constexpr (is_vector_v<X>) {
        using I = __int128_t; // !!!
        static_assert(sizeof(X) == sizeof(I));
        const auto msk = a == b;
        const auto res = std::bit_cast<I>(msk); // Not yet constexpr sadly.
        return res == I(-1);
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
 * @brief counterpart_cast (a simd-compatible static_cast to counterpart type)
 * 
 * @tparam X 
 * @param x 
 * @return counterpart_t<X> 
 */
template<typename X>
force_inline constexpr auto counterpart_cast(X x) -> counterpart_t<X>
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

} // namespace vsl

#endif /* _vsl_utils_h */
