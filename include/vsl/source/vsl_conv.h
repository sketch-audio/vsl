#pragma once

#include <bit> // bit_cast

#include "vsl_core.h"

namespace vsl {

template<typename X>
force_inline constexpr auto mask_to_bool(X x)
{
    if constexpr (is_vector_v<X>) {
        return x;
    }
    else {
        return x == true_mask_v<X>;
    }
}

template<typename X>
force_inline constexpr auto bool_to_mask(X x)
{
    if constexpr (is_vector_v<X>) {
        return x;
    }
    else {
        return x ? true_mask_v<X> : false_mask_v<X>;
    }
}

template<Integral X>
force_inline constexpr auto unsigned_to_signed(X x)
{
    if constexpr (is_vector_v<X>) {
        return x; // There is only one integral vector type.
    }
    else {
        using S = signed_t<X>;
        return std::bit_cast<S>(x);
    }
}

template<Integral X>
force_inline constexpr auto signed_to_unsigned(X x)
{
    if constexpr (is_vector_v<X>) {
        return x; // There is only one integral vector type.
    }
    else {
        using U = unsigned_t<X>;
        return std::bit_cast<U>(x);
    }
}

template<Integral X>
force_inline constexpr auto unsigned_to_float(X x)
{
    if constexpr (is_vector_v<X>) {
        using F = float_t<X>;
        return F{X::unsigned_to_float(x)};
    }
    else {
        using F = float_t<X>;
        return static_cast<F>(x);
    }
}

template<Floating_point X>
force_inline constexpr auto float_to_signed(X x)
{
    if constexpr (is_vector_v<X>) {
        using M = mask_t<X>;
        return M{X::float_to_signed(x)};
    }
    else {
        using S = signed_t<X>;
        return static_cast<S>(x);
    }
}

template<Integral X>
force_inline constexpr auto signed_to_float(X x)
{
    if constexpr (is_vector_v<X>) {
        using F = float_t<X>;
        return F{X::signed_to_float(x)};
    }
    else {
        using F = float_t<X>;
        return static_cast<F>(x);
    }
}

template<Integral X>
force_inline constexpr auto reinterpret_as_float(X x) -> float_t<X>
{
    using F = float_t<X>;
    return std::bit_cast<F>(x);
}

template<Floating_point X>
force_inline constexpr auto reinterpret_as_int(X x) -> unsigned_t<X>
{
    using U = unsigned_t<X>;
    return std::bit_cast<U>(x);
}

} // namespace vsl