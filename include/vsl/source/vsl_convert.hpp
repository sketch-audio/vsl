#pragma once

#include <bit> // bit_cast
#include <cstdint> // int32_t

#include "vsl_core.hpp"

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

static_assert(mask_to_bool(true_mask_v<int32_t>) == true);

template<Vector V>
force_inline constexpr auto bool_to_mask(V x) -> V
{
    return x;
}

template<typename M>
force_inline constexpr auto bool_to_mask(bool x) -> M
{
    return x ? true_mask_v<M> : false_mask_v<M>;
}

static_assert(bool_to_mask<uint32_t>(true) == true_mask_v<uint32_t>);

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

static_assert(unsigned_to_signed(true_mask_v<int32_t>) == -1);

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

static_assert(signed_to_unsigned(-1) == true_mask_v<int32_t>);

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

static_assert(unsigned_to_float(true_mask_v<int32_t>) == 4294967295.f);

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

static_assert(float_to_signed(1.f) == 1);

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

static_assert(signed_to_float(-1) == -1.f);

template<Integral X>
force_inline constexpr auto reinterpret_as_float(X x) -> float_t<X>
{
    using F = float_t<X>;
    return std::bit_cast<F>(x);
}

static_assert(reinterpret_as_float(0) == 0.f);

template<Floating_point X>
force_inline constexpr auto reinterpret_as_int(X x) -> unsigned_t<X>
{
    using U = unsigned_t<X>;
    return std::bit_cast<U>(x);
}

static_assert(reinterpret_as_int(0.f) == 0);

} // namespace vsl