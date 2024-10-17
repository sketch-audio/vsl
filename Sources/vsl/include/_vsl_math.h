#ifndef _vsl_math_h
#define _vsl_math_h

#include <algorithm> // min, max, clamp
#include <cmath>

#include <simd/simd.h>

#include "_vsl_core.h"
#include "_vsl_cxm.h" // sign, wrap

namespace vsl {

template<typename X>
force_inline auto abs(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::abs(x);
    }
    else {
        return std::abs(x);
    }
}

template<typename X>
force_inline auto trunc(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::trunc(x);
    }
    else {
        return std::trunc(x);
    }
}

template<typename X>
force_inline auto floor(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::floor(x);
    }
    else {
        return std::floor(x);
    }
}

template<typename X>
force_inline auto ceil(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::ceil(x);
    }
    else {
        return std::ceil(x);
    }
}

template<typename X>
force_inline auto round(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::round(x);
    }
    else {
        return std::round(x);
    }
}

template<typename X>
force_inline auto fmod(X x, X y) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::fmod(x, y);
    }
    else {
        return std::fmod(x, y);
    }
}

template<typename X>
force_inline auto cos(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::cos(x);
    }
    else {
        return std::cos(x);
    }
}

template<typename X>
force_inline auto sin(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::sin(x);
    }
    else {
        return std::sin(x);
    }
}

template<typename X>
force_inline auto tan(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::tan(x);
    }
    else {
        return std::tan(x);
    }
}

template<typename X>
force_inline auto acos(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::acos(x);
    }
    else {
        return std::acos(x);
    }
}

template<typename X>
force_inline auto asin(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::asin(x);
    }
    else {
        return std::asin(x);
    }
}

template<typename X>
force_inline auto atan(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::atan(x);
    }
    else {
        return std::atan(x);
    }
}

template<typename X>
force_inline auto cosh(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::cosh(x);
    }
    else {
        return std::cosh(x);
    }
}

template<typename X>
force_inline auto sinh(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::sinh(x);
    }
    else {
        return std::sinh(x);
    }
}

template<typename X>
force_inline auto tanh(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::tanh(x);
    }
    else {
        return std::tanh(x);
    }
}

template<typename X>
force_inline auto acosh(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::acosh(x);
    }
    else {
        return std::acosh(x);
    }
}

template<typename X>
force_inline auto asinh(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::asinh(x);
    }
    else {
        return std::asinh(x);
    }
}

template<typename X>
force_inline auto atanh(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::atanh(x);
    }
    else {
        return std::atanh(x);
    }
}

template<typename X>
force_inline auto erf(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::erf(x);
    }
    else {
        return std::erf(x);
    }
}

template<typename X>
force_inline auto exp2(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::exp2(x);
    }
    else {
        return std::exp2(x);
    }
}

template<typename X>
force_inline auto log2(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::log2(x);
    }
    else {
        return std::log2(x);
    }
}

template<typename X>
force_inline auto exp(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::exp(x);
    }
    else {
        return std::exp(x);
    }
}

template<typename X>
force_inline auto log(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::log(x);
    }
    else {
        return std::log(x);
    }
}

template<typename X>
force_inline auto log10(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::log10(x);
    }
    else {
        return std::log10(x);
    }
}

template<typename X>
force_inline auto logB(X b, X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::log2(x) / simd::log2(b);
    }
    else {
        return std::log2(x) / std::log2(b);
    }
}

template<typename X>
force_inline auto pow(X x, X y) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::pow(x, y);
    }
    else {
        return std::pow(x, y);
    }
}

template<typename X>
force_inline auto sqrt(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::sqrt(x);
    }
    else {
        return std::sqrt(x);
    }
}

template<typename X>
force_inline auto cbrt(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::cbrt(x);
    }
    else {
        return std::cbrt(x);
    }
}

template<typename X, typename C>
force_inline auto min(X a, C b) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::min(a, X(b));
    }
    else {
        return std::min(a, X(b));
    }
}

template<typename X, typename C>
force_inline auto max(X a, C b) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::max(a, X(b));
    }
    else {
        return std::max(a, X(b));
    }
}

template<typename X, typename C>
force_inline auto clamp(X x, C a, C b) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::clamp(x, X(a), X(b));
    }
    else {
        return std::clamp(x, X(a), X(b));
    }
}

template<typename X>
force_inline auto sign(X x) -> X
{
    if constexpr (is_vector_v<X>) {
        return simd::sign(x);
    }
    else {
        return cxm::sign(x);
    }
}

template<typename X>
force_inline auto wrap(X x) -> X
{
    return cxm::wrap(x);
}

template<typename X, typename B>
force_inline auto wrap(X x, B a, B b) -> X
{
    return cxm::wrap(x, a, b);
}

} // namespace vsl

#endif /* _vsl_math_h */
