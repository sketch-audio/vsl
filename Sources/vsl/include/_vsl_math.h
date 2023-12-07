#ifndef _vsl_math_h
#define _vsl_math_h

#include <algorithm>
#include <cmath>

#include <simd/simd.h>

#include "_vsl_core.h"
#include "_vsl_cxm.h"

namespace vsl {

template<typename X, bool Cxm = false>
force_inline constexpr auto abs(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::abs(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::abs(x);
    }
    else {
        return std::abs(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto trunc(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::trunc(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::trunc(x);
    }
    else {
        return std::trunc(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto floor(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::floor(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::floor(x);
    }
    else {
        return std::floor(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto ceil(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::ceil(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::ceil(x);
    }
    else {
        return std::ceil(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto round(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::round(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::round(x);
    }
    else {
        return std::round(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto fmod(X x, X y) -> X
{
    if constexpr (Cxm) {
        return cxm::fmod(x, y);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::fmod(x, y);
    }
    else {
        return std::fmod(x, y);
    }
}

template<typename X>
force_inline constexpr auto wrap(X x) -> X
{
    return cxm::wrap(x);
}

template<typename X, typename B>
force_inline constexpr auto wrap(X x, B a, B b) -> X
{
    return cxm::wrap(x, a, b);
}

template<typename X, bool Cxm = false>
force_inline constexpr auto cos(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::cos(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::cos(x);
    }
    else {
        return std::cos(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto sin(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::sin(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::sin(x);
    }
    else {
        return std::sin(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto tan(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::tan(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::tan(x);
    }
    else {
        return std::tan(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto cosh(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::cosh(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::cosh(x);
    }
    else {
        return std::cosh(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto sinh(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::sinh(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::sinh(x);
    }
    else {
        return std::sinh(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto tanh(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::tanh(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::tanh(x);
    }
    else {
        return std::tanh(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto exp2(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::exp2(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::exp2(x);
    }
    else {
        return std::exp2(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto log2(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::log2(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::log2(x);
    }
    else {
        return std::log2(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto exp(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::exp(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::exp(x);
    }
    else {
        return std::exp(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto log(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::log(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::log(x);
    }
    else {
        return std::log(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto log10(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::log10(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::log10(x);
    }
    else {
        return std::log10(x);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto logB(X b, X x) -> X
{
    if constexpr (Cxm) {
        return cxm::logB(b, x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::log2(x) / simd::log2(b);
    }
    else {
        return std::log2(x) / std::log2(b);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto pow(X x, X y) -> X
{
    if constexpr (Cxm) {
        return cxm::pow(x, y);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::pow(x, y);
    }
    else {
        return std::pow(x, y);
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto sqrt(X x) -> X
{
    if constexpr (Cxm) {
        static_assert(deferred_false_v<X>, "cxm sqrt not implemented yet");
    }
    else if constexpr (is_vector_v<X>) {
        return simd::sqrt(x);
    }
    else {
        return std::sqrt(x);
    }
}

template<typename X, typename C, bool Cxm = false>
force_inline constexpr auto min(X a, C b) -> X
{
    if constexpr (Cxm) {
        return cxm::min(a, b);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::min(a, X(b));
    }
    else {
        return std::min(a, X(b));
    }
}

template<typename X, typename C, bool Cxm = false>
force_inline constexpr auto max(X a, C b) -> X
{
    if constexpr (Cxm) {
        return cxm::max(a, b);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::max(a, X(b));
    }
    else {
        return std::max(a, X(b));
    }
}

template<typename X, typename C, bool Cxm = false>
force_inline constexpr auto clamp(X x, C a, C b) -> X
{
    if constexpr (Cxm) {
        return cxm::clamp(x, a, b);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::clamp(x, X(a), X(b));
    }
    else {
        return std::clamp(x, X(a), X(b));
    }
}

template<typename X, bool Cxm = false>
force_inline constexpr auto sign(X x) -> X
{
    if constexpr (Cxm) {
        return cxm::sign(x);
    }
    else if constexpr (is_vector_v<X>) {
        return simd::sign(x);
    }
    else {
        return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
    }
}

} // namespace vsl

#endif /* _vsl_math_h */
