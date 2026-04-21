#pragma once

#include <bit> // std::countl_zero

#include "vsl_conv.hpp"
#include "vsl_core.hpp"

namespace vsl {

// Vector compatible ternary operator.
template<typename C, typename X>
force_inline constexpr auto select(C cond, X val_t, X val_f) -> X
{
    if constexpr (is_vector_v<X>) {
        return X::select(cond, val_t, val_f);
    } else {
        return cond ? val_t : val_f;
    }
}

static_assert(select(true, 1, 0) == 1);
static_assert(select(false, 1, 0) == 0);

// Returns true if any lanes satisfy the condition, scalar compatible.
template<typename C>
force_inline constexpr auto any(C cond) -> bool
{
    if constexpr (is_vector_v<C>) {
        return C::any(cond);
    } else {
        return cond;
    }
}

static_assert(any(true));
static_assert(!any(false));

// Returns true if all lanes satisfy the condition, scalar compatible.
template<typename C>
force_inline constexpr auto all(C cond) -> bool
{
    if constexpr (is_vector_v<C>) {
        return C::all(cond);
    } else {
        return cond;
    }
}

static_assert(all(true));
static_assert(!all(false));

template<typename X>
force_inline constexpr auto all_same(X a) -> bool
{
    if constexpr (is_vector_v<X>) {
        return all(a == a[0]);
    } else {
        return true;
    }
}

template<typename X>
force_inline constexpr auto all_unique(X a) -> bool
{
    if constexpr (is_vector_v<X>) {
        constexpr auto N = num_members_v<X>;
        for (size_t i = 0; i < N; i++) {
            for (size_t j = i + 1; j < N; j++) {
                if (a[i] == a[j]) {
                    return false;
                }
            }
        }
        return true;
    } else {
        return true;
    }
}

template<typename X>
force_inline constexpr auto all_finite(X a) -> bool
{
    if constexpr (is_vector_v<X>) {
        return all(a == a) && all(a - a == X{0}); // NaN and Inf check.
    } else {
        return a == a && (a - a == 0); // NaN and Inf check.
    }
}

// static_assert(all_finite(0.f));
// static_assert(all_finite(1.f));
// static_assert(!all_finite(std::numeric_limits<float>::infinity()));
// static_assert(!all_finite(std::numeric_limits<float>::quiet_NaN()));

// MARK: - Equality

/// Are two numbers
template<typename X>
force_inline constexpr auto abs_equal(X a, X b, X tol = 1e-7f)
{
    return ((a - b) < tol) & ((b - a) < tol); // We don't have `abs` yet.
}

static_assert(abs_equal(1.f, 1 + 1e-8f));
static_assert(!abs_equal(1.f, 1 + 1e-5f));
static_assert(abs_equal(1.0, 1 + 1e-8));
static_assert(!abs_equal(1.0, 1 + 1e-5));

///
template<typename X>
force_inline constexpr auto rel_equal(X a, X b, X tol = 1e-3f)
{
    return (((a - b) / b) < tol) & (((b - a) / b) < tol);
}

static_assert(rel_equal(1.f, 1 + 1e-4f));
static_assert(!rel_equal(1.f, 1 + 1e-2f));
static_assert(rel_equal(100.1f, 100.f));
static_assert(!rel_equal(101.f, 100.f));

///
template<typename X>
force_inline constexpr auto bit_equal(X a, X b, unsigned_t<X> tol = 5) -> bool
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
    const auto n = static_cast<decltype(tol)>(matching_bits);

    return n >= tol;
}

static_assert(bit_equal(1.f, 1.01f, 6));
static_assert(bit_equal(1.f, 1.1f, 3));
static_assert(!bit_equal(200.f, 201.f, 7));
static_assert(!bit_equal(200.f, 202.f, 6));

} // namespace vsl