#pragma once

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "vsl_impl.h"

namespace vsl {

// We need two's complement.
static_assert(std::bit_cast<int32_t>(std::numeric_limits<uint32_t>::max()) == -1);
static_assert(std::bit_cast<int64_t>(std::numeric_limits<uint64_t>::max()) == -1);

template<typename T>
struct deferred_false : std::false_type {};

template<typename T>
inline constexpr bool deferred_false_v = deferred_false<T>::value;

template<typename T, typename... Ts>
struct is_one_of : std::disjunction<std::is_same<T, Ts>...> {};

template<typename T, typename... Ts>
inline constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;

// MARK: - is_scalar

template<typename T>
struct is_scalar : is_one_of<T, uint32_t, int32_t, uint64_t, int64_t, float, double> {};

template<typename T>
inline constexpr bool is_scalar_v = is_scalar<T>::value;

template<typename T>
concept Scalar = is_scalar_v<T>;

// MARK: - is_vector

template<typename T>
struct is_vector : is_one_of<T, mask32x4, mask64x2, float32x4, float64x2> {};

template<typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;

template<typename T>
concept Vector = is_vector_v<T>;

// MARK: - is_integral

template<typename T>
struct is_integral : is_one_of<T, uint32_t, int32_t, uint64_t, int64_t, mask32x4, mask64x2> {};

template<typename T>
inline constexpr bool is_integral_v = is_integral<T>::value;

template<typename T>
concept Integral = is_integral_v<T>;

// MARK: - is_floating_point

template<typename T>
struct is_floating_point : is_one_of<T, float, double, float32x4, float64x2> {};

template<typename T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

template<typename T>
concept Floating_point = is_floating_point_v<T>;

// MARK: - float equivalent

template<typename T>
struct float_equivalent { static_assert(deferred_false_v<T>); };

template<>
struct float_equivalent<uint32_t> { using type = float; };

template<>
struct float_equivalent<int32_t> { using type = float; };

template<>
struct float_equivalent<uint64_t> { using type = double; };

template<>
struct float_equivalent<int64_t> { using type = double; };

template<>
struct float_equivalent<float> { using type = float; };

template<>
struct float_equivalent<double> { using type = double; };

template<>
struct float_equivalent<mask32x4> { using type = float32x4; };

template<>
struct float_equivalent<mask64x2> { using type = float64x2; };

template<>
struct float_equivalent<float32x4> { using type = float32x4; };

template<>
struct float_equivalent<float64x2> { using type = float64x2; };

// Resolves to the floating-point type with the same bit width as T
template<typename T>
using float_t = typename float_equivalent<T>::type;

// MARK: - mask_equivalent

template<typename T>
struct mask_equivalent { static_assert(deferred_false_v<T>); };

template<>
struct mask_equivalent<uint32_t> { using type = uint32_t; };

template<>
struct mask_equivalent<int32_t> { using type = uint32_t; };

template<>
struct mask_equivalent<uint64_t> { using type = uint64_t; };

template<>
struct mask_equivalent<int64_t> { using type = uint64_t; };

template<>
struct mask_equivalent<float> { using type = uint32_t; };

template<>
struct mask_equivalent<double> { using type = uint64_t; };

template<>
struct mask_equivalent<mask32x4> { using type = mask32x4; };

template<>
struct mask_equivalent<mask64x2> { using type = mask64x2; };

template<>
struct mask_equivalent<float32x4> { using type = mask32x4; };

template<>
struct mask_equivalent<float64x2> { using type = mask64x2; };

// Resolves to the unsigned integer type with the same bit width as T
template<typename T>
using mask_t = typename mask_equivalent<T>::type;

// MARK: - signed_equivalent 

template<Scalar T>
struct signed_equivalent { static_assert(deferred_false_v<T>); };

template<>
struct signed_equivalent<uint32_t> { using type = int32_t; };

template<>
struct signed_equivalent<int32_t> { using type = int32_t; };

template<>
struct signed_equivalent<uint64_t> { using type = int64_t; };

template<>
struct signed_equivalent<int64_t> { using type = int64_t; };

template<>
struct signed_equivalent<float> { using type = int32_t; };

template<>
struct signed_equivalent<double> { using type = int64_t; };

// Resolves to the signed integer type with the same bit width as T
template<Scalar T>
using signed_t = typename signed_equivalent<T>::type;

// MARK: - unsigned_equivalent

template<typename T>
struct unsigned_equivalent { using type = mask_t<T>; };

// Resolves to the unsigned integer type with the same bit width as T
template<typename T>
using unsigned_t = typename unsigned_equivalent<T>::type;

// MARK: - scalar_equivalent

template<typename T>
struct scalar_equivalent { static_assert(deferred_false_v<T>); };

template<>
struct scalar_equivalent<uint32_t> { using type = uint32_t; };

template<>
struct scalar_equivalent<int32_t> { using type = int32_t; };

template<>
struct scalar_equivalent<uint64_t> { using type = uint64_t; };

template<>
struct scalar_equivalent<int64_t> { using type = int64_t; };

template<>
struct scalar_equivalent<float> { using type = float; };

template<>
struct scalar_equivalent<double> { using type = double; };

template<>
struct scalar_equivalent<mask32x4> { using type = uint32_t; };

template<>
struct scalar_equivalent<mask64x2> { using type = uint64_t; };

template<>
struct scalar_equivalent<float32x4> { using type = float; };

template<>
struct scalar_equivalent<float64x2> { using type = double; };

// Resolves to the scalar type with the same bit width as T (unsigned for masks).
template<typename T>
using scalar_t = typename scalar_equivalent<T>::type;

// MARK: - vector_equivalent

template<typename T>
struct vector_equivalent { static_assert(deferred_false_v<T>); };

template<>
struct vector_equivalent<uint32_t> { using type = mask32x4; };

template<>
struct vector_equivalent<int32_t> { using type = mask32x4; };

template<>
struct vector_equivalent<uint64_t> { using type = mask64x2; };

template<>
struct vector_equivalent<int64_t> { using type = mask64x2; };

template<>
struct vector_equivalent<float> { using type = float32x4; };

template<>
struct vector_equivalent<double> { using type = float64x2; };

template<>
struct vector_equivalent<mask32x4> { using type = mask32x4; };

template<>
struct vector_equivalent<mask64x2> { using type = mask64x2; };

template<>
struct vector_equivalent<float32x4> { using type = float32x4; };

template<>
struct vector_equivalent<float64x2> { using type = float64x2; };

// Resolves to the vector type with the same bit width as T
template<typename T>
using vector_t = typename vector_equivalent<T>::type;

// MARK: - true/false (scalar) masks

template<Scalar T>
struct true_mask { using M = mask_t<T>; static constexpr auto value = ~M{}; };

template<Scalar T>
inline constexpr auto true_mask_v = true_mask<T>::value;

template<Scalar T>
struct false_mask { using M = mask_t<T>; static constexpr auto value = M{}; };

template<Scalar T>
inline constexpr auto false_mask_v = false_mask<T>::value;

// MARK: - IEEE helpers

template<typename T>
struct ieee_exp_bias { static_assert(deferred_false_v<T>, "IEEE exponent bias not defined."); };

template<>
struct ieee_exp_bias<float> : std::integral_constant<uint32_t, 127> {};

template<>
struct ieee_exp_bias<double> : std::integral_constant<uint64_t, 1023> {};

template<typename T>
inline constexpr auto ieee_exp_bias_v = ieee_exp_bias<T>::value;

template<typename T>
struct ieee_sig_bits { static_assert(deferred_false_v<T>, "IEEE significand bits not defined."); };

template<>
struct ieee_sig_bits<float> : std::integral_constant<uint32_t, 23> {};

template<>
struct ieee_sig_bits<double> : std::integral_constant<uint64_t, 52> {};

template<typename T>
inline constexpr auto ieee_sig_bits_v = ieee_sig_bits<T>::value;

template<typename T>
struct ieee_exp_bits { static_assert(deferred_false_v<T>, "IEEE exponent bits not defined."); };

template<>
struct ieee_exp_bits<float> : std::integral_constant<uint32_t, 8> {};

template<>
struct ieee_exp_bits<double> : std::integral_constant<uint64_t, 11> {};

template<typename T>
inline constexpr auto ieee_exp_bits_v = ieee_exp_bits<T>::value;

// MARK: - num members

template<typename T>
struct num_members { static constexpr auto value{sizeof(T) / sizeof(scalar_t<T>)}; };

template<typename T>
inline constexpr auto num_members_v = num_members<T>::value;

static_assert(num_members_v<float> == 1);
static_assert(num_members_v<float32x4> == 4);

} // namespace vsl