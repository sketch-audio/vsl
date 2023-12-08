#ifndef _vsl_core_h
#define _vsl_core_h

#include <type_traits>

#include <simd/simd.h>

// MARK: - Macros

/// Mainly for DEBUG builds.
/// see: https://gcc.gnu.org/onlinedocs/gcc/Inline.html
#ifndef force_inline
#ifdef __GNUC__
#define force_inline inline __attribute__((always_inline))
#else
#define force_inline inline
#endif
#endif

namespace vsl {

// MARK: - Definitions and assumptions

using float4 = simd::float4;
using double2 = simd::double2;
using int4 = simd::int4;
using long2 = simd::long2;
using uint4 = simd::uint4;
using ulong2 = simd::ulong2;

using int1 = simd::int1;
using long1 = simd::long1;
using uint1 = simd::uint1;
using ulong1 = simd::ulong1;

static_assert(std::is_same_v<float, simd::float1>);
static_assert(std::is_same_v<double, simd::double1>);

// MARK: - Traits

/**
 * @brief Always returns false, but deferred until template instantiation.
 * @tparam T Any type.
 */
template<typename T>
struct deferred_false : std::false_type {};

template<typename T>
inline constexpr bool deferred_false_v = deferred_false<T>::value;

/**
 * @brief Checks if T is one of the types in Ts.
 * @tparam T The type to check.
 * @tparam Ts The types to check against.
 */
template<typename T, typename... Ts>
struct is_one_of : std::disjunction<std::is_same<T, Ts>...> {};

template<typename T, typename... Ts>
inline constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;

/**
 * @brief Checks if T is an unsigned vector type.
 * @tparam T The type to check.
 */
template<typename T>
struct is_vector_unsigned : is_one_of<T, uint4, ulong2> {};

template<typename T>
inline constexpr bool is_vector_unsigned_v = is_vector_unsigned<T>::value;

/**
 * @brief Checks if T is a signed vector type.
 * @tparam T The type to check.
 */
template<typename T>
struct is_vector_signed : is_one_of<T, int4, long2> {};

template<typename T>
inline constexpr bool is_vector_signed_v = is_vector_signed<T>::value;

/**
 * @brief Checks if T is an integral vector type.
 * @tparam T The type to check.
 */
template<typename T>
struct is_vector_integral : std::integral_constant<bool, is_vector_unsigned_v<T> || is_vector_signed_v<T>> {};

template<typename T>
inline constexpr bool is_vector_integral_v = is_vector_integral<T>::value;

/**
 * @brief Checks if T is a floating-point vector type.
 * @tparam T The type to check.
 */
template<typename T>
struct is_vector_floating_point : is_one_of<T, float4, double2> {};

template<typename T>
inline constexpr bool is_vector_floating_point_v = is_vector_floating_point<T>::value;

/**
 * @brief Checks if T is a vector type.
 * @tparam T The type to check.
 */
template<typename T>
struct is_vector : std::integral_constant<bool, is_vector_integral_v<T> || is_vector_floating_point_v<T>> {};

template<typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;

/**
 * @brief Checks if T is an unsigned scalar type. 
 * @tparam T The type to check.
 */
template<typename T>
struct is_scalar_unsigned : is_one_of<T, uint1, ulong1> {};

template<typename T>
inline constexpr bool is_scalar_unsigned_v = is_scalar_unsigned<T>::value;

/**
 * @brief Checks if T is a signed scalar type.
 * @tparam T The type to check.
 */
template<typename T>
struct is_scalar_signed : is_one_of<T, int1, long1> {};

template<typename T>
inline constexpr bool is_scalar_signed_v = is_scalar_signed<T>::value;

/**
 * @brief Checks if T is an integral scalar type.
 * @tparam T The type to check.
 */
template<typename T>
struct is_scalar_integral : std::integral_constant<bool, is_scalar_unsigned_v<T> || is_scalar_signed_v<T>> {};

template<typename T>
inline constexpr bool is_scalar_integral_v = is_scalar_integral<T>::value;

/**
 * @brief Checks if T is a floating point scalar type.
 * @tparam T The type to check.
 */
template<typename T>
struct is_scalar_floating_point : is_one_of<T, float, double> {};

template<typename T>
inline constexpr bool is_scalar_floating_point_v = is_scalar_floating_point<T>::value;

/**
 * @brief Checks if T is a scalar type.
 * @tparam T The type to check.
 */
template<typename T>
struct is_scalar : std::integral_constant<bool, is_scalar_integral_v<T> || is_scalar_floating_point_v<T>> {};

template<typename T>
inline constexpr bool is_scalar_v = is_scalar<T>::value;

/**
 * @brief Checks if T is an unsigned scalar or vector type.
 * @tparam T The type to check.
 */
template<typename T>
struct is_unsigned : std::integral_constant<bool, is_scalar_unsigned_v<T> || is_vector_unsigned_v<T>> {};

template<typename T>
inline constexpr bool is_unsigned_v = is_unsigned<T>::value;

/**
 * @brief Checks if T is a signed scalar or vector type.
 * @tparam T The type to check.
 */
template<typename T>
struct is_signed : std::integral_constant<bool, is_scalar_signed_v<T> || is_vector_signed_v<T>> {};

template<typename T>
inline constexpr bool is_signed_v = is_signed<T>::value;

/**
 * @brief Checks if T is an integral scalar or vector type.
 * @tparam T The type to check.
 */
template<typename T>
struct is_integral : std::integral_constant<bool, is_unsigned_v<T> || is_signed_v<T>> {};

template<typename T>
inline constexpr bool is_integral_v = is_integral<T>::value;

/**
 * @brief Checks if T is a floating point scalar or vector type.
 * @tparam T The type to check.
 */
template<typename T>
struct is_floating_point : std::integral_constant<bool, is_scalar_floating_point_v<T> || is_vector_floating_point_v<T>> {};

template<typename T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

/**
 * @brief Maps vector types to their corresponding scalar equivalents. (A scalar type is its own scalar equivalent.)
 * @tparam T The type to map.
 */
template<typename T>
struct scalar_eq { typedef T type; };

template<>
struct scalar_eq<float4> { typedef float type; };

template<>
struct scalar_eq<double2> { typedef double type; };

template<>
struct scalar_eq<int4> { typedef int1 type; };

template<>
struct scalar_eq<long2> { typedef long1 type; };

template<>
struct scalar_eq<uint4> { typedef uint1 type; };

template<>
struct scalar_eq<ulong2> { typedef ulong1 type; };

/**
 * @brief Alias for the scalar equivalent of a type T.
 * @tparam T The type to get the scalar equivalent of.
 */
template<typename T>
using scalar_t = typename scalar_eq<T>::type;

/**
 * @brief Maps scalar types to their corresponding vector types. (A vector type is its own vector equivalent.)
 * @tparam T The type to map.
 */
template<typename T>
struct vector_eq { typedef T type; };

template<>
struct vector_eq<float> { typedef float4 type; };

template<>
struct vector_eq<double> { typedef double2 type; };

template<>
struct vector_eq<int1> { typedef int4 type; };

template<>
struct vector_eq<long1> { typedef long2 type; };

template<>
struct vector_eq<uint1> { typedef uint4 type; };

template<>
struct vector_eq<ulong1> { typedef ulong2 type; };

/**
 * @brief Alias for the vector equivalent of a type T.
 * @tparam T The type to get the vector equivalent of.
 */
template<typename T>
using vector_t = typename vector_eq<T>::type;

/**
 * @brief Maps a type to its counterpart type.
 *
 * A counterpart type is defined as the same-sized (signed) integral type of a floating point type or vice versa.
 * For example, the counterpart of float is int1, and the counterpart of int1 is float.
 *
 * @tparam T The type to map.
 */
template<typename T>
struct counterpart { static_assert(deferred_false_v<T>, "No suitable counterpart defined."); };

template<>
struct counterpart<float> { typedef int1 type; };

template<>
struct counterpart<double> { typedef long1 type; };

template<>
struct counterpart<int1> { typedef float type; };

template<>
struct counterpart<long1> { typedef double type; };

template<>
struct counterpart<float4> { typedef int4 type; };

template<>
struct counterpart<double2> { typedef long2 type; };

template<>
struct counterpart<int4> { typedef float4 type; };

template<>
struct counterpart<long2> { typedef double2 type; };

template<typename T>
using counterpart_t = typename counterpart<T>::type;

/**
 * @brief Maps a type to its unsigned counterpart type.
 *
 * An unsigned counterpart type is defined as the same-sized unsigned integral type of a floating point type or vice versa.
 * For example, the unsigned counterpart of float is uint1, and the unsigned counterpart of uint1 is float.
 *
 * @tparam T The type to map.
 */
template<typename T>
struct unsigned_counterpart { static_assert(deferred_false_v<T>, "No suitable unsigned counterpart defined."); };

template<>
struct unsigned_counterpart<float> { typedef uint1 type; };

template<>
struct unsigned_counterpart<double> { typedef ulong1 type; };

template<>
struct unsigned_counterpart<uint1> { typedef float type; };

template<>
struct unsigned_counterpart<ulong1> { typedef double type; };

template<>
struct unsigned_counterpart<float4> { typedef uint4 type; };

template<>
struct unsigned_counterpart<double2> { typedef ulong2 type; };

template<>
struct unsigned_counterpart<uint4> { typedef float4 type; };

template<>
struct unsigned_counterpart<ulong2> { typedef double2 type; };

template<typename T>
using unsigned_counterpart_t = typename unsigned_counterpart<T>::type;

/**
 * @brief Maps a signed type to its unsigned counterpart.
 *
 * This is used to convert signed integral types to their corresponding unsigned types.
 * For example, the unsigned counterpart of int1 is uint1, and the unsigned counterpart of long1 is ulong1.
 *
 * @tparam T The type to map.
 */
template<typename T>
struct su_cvt { static_assert(deferred_false_v<T>, "No suitable unsigned type defined."); };

template<>
struct su_cvt<int1> { typedef uint1 type; };

template<>
struct su_cvt<long1> { typedef ulong1 type; };

template<>
struct su_cvt<int4> { typedef uint4 type; };

template<>
struct su_cvt<long2> { typedef ulong2 type; };

template<typename T>
using su_cvt_t = typename su_cvt<T>::type;

/**
 * @brief Maps an unsigned type to its signed counterpart.
 *
 * This is used to convert unsigned integral types to their corresponding signed types.
 * For example, the signed counterpart of uint1 is int1, and the signed counterpart of ulong1 is long1.
 *
 * @tparam T The type to map.
 */
template<typename T>
struct us_cvt { static_assert(deferred_false_v<T>, "No suitable signed type defined."); };

template<>
struct us_cvt<uint1> { typedef int1 type; };

template<>
struct us_cvt<ulong1> { typedef long1 type; };

template<>
struct us_cvt<uint4> { typedef int4 type; };

template<>
struct us_cvt<ulong2> { typedef long2 type; };

template<typename T>
using us_cvt_t = typename us_cvt<T>::type;

/**
 * @brief Checks if T is an unsigned scalar or vector type.
 * @tparam T The type to check.
 */
template<typename T>
concept Unsigned = is_unsigned_v<T>;

/**
 * @brief Checks if T is a signed scalar or vector type.
 * @tparam T The type to check.
 */
template<typename T>
concept Signed = is_signed_v<T>;

/**
 * @brief Checks if T is an integral scalar or vector type.
 * @tparam T The type to check.
 */
template<typename T>
concept Integral = is_integral_v<T>;

/**
 * @brief Checks if T is a floating-point scalar or vector type.
 * @tparam T The type to check.
 */
template<typename T>
concept FloatingPoint = is_floating_point_v<T>;

/**
 * @brief Checks if T is a scalar type.
 * @tparam T The type to check.
 */
template<typename T>
concept Scalar = is_scalar_v<T>;

/**
 * @brief Checks if T is a vector type.
 * @tparam T The type to check.
 */
template<typename T>
concept Vector = is_vector_v<T>;

/**
 * @brief Gets the mask type for a floating-point type.
 * @tparam T A floating-point type.
 */
template<FloatingPoint T>
using mask_t = counterpart_t<T>;

/**
 * @brief Gets the (signed) integral type for a floating-point type.
 * @tparam T A floating-point type.
 */
template<FloatingPoint T>
using int_t = counterpart_t<T>;

/**
 * @brief Gets the (unsigned) integral type for a floating-point type.
 * @tparam T A floating-point type.
 */
template<FloatingPoint T>
using uint_t = unsigned_counterpart_t<T>;

/**
 * @brief Get the number of members in a vector type. (Returns 1 for scalar types.)
 * @tparam T 
 */
template<typename T>
struct num_members { static constexpr auto value{sizeof(T) / sizeof(scalar_t<T>)}; };

template<typename T>
inline constexpr auto num_members_v = num_members<T>::value;

static_assert(num_members_v<float> == 1);
static_assert(num_members_v<float4> == 4);

/**
 * @brief Get the "true" mask for a floating-point type.
 * @tparam T A floating-point type.
 */
template<FloatingPoint T>
struct true_mask { static constexpr auto value = mask_t<T>(-1); };

template<typename T>
inline constexpr auto true_mask_v = true_mask<T>::value;

/**
 * @brief Get the "false" mask for a floating-point type.
 * @tparam T A floating-point type.
 */
template<FloatingPoint T>
struct false_mask { static constexpr auto value = mask_t<T>(0); };

template<typename T>
inline constexpr auto false_mask_v = false_mask<T>::value;

/**
 * @brief Maps a type to its IEEE 754 exponent bias.
 *
 * The exponent bias is a specific number subtracted from the actual exponent to get the stored exponent in IEEE 754 floating-point numbers.
 * For single-precision (float), the bias is 127.
 * For double-precision (double), the bias is 1023.
 *
 * @tparam T The type to map. (A scalar, floating-point type.)
 */
template<typename T>
struct ieee_exp_bias { static_assert(deferred_false_v<T>, "IEEE exponent bias not defined."); };

template<>
struct ieee_exp_bias<float> : std::integral_constant<uint1, 127> {};

template<>
struct ieee_exp_bias<double> : std::integral_constant<ulong1, 1023> {};

template<typename T>
inline constexpr auto ieee_exp_bias_v = ieee_exp_bias<T>::value;

/**
 * @brief Maps a type to its IEEE 754 significand bits.
 *
 * The significand (also known as the mantissa) is the part of a floating-point number that contains its significant digits.
 * For single-precision (float), there are 23 explicit significand bits.
 * For double-precision (double), there are 52 explicit significand bits.
 *
 * @tparam T The type to map. (A scalar, floating-point type.)
 */
template<typename T>
struct ieee_sig_bits { static_assert(deferred_false_v<T>, "IEEE significand bits not defined."); };

template<>
struct ieee_sig_bits<float> : std::integral_constant<uint1, 23> {};

template<>
struct ieee_sig_bits<double> : std::integral_constant<ulong1, 52> {};

template<typename T>
inline constexpr auto ieee_sig_bits_v = ieee_sig_bits<T>::value;

/**
 * @brief Maps a type to its IEEE 754 exponent bits.
 *
 * The exponent is the part of a floating-point number that determines the number's magnitude.
 * For single-precision (float), there are 8 exponent bits.
 * For double-precision (double), there are 11 exponent bits.
 *
 * @tparam T The type to map. (A scalar, floating-point type.)
 */
template<typename T>
struct ieee_exp_bits { static_assert(deferred_false_v<T>, "IEEE exponent bits not defined."); };

template<>
struct ieee_exp_bits<float> : std::integral_constant<uint1, 8> {};

template<>
struct ieee_exp_bits<double> : std::integral_constant<ulong1, 11> {};

template<typename T>
inline constexpr auto ieee_exp_bits_v = ieee_exp_bits<T>::value;

} // namespace cxm 

#endif /* _vsl_core_h */
