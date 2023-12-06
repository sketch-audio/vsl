#ifndef _vsl_core_h
#define _vsl_core_h

#include <cstdint>
#include <type_traits>

#include <simd/simd.h>

// MARK: - Macros

/// Mainly for performance in DEBUG builds. Questionable benefit.
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

static_assert(sizeof(float) == 4, "Size of float must be 4 bytes.");
static_assert(sizeof(double) == 8, "Size of double must be 8 bytes.");

// MARK: - Traits

/**
 * @brief deferred_false
 * 
 * @tparam T 
 */
template<typename T>
struct deferred_false : std::false_type {};

template<typename T>
inline constexpr bool deferred_false_v = deferred_false<T>::value;

/**
 * @brief is_one_of
 * 
 * @tparam T 
 * @tparam Ts 
 */
template<typename T, typename... Ts>
struct is_one_of : std::disjunction<std::is_same<T, Ts>...> {};

template<typename T, typename... Ts>
inline constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;

/**
 * @brief is_vector_unsigned
 * 
 * @tparam T 
 */
template<typename T>
struct is_vector_unsigned : is_one_of<T, uint4, ulong2> {};

template<typename T>
inline constexpr bool is_vector_unsigned_v = is_vector_unsigned<T>::value;

/**
 * @brief is_vector_signed
 * 
 * @tparam T 
 */
template<typename T>
struct is_vector_signed : is_one_of<T, int4, long2> {};

template<typename T>
inline constexpr bool is_vector_signed_v = is_vector_signed<T>::value;

/**
 * @brief is_vector_integral
 * 
 * @tparam T 
 */
template<typename T>
struct is_vector_integral : std::integral_constant<bool, is_vector_unsigned_v<T> || is_vector_signed_v<T>> {};

template<typename T>
inline constexpr bool is_vector_integral_v = is_vector_integral<T>::value;

/**
 * @brief is_vector_floating_point
 * 
 * @tparam T 
 */
template<typename T>
struct is_vector_floating_point : is_one_of<T, float4, double2> {};

template<typename T>
inline constexpr bool is_vector_floating_point_v = is_vector_floating_point<T>::value;

/**
 * @brief is_vector
 * 
 * @tparam T 
 */
template<typename T>
struct is_vector : std::integral_constant<bool, is_vector_integral_v<T> || is_vector_floating_point_v<T>> {};

template<typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;

/**
 * @brief is_scalar_usigned
 * 
 */
template<typename T>
struct is_scalar_unsigned : is_one_of<T, uint32_t, uint64_t> {};

template<typename T>
inline constexpr bool is_scalar_unsigned_v = is_scalar_unsigned<T>::value;

/**
 * @brief is_scalar_signed
 * 
 * @tparam T 
 */
template<typename T>
struct is_scalar_signed : is_one_of<T, int32_t, int64_t> {};

template<typename T>
inline constexpr bool is_scalar_signed_v = is_scalar_signed<T>::value;

/**
 * @brief is_scalar_integral
 * 
 * @tparam T 
 */
template<typename T>
struct is_scalar_integral : std::integral_constant<bool, is_scalar_unsigned_v<T> || is_scalar_signed_v<T>> {};

template<typename T>
inline constexpr bool is_scalar_integral_v = is_scalar_integral<T>::value;

/**
 * @brief is_scalar_floating_point
 * 
 * @tparam T 
 */
template<typename T>
struct is_scalar_floating_point : is_one_of<T, float, double> {};

template<typename T>
inline constexpr bool is_scalar_floating_point_v = is_scalar_floating_point<T>::value;

/**
 * @brief is_scalar
 * 
 * @tparam T 
 */
template<typename T>
struct is_scalar : std::integral_constant<bool, is_scalar_integral_v<T> || is_scalar_floating_point_v<T>> {};

template<typename T>
inline constexpr bool is_scalar_v = is_scalar<T>::value;

/**
 * @brief is_unsigned
 * 
 */
template<typename T>
struct is_unsigned : std::integral_constant<bool, is_scalar_unsigned_v<T> || is_vector_unsigned_v<T>> {};

template<typename T>
inline constexpr bool is_unsigned_v = is_unsigned<T>::value;

/**
 * @brief is_signed
 * 
 * @tparam T 
 */
template<typename T>
struct is_signed : std::integral_constant<bool, is_scalar_signed_v<T> || is_vector_signed_v<T>> {};

template<typename T>
inline constexpr bool is_signed_v = is_signed<T>::value;

/**
 * @brief is_integral
 * 
 * @tparam T 
 */
template<typename T>
struct is_integral : std::integral_constant<bool, is_unsigned_v<T> || is_signed_v<T>> {};

template<typename T>
inline constexpr bool is_integral_v = is_integral<T>::value;

/**
 * @brief is_floating_point
 * 
 * @tparam T 
 */
template<typename T>
struct is_floating_point : std::integral_constant<bool, is_scalar_floating_point_v<T> || is_vector_floating_point_v<T>> {};

template<typename T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

/**
 * @brief scalar_eq (get the scalar equivalent of a type)
 * 
 * @tparam T 
 */
template<typename T>
struct scalar_eq { typedef T type; };

template<>
struct scalar_eq<float4> { typedef float type; };

template<>
struct scalar_eq<double2> { typedef double type; };

template<>
struct scalar_eq<int4> { typedef int32_t type; };

template<>
struct scalar_eq<long2> { typedef int64_t type; };

template<>
struct scalar_eq<uint4> { typedef uint32_t type; };

template<>
struct scalar_eq<ulong2> { typedef uint64_t type; };

/**
 * @brief scalar_t
 * 
 * @tparam T 
 */
template<typename T>
using scalar_t = typename scalar_eq<T>::type;

/**
 * @brief vector_eq (get the vector equivalent of a type)
 * 
 * @tparam T 
 */
template<typename T>
struct vector_eq { typedef T type; };

template<>
struct vector_eq<float> { typedef float4 type; };

template<>
struct vector_eq<double> { typedef double2 type; };

template<>
struct vector_eq<int32_t> { typedef int4 type; };

template<>
struct vector_eq<int64_t> { typedef long2 type; };

template<>
struct vector_eq<uint32_t> { typedef uint4 type; };

template<>
struct vector_eq<uint64_t> { typedef ulong2 type; };

/**
 * @brief vector_t
 * 
 * @tparam T 
 */
template<typename T>
using vector_t = typename vector_eq<T>::type;

/**
 * @brief counterpart (get the same-sized (signed) integral type of a floating point type or vice versa)
 *
 * @tparam T 
 */
template<typename T>
struct counterpart { static_assert(deferred_false_v<T>, "No suitable counterpart defined."); };

template<>
struct counterpart<float> { typedef int32_t type; };

template<>
struct counterpart<double> { typedef int64_t type; };

template<>
struct counterpart<int32_t> { typedef float type; };

template<>
struct counterpart<int64_t> { typedef double type; };

template<>
struct counterpart<float4> { typedef int4 type; };

template<>
struct counterpart<double2> { typedef long2 type; };

template<>
struct counterpart<int4> { typedef float4 type; };

template<>
struct counterpart<long2> { typedef double2 type; };

/**
 * @brief counterpart_t
 *
 * @tparam T
 */
template<typename T>
using counterpart_t = typename counterpart<T>::type;

/**
 * @brief unsigned_counterpart (get the same-sized (unsigned) integral type of a floating point type or vice versa)
 *
 * @tparam T
 */
template<typename T>
struct unsigned_counterpart { static_assert(deferred_false_v<T>, "No suitable unsigned counterpart defined."); };

template<>
struct unsigned_counterpart<float> { typedef uint32_t type; };

template<>
struct unsigned_counterpart<double> { typedef uint64_t type; };

template<>
struct unsigned_counterpart<uint32_t> { typedef float type; };

template<>
struct unsigned_counterpart<uint64_t> { typedef double type; };

template<>
struct unsigned_counterpart<float4> { typedef uint4 type; };

template<>
struct unsigned_counterpart<double2> { typedef ulong2 type; };

template<>
struct unsigned_counterpart<uint4> { typedef float4 type; };

template<>
struct unsigned_counterpart<ulong2> { typedef double2 type; };

/**
 * @brief unsigned_counterpart_t
 *
 * @tparam T 
 */
template<typename T>
using unsigned_counterpart_t = typename unsigned_counterpart<T>::type;

/**
 * @brief sucvt (signed to unsigned convert)
 *
 * @tparam T
 */
template<typename T>
struct su_cvt { static_assert(deferred_false_v<T>, "No suitable counterpart defined."); };

template<>
struct su_cvt<int32_t> { typedef uint32_t type; };

template<>
struct su_cvt<int64_t> { typedef uint64_t type; };

template<>
struct su_cvt<int4> { typedef uint4 type; };

template<>
struct su_cvt<long2> { typedef ulong2 type; };

/**
 * @brief su_cvt_t (signed to unsigned convert)
 *
 * @tparam T
 */
template<typename T>
using su_cvt_t = typename su_cvt<T>::type;

/**
 * @brief us_cvt (unsigned to signed convert)
 *
 * @tparam T
 */
template<typename T>
struct us_cvt { static_assert(deferred_false_v<T>, "No suitable counterpart defined."); };

template<>
struct us_cvt<uint32_t> { typedef int32_t type; };

template<>
struct us_cvt<uint64_t> { typedef int64_t type; };

template<>
struct us_cvt<uint4> { typedef int4 type; };

template<>
struct us_cvt<ulong2> { typedef long2 type; };

/**
 * @brief us_cvt_t (unsigned to signed convert)
 *
 * @tparam T
 */
template<typename T>
using us_cvt_t = typename us_cvt<T>::type;

/**
 * @brief Unsigned
 *
 * @tparam T
 */
template<typename T>
concept Unsigned = is_unsigned_v<T>;

/**
 * @brief Signed
 *
 * @tparam T
 */
template<typename T>
concept Signed = is_signed_v<T>;

/**
 * @brief Integral
 * 
 * @tparam T 
 */
template<typename T>
concept Integral = is_integral_v<T>;

/**
 * @brief FloatingPoint
 * 
 * @tparam T 
 */
template<typename T>
concept FloatingPoint = is_floating_point_v<T>;

/**
 * @brief Scalar
 * 
 * @tparam T 
 */
template<typename T>
concept Scalar = is_scalar_v<T>;

/**
 * @brief Vector
 * 
 * @tparam T 
 */
template<typename T>
concept Vector = is_vector_v<T>;

/**
 * @brief mask_t (get the mask for a floating-point type)
 * 
 * @tparam T 
 */
template<FloatingPoint T>
using mask_t = counterpart_t<T>;

/**
 * @brief int_t (get the same-sized (signed) integral type for a floating-point type)
 *
 * @tparam T 
 */
template<FloatingPoint T>
using int_t = counterpart_t<T>;

/**
 * @brief uint_t (get the same-sized (unsigned) integral type for a floating-point type)
 *
 * @tparam T
 */
template<FloatingPoint T>
using uint_t = unsigned_counterpart_t<T>;

/**
 * @brief num_members (get the number of members or lanes in a type)
 * 
 * @tparam T 
 */
template<typename T>
struct num_members { static constexpr auto value{sizeof(T) / sizeof(scalar_t<T>)}; };

template<typename T>
inline constexpr auto num_members_v = num_members<T>::value;

/**
 * @brief true mask
 * 
 * @tparam T 
 */
template<FloatingPoint T>
struct true_mask { static constexpr auto value = mask_t<T>(-1); };

template<typename T>
inline constexpr auto true_mask_v = true_mask<T>::value;

/**
 * @brief false mask
 * 
 * @tparam T 
 */
template<FloatingPoint T>
struct false_mask { static constexpr auto value = mask_t<T>(0); };

template<typename T>
inline constexpr auto false_mask_v = false_mask<T>::value;

/**
 * @brief IEEE exponent bias
 * 
 * @tparam T 
 */
template<typename T>
struct ieee_exp_bias { static_assert(deferred_false_v<T>, "IEEE exponent bias not defined."); };

template<>
struct ieee_exp_bias<float> : std::integral_constant<uint32_t, 127> {};

template<>
struct ieee_exp_bias<double> : std::integral_constant<uint64_t, 1023> {};

template<typename T>
inline constexpr auto ieee_exp_bias_v = ieee_exp_bias<T>::value;

/**
 * @brief IEEE significand bits (explicit)
 *
 * @tparam T 
 */
template<typename T>
struct ieee_sig_bits { static_assert(deferred_false_v<T>, "IEEE significand bits not defined."); };

template<>
struct ieee_sig_bits<float> : std::integral_constant<uint32_t, 23> {};

template<>
struct ieee_sig_bits<double> : std::integral_constant<uint64_t, 52> {};

template<typename T>
inline constexpr auto ieee_sig_bits_v = ieee_sig_bits<T>::value;

/**
 * @brief IEEE exponent bits
 *
 * @tparam T
 */
template<typename T>
struct ieee_exp_bits { static_assert(deferred_false_v<T>, "IEEE exponent bits not defined."); };

template<>
struct ieee_exp_bits<float> : std::integral_constant<uint32_t, 8> {};

template<>
struct ieee_exp_bits<double> : std::integral_constant<uint64_t, 11> {};

template<typename T>
inline constexpr auto ieee_exp_bits_v = ieee_exp_bits<T>::value;


} // namespace cxm 

#endif /* _vsl_core_h */
