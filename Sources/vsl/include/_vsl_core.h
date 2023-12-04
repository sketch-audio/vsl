#ifndef _vsl_core_h
#define _vsl_core_h

#include <type_traits>

// TODO: - Write a cross-platform drop-in for Apple SIMD library.

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

static_assert(sizeof(int) == 4, "Size of int must be 4 bytes.");
static_assert(sizeof(long) == 8, "Size of long must be 8 bytes.");
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
 * @brief is_vector_integral
 * 
 * @tparam T 
 */
template<typename T>
struct is_vector_integral : is_one_of<T, int4, long2> {};

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
 * @brief is_scalar (a type is scalar if it is not a vector)
 * 
 * @tparam T 
 */
template<typename T>
struct is_scalar : std::integral_constant<bool, !is_vector_v<T>> {};

template<typename T>
inline constexpr bool is_scalar_v = is_scalar<T>::value;

/**
 * @brief is_floating_point
 * 
 * @tparam T 
 */
template<typename T>
struct is_floating_point : std::integral_constant<bool, is_one_of_v<T, float, double> || is_vector_floating_point_v<T>> {};

template<typename T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

/**
 * @brief is_integral (a type is integral if it is not floating point)
 * 
 * @tparam T 
 */
template<typename T>
struct is_integral : std::integral_constant<bool, !is_floating_point_v<T>> {};

template<typename T>
inline constexpr bool is_integral_v = is_integral<T>::value;

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
struct scalar_eq<int4> { typedef int type; };

template<>
struct scalar_eq<long2> { typedef long type; };

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
struct vector_eq<int> { typedef int4 type; };

template<>
struct vector_eq<long> { typedef long2 type; };

/**
 * @brief vector_t
 * 
 * @tparam T 
 */
template<typename T>
using vector_t = typename vector_eq<T>::type;

/**
 * @brief counterpart (get the same sized integral type of a floating point type or vice versa)
 * 
 * @tparam T 
 */
template<typename T>
struct counterpart { static_assert(deferred_false_v<T>, "No suitable counterpart defined."); };

template<>
struct counterpart<float> { typedef int type; };

template<>
struct counterpart<double> { typedef long type; };

template<>
struct counterpart<int> { typedef float type; };

template<>
struct counterpart<long> { typedef double type; };

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
 * @brief 
 * 
 * @tparam T 
 */
template<typename T>
concept Integral = is_integral_v<T>;

/**
 * @brief 
 * 
 * @tparam T 
 */
template<typename T>
concept FloatingPoint = is_floating_point_v<T>;

/**
 * @brief 
 * 
 * @tparam T 
 */
template<typename T>
concept Scalar = is_scalar_v<T>;

/**
 * @brief 
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
 * @brief int_t (get the same-sized integral type for a floating-point type)
 * 
 * @tparam T 
 */
template<FloatingPoint T>
using int_t = counterpart_t<T>;

/**
 * @brief float_t (get the same-sized floating-point type for an integral type)
 * 
 * @tparam T 
 */
template<Integral T>
using float_t = counterpart_t<T>;

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
struct ieee_exp_bias<float> : std::integral_constant<int, 127> {};

template<>
struct ieee_exp_bias<double> : std::integral_constant<long, 1023> {};

template<typename T>
inline constexpr auto ieee_exp_bias_v = ieee_exp_bias<T>::value;

/**
 * @brief IEEE significand bits
 * 
 * @tparam T 
 */
template<typename T>
struct ieee_sig_bits { static_assert(deferred_false_v<T>, "IEEE significand bits not defined."); };

template<>
struct ieee_sig_bits<float> : std::integral_constant<int, 23> {};

template<>
struct ieee_sig_bits<double> : std::integral_constant<long, 52> {};

template<typename T>
inline constexpr auto ieee_sig_bits_v = ieee_sig_bits<T>::value;

} // namespace cxm 

#endif /* _vsl_core_h */
