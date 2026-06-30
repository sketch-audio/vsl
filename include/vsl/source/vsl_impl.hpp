#pragma once

#include <array>
#include <cstdint>

#if defined(__ARM_NEON__)
#define VSL_NEON 1
#include <arm_neon.h>
#elif defined(__SSE4_2__) || (defined(_MSC_VER) && defined(_M_X64)) // Assume X64 has SSE4.2
#define VSL_SSE4 1
#include <immintrin.h>
#else
#error "No SIMD support detected."
#endif

// We need FMA.
#if defined(VSL_SSE4) && !defined(__FMA__) && !defined(_MSC_VER)
#error "No FMA support detected. May need to add -mfma to your compiler flags."
#endif

#if defined(_MSC_VER)
#define force_inline inline __forceinline
#define vector_call __vectorcall
#else
#define force_inline inline __attribute__((always_inline))
#define vector_call
#endif

namespace vsl {

// MARK: - mask64x2

struct mask64x2 {
#if VSL_NEON
    using Vector = uint64x2_t;
    using Float = float64x2_t;
#elif VSL_SSE4
    using Vector = __m128i;
    using Float = __m128d;
#endif
    Vector data{};

    using Scalar = uint64_t;
    static constexpr auto num_lanes = size_t{2};
    using Scalar_arr = std::array<Scalar, num_lanes>;

private:
    static force_inline auto vector_call make_vector(Scalar x) -> Vector
    {
#if VSL_NEON
        return vld1q_dup_u64(&x);
#elif VSL_SSE4
        return _mm_set1_epi64x(std::bit_cast<int64_t>(x));
#endif
    }
    
    static force_inline auto vector_call make_vector(const Scalar_arr& arr) -> Vector
    {
#if VSL_NEON
        return vld1q_u64(arr.data());
#elif VSL_SSE4
        return _mm_set_epi64x(std::bit_cast<int64_t>(arr[1]), std::bit_cast<int64_t>(arr[0]));
#endif
    }

    static force_inline auto vector_call access(Vector data, size_t i) -> Scalar
    {
#if VSL_NEON
        switch (i) {
            case 0: return vgetq_lane_u64(data, 0);
            case 1: return vgetq_lane_u64(data, 1);
            default: return 0;
        }
#elif VSL_SSE4
        switch (i) {
            case 0: return std::bit_cast<Scalar>(_mm_extract_epi64(data, 0));
            case 1: return std::bit_cast<Scalar>(_mm_extract_epi64(data, 1));
            default: return Scalar{};
        }
#endif
    }

    static force_inline auto vector_call add(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vaddq_u64(a, b);
#elif VSL_SSE4
        return _mm_add_epi64(a, b);
#endif
    }

    static force_inline auto vector_call sub(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vsubq_u64(a, b);
#elif VSL_SSE4
        return _mm_sub_epi64(a, b);
#endif
    }

    static force_inline auto vector_call mul(Vector ab, Vector cd) -> Vector
    {
#if VSL_NEON
        // See: https://stackoverflow.com/questions/60236627/
        const auto a = vmovn_u64(ab);
        const auto c = vmovn_u64(cd);
        const auto hi = vmulq_u32(cd, vrev64q_u32(ab));
        return vmlal_u32(vshlq_n_u64(vpaddlq_u32(hi), 32), a, c);
#elif VSL_SSE4
        // See: https://stackoverflow.com/questions/17863411/
        const auto ac = _mm_mul_epu32(ab, cd);
        const auto b = _mm_srli_epi64(ab, 32);
        const auto bc = _mm_mul_epu32(b, cd);
        const auto d = _mm_srli_epi64(cd, 32);
        const auto ad = _mm_mul_epu32(ab, d);
        auto hi = _mm_add_epi64(bc, ad);
        hi = _mm_slli_epi64(hi, 32);
        return _mm_add_epi64(ac, hi);
#endif
    }

    static force_inline auto vector_call bit_and(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vandq_u64(a, b);
#elif VSL_SSE4
        return _mm_and_si128(a, b);
#endif
    }

    static force_inline auto vector_call bit_or(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vorrq_u64(a, b);
#elif VSL_SSE4
        return _mm_or_si128(a, b);
#endif
    }

    static force_inline auto vector_call bit_xor(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return veorq_u64(a, b);
#elif VSL_SSE4
        return _mm_xor_si128(a, b);
#endif
    }

    static force_inline auto vector_call bit_not(Vector a) -> Vector
    {
        return bit_xor(a, make_vector(0xffffffffffffffff));
    }

    static force_inline auto vector_call rshift(Vector a, int n) -> Vector
    {
#if VSL_NEON
        return vshlq_u64(a, vdupq_n_s64(-n));
#elif VSL_SSE4
        return _mm_srli_epi64(a, n);
#endif
    }

    static force_inline auto vector_call lshift(Vector a, int n) -> Vector
    {
#if VSL_NEON
        return vshlq_u64(a, vdupq_n_s64(n));
#elif VSL_SSE4
        return _mm_slli_epi64(a, n);
#endif
    }

    static force_inline auto vector_call equal(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vceqq_u64(a, b);
#elif VSL_SSE4
        return _mm_cmpeq_epi64(a, b);
#endif
    }

    static force_inline auto vector_call not_equal(Vector a, Vector b) -> Vector
    {
        return bit_not(equal(a, b));
    }

    static force_inline auto vector_call greater(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vcgtq_u64(a, b);
#elif VSL_SSE4
        return _mm_cmpgt_epi64(a, b); // SSE4.2
#endif
    }

    static force_inline auto vector_call greater_equal(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vcgeq_u64(a, b);
#elif VSL_SSE4
        return bit_or(greater(a, b), equal(a, b));
#endif
    }

    static force_inline auto vector_call less(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vcltq_u64(a, b);
#elif VSL_SSE4
        return bit_not(greater_equal(a, b));
#endif
    }

    static force_inline auto vector_call less_equal(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vcleq_u64(a, b);
#elif VSL_SSE4
        return bit_not(greater(a, b));
#endif
    }

    static force_inline auto vector_call select(Vector mask, Vector val_t, Vector val_f) -> Vector
    {
#if VSL_NEON
        return vbslq_u64(mask, val_t, val_f);
#elif VSL_SSE4
        return _mm_blendv_epi8(val_f, val_t, mask);
#endif
    }

    static force_inline auto vector_call any(Vector x) -> bool
    {
#if VSL_NEON
        return (vgetq_lane_u64(x, 0) | vgetq_lane_u64(x, 1)) != 0;
#elif VSL_SSE4
        return _mm_movemask_epi8(x) != 0;
#endif
    }

    static force_inline auto vector_call all(Vector x) -> bool
    {
#if VSL_NEON
        return (vgetq_lane_u64(x, 0) & vgetq_lane_u64(x, 1)) != 0;
#elif VSL_SSE4
        return _mm_movemask_epi8(x) == 0xffff;
#endif
    }

    // Convert (static_cast) as if unsigned to float
    static force_inline auto vector_call unsigned_to_float(Vector x) -> Float
    {
#if VSL_NEON
        return vcvtq_f64_u64(x);
#elif VSL_SSE4
        // See: https://stackoverflow.com/questions/41144668/
        auto xh = _mm_srli_epi64(x, 32);
        xh = _mm_or_si128(xh, _mm_castpd_si128(_mm_set1_pd(19342813113834066795298816.)));
        const auto xl = _mm_blend_epi16(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)), 0xcc);
        const auto f = _mm_sub_pd(_mm_castsi128_pd(xh), _mm_set1_pd(19342813118337666422669312.));
        return _mm_add_pd(f, _mm_castsi128_pd(xl));
#endif
    }

    // Convert (static_cast) as if signed to float
    static force_inline auto vector_call signed_to_float(Vector x) -> Float
    {
#if VSL_NEON
        return vcvtq_f64_s64(vreinterpretq_s64_u64(x));
#elif VSL_SSE4
        // See: https://stackoverflow.com/questions/41144668/
        auto xh = _mm_srai_epi32(x, 16);
        xh = _mm_blend_epi16(xh, _mm_setzero_si128(), 0x33);
        xh = _mm_add_epi64(xh, _mm_castpd_si128(_mm_set1_pd(442721857769029238784.)));              //  3*2^67
        const auto xl = _mm_blend_epi16(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)), 0x88);   //  2^52
        const auto f = _mm_sub_pd(_mm_castsi128_pd(xh), _mm_set1_pd(442726361368656609280.));          //  3*2^67 + 2^52
        return _mm_add_pd(f, _mm_castsi128_pd(xl));
#endif
    }

public:
    force_inline mask64x2() noexcept = default;
    force_inline mask64x2(Scalar x) noexcept : data{make_vector(x)} {}
    force_inline mask64x2(Scalar x0, Scalar x1) noexcept : data{make_vector(Scalar_arr{{x0, x1}})} {}
    force_inline mask64x2(const Scalar_arr& arr) noexcept : data{make_vector(arr)} {}
    force_inline mask64x2(Vector x) noexcept : data{x} {}

    force_inline auto vector_call operator[](size_t i) const noexcept -> Scalar { return access(data, i); }

    friend force_inline auto vector_call operator+(const mask64x2& lhs, const mask64x2& rhs) noexcept -> mask64x2 { return add(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator-(const mask64x2& lhs, const mask64x2& rhs) noexcept -> mask64x2 { return sub(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator*(const mask64x2& lhs, const mask64x2& rhs) noexcept -> mask64x2 { return mul(lhs.data, rhs.data); }

    force_inline auto vector_call operator+=(const mask64x2& other) noexcept -> mask64x2& { data = add(data, other.data); return *this; }
    force_inline auto vector_call operator-=(const mask64x2& other) noexcept -> mask64x2& { data = sub(data, other.data); return *this; }
    force_inline auto vector_call operator*=(const mask64x2& other) noexcept -> mask64x2& { data = mul(data, other.data); return *this; }

    friend force_inline auto vector_call operator&(const mask64x2& lhs, const mask64x2& rhs) noexcept -> mask64x2 { return bit_and(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator|(const mask64x2& lhs, const mask64x2& rhs) noexcept -> mask64x2 { return bit_or(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator^(const mask64x2& lhs, const mask64x2& rhs) noexcept -> mask64x2 { return bit_xor(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator~(const mask64x2& x) noexcept -> mask64x2 { return bit_not(x.data); }

    force_inline auto vector_call operator&=(const mask64x2& other) noexcept -> mask64x2& { data = bit_and(data, other.data); return *this; }
    force_inline auto vector_call operator|=(const mask64x2& other) noexcept -> mask64x2& { data = bit_or(data, other.data); return *this; }
    force_inline auto vector_call operator^=(const mask64x2& other) noexcept -> mask64x2& { data = bit_xor(data, other.data); return *this; }

    force_inline auto vector_call operator>>(int n) const noexcept -> mask64x2 { return rshift(data, n); }
    force_inline auto vector_call operator<<(int n) const noexcept -> mask64x2 { return lshift(data, n); }

    force_inline auto vector_call operator>>=(int n) noexcept -> mask64x2& { data = rshift(data, n); return *this; }
    force_inline auto vector_call operator<<=(int n) noexcept -> mask64x2& { data = lshift(data, n); return *this; }

    force_inline auto vector_call operator==(const mask64x2& other) const noexcept -> mask64x2 { return equal(data, other.data); }
    force_inline auto vector_call operator!=(const mask64x2& other) const noexcept -> mask64x2 { return not_equal(data, other.data); }
    force_inline auto vector_call operator<(const mask64x2& other) const noexcept -> mask64x2 { return less(data, other.data); }
    force_inline auto vector_call operator<=(const mask64x2& other) const noexcept -> mask64x2 { return less_equal(data, other.data); }
    force_inline auto vector_call operator>(const mask64x2& other) const noexcept -> mask64x2 { return greater(data, other.data); }
    force_inline auto vector_call operator>=(const mask64x2& other) const noexcept -> mask64x2 { return greater_equal(data, other.data); }

    static force_inline auto vector_call select(mask64x2 mask, mask64x2 val_t, mask64x2 val_f) -> mask64x2 { return select(mask.data, val_t.data, val_f.data); }
    static force_inline auto vector_call any(mask64x2 x) -> bool { return any(x.data); }
    static force_inline auto vector_call all(mask64x2 x) -> bool { return all(x.data); }

    static force_inline auto vector_call unsigned_to_float(mask64x2 x) -> Float { return unsigned_to_float(x.data); }
    static force_inline auto vector_call signed_to_float(mask64x2 x) -> Float { return signed_to_float(x.data); }

};

// MARK: - float64x2

struct float64x2 {
#if VSL_NEON
    using Vector = float64x2_t;
    using Mask = uint64x2_t;
#elif VSL_SSE4
    using Vector = __m128d;
    using Mask = __m128i;
#endif
    Vector data{};

    using Scalar = double;
    static constexpr auto num_lanes = size_t{2};
    using Scalar_arr = std::array<Scalar, num_lanes>;

private:
    static force_inline auto vector_call make_vector(Scalar x) -> Vector
    {
#if VSL_NEON
        return vld1q_dup_f64(&x);
#elif VSL_SSE4
        return _mm_set1_pd(x);
#endif
    }

    static force_inline auto vector_call make_vector(const Scalar_arr& arr) -> Vector
    {
#if VSL_NEON
        return vld1q_f64(arr.data());
#elif VSL_SSE4
        return _mm_set_pd(arr[1], arr[0]);
#endif
    }
    
    static force_inline auto vector_call access(Vector data, size_t i) -> Scalar
    {
#if VSL_NEON
        switch (i) {
            case 0: return vgetq_lane_f64(data, 0);
            case 1: return vgetq_lane_f64(data, 1);
            default: return 0;
        }
#elif VSL_SSE4
        Scalar x;
        switch (i) {
            case 0: _mm_store_sd(&x, _mm_shuffle_pd(data, data, _MM_SHUFFLE2(0, 0))); return x;
            case 1: _mm_store_sd(&x, _mm_shuffle_pd(data, data, _MM_SHUFFLE2(1, 1))); return x;
            default: return 0;
        }
#endif
    }

    static force_inline auto vector_call add(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vaddq_f64(a, b);
#elif VSL_SSE4
        return _mm_add_pd(a, b);
#endif
    }

    static force_inline auto vector_call sub(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vsubq_f64(a, b);
#elif VSL_SSE4
        return _mm_sub_pd(a, b);
#endif
    }

    static force_inline auto vector_call mul(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vmulq_f64(a, b);
#elif VSL_SSE4
        return _mm_mul_pd(a, b);
#endif
    }

    static force_inline auto vector_call fma(Vector a, Vector b, Vector c) -> Vector
    {
#if VSL_NEON
        return vfmaq_f64(c, a, b);
#elif VSL_SSE4
        return _mm_fmadd_pd(a, b, c);
#endif
    }

    static force_inline auto vector_call div(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vdivq_f64(a, b);
#elif VSL_SSE4
        return _mm_div_pd(a, b);
#endif
    }

    static force_inline auto vector_call neg(Vector a) -> Vector
    {
        return mul(a, make_vector(-1));
    }

    static force_inline auto vector_call as_mask(Vector a) -> Mask
    {
#if VSL_NEON
        return vreinterpretq_u64_f64(a);
#elif VSL_SSE4
        return _mm_castpd_si128(a);
#endif
    }

    static force_inline auto vector_call as_vector(Mask a) -> Vector
    {
#if VSL_NEON
        return vreinterpretq_f64_u64(a);
#elif VSL_SSE4
        return _mm_castsi128_pd(a);
#endif
    }

    static force_inline auto vector_call equal(Vector a, Vector b) -> Mask
    {
#if VSL_NEON
        return vceqq_f64(a, b);
#elif VSL_SSE4
        return as_mask(_mm_cmpeq_pd(a, b));
#endif
    }

    static force_inline auto vector_call not_equal(Vector a, Vector b) -> Mask
    {
#if VSL_NEON
        return veorq_u64(equal(a, b), vdupq_n_u64(0xffffffffffffffff));
#elif VSL_SSE4
        return as_mask(_mm_cmpneq_pd(a, b));
#endif
    }

    static force_inline auto vector_call less(Vector a, Vector b) -> Mask
    {
#if VSL_NEON
        return vcltq_f64(a, b);
#elif VSL_SSE4
        return as_mask(_mm_cmplt_pd(a, b));
#endif
    }

    static force_inline auto vector_call less_equal(Vector a, Vector b) -> Mask
    {
#if VSL_NEON
        return vcleq_f64(a, b);
#elif VSL_SSE4
        return as_mask(_mm_cmple_pd(a, b));
#endif
    }

    static force_inline auto vector_call greater(Vector a, Vector b) -> Mask
    {
#if VSL_NEON
        return vcgtq_f64(a, b);
#elif VSL_SSE4
        return as_mask(_mm_cmpgt_pd(a, b));
#endif
    }

    static force_inline auto vector_call greater_equal(Vector a, Vector b) -> Mask
    {
#if VSL_NEON
        return vcgeq_f64(a, b);
#elif VSL_SSE4
        return as_mask(_mm_cmpge_pd(a, b));
#endif
    }

    static force_inline auto vector_call select(Mask mask, Vector val_t, Vector val_f) -> Vector
    {
#if VSL_NEON
        return vbslq_f64(mask, val_t, val_f);
#elif VSL_SSE4
        return _mm_blendv_pd(val_f, val_t, as_vector(mask));
#endif
    }

    // Convert (static_cast) to signed.
    static force_inline auto vector_call float_to_signed(Vector x) -> Mask
    {
#if VSL_NEON
        return vcvtq_s64_f64(x);
#elif VSL_SSE4
        // See: https://stackoverflow.com/questions/41144668/
        // Only works for inputs in the range: [-2^51, 2^51]
        x = _mm_round_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        x = _mm_add_pd(x, _mm_set1_pd(0x0018000000000000));
        return _mm_sub_epi64(_mm_castpd_si128(x), _mm_castpd_si128(_mm_set1_pd(0x0018000000000000)));
#endif
    }

    // Convert (static_cast) to unsigned.
    static force_inline auto vector_call float_to_unsigned(Vector x) -> Mask
    {
#if VSL_NEON
        return vcvtq_u64_f64(x);
#elif VSL_SSE4
        // See: https://stackoverflow.com/questions/41144668/
        // Only works for inputs in the range: [0, 2^52]
        x = _mm_round_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        x = _mm_add_pd(x, _mm_set1_pd(0x0010000000000000));
        return _mm_sub_epi64(_mm_castpd_si128(x), _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)));
#endif
    }

    static force_inline auto vector_call trunc(Vector x) -> Vector
    {
#if VSL_NEON
        return vrndq_f64(x); // round toward zero (FRINTZ)
#elif VSL_SSE4
        return _mm_round_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#endif
    }

    static force_inline auto vector_call floor(Vector x) -> Vector
    {
#if VSL_NEON
        return vrndmq_f64(x);
#elif VSL_SSE4
        return _mm_floor_pd(x);
#endif
    }

    static force_inline auto vector_call ceil(Vector x) -> Vector
    {
#if VSL_NEON
        return vrndpq_f64(x);
#elif VSL_SSE4
        return _mm_ceil_pd(x);
#endif
    }

    static force_inline auto vector_call round(Vector x) -> Vector
    {
#if VSL_NEON
        return vrndaq_f64(x); // ties away from zero (FRINTA), matches std::round
#elif VSL_SSE4
        // ROUNDPD only offers ties-to-even; std::round is ties-away. Build it from
        // trunc plus a copysign(1) bump when the (exact) fractional part is >= 0.5.
        const auto sign_mask = _mm_set1_pd(-0.0);
        const auto tz = _mm_round_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC); // toward zero
        const auto frac = _mm_andnot_pd(sign_mask, _mm_sub_pd(x, tz)); // |x - trunc(x)|
        const auto bump = _mm_or_pd(_mm_and_pd(x, sign_mask), _mm_set1_pd(1.0)); // copysign(1, x)
        const auto ge_half = _mm_cmpge_pd(frac, _mm_set1_pd(0.5));
        return _mm_add_pd(tz, _mm_and_pd(ge_half, bump));
#endif
    }

    static force_inline auto vector_call abs(Vector x) -> Vector
    {
#if VSL_NEON
        return vabsq_f64(x);
#elif VSL_SSE4
        return _mm_andnot_pd(_mm_set1_pd(-0.), x);
#endif
    }

    static force_inline auto vector_call sqrt(Vector x) -> Vector
    {
#if VSL_NEON
        return vsqrtq_f64(x);
#elif VSL_SSE4
        return _mm_sqrt_pd(x);
#endif
    }

    static force_inline auto vector_call rsqrt(Vector x) -> Vector
    {
#if VSL_NEON
        return vrsqrteq_f64(x);
#elif VSL_SSE4
        return _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(x));
#endif
    }

    static force_inline auto vector_call min(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vminq_f64(a, b);
#elif VSL_SSE4
        return _mm_min_pd(a, b);
#endif
    }

    static force_inline auto vector_call max(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vmaxq_f64(a, b);
#elif VSL_SSE4
        return _mm_max_pd(a, b);
#endif
    }

public:
    force_inline float64x2() noexcept = default;
    force_inline float64x2(Scalar x) noexcept : data{make_vector(x)} {}
    force_inline float64x2(Scalar x0, Scalar x1) noexcept : data{make_vector(Scalar_arr{{x0, x1}})} {}
    force_inline float64x2(const Scalar_arr& arr) noexcept : data{make_vector(arr)} {}
    force_inline float64x2(Vector x) noexcept : data{x} {}

    force_inline auto vector_call operator[](size_t i) const noexcept -> Scalar { return access(data, i); }

    friend force_inline auto vector_call operator+(const float64x2& lhs, const float64x2& rhs) noexcept -> float64x2 { return add(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator-(const float64x2& lhs, const float64x2& rhs) noexcept -> float64x2 { return sub(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator*(const float64x2& lhs, const float64x2& rhs) noexcept -> float64x2 { return mul(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator/(const float64x2& lhs, const float64x2& rhs) noexcept -> float64x2 { return div(lhs.data, rhs.data); }

    force_inline auto vector_call operator+=(const float64x2& other) noexcept -> float64x2& { data = add(data, other.data); return *this; }
    force_inline auto vector_call operator-=(const float64x2& other) noexcept -> float64x2& { data = sub(data, other.data); return *this; }
    force_inline auto vector_call operator*=(const float64x2& other) noexcept -> float64x2& { data = mul(data, other.data); return *this; }
    force_inline auto vector_call operator/=(const float64x2& other) noexcept -> float64x2& { data = div(data, other.data); return *this; }

    force_inline auto vector_call operator+() const noexcept -> float64x2 { return *this; }
    force_inline auto vector_call operator-() const noexcept -> float64x2 { return neg(data); }

    friend force_inline auto vector_call operator==(const float64x2& a, const float64x2& b) noexcept -> mask64x2 { return equal(a.data, b.data); }
    friend force_inline auto vector_call operator!=(const float64x2& a, const float64x2& b) noexcept -> mask64x2 { return not_equal(a.data, b.data); }
    friend force_inline auto vector_call operator<(const float64x2& a, const float64x2& b) noexcept -> mask64x2 { return less(a.data, b.data); }
    friend force_inline auto vector_call operator<=(const float64x2& a, const float64x2& b) noexcept -> mask64x2 { return less_equal(a.data, b.data); }
    friend force_inline auto vector_call operator>(const float64x2& a, const float64x2& b) noexcept -> mask64x2 { return greater(a.data, b.data); }
    friend force_inline auto vector_call operator>=(const float64x2& a, const float64x2& b) noexcept -> mask64x2 { return greater_equal(a.data, b.data); }

    static force_inline auto vector_call select(mask64x2 mask, float64x2 val_t, float64x2 val_f) noexcept -> float64x2 { return select(mask.data, val_t.data, val_f.data); }

    static force_inline auto vector_call float_to_signed(float64x2 x) noexcept -> Mask { return float_to_signed(x.data); }
    static force_inline auto vector_call float_to_unsigned(float64x2 x) noexcept -> Mask { return float_to_unsigned(x.data); }

    static force_inline auto vector_call trunc(float64x2 x) noexcept -> float64x2 { return trunc(x.data); }
    static force_inline auto vector_call floor(float64x2 x) noexcept -> float64x2 { return floor(x.data); }
    static force_inline auto vector_call ceil(float64x2 x) noexcept -> float64x2 { return ceil(x.data); }
    static force_inline auto vector_call round(float64x2 x) noexcept -> float64x2 { return round(x.data); }
    static force_inline auto vector_call abs(float64x2 x) noexcept -> float64x2 { return abs(x.data); }

    static force_inline auto vector_call sqrt(float64x2 x) noexcept -> float64x2 { return sqrt(x.data); }
    static force_inline auto vector_call rsqrt(float64x2 x) noexcept -> float64x2 { return rsqrt(x.data); }
    static force_inline auto vector_call min(float64x2 a, float64x2 b) noexcept -> float64x2 { return min(a.data, b.data); }
    static force_inline auto vector_call max(float64x2 a, float64x2 b) noexcept -> float64x2 { return max(a.data, b.data); }
    static force_inline auto vector_call fma(float64x2 a, float64x2 b, float64x2 c) noexcept -> float64x2 { return fma(a.data, b.data, c.data); }

};

// MARK: - mask32x4

struct mask32x4 {
#if VSL_NEON
    using Vector = uint32x4_t;
    using Float = float32x4_t;
#elif VSL_SSE4
    using Vector = __m128i;
    using Float = __m128;
#endif
    Vector data{};

    using Scalar = uint32_t;
    static constexpr auto num_lanes = size_t{4};
    using Scalar_arr = std::array<Scalar, num_lanes>;

private:
    static force_inline auto vector_call make_vector(Scalar x) -> Vector
    {
#if VSL_NEON
        return vld1q_dup_u32(&x);
#elif VSL_SSE4
        return _mm_set1_epi32(std::bit_cast<int32_t>(x));
#endif
    }

    static force_inline auto vector_call make_vector(const Scalar_arr& arr) -> Vector
    {
#if VSL_NEON
        return vld1q_u32(arr.data());
#elif VSL_SSE4
        return _mm_set_epi32(std::bit_cast<int32_t>(arr[3]), std::bit_cast<int32_t>(arr[2]), std::bit_cast<int32_t>(arr[1]), std::bit_cast<int32_t>(arr[0]));
#endif
    }

    static force_inline auto vector_call access(Vector data, size_t i) -> Scalar
    {
#if VSL_NEON
        switch (i) {
            case 0: return vgetq_lane_u32(data, 0);
            case 1: return vgetq_lane_u32(data, 1);
            case 2: return vgetq_lane_u32(data, 2);
            case 3: return vgetq_lane_u32(data, 3);
            default: return 0;
        }
#elif VSL_SSE4
        switch (i) {
            case 0: return std::bit_cast<Scalar>(_mm_extract_epi32(data, 0));
            case 1: return std::bit_cast<Scalar>(_mm_extract_epi32(data, 1));
            case 2: return std::bit_cast<Scalar>(_mm_extract_epi32(data, 2));
            case 3: return std::bit_cast<Scalar>(_mm_extract_epi32(data, 3));
            default: return Scalar{};
        }
#endif
    }

    static force_inline auto vector_call add(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vaddq_u32(a, b);
#elif VSL_SSE4
        return _mm_add_epi32(a, b);
#endif
    }

    static force_inline auto vector_call sub(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vsubq_u32(a, b);
#elif VSL_SSE4
        return _mm_sub_epi32(a, b);
#endif
    }

    static force_inline auto vector_call mul(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vmulq_u32(a, b);
#elif VSL_SSE4
        return _mm_mullo_epi32(a, b);
#endif
    }

    static force_inline auto vector_call bit_and(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vandq_u32(a, b);
#elif VSL_SSE4
        return _mm_and_si128(a, b);
#endif
    }

    static force_inline auto vector_call bit_or(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vorrq_u32(a, b);
#elif VSL_SSE4
        return _mm_or_si128(a, b);
#endif
    }

    static force_inline auto vector_call bit_xor(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return veorq_u32(a, b);
#elif VSL_SSE4
        return _mm_xor_si128(a, b);
#endif
    }

    static force_inline auto vector_call bit_not(Vector a) -> Vector
    {
        return bit_xor(a, make_vector(0xffffffff));
    }

    static force_inline auto vector_call rshift(Vector a, int n) -> Vector
    {
#if VSL_NEON
        return vshlq_u32(a, vdupq_n_s32(-n));
#elif VSL_SSE4
        return _mm_srli_epi32(a, n);
#endif
    }

    static force_inline auto vector_call lshift(Vector a, int n) -> Vector
    {
#if VSL_NEON
        return vshlq_u32(a, vdupq_n_s32(n));
#elif VSL_SSE4
        return _mm_slli_epi32(a, n);
#endif
    }

    static force_inline auto vector_call equal(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vceqq_u32(a, b);
#elif VSL_SSE4
        return _mm_cmpeq_epi32(a, b);
#endif
    }

    static force_inline auto vector_call not_equal(Vector a, Vector b) -> Vector
    {
        return bit_not(equal(a, b));
    }

    static force_inline auto vector_call greater(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vcgtq_u32(a, b);
#elif VSL_SSE4
        return _mm_cmpgt_epi32(a, b);
#endif
    }

    static force_inline auto vector_call greater_equal(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vcgeq_u32(a, b);
#elif VSL_SSE4
        return bit_or(greater(a, b), equal(a, b));
#endif
    }

    static force_inline auto vector_call less(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vcltq_u32(a, b);
#elif VSL_SSE4
        return _mm_cmplt_epi32(a, b);
#endif
    }

    static force_inline auto vector_call less_equal(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vcleq_u32(a, b);
#elif VSL_SSE4
        return bit_or(less(a, b), equal(a, b));
#endif
    }

    static force_inline auto vector_call select(Vector mask, Vector val_t, Vector val_f) -> Vector
    {
#if VSL_NEON
        return vbslq_u32(mask, val_t, val_f);
#elif VSL_SSE4
        return _mm_blendv_epi8(val_f, val_t, mask);
#endif
    }

    static force_inline auto vector_call any(Vector x) -> bool
    {
#if VSL_NEON
        return vmaxvq_u32(x) != 0;
#elif VSL_SSE4
        return _mm_movemask_epi8(x) != 0;
#endif
    }

    static force_inline auto vector_call all(Vector x) -> bool
    {
#if VSL_NEON
        return vminvq_u32(x) != 0;
#elif VSL_SSE4
        return _mm_movemask_epi8(x) == 0xffff;
#endif
    }

    // Convert (static_cast) as if unsigned to float
    static force_inline auto vector_call unsigned_to_float(Vector x) -> Float
    {
#if VSL_NEON
        return vcvtq_f32_u32(x);
#elif VSL_SSE4
        // See: https://stackoverflow.com/questions/34066228
        const auto x2 = _mm_srli_epi32(x, 1);
        const auto x1 = _mm_and_si128(x, _mm_set1_epi32(1));
        const auto x2f = _mm_cvtepi32_ps(x2);
        const auto x1f = _mm_cvtepi32_ps(x1);
        return _mm_add_ps(_mm_add_ps(x2f, x2f), x1f);
#endif
    }

    // Convert (static_cast) as if signed to float
    static force_inline auto vector_call signed_to_float(Vector x) -> Float
    {
#if VSL_NEON
        return vcvtq_f32_s32(vreinterpretq_s32_u32(x));
#elif VSL_SSE4
        return _mm_cvtepi32_ps(x);
#endif
    }

public:
    force_inline mask32x4() noexcept = default;
    force_inline mask32x4(Scalar x) noexcept : data{make_vector(x)} {}
    force_inline mask32x4(Scalar x0, Scalar x1, Scalar x2, Scalar x3) noexcept : data{make_vector(Scalar_arr{{x0, x1, x2, x3}})} {}
    force_inline mask32x4(const Scalar_arr& arr) noexcept : data{make_vector(arr)} {}
    force_inline mask32x4(Vector x) noexcept : data{x} {}

    force_inline auto vector_call operator[](size_t i) const noexcept -> Scalar { return access(data, i); }

    friend force_inline auto vector_call operator+(const mask32x4& lhs, const mask32x4& rhs) noexcept -> mask32x4 { return add(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator-(const mask32x4& lhs, const mask32x4& rhs) noexcept -> mask32x4 { return sub(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator*(const mask32x4& lhs, const mask32x4& rhs) noexcept -> mask32x4 { return mul(lhs.data, rhs.data); }

    force_inline auto vector_call operator+=(const mask32x4& other) noexcept -> mask32x4& { data = add(data, other.data); return *this; }
    force_inline auto vector_call operator-=(const mask32x4& other) noexcept -> mask32x4& { data = sub(data, other.data); return *this; }
    force_inline auto vector_call operator*=(const mask32x4& other) noexcept -> mask32x4& { data = mul(data, other.data); return *this; }

    friend force_inline auto vector_call operator&(const mask32x4& lhs, const mask32x4& rhs) noexcept -> mask32x4 { return bit_and(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator|(const mask32x4& lhs, const mask32x4& rhs) noexcept -> mask32x4 { return bit_or(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator^(const mask32x4& lhs, const mask32x4& rhs) noexcept -> mask32x4 { return bit_xor(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator~(const mask32x4& x) noexcept -> mask32x4 { return bit_not(x.data); }

    force_inline auto vector_call operator&=(const mask32x4& other) noexcept -> mask32x4& { data = bit_and(data, other.data); return *this; }
    force_inline auto vector_call operator|=(const mask32x4& other) noexcept -> mask32x4& { data = bit_or(data, other.data); return *this; }
    force_inline auto vector_call operator^=(const mask32x4& other) noexcept -> mask32x4& { data = bit_xor(data, other.data); return *this; }

    force_inline auto vector_call operator>>(int n) const noexcept -> mask32x4 { return rshift(data, n); }
    force_inline auto vector_call operator<<(int n) const noexcept -> mask32x4 { return lshift(data, n); }

    force_inline auto vector_call operator>>=(int n) noexcept -> mask32x4& { data = rshift(data, n); return *this; }
    force_inline auto vector_call operator<<=(int n) noexcept -> mask32x4& { data = lshift(data, n); return *this; }

    force_inline auto vector_call operator==(const mask32x4& other) const noexcept -> mask32x4 { return equal(data, other.data); }
    force_inline auto vector_call operator!=(const mask32x4& other) const noexcept -> mask32x4 { return not_equal(data, other.data); }
    force_inline auto vector_call operator<(const mask32x4& other) const noexcept -> mask32x4 { return less(data, other.data); }
    force_inline auto vector_call operator<=(const mask32x4& other) const noexcept -> mask32x4 { return less_equal(data, other.data); }
    force_inline auto vector_call operator>(const mask32x4& other) const noexcept -> mask32x4 { return greater(data, other.data); }
    force_inline auto vector_call operator>=(const mask32x4& other) const noexcept -> mask32x4 { return greater_equal(data, other.data); }

    static force_inline auto vector_call select(mask32x4 mask, mask32x4 val_t, mask32x4 val_f) -> mask32x4 { return select(mask.data, val_t.data, val_f.data); }
    static force_inline auto vector_call any(mask32x4 x) -> bool { return any(x.data); }
    static force_inline auto vector_call all(mask32x4 x) -> bool { return all(x.data); }

    static force_inline auto vector_call unsigned_to_float(mask32x4 x) -> Float { return unsigned_to_float(x.data); }
    static force_inline auto vector_call signed_to_float(mask32x4 x) -> Float { return signed_to_float(x.data); }

};

// MARK: - float32x4

struct float32x4 {
#if VSL_NEON
    using Vector = float32x4_t;
    using Mask = int32x4_t;
#elif VSL_SSE4
    using Vector = __m128;
    using Mask = __m128i;
#endif
    Vector data{};

    using Scalar = float;
    static constexpr auto num_lanes = size_t{4};
    using Scalar_arr = std::array<Scalar, num_lanes>;

private:
    static force_inline auto vector_call make_vector(Scalar x) -> Vector
    {
#if VSL_NEON
        return vld1q_dup_f32(&x);
#elif VSL_SSE4
        return _mm_set1_ps(x);
#endif
    }

    static force_inline auto vector_call make_vector(const Scalar_arr& arr) -> Vector
    {
#if VSL_NEON
        return vld1q_f32(arr.data());
#elif VSL_SSE4
        return _mm_set_ps(arr[3], arr[2], arr[1], arr[0]);
#endif
    }

    static force_inline auto vector_call access(Vector data, size_t i) -> Scalar
    {
#if VSL_NEON
        switch (i) {
            case 0: return vgetq_lane_f32(data, 0);
            case 1: return vgetq_lane_f32(data, 1);
            case 2: return vgetq_lane_f32(data, 2);
            case 3: return vgetq_lane_f32(data, 3);
            default: return 0;
        }
#elif VSL_SSE4
        Scalar x;
        switch (i) {
            case 0: _mm_store_ss(&x, _mm_shuffle_ps(data, data, _MM_SHUFFLE(0, 0, 0, 0))); return x;
            case 1: _mm_store_ss(&x, _mm_shuffle_ps(data, data, _MM_SHUFFLE(1, 1, 1, 1))); return x;
            case 2: _mm_store_ss(&x, _mm_shuffle_ps(data, data, _MM_SHUFFLE(2, 2, 2, 2))); return x;
            case 3: _mm_store_ss(&x, _mm_shuffle_ps(data, data, _MM_SHUFFLE(3, 3, 3, 3))); return x;
            default: return 0;
        }
#endif
    }

    static force_inline auto vector_call add(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vaddq_f32(a, b);
#elif VSL_SSE4
        return _mm_add_ps(a, b);
#endif
    }

    static force_inline auto vector_call sub(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vsubq_f32(a, b);
#elif VSL_SSE4
        return _mm_sub_ps(a, b);
#endif
    }

    static force_inline auto vector_call mul(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vmulq_f32(a, b);
#elif VSL_SSE4
        return _mm_mul_ps(a, b);
#endif
    }

    static force_inline auto vector_call fma(Vector a, Vector b, Vector c) -> Vector
    {
#if VSL_NEON
        return vfmaq_f32(c, a, b);
#elif VSL_SSE4
        return _mm_fmadd_ps(a, b, c);
#endif
    }

    static force_inline auto vector_call div(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vdivq_f32(a, b);
#elif VSL_SSE4
        return _mm_div_ps(a, b);
#endif
    }

    static force_inline auto vector_call neg(Vector a) -> Vector
    {
        return mul(a, make_vector(-1));
    }

    static force_inline auto vector_call as_mask(Vector a) -> Mask
    {
#if VSL_NEON
        return vreinterpretq_u32_f32(a);
#elif VSL_SSE4
        return _mm_castps_si128(a);
#endif
    }

    static force_inline auto vector_call as_vector(Mask a) -> Vector
    {
#if VSL_NEON
        return vreinterpretq_f32_u32(a);
#elif VSL_SSE4
        return _mm_castsi128_ps(a);
#endif
    }

    static force_inline auto vector_call equal(Vector a, Vector b) -> Mask
    {
#if VSL_NEON
        return vceqq_f32(a, b);
#elif VSL_SSE4
        return as_mask(_mm_cmpeq_ps(a, b));
#endif
    }

    static force_inline auto vector_call not_equal(Vector a, Vector b) -> Mask
    {
#if VSL_NEON
        return veorq_u32(equal(a, b), vdupq_n_u32(0xffffffff));
#elif VSL_SSE4
        return as_mask(_mm_cmpneq_ps(a, b));
#endif
    }

    static force_inline auto vector_call less(Vector a, Vector b) -> Mask
    {
#if VSL_NEON
        return vcltq_f32(a, b);
#elif VSL_SSE4
        return as_mask(_mm_cmplt_ps(a, b));
#endif
    }

    static force_inline auto vector_call less_equal(Vector a, Vector b) -> Mask
    {
#if VSL_NEON
        return vcleq_f32(a, b);
#elif VSL_SSE4
        return as_mask(_mm_cmple_ps(a, b));
#endif
    }

    static force_inline auto vector_call greater(Vector a, Vector b) -> Mask
    {
#if VSL_NEON
        return vcgtq_f32(a, b);
#elif VSL_SSE4
        return as_mask(_mm_cmpgt_ps(a, b));
#endif
    }

    static force_inline auto vector_call greater_equal(Vector a, Vector b) -> Mask
    {
#if VSL_NEON    
        return vcgeq_f32(a, b);
#elif VSL_SSE4
        return as_mask(_mm_cmpge_ps(a, b));
#endif
    }

    static force_inline auto vector_call select(Mask mask, Vector val_t, Vector val_f) -> Vector
    {
#if VSL_NEON
        return vbslq_f32(mask, val_t, val_f);
#elif VSL_SSE4
        return _mm_blendv_ps(val_f, val_t, as_vector(mask));
#endif
    }

    // Convert (static_cast) to signed.
    static force_inline auto vector_call float_to_signed(Vector x) -> Mask
    {
#if VSL_NEON
        return vcvtq_s32_f32(x);
#elif VSL_SSE4
        return _mm_cvttps_epi32(x);
#endif
    }

    // Convert (static_cast) to unsigned.
    static force_inline auto vector_call float_to_unsigned(Vector x) -> Mask
    {
#if VSL_NEON
        return vcvtq_u32_f32(x);
#elif VSL_SSE4
        return _mm_cvttps_epi32(x); // For now.
#endif
    }

    static force_inline auto vector_call trunc(Vector x) -> Vector
    {
#if VSL_NEON
        return vrndq_f32(x); // round toward zero (FRINTZ)
#elif VSL_SSE4
        return _mm_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#endif
    }

    static force_inline auto vector_call floor(Vector x) -> Vector
    {
#if VSL_NEON
        return vrndmq_f32(x);
#elif VSL_SSE4
        return _mm_floor_ps(x);
#endif
    }

    static force_inline auto vector_call ceil(Vector x) -> Vector
    {
#if VSL_NEON
        return vrndpq_f32(x);
#elif VSL_SSE4
        return _mm_ceil_ps(x);
#endif
    }

    static force_inline auto vector_call round(Vector x) -> Vector
    {
#if VSL_NEON
        return vrndaq_f32(x); // ties away from zero (FRINTA), matches std::round
#elif VSL_SSE4
        // ROUNDPS only offers ties-to-even; std::round is ties-away. Build it from
        // trunc plus a copysign(1) bump when the (exact) fractional part is >= 0.5.
        const auto sign_mask = _mm_set1_ps(-0.0f);
        const auto tz = _mm_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC); // toward zero
        const auto frac = _mm_andnot_ps(sign_mask, _mm_sub_ps(x, tz)); // |x - trunc(x)|
        const auto bump = _mm_or_ps(_mm_and_ps(x, sign_mask), _mm_set1_ps(1.0f)); // copysign(1, x)
        const auto ge_half = _mm_cmpge_ps(frac, _mm_set1_ps(0.5f));
        return _mm_add_ps(tz, _mm_and_ps(ge_half, bump));
#endif
    }

    static force_inline auto vector_call abs(Vector x) -> Vector
    {
#if VSL_NEON
        return vabsq_f32(x);
#elif VSL_SSE4
        return _mm_andnot_ps(_mm_set1_ps(-0.f), x);
#endif
    }

    static force_inline auto vector_call sqrt(Vector x) -> Vector
    {
#if VSL_NEON
        return vsqrtq_f32(x);
#elif VSL_SSE4
        return _mm_sqrt_ps(x);
#endif
    }

    static force_inline auto vector_call rsqrt(Vector x) -> Vector
    {
#if VSL_NEON
        return vrsqrteq_f32(x);
#elif VSL_SSE4
        return _mm_rsqrt_ps(x);
#endif
    }

    static force_inline auto vector_call min(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vminq_f32(a, b);
#elif VSL_SSE4
        return _mm_min_ps(a, b);
#endif
    }

    static force_inline auto vector_call max(Vector a, Vector b) -> Vector
    {
#if VSL_NEON
        return vmaxq_f32(a, b);
#elif VSL_SSE4
        return _mm_max_ps(a, b);
#endif
    }

public:
    force_inline float32x4() noexcept = default;
    force_inline float32x4(Scalar x) noexcept : data{make_vector(x)} {}
    force_inline float32x4(Scalar x0, Scalar x1, Scalar x2, Scalar x3) noexcept : data{make_vector(Scalar_arr{{x0, x1, x2, x3}})} {}
    force_inline float32x4(const Scalar_arr& arr) noexcept : data{make_vector(arr)} {}
    force_inline float32x4(Vector x) noexcept : data{x} {}

    force_inline auto vector_call operator[](size_t i) const noexcept -> Scalar { return access(data, i); }

    friend force_inline auto vector_call operator+(const float32x4& lhs, const float32x4& rhs) noexcept -> float32x4 { return add(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator-(const float32x4& lhs, const float32x4& rhs) noexcept -> float32x4 { return sub(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator*(const float32x4& lhs, const float32x4& rhs) noexcept -> float32x4 { return mul(lhs.data, rhs.data); }
    friend force_inline auto vector_call operator/(const float32x4& lhs, const float32x4& rhs) noexcept -> float32x4 { return div(lhs.data, rhs.data); }

    force_inline auto vector_call operator+=(const float32x4& other) noexcept -> float32x4& { data = add(data, other.data); return *this; }
    force_inline auto vector_call operator-=(const float32x4& other) noexcept -> float32x4& { data = sub(data, other.data); return *this; }
    force_inline auto vector_call operator*=(const float32x4& other) noexcept -> float32x4& { data = mul(data, other.data); return *this; }
    force_inline auto vector_call operator/=(const float32x4& other) noexcept -> float32x4& { data = div(data, other.data); return *this; }

    force_inline auto vector_call operator+() const noexcept -> float32x4 { return *this; }
    force_inline auto vector_call operator-() const noexcept -> float32x4 { return neg(data); }

    friend force_inline auto vector_call operator==(const float32x4& a, const float32x4& b) noexcept -> mask32x4 { return equal(a.data, b.data); }
    friend force_inline auto vector_call operator!=(const float32x4& a, const float32x4& b) noexcept -> mask32x4 { return not_equal(a.data, b.data); }
    friend force_inline auto vector_call operator<(const float32x4& a, const float32x4& b) noexcept -> mask32x4 { return less(a.data, b.data); }
    friend force_inline auto vector_call operator<=(const float32x4& a, const float32x4& b) noexcept -> mask32x4 { return less_equal(a.data, b.data); }
    friend force_inline auto vector_call operator>(const float32x4& a, const float32x4& b) noexcept -> mask32x4 { return greater(a.data, b.data); }
    friend force_inline auto vector_call operator>=(const float32x4& a, const float32x4& b) noexcept -> mask32x4 { return greater_equal(a.data, b.data); }

    static force_inline auto vector_call select(mask32x4 mask, float32x4 val_t, float32x4 val_f) noexcept -> float32x4 { return select(mask.data, val_t.data, val_f.data); }

    static force_inline auto vector_call float_to_signed(float32x4 x) noexcept -> Mask { return float_to_signed(x.data); }
    static force_inline auto vector_call float_to_unsigned(float32x4 x) noexcept -> Mask { return float_to_unsigned(x.data); }

    static force_inline auto vector_call trunc(float32x4 x) noexcept -> float32x4 { return trunc(x.data); }
    static force_inline auto vector_call floor(float32x4 x) noexcept -> float32x4 { return floor(x.data); }
    static force_inline auto vector_call ceil(float32x4 x) noexcept -> float32x4 { return ceil(x.data); }
    static force_inline auto vector_call round(float32x4 x) noexcept -> float32x4 { return round(x.data); }
    static force_inline auto vector_call abs(float32x4 x) noexcept -> float32x4 { return abs(x.data); }

    static force_inline auto vector_call sqrt(float32x4 x) noexcept -> float32x4 { return sqrt(x.data); }
    static force_inline auto vector_call rsqrt(float32x4 x) noexcept -> float32x4 { return rsqrt(x.data); }
    static force_inline auto vector_call min(float32x4 a, float32x4 b) noexcept -> float32x4 { return min(a.data, b.data); }
    static force_inline auto vector_call max(float32x4 a, float32x4 b) noexcept -> float32x4 { return max(a.data, b.data); }
    static force_inline auto vector_call fma(float32x4 a, float32x4 b, float32x4 c) noexcept -> float32x4 { return fma(a.data, b.data, c.data); }

};

} // namespace vsl