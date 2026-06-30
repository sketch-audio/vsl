#include <array>
#include <string>

#include <vsl/vsl.hpp>
#include <audio_bench/audio_bench.hpp>

namespace {

// Conversions are tested for scalar/vector parity: every value is run through
// the scalar overload and through a broadcast vector, then the lanes are
// checked to agree (all_same) and to match the scalar result.
template<typename F>
auto test_conv(const std::string& name) -> void
{
    using Tests = audio_bench::Tests;
    using V = vsl::vector_t<F>;   // float32x4 / float64x2
    using S = vsl::signed_t<F>;   // int32_t   / int64_t
    using U = vsl::unsigned_t<F>; // uint32_t  / uint64_t
    using M = vsl::vector_t<S>;   // mask32x4  / mask64x2 (the one integral vector)

    constexpr auto bits = sizeof(F) * 8;
    constexpr U one = U{1};
    constexpr U sign_bit = one << (bits - 1);
    constexpr U all_ones = ~U{0};
    // exponent all ones + non-zero significand -> a NaN bit pattern.
    constexpr U nan_bits = (all_ones << 1) | one;

    Tests::add("conv - reinterpret round-trip (" + name + ")", [=]() {
        const std::array<U, 6> pats = {U{0}, one, all_ones, sign_bit, sign_bit | one, nan_bits};
        for (auto u : pats) {
            // int -> float -> int is bit preserving, even for sign bit / NaN.
            const auto f = vsl::reinterpret_as_float(u);
            const auto back = vsl::reinterpret_as_int(f);
            audio_bench::expect_true(back == u, "scalar reinterpret int->float->int");

            // Vector parity: same round-trip through a broadcast lane. Checks stay
            // in the integer domain since some patterns are NaN (NaN != NaN, so
            // all_same / float compares can't be used here).
            const auto vf = vsl::reinterpret_as_float(M{u});
            const auto vback = vsl::reinterpret_as_int(vf);
            audio_bench::expect_true(vsl::all(vback == u), "vector reinterpret round-trip");
            audio_bench::expect_true(vback[0] == back, "scalar/vector reinterpret parity");
        }
    });

    Tests::add("conv - signed/unsigned inverse (" + name + ")", [=]() {
        const std::array<U, 5> us = {U{0}, one, all_ones, sign_bit, sign_bit | U{7}};
        for (auto u : us) {
            const auto s = vsl::unsigned_to_signed(u);
            audio_bench::expect_true(vsl::signed_to_unsigned(s) == u, "scalar u->s->u");
        }

        const std::array<S, 5> ss = {S{0}, S{1}, S{-1}, S{42}, S{-42}};
        for (auto s : ss) {
            const auto u = vsl::signed_to_unsigned(s);
            audio_bench::expect_true(vsl::unsigned_to_signed(u) == s, "scalar s->u->s");
        }

        // Two's-complement edge: all-ones bits are -1.
        audio_bench::expect_true(vsl::unsigned_to_signed(all_ones) == S{-1}, "all-ones -> -1");
        audio_bench::expect_true(vsl::signed_to_unsigned(S{-1}) == all_ones, "-1 -> all-ones");

        // Vector conversions are bit-preserving identities (one integral vector).
        const auto m = M{all_ones};
        audio_bench::expect_true(vsl::all(vsl::unsigned_to_signed(m) == m), "vector u->s identity");
        audio_bench::expect_true(vsl::all(vsl::signed_to_unsigned(m) == m), "vector s->u identity");
    });

    Tests::add("conv - bool/mask round-trip (" + name + ")", [=]() {
        // Scalar: true <-> all-ones, false <-> zero.
        audio_bench::expect_true(vsl::mask_to_bool(vsl::bool_to_mask<U>(true)) == true, "scalar true round-trip");
        audio_bench::expect_true(vsl::mask_to_bool(vsl::bool_to_mask<U>(false)) == false, "scalar false round-trip");
        audio_bench::expect_true(vsl::bool_to_mask<U>(true) == all_ones, "scalar true is all-ones");
        audio_bench::expect_true(vsl::bool_to_mask<U>(false) == U{0}, "scalar false is zero");

        // Vector: a comparison mask survives mask <-> bool unchanged.
        const auto all_true = (V{1} == V{1});
        const auto all_false = (V{1} == V{2});
        audio_bench::expect_true(vsl::all(vsl::bool_to_mask(vsl::mask_to_bool(all_true)) == all_true), "vector true-mask round-trip");
        audio_bench::expect_true(vsl::all(vsl::bool_to_mask(vsl::mask_to_bool(all_false)) == all_false), "vector false-mask round-trip");
    });

    Tests::add("conv - unsigned_to_float edge values (" + name + ")", [=]() {
        // Includes the large, mantissa-losing value Generator::next relies on.
        const std::array<U, 5> us = {U{0}, one, U{1000}, sign_bit, all_ones};
        for (auto u : us) {
            const auto fs = vsl::unsigned_to_float(u);
            const auto fv = vsl::unsigned_to_float(M{u});
            audio_bench::expect_true(fs == static_cast<F>(u), "u2f scalar value");
            audio_bench::expect_true(vsl::all_same(fv), "u2f all_same");
            audio_bench::expect_true(fv[0] == fs, "u2f scalar/vector parity");
        }
    });

    Tests::add("conv - signed_to_float (" + name + ")", [=]() {
        const std::array<S, 5> ss = {S{0}, S{1}, S{-1}, S{1000}, S{-1000}};
        for (auto s : ss) {
            const auto fs = vsl::signed_to_float(s);
            // Feed the signed bit pattern through the integral vector.
            const auto fv = vsl::signed_to_float(M{vsl::signed_to_unsigned(s)});
            audio_bench::expect_true(fs == static_cast<F>(s), "s2f scalar value");
            audio_bench::expect_true(vsl::all_same(fv), "s2f all_same");
            audio_bench::expect_true(fv[0] == fs, "s2f scalar/vector parity");
        }
    });

    Tests::add("conv - float_to_signed (" + name + ")", [=]() {
        // Scalar truncates toward zero.
        audio_bench::expect_true(vsl::float_to_signed(static_cast<F>(1.9)) == S{1}, "scalar trunc positive");
        audio_bench::expect_true(vsl::float_to_signed(static_cast<F>(-1.9)) == S{-1}, "scalar trunc negative");

        // Parity uses integer-valued inputs: the vector rounding mode differs by
        // platform (SSE truncates, NEON rounds), so fractional inputs aren't comparable.
        const std::array<F, 5> xs = {F{0}, F{1}, F{-1}, F{100}, F{-100}};
        for (auto x : xs) {
            const auto ss = vsl::float_to_signed(x);
            const auto sv = vsl::float_to_signed(V{x}); // mask lanes hold the signed bits.
            audio_bench::expect_true(ss == static_cast<S>(x), "f2s scalar value");
            audio_bench::expect_true(vsl::all_same(sv), "f2s all_same");
            audio_bench::expect_true(vsl::unsigned_to_signed(sv[0]) == ss, "f2s scalar/vector parity");
        }
    });
}

struct Register {
    Register() {
        test_conv<float>("float32");
        test_conv<double>("float64");
    }
} register_{};

} // namespace
