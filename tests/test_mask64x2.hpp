#pragma once

#include <vsl/vsl.hpp>
#include <audio_bench/audio_bench.hpp>

inline auto test_mask64x2() -> void
{
    using Tests = audio_bench::Tests;

    Tests::add("mask64x2 - constructors and basic ops", []() {
        const auto v0 = vsl::mask64x2{};
        audio_bench::expect_true(vsl::all(v0 == 0), "Default constructor: all lanes zero");

        const auto v1 = vsl::mask64x2{5};
        audio_bench::expect_true(vsl::all(v1 == 5), "Single value constructor: all lanes equal to argument");

        const auto ve = vsl::mask64x2{3, 7};
        audio_bench::expect_true(ve[0] == 3, "Array constructor: lane 0");
        audio_bench::expect_true(ve[1] == 7, "Array constructor: lane 1");
    });

    Tests::add("mask64x2 - arithmetic", []() {
        const auto a = vsl::mask64x2{3, 7};
        const auto b = vsl::mask64x2{7, 5};
        const auto add = a + b;
        audio_bench::expect_true(add[0] == 10, "Addition operator: lane 0");
        audio_bench::expect_true(add[1] == 12, "Addition operator: lane 1");

        const auto aa = vsl::mask64x2{1, 2};
        const auto lhs = 5 + aa;
        const auto rhs = aa + 5;
        audio_bench::expect_true(lhs[0] == rhs[0] && rhs[0] == 6, "Scalar addition: both lhs and rhs");

        // Subtraction with wraparound
        const auto s_a = vsl::mask64x2{10, 8};
        const auto s_b = vsl::mask64x2{4, 12};
        const auto sub = s_a - s_b;
        audio_bench::expect_true(sub[0] == 6, "Subtraction operator: lane 0");
        audio_bench::expect_true(vsl::unsigned_to_signed(sub[1]) == -4, "Subtraction operator: lane 1");

        // Scalar subtraction both ways
        const auto sa = vsl::mask64x2{1, 2};
        const auto lhs_sub = 5 - sa;
        const auto rhs_sub = sa - 5;
        audio_bench::expect_true(lhs_sub[0] == 4, "Scalar subtraction: lhs lane 0");
        audio_bench::expect_true(lhs_sub[1] == 3, "Scalar subtraction: lhs lane 1");
        audio_bench::expect_true(vsl::unsigned_to_signed(rhs_sub[0]) == -4, "Scalar subtraction: rhs lane 0");
        audio_bench::expect_true(vsl::unsigned_to_signed(rhs_sub[1]) == -3, "Scalar subtraction: rhs lane 1");

        // Multiplication
        const auto m_a = vsl::mask64x2{2, 3};
        const auto m_b = vsl::mask64x2{4, 5};
        const auto mul = m_a * m_b;
        audio_bench::expect_true(mul[0] == 8 && mul[1] == 15, "Multiplication operator: lane-wise multiplication");

        // Scalar multiplication
        const auto mm_l = 4 * m_a;
        const auto mm_r = m_a * 4;
        audio_bench::expect_true(mm_l[0] == mm_r[0] && mm_r[0] == 8, "Scalar multiplication: both lhs and rhs");
    });

    Tests::add("mask64x2 - select & compare", []() {
        const auto a = vsl::mask64x2{3, 7};
        const auto t = vsl::mask64x2{2};
        const auto f = vsl::mask64x2{5};

        const auto x0 = vsl::select(a == 3, t, f);
        const auto x1 = vsl::select(a != 3, t, f);
        const auto x2 = vsl::select(a < 10, t, f);
        const auto x3 = vsl::select(a <= 3, t, f);
        const auto x4 = vsl::select(a > 10, t, f);
        const auto x5 = vsl::select(a >= 7, t, f);

        audio_bench::expect_true(x0[0] == 2 && x0[1] == 5, "select eq");
        audio_bench::expect_true(x1[0] == 5 && x1[1] == 2, "select neq");
        audio_bench::expect_true(x2[0] == 2 && x2[1] == 2, "select lt");
        audio_bench::expect_true(x3[0] == 2 && x3[1] == 5, "select le");
        audio_bench::expect_true(x4[0] == 5 && x4[1] == 5, "select gt");
        audio_bench::expect_true(x5[0] == 5 && x5[1] == 2, "select ge");
    });

    Tests::add("mask64x2 - any/all & convert", []() {
        const auto a = vsl::mask64x2{3, 7};
        audio_bench::expect_true(vsl::any(a == 3), "any(): at least one lane is true");
        audio_bench::expect_true(!vsl::any(a < 0), "any(): no lanes are true");
        audio_bench::expect_true(vsl::all(a >= 0), "all(): all lanes are true");
        audio_bench::expect_true(!vsl::all(a > 3), "all(): not all lanes are true");

        const auto b = vsl::mask64x2{8, 16};
        const auto f = vsl::unsigned_to_float(b);
        audio_bench::expect_true(f[0] == 8.0 && f[1] == 16.0, "Convert to float");
    });

    Tests::add("mask64x2 - bitwise ops", []() {
        const auto a = vsl::mask64x2{0x0F0F0F0Fu, 0xFFFF0000u};
        const auto b = vsl::mask64x2{0x00FF00FFu, 0x0000FFFFu};

        // idempotence
        audio_bench::expect_true((a & a)[0] == a[0] && (a | a)[1] == a[1], "AND/OR idempotence");

        // XOR with self -> zero
        audio_bench::expect_true((a ^ a)[0] == 0 && (a ^ a)[1] == 0, "XOR self -> zero");

        // commutativity
        audio_bench::expect_true((a & b)[0] == (b & a)[0] && (a | b)[1] == (b | a)[1], "AND/OR commutative");

        // complements
        const auto all_ones = ~vsl::mask64x2{0};
        audio_bench::expect_true(((~a) & a)[0] == 0 && ((~a) & a)[1] == 0, "~a & a == 0");
        audio_bench::expect_true(((~a) | a)[0] == all_ones[0] && ((~a) | a)[1] == all_ones[1], "~a | a == all ones");
    });
}