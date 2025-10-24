#pragma once

#include <vsl/vsl.h>
#include <audio_bench/audio_bench.h>

inline auto test_mask32x4() -> void
{
    audio_bench::add_test("mask32x4 - constructors and basic ops", []() {
        // Default constructor
        const auto v0 = vsl::mask32x4{};
        audio_bench::expect_true(vsl::all(v0 == 0), "Default constructor: all lanes zero");

        // Single value
        const auto v1 = vsl::mask32x4{5};
        audio_bench::expect_true(vsl::all(v1 == 5), "Single value constructor: all lanes equal to argument");

        // Array constructor
        const auto ve = vsl::mask32x4{3, 7, 11, 13};
        audio_bench::expect_true(ve[0] == 3, "Array constructor: lane 0");
        audio_bench::expect_true(ve[1] == 7, "Array constructor: lane 1");
        audio_bench::expect_true(ve[2] == 11, "Array constructor: lane 2");
        audio_bench::expect_true(ve[3] == 13, "Array constructor: lane 3");
    });

    audio_bench::add_test("mask32x4 - arithmetic", []() {
        const auto a = vsl::mask32x4{3, 7, 11, 13};
        const auto b = vsl::mask32x4{7, 5, 3, 1};
        const auto add = a + b;
        audio_bench::expect_true(add[0] == 10, "Addition operator: lane 0");
        audio_bench::expect_true(add[1] == 12, "Addition operator: lane 1");
        audio_bench::expect_true(add[2] == 14, "Addition operator: lane 2");
        audio_bench::expect_true(add[3] == 14, "Addition operator: lane 3");

        const auto aa = vsl::mask32x4{1, 2, 3, 4};
        const auto lhs = 5 + aa;
        const auto rhs = aa + 5;
        audio_bench::expect_true(lhs[0] == rhs[0] && rhs[0] == 6, "Scalar addition: both lhs and rhs");

        // Subtraction with wraparound
        const auto s_a = vsl::mask32x4{10, 8, 6, 4};
        const auto s_b = vsl::mask32x4{4, 12, 14, 16};
        const auto sub = s_a - s_b;
        audio_bench::expect_true(sub[0] == 6, "Subtraction operator: lane 0");
        audio_bench::expect_true(vsl::unsigned_to_signed(sub[1]) == -4, "Subtraction operator: lane 1");
        audio_bench::expect_true(vsl::unsigned_to_signed(sub[2]) == -8, "Subtraction operator: lane 2");
        audio_bench::expect_true(vsl::unsigned_to_signed(sub[3]) == -12, "Subtraction operator: lane 3");

        // Scalar subtraction both ways
        const auto sa = vsl::mask32x4{1, 2, 3, 4};
        const auto lhs_sub = 5 - sa;
        const auto rhs_sub = sa - 5;
        audio_bench::expect_true(lhs_sub[0] == 4, "Scalar subtraction: lhs lane 0");
        audio_bench::expect_true(vsl::unsigned_to_signed(rhs_sub[0]) == -4, "Scalar subtraction: rhs lane 0");

        // Multiplication
        const auto m_a = vsl::mask32x4{2, 3, 4, 5};
        const auto m_b = vsl::mask32x4{4, 5, 6, 7};
        const auto mul = m_a * m_b;
        audio_bench::expect_true(mul[0] == 8, "Multiplication operator: lane 0");
        audio_bench::expect_true(mul[1] == 15, "Multiplication operator: lane 1");
        audio_bench::expect_true(mul[2] == 24, "Multiplication operator: lane 2");
        audio_bench::expect_true(mul[3] == 35, "Multiplication operator: lane 3");

        // Scalar multiplication
        const auto mm_l = 4 * m_a;
        const auto mm_r = m_a * 4;
        audio_bench::expect_true(mm_l[0] == mm_r[0] && mm_r[0] == 8, "Scalar multiplication: both lhs and rhs");
    });

    audio_bench::add_test("mask32x4 - select & compare", []() {
        const auto a = vsl::mask32x4{3, 7, 11, 13};
        const auto t = vsl::mask32x4{2};
        const auto f = vsl::mask32x4{5};

        const auto x0 = vsl::select(a == 3, t, f);
        const auto x1 = vsl::select(a != 3, t, f);
        const auto x2 = vsl::select(a < 10, t, f);
        const auto x3 = vsl::select(a <= 3, t, f);
        const auto x4 = vsl::select(a > 10, t, f);
        const auto x5 = vsl::select(a >= 13, t, f);

        audio_bench::expect_true(x0[0] == 2 && x0[1] == 5 && x0[2] == 5 && x0[3] == 5, "select eq");
        audio_bench::expect_true(x1[0] == 5 && x1[1] == 2 && x1[2] == 2 && x1[3] == 2, "select neq");
        audio_bench::expect_true(x2[0] == 2 && x2[1] == 2 && x2[2] == 5 && x2[3] == 5, "select lt");
        audio_bench::expect_true(x3[0] == 2 && x3[1] == 5 && x3[2] == 5 && x3[3] == 5, "select le");
        audio_bench::expect_true(x4[0] == 5 && x4[1] == 5 && x4[2] == 2 && x4[3] == 2, "select gt");
        audio_bench::expect_true(x5[0] == 5 && x5[1] == 5 && x5[2] == 5 && x5[3] == 2, "select ge");
    });

    audio_bench::add_test("mask32x4 - any/all & convert", []() {
        const auto a = vsl::mask32x4{3, 7, 11, 13};
        audio_bench::expect_true(vsl::any(a == 3), "any(): at least one lane is true");
        audio_bench::expect_true(!vsl::any(a < 0), "any(): no lanes are true");
        audio_bench::expect_true(vsl::all(a >= 0), "all(): all lanes are true");
        audio_bench::expect_true(!vsl::all(a > 7), "all(): not all lanes are true");

        const auto b = vsl::mask32x4{8, 16, 32, 64};
        const auto f = vsl::unsigned_to_float(b);
        audio_bench::expect_true(f[0] == 8.f && f[1] == 16.f && f[2] == 32.f && f[3] == 64.f, "Convert to float");
    });

    audio_bench::add_test("mask32x4 - bitwise ops", []() {
        const auto a = vsl::mask32x4{0x0F0F0F0Fu, 0xFFFF0000u, 0xAAAAAAAAu, 0x55555555u};
        const auto b = vsl::mask32x4{0x00FF00FFu, 0x0000FFFFu, 0x33333333u, 0xCCCCCCCCu};

        // idempotence
        audio_bench::expect_true((a & a)[0] == a[0] && (a | a)[1] == a[1], "AND/OR idempotence");

        // XOR with self -> zero
        audio_bench::expect_true((a ^ a)[0] == 0 && (a ^ a)[1] == 0 && (a ^ a)[2] == 0 && (a ^ a)[3] == 0, "XOR self -> zero");

        // commutativity
        audio_bench::expect_true((a & b)[0] == (b & a)[0] && (a | b)[2] == (b | a)[2], "AND/OR commutative");

        // complements
        const auto all_ones = ~vsl::mask32x4{0};
        audio_bench::expect_true(((~a) & a)[0] == 0 && ((~a) & a)[1] == 0, "~a & a == 0");
        audio_bench::expect_true(((~a) | a)[0] == all_ones[0] && ((~a) | a)[2] == all_ones[2], "~a | a == all ones");
    });
}