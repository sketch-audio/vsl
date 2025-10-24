#pragma once

#include <vsl/vsl.h>
#include <audio_bench/audio_bench.h>

inline auto test_float64x2() -> void
{
    audio_bench::add_test("float64x2 - constructors and access", []() {
        auto v = vsl::float64x2{};
        audio_bench::expect_true(vsl::all(v == 0), "Default constructor should zero lanes");

        v = vsl::float64x2{3.5};
        audio_bench::expect_true(vsl::all(v == 3.5), "Constructor with single value should set all lanes");

        v = vsl::float64x2{2.5, 7.5};
        audio_bench::expect_true(v[0] == 2.5, "ctor array lane 0");
        audio_bench::expect_true(v[1] == 7.5, "ctor array lane 1");
    });

    audio_bench::add_test("float64x2 - arithmetic ops", []() {
        auto a = vsl::float64x2{2.5, 7.5};
        auto b = vsl::float64x2{7.5, 2.5};
        auto c = a + b;
        audio_bench::expect_true(c[0] == 10, "add lane 0");
        audio_bench::expect_true(c[1] == 10, "add lane 1");

        // Scalar addition
        a = vsl::float64x2{1.5, 2.5};
        auto s1 = 5 + a;
        auto s2 = a + 5;
        audio_bench::expect_true(s1[0] == s2[0], "scalar add comm lane 0");
        audio_bench::expect_true(s2[0] == 6.5, "scalar add value lane 0");

        // Subtraction and negation
        a = vsl::float64x2{10.5, 8.5};
        b = vsl::float64x2{4.5, 12.5};
        c = a - b;
        audio_bench::expect_true(c[0] == 6, "sub lane 0");
        audio_bench::expect_true(c[1] == -4, "sub lane 1");
        auto d = -a;
        audio_bench::expect_true(d[0] == -10.5, "neg lane 0");

        // Scalar subtraction
        a = vsl::float64x2{1.5, 2.5};
        auto r1 = 5 - a;
        auto r2 = a - 5;
        audio_bench::expect_true(r1[0] == 3.5, "scalar sub lane 0");
        audio_bench::expect_true(r2[0] == -3.5, "scalar sub lane 0 neg");

        // Multiplication
        a = vsl::float64x2{2, 3};
        b = vsl::float64x2{4, 5};
        c = a * b;
        audio_bench::expect_true(c[0] == 8, "mul lane 0");

        // Scalar multiplication
        a = vsl::float64x2{2, 3};
        auto m1 = 4 * a;
        auto m2 = a * 4;
        audio_bench::expect_true(m1[0] == m2[0], "scalar mul comm lane 0");

        // Division
        a = vsl::float64x2{9, 15};
        b = vsl::float64x2{3, 5};
        c = a / b;
        audio_bench::expect_true(c[0] == 3, "div lane 0");

        // Scalar division
        a = vsl::float64x2{8, 16};
        auto sd1 = 2 / a;
        auto sd2 = a / 2;
        audio_bench::expect_true(sd1[0] == 0.25, "scalar div lane 0");
        audio_bench::expect_true(sd2[1] == 8, "scalar div lane 1");
    });

    audio_bench::add_test("float64x2 - compare & select", []() {
        auto a = vsl::float64x2{8, 16};
        const auto t = vsl::float64x2{2};
        const auto f = vsl::float64x2{5};

        const auto x0 = vsl::select(a == 8, t, f);
        const auto x1 = vsl::select(a != 8, t, f);
        const auto x2 = vsl::select(a < 20, t, f);
        const auto x3 = vsl::select(a <= 8, t, f);
        const auto x4 = vsl::select(a > 12, t, f);
        const auto x5 = vsl::select(a >= 16, t, f);

        audio_bench::expect_true(x0[0] == 2 && x0[1] == 5, "select eq lanes");
        audio_bench::expect_true(x1[0] == 5 && x1[1] == 2, "select neq lanes");
        audio_bench::expect_true(x2[0] == 2 && x2[1] == 2, "select lt lanes");
        audio_bench::expect_true(x3[0] == 2 && x3[1] == 5, "select le lanes");
        audio_bench::expect_true(x4[0] == 5 && x4[1] == 2, "select gt lanes");
        audio_bench::expect_true(x5[0] == 5 && x5[1] == 2, "select ge lanes");

        audio_bench::expect_true(vsl::any(a == 8), "any should be true for a == 8");
        audio_bench::expect_true(!vsl::any(a < 4), "no lane should be < 4");
        audio_bench::expect_true(vsl::all(a < 20), "all lanes < 20");
        audio_bench::expect_true(!vsl::all(a > 9), "not all lanes > 9");
    });
}