#pragma once

#include <vsl/vsl.h>
#include <audio_bench/audio_bench.hpp>

inline auto test_float32x4() -> void
{
    using Tests = audio_bench::Tests;
    
    Tests::add("float32x4 - constructors and access", []() {
        // Default constructor
        auto vec = vsl::float32x4{};
        audio_bench::expect_true(vsl::all(vec == 0), "Default constructor should zero lanes");

        // Single value
        vec = vsl::float32x4{3.5f};
        audio_bench::expect_true(vsl::all(vec == 3.5f), "Ctor single value should set all lanes");

        // Array constructor
        vec = vsl::float32x4{2.5f, 7.5f, 11.5f, 13.5f};
        audio_bench::expect_true(vec[0] == 2.5f, "Fail ctor array lane 0");
        audio_bench::expect_true(vec[1] == 7.5f, "Fail ctor array lane 1");
        audio_bench::expect_true(vec[2] == 11.5f, "Fail ctor array lane 2");
        audio_bench::expect_true(vec[3] == 13.5f, "Fail ctor array lane 3");
    });

    Tests::add("float32x4 - arithmetic ops", []() {
        // Addition
        auto a = vsl::float32x4{2.5f, 7.5f, 11.5f, 13.5f};
        auto b = vsl::float32x4{7.5f, 2.5f, 3.5f, 1.5f};
        auto c = a + b;
        audio_bench::expect_true(c[0] == 10, "add lane 0");
        audio_bench::expect_true(c[1] == 10, "add lane 1");
        audio_bench::expect_true(c[2] == 15, "add lane 2");
        audio_bench::expect_true(c[3] == 15, "add lane 3");

        // Scalar addition
        a = vsl::float32x4{1.5f, 2.5f, 3.5f, 4.5f};
        const auto s1 = 5 + a;
        const auto s2 = a + 5;
        audio_bench::expect_true(s1[0] == s2[0], "scalar add commutative lane 0");
        audio_bench::expect_true(s2[0] == 6.5f, "scalar add value lane 0");

        // Subtraction and negation
        a = vsl::float32x4{10.5f, 8.5f, 15.5f, 20.5f};
        b = vsl::float32x4{4.5f, 12.5f, 5.5f, 25.5f};
        c = a - b;
        audio_bench::expect_true(c[0] == 6, "sub lane 0");
        audio_bench::expect_true(c[1] == -4, "sub lane 1");
        audio_bench::expect_true(c[2] == 10, "sub lane 2");
        audio_bench::expect_true(c[3] == -5, "sub lane 3");
        const auto d = -a;
        audio_bench::expect_true(d[0] == -10.5f, "neg lane 0");

        // Scalar subtraction
        a = vsl::float32x4{1.5f, 2.5f, 3.5f, 4.5f};
        const auto r1 = 5 - a;
        const auto r2 = a - 5;
        audio_bench::expect_true(r1[0] == 3.5f, "scalar sub lane 0");
        audio_bench::expect_true(r2[0] == -3.5f, "scalar sub lane 0 neg");

        // Multiplication
        a = vsl::float32x4{2, 3, 4, 5};
        b = vsl::float32x4{4, 5, 6, 7};
        c = a * b;
        audio_bench::expect_true(c[0] == 8, "mul lane 0");
        audio_bench::expect_true(c[1] == 15, "mul lane 1");
        audio_bench::expect_true(c[2] == 24, "mul lane 2");
        audio_bench::expect_true(c[3] == 35, "mul lane 3");

        // Scalar multiplication
        a = vsl::float32x4{2, 3, 4, 5};
        const auto m1 = 4 * a;
        const auto m2 = a * 4;
        audio_bench::expect_true(m1[0] == m2[0], "scalar mul comm lane 0");
        audio_bench::expect_true(m1[0] == 8, "scalar mul value lane 0");

        // Division
        a = vsl::float32x4{8, 15, 24, 35};
        b = vsl::float32x4{2, 3, 4, 5};
        c = a / b;
        audio_bench::expect_true(c[0] == 4, "div lane 0");

        // Scalar division
        const auto sd = a / 2;
        audio_bench::expect_true(sd[0] == 4, "scalar div lane 0");
    });

    Tests::add("float32x4 - compare & select", []() {
        auto a = vsl::float32x4{8, 15, 24, 35};
        const auto t = vsl::float32x4{2};
        const auto f = vsl::float32x4{5};

        const auto x0 = vsl::select(a == 15, t, f);
        const auto x1 = vsl::select(a != 15, t, f);
        const auto x2 = vsl::select(a < 20, t, f);
        const auto x3 = vsl::select(a <= 8, t, f);
        const auto x4 = vsl::select(a > 20, t, f);
        const auto x5 = vsl::select(a >= 35, t, f);

        audio_bench::expect_true(x0[0] == 5 && x0[1] == 2 && x0[2] == 5 && x0[3] == 5, "select eq");
        audio_bench::expect_true(x1[0] == 2 && x1[1] == 5 && x1[2] == 2 && x1[3] == 2, "select neq");
        audio_bench::expect_true(x2[0] == 2 && x2[1] == 2 && x2[2] == 5 && x2[3] == 5, "select lt");
        audio_bench::expect_true(x3[0] == 2 && x3[1] == 5 && x3[2] == 5 && x3[3] == 5, "select le");
        audio_bench::expect_true(x4[0] == 5 && x4[1] == 5 && x4[2] == 2 && x4[3] == 2, "select gt");
        audio_bench::expect_true(x5[0] == 5 && x5[1] == 5 && x5[2] == 5 && x5[3] == 2, "select ge");

        audio_bench::expect_true(vsl::any(a == 15), "any should be true for a == 15");
        audio_bench::expect_true(!vsl::any(a < 4), "no lane should be < 4");
        audio_bench::expect_true(vsl::all(a >= 2), "all lanes >= 2");
        audio_bench::expect_true(!vsl::all(a > 8), "not all lanes > 8");
    });
}