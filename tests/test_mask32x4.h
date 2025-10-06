#pragma once

#include <cassert>
#include "vsl.h"

inline auto test_mask32x4() -> void
{
    {
        const auto vec = vsl::mask32x4{};
        assert(vsl::all(vec == 0) && "Default constructor: all lanes zero.");
    }

    {
        const auto vec = vsl::mask32x4{5};
        assert(vsl::all(vec == 5) && "Single value constructor: all lanes equal to argument.");
    }

    {
        const auto vec = vsl::mask32x4{3, 7, 11, 13};
        const auto res = (vec[0] == 3) && (vec[1] == 7) && (vec[2] == 11) && (vec[3] == 13);
        assert(res && "Array constructor: lane indexing matches argument order.");
    }

    {
        const auto a = vsl::mask32x4{3, 7, 11, 13};
        const auto b = vsl::mask32x4{7, 5, 3, 1};
        const auto c = a + b;
        const auto res = (c[0] == 10) && (c[1] == 12) && (c[2] == 14) && (c[3] == 14);
        assert(res && "Addition operator: lane-wise addition.");
    }

    {
        const auto a = vsl::mask32x4{1, 2, 3, 4};
        const auto b = 5 + a;
        const auto c = a + 5;
        const auto res = (b[0] == c[0]) && (c[0] == 6) && (b[1] == c[1]) && (c[1] == 7) && (b[2] == c[2]) && (c[2] == 8) && (b[3] == c[3]) && (c[3] == 9);
        assert(res && "Scalar addition: both lhs and rhs.");
    }

    {
        auto u_to_s = [](auto x) { return vsl::unsigned_to_signed(x); };
        const auto a = vsl::mask32x4{10, 8, 6, 4};
        const auto b = vsl::mask32x4{4, 12, 14, 16};
        const auto c = a - b;
        const auto res = (c[0] == 6) && (u_to_s(c[1]) == -4) && (u_to_s(c[2]) == -8) && (u_to_s(c[3]) == -12);
        assert(res && "Subtraction operator: lane-wise subtraction with wraparound.");
    }

    {
        auto u_to_s = [](auto x) { return vsl::unsigned_to_signed(x); };
        const auto a = vsl::mask32x4{1, 2, 3, 4};
        const auto b = 5 - a;
        const auto c = a - 5;
        const auto res0 = (b[0] == 4) && (b[1] == 3) && (b[2] == 2) && (b[3] == 1);
        const auto res1 = (u_to_s(c[0]) == -4) && (u_to_s(c[1]) == -3) && (u_to_s(c[2]) == -2) && (u_to_s(c[3]) == -1);
        assert(res0 && res1 && "Scalar subtraction: both lhs and rhs with wraparound.");
    }

    {
        const auto a = vsl::mask32x4{2, 3, 4, 5};
        const auto b = vsl::mask32x4{4, 5, 6, 7};
        const auto c = a * b;
        const auto res = (c[0] == 8) && (c[1] == 15) && (c[2] == 24) && (c[3] == 35);
        assert(res && "Multiplication operator: lane-wise multiplication.");
    }

    {
        const auto a = vsl::mask32x4{2, 3, 4, 5};
        const auto b = 4 * a;
        const auto c = a * 4;
        const auto res = (b[0] == c[0]) && (c[0] == 8) && (b[1] == c[1]) && (c[1] == 12) && (b[2] == c[2]) && (c[2] == 16) && (b[3] == c[3]) && (c[3] == 20);
        assert(res && "Scalar multiplication: both lhs and rhs.");
    }

    {
        // Select and compare.
        const auto a = vsl::mask32x4{3, 7, 11, 13};
        const auto t = vsl::mask32x4{2};
        const auto f = vsl::mask32x4{5};

        const auto x0 = vsl::select(a == 3, t, f);
        const auto x1 = vsl::select(a != 3, t, f);
        const auto x2 = vsl::select(a < 10, t, f);
        const auto x3 = vsl::select(a <= 3, t, f);
        const auto x4 = vsl::select(a > 10, t, f);
        const auto x5 = vsl::select(a >= 13, t, f);

        assert((x0[0] == 2) && (x0[1] == 5) && (x0[2] == 5) && (x0[3] == 5));
        assert((x1[0] == 5) && (x1[1] == 2) && (x1[2] == 2) && (x1[3] == 2));
        assert((x2[0] == 2) && (x2[1] == 2) && (x2[2] == 5) && (x2[3] == 5));
        assert((x3[0] == 2) && (x3[1] == 5) && (x3[2] == 5) && (x3[3] == 5));
        assert((x4[0] == 5) && (x4[1] == 5) && (x4[2] == 2) && (x4[3] == 2));
        assert((x5[0] == 5) && (x5[1] == 5) && (x5[2] == 5) && (x5[3] == 2));
    }

    {
        // Any, all.
        const auto a = vsl::mask32x4{3, 7, 11, 13};
        assert(vsl::any(a == 3) && "any(): at least one lane is true.");
        assert(!vsl::any(a < 0) && "any(): no lanes are true.");
        assert(vsl::all(a >= 0) && "all(): all lanes are true.");
        assert(!vsl::all(a > 7) && "all(): not all lanes are true.");
    }

    {
        // Convert to float.
        const auto a = vsl::mask32x4{8, 16, 32, 64};
        const auto f = vsl::unsigned_to_float(a);
        assert((f[0] == 8.f) && (f[1] == 16.f) && (f[2] == 32.f) && (f[3] == 64.f) && "Convert to float.");
    }
}