#pragma once

#include <cassert>
#include "vsl/vsl.h"

inline auto test_mask64x2() -> void
{
    {
        const auto vec = vsl::mask64x2{};
        assert(vsl::all(vec == 0) && "Default constructor: all lanes zero.");
    }

    {
        const auto vec = vsl::mask64x2{5};
        assert(vsl::all(vec == 5) && "Single value constructor: all lanes equal to argument.");
    }

    {
        const auto vec = vsl::mask64x2{3, 7};
        assert((vec[0] == 3) && (vec[1] == 7) && "Array constructor: lane indexing matches argument order.");
    }

    {
        const auto a = vsl::mask64x2{3, 7};
        const auto b = vsl::mask64x2{7, 5};
        const auto c = a + b;
        assert((c[0] == 10) && (c[1] == 12) && "Addition operator: lane-wise addition.");
    }

    {
        const auto a = vsl::mask64x2{1, 2};
        const auto b = 5 + a;
        const auto c = a + 5;
        assert((b[0] == c[0]) && (c[0] == 6) && (b[1] == c[1]) && (c[1] == 7) && "Scalar addition: both lhs and rhs.");
    }

    {
        auto u_to_s = [](auto x) { return vsl::unsigned_to_signed(x); };
        const auto a = vsl::mask64x2{10, 8};
        const auto b = vsl::mask64x2{4, 12};
        const auto c = a - b;
        assert((c[0] == 6) && (u_to_s(c[1]) == -4) && "Subtraction operator: lane-wise subtraction with wraparound.");
    }

    {
        auto u_to_s = [](auto x) { return vsl::unsigned_to_signed(x); };
        const auto a = vsl::mask64x2{1, 2};
        const auto b = 5 - a;
        const auto c = a - 5;
        const auto res0 = (b[0] == 4) && (b[1] == 3);
        const auto res1 = (u_to_s(c[0]) == -4) && (u_to_s(c[1]) == -3);
        assert(res0 && res1 && "Scalar subtraction: both lhs and rhs with wraparound.");
    }

    {
        const auto a = vsl::mask64x2{2, 3};
        const auto b = vsl::mask64x2{4, 5};
        const auto c = a * b;
        assert((c[0] == 8) && (c[1] == 15) && "Multiplication operator: lane-wise multiplication.");
    }

    {
        const auto a = vsl::mask64x2{2, 3};
        const auto b = 4 * a;
        const auto c = a * 4;
        const auto res = (b[0] == c[0]) && (c[0] == 8) && (b[1] == c[1]) && (c[1] == 12);
        assert(res && "Scalar multiplication: both lhs and rhs.");
    }

    {
        // Select and compare.
        const auto a = vsl::mask64x2{3, 7};
        const auto t = vsl::mask64x2{2};
        const auto f = vsl::mask64x2{5};

        const auto x0 = vsl::select(a == 3, t, f);
        const auto x1 = vsl::select(a != 3, t, f);
        const auto x2 = vsl::select(a < 10, t, f);
        const auto x3 = vsl::select(a <= 3, t, f);
        const auto x4 = vsl::select(a > 10, t, f);
        const auto x5 = vsl::select(a >= 7, t, f);

        assert((x0[0] == 2) && (x0[1] == 5));
        assert((x1[0] == 5) && (x1[1] == 2));
        assert((x2[0] == 2) && (x2[1] == 2));
        assert((x3[0] == 2) && (x3[1] == 5));
        assert((x4[0] == 5) && (x4[1] == 5));
        assert((x5[0] == 5) && (x5[1] == 2));
    }

    {
        // Any, all.
        const auto a = vsl::mask64x2{3, 7};
        assert(vsl::any(a == 3) && "any(): at least one lane is true.");
        assert(!vsl::any(a < 0) && "any(): no lanes are true.");
        assert(vsl::all(a >= 0) && "all(): all lanes are true.");
        assert(!vsl::all(a > 3) && "all(): not all lanes are true.");
    }

    {
        // Convert to float.
        const auto a = vsl::mask64x2{8, 16};
        const auto f = vsl::unsigned_to_float(a);
        assert((f[0] == 8.0) && (f[1] == 16.0) && "Convert to float.");
    }
}