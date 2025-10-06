#pragma once

#include <cassert>
#include "vsl.h"

inline auto test_float64x2() -> void
{
    {
        // Default constructor.
        auto vec = vsl::float64x2{};
        assert(vec[0] == 0);
        assert(vec[1] == 0);
    }

    {
        // Construct with a single value.
        auto vec = vsl::float64x2{3.5};
        assert(vec[0] == 3.5);
        assert(vec[1] == 3.5);
    }

    {
        // Construct with an array.
        auto vec = vsl::float64x2{2.5, 7.5};
        assert(vec[0] == 2.5);
        assert(vec[1] == 7.5);
    }

    {
        // Addition.
        auto a = vsl::float64x2{2.5, 7.5};
        auto b = vsl::float64x2{7.5, 2.5};
        auto c = a + b;
        assert(c[0] == 10.0);
        assert(c[1] == 10.0);
    }

    {
        // Scalar addition.
        auto a = vsl::float64x2{1.5, 2.5};
        auto b = 5 + a;
        auto c = a + 5;
        assert(b[0] == c[0]);
        assert(c[0] == 6.5);
        assert(b[1] == c[1]);
        assert(c[1] == 7.5);
    }

    {
        // Subtraction.
        auto a = vsl::float64x2{10.5, 8.5};
        auto b = vsl::float64x2{4.5, 12.5};
        auto c = a - b;
        assert(c[0] == 6.0);
        assert(c[1] == -4.0);
        auto d = -a;
        assert(d[0] == -10.5);
        assert(d[1] == -8.5);
    }

    {
        // Scalar subtraction.
        auto a = vsl::float64x2{1.5, 2.5};
        auto b = 5 - a;
        auto c = a - 5;
        assert(b[0] == 3.5);
        assert(c[0] == -3.5);
        assert(b[1] == 2.5);
        assert(c[1] == -2.5);
    }

    {
        // Multiplication.
        auto a = vsl::float64x2{2.0, 3.0};
        auto b = vsl::float64x2{4.0, 5.0};
        auto c = a * b;
        assert(c[0] == 8.0);
        assert(c[1] == 15.0);
    }

    {
        // Scalar multiplication.
        auto a = vsl::float64x2{2.0, 3.0};
        auto b = 4.0 * a;
        auto c = a * 4.0;
        assert(b[0] == c[0]);
        assert(c[0] == 8.0);
        assert(b[1] == c[1]);
        assert(c[1] == 12.0);
    }

    {
        // Division.
        auto a = vsl::float64x2{9.0, 15.0};
        auto b = vsl::float64x2{3.0, 5.0};
        auto c = a / b;
        assert(c[0] == 3.0);
        assert(c[1] == 3.0);
    }

    {
        // Scalar division.
        auto a = vsl::float64x2{8.0, 16.0};
        auto b = 2.0 / a;
        auto c = a / 2.0;
        assert(b[0] == 0.25);
        assert(c[0] == 4.0);
        assert(b[1] == 0.125);
        assert(c[1] == 8.0);
    }

    {
        // Compare, select
        auto a = vsl::float64x2{8.0, 16.0};
        auto t = vsl::float64x2{2.0};
        auto f = vsl::float64x2{5.0};

        auto x0 = vsl::select(a == 8, t, f);
        auto x1 = vsl::select(a != 8, t, f);
        auto x2 = vsl::select(a < 20, t, f);
        auto x3 = vsl::select(a <= 8, t, f);
        auto x4 = vsl::select(a > 12, t, f);
        auto x5 = vsl::select(a >= 16, t, f);

        assert((x0[0] == 2.0) && (x0[1] == 5.0));
        assert((x1[0] == 5.0) && (x1[1] == 2.0));
        assert((x2[0] == 2.0) && (x2[1] == 2.0));
        assert((x3[0] == 2.0) && (x3[1] == 5.0));
        assert((x4[0] == 5.0) && (x4[1] == 2.0));
        assert((x5[0] == 5.0) && (x5[1] == 2.0));

        assert(vsl::any(a == 8));
        assert(!vsl::any(a < 4));
        assert(vsl::all(a < 20));
        assert(!vsl::all(a > 9));
    }
}