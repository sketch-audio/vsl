#pragma once

#include <cassert>
#include "vsl/vsl.h"

inline auto test_float32x4() -> void
{
    {
        // Default constructor.
        auto vec = vsl::float32x4{};
        assert(vec[0] == 0);
        assert(vec[1] == 0);
        assert(vec[2] == 0);
        assert(vec[3] == 0);
    }

    {
        // Construct with a single value.
        auto vec = vsl::float32x4{3.5f};
        assert(vec[0] == 3.5f);
        assert(vec[1] == 3.5f);
        assert(vec[2] == 3.5f);
        assert(vec[3] == 3.5f);
    }

    {
        // Construct with an array.
        auto vec = vsl::float32x4{2.5f, 7.5f, 11.5f, 13.5f};
        assert(vec[0] == 2.5f);
        assert(vec[1] == 7.5f);
        assert(vec[2] == 11.5f);
        assert(vec[3] == 13.5f);
    }

    {
        // Addition.
        auto a = vsl::float32x4{2.5f, 7.5f, 11.5f, 13.5f};
        auto b = vsl::float32x4{7.5f, 2.5f, 3.5f, 1.5f};
        auto c = a + b;
        assert(c[0] == 10.f);
        assert(c[1] == 10.f);
        assert(c[2] == 15.f);
        assert(c[3] == 15.f);
    }

    {
        // Scalar addition.
        auto a = vsl::float32x4{1.5f, 2.5f, 3.5f, 4.5f};
        auto b = 5.f + a;
        auto c = a + 5.f;
        assert(b[0] == c[0]);
        assert(c[0] == 6.5f);
        assert(b[1] == c[1]);
        assert(c[1] == 7.5f);
        assert(b[2] == c[2]);
        assert(c[2] == 8.5f);
        assert(b[3] == c[3]);
        assert(c[3] == 9.5f);
    }

    {
        // Subtraction.
        auto a = vsl::float32x4{10.5f, 8.5f, 15.5f, 20.5f};
        auto b = vsl::float32x4{4.5f, 12.5f, 5.5f, 25.5f};
        auto c = a - b;
        assert(c[0] == 6.f);
        assert(c[1] == -4.f);
        assert(c[2] == 10.f);
        assert(c[3] == -5.f);
        auto d = -a;
        assert(d[0] == -10.5f);
        assert(d[1] == -8.5f);
        assert(d[2] == -15.5f);
        assert(d[3] == -20.5f);
    }

    {
        // Scalar subtraction.
        auto a = vsl::float32x4{1.5f, 2.5f, 3.5f, 4.5f};
        auto b = 5.f - a;
        auto c = a - 5.f;
        assert(b[0] == 3.5f);
        assert(c[0] == -3.5f);
        assert(b[1] == 2.5f);
        assert(c[1] == -2.5f);
        assert(b[2] == 1.5f);
        assert(c[2] == -1.5f);
        assert(b[3] == 0.5f);
        assert(c[3] == -0.5f);
    }

    {
        // Multiplication.
        auto a = vsl::float32x4{2.f, 3.f, 4.f, 5.f};
        auto b = vsl::float32x4{4.f, 5.f, 6.f, 7.f};
        auto c = a * b;
        assert(c[0] == 8.f);
        assert(c[1] == 15.f);
        assert(c[2] == 24.f);
        assert(c[3] == 35.f);
    }

    {
        // Scalar multiplication.
        auto a = vsl::float32x4{2.f, 3.f, 4.f, 5.f};
        auto b = 4.f * a;
        auto c = a * 4.f;
        assert(b[0] == c[0]);
        assert(c[0] == 8.f);
        assert(b[1] == c[1]);
        assert(c[1] == 12.f);
        assert(b[2] == c[2]);
        assert(c[2] == 16.f);
        assert(b[3] == c[3]);
        assert(c[3] == 20.f);
    }

    {
        // Division.
        auto a = vsl::float32x4{8.f, 15.f, 24.f, 35.f};
        auto b = vsl::float32x4{2.f, 3.f, 4.f, 5.f};
        auto c = a / b;
        assert(c[0] == 4.f);
        assert(c[1] == 5.f);
        assert(c[2] == 6.f);
        assert(c[3] == 7.f);
    }

    {
        // Scalar division.
        auto a = vsl::float32x4{8.f, 15.f, 24.f, 35.f};
        auto b = a / 2.f;
        assert(b[0] == 4.f);
        assert(b[1] == 7.5f);
        assert(b[2] == 12.f);
        assert(b[3] == 17.5f);
    }

    {
        // Compare, select
        auto a = vsl::float32x4{8.f, 15.f, 24.f, 35.f};
        auto t = vsl::float32x4{2.f};
        auto f = vsl::float32x4{5.f};

        auto x0 = vsl::select(a == 15.f, t, f);
        auto x1 = vsl::select(a != 15.f, t, f);
        auto x2 = vsl::select(a < 20.f, t, f);
        auto x3 = vsl::select(a <= 8.f, t, f);
        auto x4 = vsl::select(a > 20.f, t, f);
        auto x5 = vsl::select(a >= 35.f, t, f);

        assert((x0[0] == 5.f) && (x0[1] == 2.f) && (x0[2] == 5.f) && (x0[3] == 5.f));
        assert((x1[0] == 2.f) && (x1[1] == 5.f) && (x1[2] == 2.f) && (x1[3] == 2.f));
        assert((x2[0] == 2.f) && (x2[1] == 2.f) && (x2[2] == 5.f) && (x2[3] == 5.f));
        assert((x3[0] == 2.f) && (x3[1] == 5.f) && (x3[2] == 5.f) && (x3[3] == 5.f));
        assert((x4[0] == 5.f) && (x4[1] == 5.f) && (x4[2] == 2.f) && (x4[3] == 2.f));
        assert((x5[0] == 5.f) && (x5[1] == 5.f) && (x5[2] == 5.f) && (x5[3] == 2.f));

        assert(vsl::any(a == 15.f));
        assert(!vsl::any(a < 4.f));
        assert(vsl::all(a >= 2.f));
        assert(!vsl::all(a > 8.f));
    }
}