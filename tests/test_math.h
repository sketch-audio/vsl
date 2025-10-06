#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "vsl.h"

#include "test_helpers.h"

#define VSL_TEST_VECTOR_FUNC(FNAME, VTYPE, INPUT_LIST, TOL) \
{                                                           \
    using V = VTYPE;                                        \
    using S = vsl::scalar_t<V>;                             \
    const auto tol = S{TOL};                                \
                                                            \
    const auto xs = std::vector<S>INPUT_LIST;               \
    for (const auto x : xs) {                               \
        const auto v = vsl::FNAME(V{x});                    \
        const auto s = std::FNAME(x);                       \
        const auto res = vsl::all(vsl::abs_equal(v, V{s}, V{tol})); \
        if (!res) {                                         \
            printf("FNAME: %s, VTYPE: %s, x: %f, v: %f, s: %f\n", #FNAME, #VTYPE, x, v[0], s); \
            printf("abs delta: %f\n", vsl::abs(v - V{s})[0]); \
            assert(false && "Vector math test failed");     \
        }                                                   \
    }                                                       \
}

#define VSL_TEST_VECTOR_FUNC_2ARGS(FNAME, VTYPE, INPUT_LIST1, INPUT_LIST2, TOL) \
{                                                                               \
    using V = VTYPE;                                                            \
    using S = vsl::scalar_t<V>;                                                 \
    const auto tol = S{TOL};                                                    \
                                                                                \
    const auto xs = std::vector<S>INPUT_LIST1;                                  \
    const auto ys = std::vector<S>INPUT_LIST2;                                  \
    assert(xs.size() == ys.size());                                             \
    for (size_t i = 0; i < xs.size(); i++) {                                    \
        const auto x = xs[i];                                                   \
        const auto y = ys[i];                                                   \
        const auto v = vsl::FNAME(V{x}, V{y});                                  \
        const auto s = std::FNAME(x, y);                                        \
        const auto res = vsl::all(vsl::abs_equal(v, V{s}, V{tol}));             \
        if (!res) {                                                             \
            printf("FNAME: %s, VTYPE: %s, x: %f, y: %f, v: %f, s: %f\n", #FNAME, #VTYPE, x, y, v[0], s); \
            printf("abs delta: %f\n", vsl::abs(v - V{s})[0]);                   \
            assert(false && "Vector math test failed");                         \
        }                                                                       \
    }                                                                           \
}

inline auto test_math() -> void
{
    auto f_prec = 1e-7f;
    auto d_prec = 1e-15;
    VSL_TEST_VECTOR_FUNC(abs, vsl::float32x4, ({-1, 0, 2}), f_prec);
    VSL_TEST_VECTOR_FUNC(abs, vsl::float64x2, ({-1, 0, 2}), d_prec);
    VSL_TEST_VECTOR_FUNC(trunc, vsl::float32x4, ({0, 3.621f, 5, -1.2f, 1e20f}), f_prec);
    VSL_TEST_VECTOR_FUNC(trunc, vsl::float64x2, ({0, 3.621, 5, -1.2, 1e20}), d_prec);
    VSL_TEST_VECTOR_FUNC(floor, vsl::float32x4, ({2.35f, -2.3f, 5.f, -1.f}), f_prec);
    VSL_TEST_VECTOR_FUNC(floor, vsl::float64x2, ({2.35, -2.3, 5.0, -1.0}), d_prec);
    VSL_TEST_VECTOR_FUNC(ceil, vsl::float32x4, ({0, 3.25f, -2.1f, 5.f}), f_prec);
    VSL_TEST_VECTOR_FUNC(ceil, vsl::float64x2, ({0, 3.25, -2.1, 5.0}), d_prec);
    VSL_TEST_VECTOR_FUNC(round, vsl::float32x4, ({1.5f, 1.49f, -1.5f, -2.9f}), f_prec);
    VSL_TEST_VECTOR_FUNC(round, vsl::float64x2, ({1.5, 1.49, -1.5, -2.9}), d_prec);
    VSL_TEST_VECTOR_FUNC_2ARGS(fmod, vsl::float32x4, ({5.3f, 18.5f, -7.2f, -15.3f}), ({2.2f, 4.2f, -3.1f, -6.4f}), f_prec);
    VSL_TEST_VECTOR_FUNC_2ARGS(fmod, vsl::float64x2, ({5.3, 18.5, -7.2, -15.3}), ({2.2, 4.2, -3.1, -6.4}), d_prec);

    f_prec = 1e-6f;
    d_prec = 1e-6;
    VSL_TEST_VECTOR_FUNC(cos, vsl::float32x4, ({0, 1, -1, 10, -10}), f_prec);
    VSL_TEST_VECTOR_FUNC(cos, vsl::float64x2, ({0, 1, -1, 10, -10}), d_prec);
    VSL_TEST_VECTOR_FUNC(sin, vsl::float32x4, ({0, 1, -1, 10, -10}), f_prec);
    VSL_TEST_VECTOR_FUNC(sin, vsl::float64x2, ({0, 1, -1, 10, -10}), d_prec);

    // typical max value for audio
    // pi * 20000 / 44100 = 1.424132f
    // pi * 20000 / 48000 = 1.308997f
    VSL_TEST_VECTOR_FUNC(tan, vsl::float32x4, ({0, 1, -1, 0.5f, -0.5f, 1.308997f}), f_prec);
    VSL_TEST_VECTOR_FUNC(tan, vsl::float64x2, ({0, 1, -1, 0.5, -0.5, 1.308997}), d_prec);

    // These are just approximations, don't test endpoints yet.
    f_prec = 1e-6f;
    d_prec = 1e-6;
    VSL_TEST_VECTOR_FUNC(asin, vsl::float32x4, ({0, 0.5f, -0.5f}), f_prec);
    VSL_TEST_VECTOR_FUNC(asin, vsl::float64x2, ({0, 0.5, -0.5}), d_prec);
    VSL_TEST_VECTOR_FUNC(acos, vsl::float32x4, ({0, 0.5f, -0.5f}), f_prec);
    VSL_TEST_VECTOR_FUNC(acos, vsl::float64x2, ({0, 0.5, -0.5}), d_prec);
    VSL_TEST_VECTOR_FUNC(atan, vsl::float32x4, ({0, 0.5f, -0.5f}), f_prec);
    VSL_TEST_VECTOR_FUNC(atan, vsl::float64x2, ({0, 0.5, -0.5}), d_prec);

    VSL_TEST_VECTOR_FUNC(cosh, vsl::float32x4, ({0, 1, -1}), f_prec);
    VSL_TEST_VECTOR_FUNC(cosh, vsl::float64x2, ({0, 1, -1}), d_prec);
    VSL_TEST_VECTOR_FUNC(sinh, vsl::float32x4, ({0, 1, -1}), f_prec);
    VSL_TEST_VECTOR_FUNC(sinh, vsl::float64x2, ({0, 1, -1}), d_prec);
    VSL_TEST_VECTOR_FUNC(tanh, vsl::float32x4, ({0, 1, -1, 3, -3}), f_prec);
    VSL_TEST_VECTOR_FUNC(tanh, vsl::float64x2, ({0, 1, -1, 3, -3}), d_prec);

    f_prec = 1e-6f;
    d_prec = 1e-6;
    VSL_TEST_VECTOR_FUNC(exp2, vsl::float32x4, ({-3, -1, 0, 1, 2, 3.5f, 7.1f}), f_prec);
    VSL_TEST_VECTOR_FUNC(exp2, vsl::float64x2, ({-3, -1, 0, 1, 2, 3.5, 7.1}), d_prec);
    VSL_TEST_VECTOR_FUNC(log2, vsl::float32x4, ({0.1f, 0.5f, 1, 2, 8, 3.15f, 69, 1/440.f, 20000/440.f}), f_prec);
    VSL_TEST_VECTOR_FUNC(log2, vsl::float64x2, ({0.1, 0.5, 1, 2, 8, 3.15, 69, 1/440.0, 20000/440.0}), d_prec);

    f_prec = 1e-6f;
    d_prec = 1e-6;
    VSL_TEST_VECTOR_FUNC(exp, vsl::float32x4, ({-6.55f, -3, -1, -0.277f, 0, 1, 1.15f, 2, 3.5f, 7.1f}), f_prec);
    VSL_TEST_VECTOR_FUNC(exp, vsl::float64x2, ({-6.55, -3, -1, -0.277, 0, 1, 1.15, 2, 3.5, 7.1}), d_prec);

    f_prec = 1e-6f;
    d_prec = 1e-6;
    VSL_TEST_VECTOR_FUNC(log, vsl::float32x4, ({0.1f, 0.5f, 1, 2, 8, 69}), f_prec);
    VSL_TEST_VECTOR_FUNC(log, vsl::float64x2, ({0.1, 0.5, 1, 2, 8, 69}), d_prec);
    VSL_TEST_VECTOR_FUNC(log10, vsl::float32x4, ({0.1f, 0.5f, 1, 2, 8, 69}), f_prec);
    VSL_TEST_VECTOR_FUNC(log10, vsl::float64x2, ({0.1, 0.5, 1, 2, 8, 69}), d_prec);

    f_prec = 1e-6f;
    d_prec = 1e-6;
    VSL_TEST_VECTOR_FUNC_2ARGS(pow, vsl::float32x4, ({2.f, 10.f, 5.f}), ({1.5f, -48 / 20.f, 0.f}), f_prec);
    VSL_TEST_VECTOR_FUNC_2ARGS(pow, vsl::float64x2, ({2.0, 10.0, 5.0}), ({1.5, -48 / 20.0, 0.0}), d_prec);

    // We're not hitting full accuracy for high gain values but we're still close.
    VSL_TEST_VECTOR_FUNC_2ARGS(pow, vsl::float32x4, ({10.f, 10.f, 10.f, 10.f, 10.f, 10.f, 10.f, 10.f}), ({-96.f/20.f, -48.f/20.f, -36.f/20.f, -24.f/20.f, -12.f/20.f, 0.f, 12.f/20.f, 24.f/20.f}), f_prec);
    VSL_TEST_VECTOR_FUNC_2ARGS(pow, vsl::float64x2, ({10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}), ({-96.0/20.0, -48.0/20.0, -36.0/20.0, -24.0/20.0, -12.0/20.0, 0.0, 12.0/20.0, 24.0/20.0}), d_prec);

    f_prec = 1e-7f;
    d_prec = 1e-15;
    VSL_TEST_VECTOR_FUNC_2ARGS(min, vsl::float32x4, ({1.f, 2.f, -1.f, 0.f}), ({0.f, 3.f, -2.f, 0.f}), f_prec);
    VSL_TEST_VECTOR_FUNC_2ARGS(min, vsl::float64x2, ({1.0, 2.0, -1.0, 0.0}), ({0.0, 3.0, -2.0, 0.0}), d_prec);
    VSL_TEST_VECTOR_FUNC_2ARGS(max, vsl::float32x4, ({1.f, 2.f, -1.f, 0.f}), ({0.f, 3.f, -2.f, 0.f}), f_prec);
    VSL_TEST_VECTOR_FUNC_2ARGS(max, vsl::float64x2, ({1.0, 2.0, -1.0, 0.0}), ({0.0, 3.0, -2.0, 0.0}), d_prec);
}