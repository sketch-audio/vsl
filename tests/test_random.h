#pragma once

#include <cassert>
#include "vsl.h"

template<typename S, typename V, vsl::Random_engine Engine>
inline auto test_engine() -> void
{
    {
        // Test that scalar and vector generators give the same output for the same seed.
        auto srng = vsl::Random_gen<S, Engine>{};
        auto vrng = vsl::Random_gen<V, Engine>{};
        srng.reset();
        vrng.reset();
        auto sprev = S{};
        for (size_t i = 0; i < 10000; ++i) {
            const auto s = srng.next();
            const auto v = vrng.next();
            assert(s >= 0 && s <= 1 && "Scalar output in [0, 1].");
            assert(s != sprev && "No repeats in scalar output.");

            for (size_t j = 0; j < V::num_lanes; ++j) {
                assert(s == v[j] && "Scalar and vector outputs the same.");
            }
            sprev = s;
        }
    }

    {
        // Test that outputs are in min/max range.
        const auto min = S{5};
        const auto max = S{10};
        auto srng = vsl::Random_gen<S>{min, max};
        auto vrng = vsl::Random_gen<V>{min, max};
        srng.reset();
        vrng.reset();

        for (size_t i = 0; i < 1000; ++i) {
            const auto s = srng.next();
            const auto v = vrng.next();
            assert((s >= min && s <= max) && "Scalar output in [min, max].");
            for (size_t j = 0; j < V::num_lanes; ++j) {
                assert((v[j] >= min && v[j] <= max) && "Vector output in [min, max].");
            }
        }
    }

    {
        // Test that vector generator with vector seed gives different outputs per lane.
        auto make_seed = []() {
            using M = vsl::mask_t<V>;
            if constexpr (M::num_lanes == 2) {
                return M{808, 909};
            } else if constexpr (M::num_lanes == 4) {
                return M{303, 404, 808, 909};
            } else {
                static_assert(vsl::deferred_false_v<M>, "Unhandled number of lanes.");
            }
        };
        auto vrng = vsl::Random_gen<V>{make_seed()};
        vrng.reset();

        for (size_t i = 0; i < 1000; ++i) {
            const auto x = vrng.next();
            for (size_t j = 0; j < V::num_lanes; ++j) {
                for (size_t k = j + 1; k < V::num_lanes; ++k) {
                    assert((x[j] != x[k]) && "Different lanes give different outputs.");
                }
            }
        }
    }
}

inline auto test_random() -> void
{
    test_engine<float, vsl::float32x4, vsl::Random_engine::lin_cong>();
    test_engine<float, vsl::float32x4, vsl::Random_engine::xor_shift>();
    test_engine<double, vsl::float64x2, vsl::Random_engine::lin_cong>();
    test_engine<double, vsl::float64x2, vsl::Random_engine::xor_shift>();
}