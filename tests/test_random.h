#pragma once

#include <format>

#include <vsl/vsl.h>
#include <audio_bench/audio_bench.h>

template<typename S, typename V, vsl::Random_engine Engine>
inline auto test_engine() -> void
{
    using Tests = audio_bench::Tests;

    // Scalar vs vector equality for same seed
    Tests::add("random - scalar/vector equality", []() {
        auto srng = vsl::Random_gen<S, Engine>{};
        auto vrng = vsl::Random_gen<V, Engine>{};
        srng.reset();
        vrng.reset();

        auto sprev = S{};
        for (size_t i = 0; i < 10000; ++i) {
            const auto s = srng.next();
            const auto v = vrng.next();
            audio_bench::expect_true((s >= 0 && s <= 1), std::format("Scalar output in [0,1], got {}", s));
            audio_bench::expect_true((s != sprev), "No repeats in scalar output");

            for (size_t j = 0; j < V::num_lanes; ++j) {
                audio_bench::expect_true(s == v[j], "Scalar and vector outputs the same");
            }
            sprev = s;
        }
    });

    // Range test
    Tests::add("random - min/max range", []() {
        const auto min = S{5};
        const auto max = S{10};
        auto srng = vsl::Random_gen<S>{min, max};
        auto vrng = vsl::Random_gen<V>{min, max};
        srng.reset();
        vrng.reset();

        for (size_t i = 0; i < 1000; ++i) {
            const auto s = srng.next();
            const auto v = vrng.next();
            audio_bench::expect_true((s >= min && s <= max), "Scalar output in [min, max]");
            for (size_t j = 0; j < V::num_lanes; ++j) {
                audio_bench::expect_true((v[j] >= min && v[j] <= max), "Vector output in [min, max]");
            }
        }
    });

    // Per-lane variability with vector seed
    Tests::add("random - per-lane variability", []() {
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
                    audio_bench::expect_true((x[j] != x[k]), "Different lanes give different outputs");
                }
            }
        }
    });
}

inline auto test_random() -> void
{
    test_engine<float, vsl::float32x4, vsl::Random_engine::lin_cong>();
    test_engine<float, vsl::float32x4, vsl::Random_engine::xor_shift>();
    test_engine<double, vsl::float64x2, vsl::Random_engine::lin_cong>();
    test_engine<double, vsl::float64x2, vsl::Random_engine::xor_shift>();
}