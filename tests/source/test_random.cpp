#include <format>

#include <vsl/vsl.hpp>
#include <audio_bench/audio_bench.hpp>

namespace {

template<typename S, typename V, vsl::random::Engine Engine>
auto test_engine() -> void
{
    using Tests = audio_bench::Tests;

    // Scalar vs vector equality for same seed
    Tests::add("random - scalar/vector equality", []() {
        auto srng = vsl::random::Generator<S, Engine>{};
        auto vrng = vsl::random::Generator<V, Engine>{};
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
        auto srng = vsl::random::Generator<S>{min, max};
        auto vrng = vsl::random::Generator<V>{min, max};
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

        auto vrng = vsl::random::Generator<V>{make_seed()};
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

auto test_random() -> void
{
    using enum vsl::random::Engine;
    test_engine<float, vsl::float32x4, Lin_cong>();
    test_engine<float, vsl::float32x4, Xor_shift>();
    test_engine<double, vsl::float64x2, Lin_cong>();
    test_engine<double, vsl::float64x2, Xor_shift>();
}

struct Register { Register() { test_random(); } } register_{};

} // namespace