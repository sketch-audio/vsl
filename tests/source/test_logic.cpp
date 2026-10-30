#include <limits>
#include <string>

#include <vsl/vsl.hpp>
#include <audio_bench/audio_bench.hpp>

namespace {

template<typename F>
auto test_logic(const std::string& name) -> void
{
    using Tests = audio_bench::Tests;
    using V = vsl::vector_t<F>;
    constexpr auto N = vsl::num_members_v<V>;

    // Build a vector with the given lanes, ignoring the extra args on 2-lane types.
    auto make = [](F a, F b, F c, F d) -> V {
        (void)c;
        (void)d;
        if constexpr (N == 4) {
            return V{a, b, c, d};
        } else {
            return V{a, b};
        }
    };

    Tests::add("logic - all_same (" + name + ")", [=]() {
        audio_bench::expect_true(vsl::all_same(F{42}), "scalar is trivially all_same");
        audio_bench::expect_true(vsl::all_same(V{7}), "broadcast lanes are all_same");
        audio_bench::expect_true(!vsl::all_same(make(1, 2, 3, 4)), "distinct lanes are not all_same");
    });

    Tests::add("logic - all_unique (" + name + ")", [=]() {
        audio_bench::expect_true(vsl::all_unique(F{42}), "scalar is trivially unique");
        audio_bench::expect_true(vsl::all_unique(make(1, 2, 3, 4)), "distinct lanes are unique");
        // make(1, 1, ...) duplicates lane 0 and 1 on both widths.
        audio_bench::expect_true(!vsl::all_unique(make(1, 1, 2, 3)), "duplicate lane breaks uniqueness");
    });

    // Infinities and NaN are UB under -ffinite-math-only (VSL_FINITE_MATH), so this can't run there.
#if !VSL_FINITE_MATH
    Tests::add("logic - all_finite (" + name + ")", [=]() {
        const auto inf = std::numeric_limits<F>::infinity();
        const auto nan = std::numeric_limits<F>::quiet_NaN();

        audio_bench::expect_true(vsl::all_finite(F{1}), "scalar finite");
        audio_bench::expect_true(!vsl::all_finite(inf), "scalar +inf not finite");
        audio_bench::expect_true(!vsl::all_finite(-inf), "scalar -inf not finite");
        audio_bench::expect_true(!vsl::all_finite(nan), "scalar NaN not finite");

        audio_bench::expect_true(vsl::all_finite(make(1, 2, 3, 4)), "all-finite vector");
        audio_bench::expect_true(!vsl::all_finite(make(1, inf, 2, 3)), "one inf lane breaks finite");
        audio_bench::expect_true(!vsl::all_finite(make(1, nan, 2, 3)), "one NaN lane breaks finite");
    });
#endif

    Tests::add("logic - abs_equal (" + name + ")", [=]() {
        // Boundary either side of the default 1e-7 tolerance.
        const auto near = F{1} + static_cast<F>(1e-8);
        const auto far = F{1} + static_cast<F>(1e-5);
        audio_bench::expect_true(vsl::abs_equal(F{1}, near), "scalar within tol");
        audio_bench::expect_true(!vsl::abs_equal(F{1}, far), "scalar outside tol");

        audio_bench::expect_true(vsl::all(vsl::abs_equal(V{1}, V{near})), "vector within tol");
        audio_bench::expect_true(!vsl::any(vsl::abs_equal(V{1}, V{far})), "vector outside tol");
    });

    Tests::add("logic - rel_equal (" + name + ")", [=]() {
        // Boundary either side of the default 1e-3 relative tolerance.
        audio_bench::expect_true(vsl::rel_equal(F{1}, F{1} + static_cast<F>(1e-4)), "scalar within tol");
        audio_bench::expect_true(!vsl::rel_equal(F{1}, F{1} + static_cast<F>(1e-2)), "scalar outside tol");

        audio_bench::expect_true(vsl::all(vsl::rel_equal(V{static_cast<F>(100.1)}, V{100})), "vector within tol");
        audio_bench::expect_true(!vsl::any(vsl::rel_equal(V{101}, V{100})), "vector outside tol");
    });

    Tests::add("logic - bit_equal (" + name + ")", [=]() {
        // bit_equal is scalar-only (it branches on a scalar exponent compare).
        audio_bench::expect_true(vsl::bit_equal(F{1}, F{1}), "identical bits match");
        // Differing exponents are rejected outright.
        audio_bench::expect_true(!vsl::bit_equal(F{1}, F{2}), "different exponent never matches");
        // Close-but-not-identical: enough significand bits agree for a loose tol,
        // not enough for a strict one.
        audio_bench::expect_true(vsl::bit_equal(F{1}, F{1} + static_cast<F>(1e-3), 6), "loose tol matches near values");
        audio_bench::expect_true(!vsl::bit_equal(F{1}, F{1} + static_cast<F>(1e-3), 20), "strict tol rejects near values");
    });
}

struct Register {
    Register() {
        test_logic<float>("float32");
        test_logic<double>("float64");
    }
} register_{};

} // namespace
