#include <array>
#include <cmath>
#include <format>

#include <vsl/vsl.hpp>
#include <audio_bench/audio_bench.hpp>

namespace {

template<typename T>
using Tracker = audio_bench::Error_tracker<T>; // Dumb can't do this in function scope.

auto test_math() -> void
{
    // MARK: - basic
    using Tests = audio_bench::Tests;
    using Error = audio_bench::Error;
    using Range = audio_bench::signals::Range;

    Tests::add("Test vsl::abs.", []() {
        auto xs = Range::make<float>({-10, 10, 100000});
        for (auto x : xs) {
            const auto v = vsl::abs(vsl::float32x4{x});
            const auto s = std::abs(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_true(v[0] == s);
        }

        auto xs_d = Range::make<double>({-10, 10, 100000});
        for (auto x : xs_d) {
            const auto v = vsl::abs(vsl::float64x2{x});
            const auto s = std::abs(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_true(v[0] == s);
        }
    });

    Tests::add("Test vsl::trunc.", []() {
        auto xs = Range::make<float>({-10, 10, 100000});
        for (auto x : xs) {
            const auto v = vsl::trunc(vsl::float32x4{x});
            const auto s = std::trunc(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_true(v[0] == s);
        }

        auto xs_d = Range::make<double>({-10, 10, 100000});
        for (auto x : xs_d) {
            const auto v = vsl::trunc(vsl::float64x2{x});
            const auto s = std::trunc(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_true(v[0] == s);
        }
    });

    Tests::add("Test vsl::floor.", []() {
        auto xs = Range::make<float>({-10, 10, 100000});
        for (auto x : xs) {
            const auto v = vsl::floor(vsl::float32x4{x});
            const auto s = std::floor(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_true(v[0] == s);
        }

        auto xs_d = Range::make<double>({-10, 10, 100000});
        for (auto x : xs_d) {
            const auto v = vsl::floor(vsl::float64x2{x});
            const auto s = std::floor(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_true(v[0] == s);
        }
    });

    Tests::add("Test vsl::ceil.", []() {
        auto xs = Range::make<float>({-10, 10, 100000});
        for (auto x : xs) {
            const auto v = vsl::ceil(vsl::float32x4{x});
            const auto s = std::ceil(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_true(v[0] == s);
        }

        auto xs_d = Range::make<double>({-10, 10, 100000});
        for (auto x : xs_d) {
            const auto v = vsl::ceil(vsl::float64x2{x});
            const auto s = std::ceil(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_true(v[0] == s);
        }
    });

    Tests::add("Test vsl::round.", []() {
        auto xs = Range::make<float>({-10, 10, 100000});
        for (auto x : xs) {
            const auto v = vsl::round(vsl::float32x4{x});
            const auto s = std::round(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_true(v[0] == s);
        }

        auto xs_d = Range::make<double>({-10, 10, 100000});
        for (auto x : xs_d) {
            const auto v = vsl::round(vsl::float64x2{x});
            const auto s = std::round(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_true(v[0] == s);
        }
    });

    Tests::add("Test vsl::round exact halves.", []() {
        const auto halves = std::array{0.5, 1.5, 2.5, 3.5, 4.5, -0.5, -1.5, -2.5, -3.5};
        for (auto x : halves) {
            const auto vf = vsl::round(vsl::float32x4{static_cast<float>(x)});
            const auto sf = std::round(static_cast<float>(x));
            audio_bench::expect_true(vsl::all_same(vf));
            audio_bench::expect_true(vf[0] == sf, std::format("round f32({}) = {}, std::round = {}", x, vf[0], sf));

            const auto vd = vsl::round(vsl::float64x2{x});
            const auto sd = std::round(x);
            audio_bench::expect_true(vsl::all_same(vd));
            audio_bench::expect_true(vd[0] == sd, std::format("round f64({}) = {}, std::round = {}", x, vd[0], sd));
        }
    });

    // MARK: - trig

    Tests::add("Test vsl::cos.", []() {
        audio_bench::log("Starting Test vsl::cos.");
        constexpr auto pi = std::numbers::pi_v<float>;
        auto xs = Range::make<float>({-pi, pi, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::cos<false>(vsl::float32x4{x});
            const auto s = std::cos(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{4e-7f}, std::format("Fail cos({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{4e-7f}, std::format("Fail cos({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{7}, std::format("Fail cos({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::sin.", []() {
        audio_bench::log("Starting Test vsl::sin.");
        constexpr auto pi = std::numbers::pi_v<float>;
        auto xs = Range::make<float>({-pi, pi, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::sin<false>(vsl::float32x4{x});
            const auto s = std::sin(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{3e-7f}, std::format("Fail sin({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{4e-7f}, std::format("Fail sin({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{5}, std::format("Fail sin({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::tan.", []() {
        audio_bench::log("Starting Test vsl::tan.");
        // Typical max values for filter calculations.
        // pi * 20000 / 44100 = ~1.424132f
        // pi * 20000 / 48000 = ~1.308997f
        auto xs = Range::make<float>({-1.424132f, 1.424132f, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::tan<false>(vsl::float32x4{x});
            const auto s = std::tan(x);
            audio_bench::expect_close(v[0], s, Error::Absolute{2e-6f}, std::format("Fail tan({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{4e-7f}, std::format("Fail tan({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{5}, std::format("Fail tan({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::cos_2pi.", []() {
        audio_bench::log("Starting Test vsl::cos_2pi.");
        auto xs = Range::make<float>({0, 1, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::cos_2pi<false>(vsl::float32x4{x});
            const auto s = static_cast<float>(std::cos(2 * std::numbers::pi_v<double> * x));
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{5e-7f}, std::format("Fail cos_2pi({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{5e-7f}, std::format("Fail cos_2pi({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{8}, std::format("Fail cos_2pi({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::sin_2pi.", []() {
        audio_bench::log("Starting Test vsl::sin_2pi.");
        auto xs = Range::make<float>({0, 1, 100001, false}); // We're exact at 0.5 and 1. Avoid those in the test data.

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::sin_2pi<false>(vsl::float32x4{x});
            const auto s = static_cast<float>(std::sin(2 * std::numbers::pi_v<double> * x));
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{2e-7f}, std::format("Fail sin_2pi({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{3e-7f}, std::format("Fail sin_2pi({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{5}, std::format("Fail sin_2pi({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::tan_pi.", []() {
        audio_bench::log("Starting Test vsl::tan_pi.");
        // tan_pi(f/fs) = tan(pi * f/fs), the filter prewarp; 20000/44100 = ~0.4535f.
        auto xs = Range::make<float>({-0.4535f, 0.4535f, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::tan_pi<false>(vsl::float32x4{x});
            const auto s = static_cast<float>(std::tan(std::numbers::pi_v<double> * x));
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{2e-6f}, std::format("Fail tan_pi({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{4e-7f}, std::format("Fail tan_pi({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{5}, std::format("Fail tan_pi({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    // MARK: - inverse trig

    Tests::add("Test vsl::acos.", []() {
        audio_bench::log("Starting Test vsl::acos.");
        // We're using this in butterworth filter to calculate alpha.
        // So in practice we just need -0.707...0.707.
        auto xs = Range::make<float>({-1.f, 1.f, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::acos(vsl::float32x4{x});
            const auto s = std::acos(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{5e-7f}, std::format("Fail acos({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{6e-6f}, std::format("Fail acos({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{5}, std::format("Fail acos({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::asin.", []() {
        audio_bench::log("Starting Test vsl::asin.");
        // We're using this in butterworth filter to calculate alpha.
        // So in practice we just need -0.707...0.707.
        auto xs = Range::make<float>({-1.f, 1.f, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::asin(vsl::float32x4{x});
            const auto s = std::asin(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{3e-7f}, std::format("Fail asin({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{6e-7f}, std::format("Fail asin({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{6}, std::format("Fail asin({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::atan.", []() {
        audio_bench::log("Starting Test vsl::atan.");
        auto xs = Range::make<float>({-10.f, 10.f, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::atan(vsl::float32x4{x});
            const auto s = std::atan(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{3e-7f}, std::format("Fail atan({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{3e-7f}, std::format("Fail atan({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{3}, std::format("Fail atan({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    // MARK: - exp, log, pow

    Tests::add("Test vsl::exp2.", []() {
        audio_bench::log("Starting Test vsl::exp2.");
        auto xs = Range::make<float>({-100, 100, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::exp2(vsl::float32x4{x});
            const auto s = std::exp2(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Relative{2e-7f}, std::format("Fail exp2({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{2}, std::format("Fail exp2({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::log2.", []() {
        audio_bench::log("Starting Test vsl::log2.");

        audio_bench::log("Testing range 0...1");
        auto xs = Range::make<float>({1e-10f, 1, 1000000}); // 1e-10f -> -200 dB

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::log2(vsl::float32x4{x});
            const auto s = std::log2(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-6f}, std::format("Fail log2({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{3e-7f}, std::format("Fail log2({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{3}, std::format("Fail log2({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();

        audio_bench::log("Testing range 1...1000");
        xs = Range::make<float>({1, 1000, 100000});

        errors = Tracker<float>{}; // reset
        for (auto x : xs) {
            const auto v = vsl::log2(vsl::float32x4{x});
            const auto s = std::log2(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-6f}, std::format("Fail log2({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{2e-7f}, std::format("Fail log2({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{1}, std::format("Fail log2({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::exp.", []() {
        audio_bench::log("Starting Test vsl::exp.");
        auto xs = Range::make<float>({-85, 85, 100000}); // We're flushing to zero at -87.34

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::exp(vsl::float32x4{x});
            const auto s = std::exp(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Relative{2e-7f}, std::format("Fail exp({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{2}, std::format("Fail exp({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::log.", []() {
        audio_bench::log("Starting Test vsl::log.");

        audio_bench::log("Testing range 0...1");
        auto xs = Range::make<float>({1e-10f, 1, 1000000}); // 1e-10f -> -200 dB

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::log(vsl::float32x4{x});
            const auto s = std::log(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-6f}, std::format("Fail log({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{3e-7f}, std::format("Fail log({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{3}, std::format("Fail log({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();

        audio_bench::log("Testing range 1...1000");
        xs = Range::make<float>({1, 1000, 100000});

        errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::log(vsl::float32x4{x});
            const auto s = std::log(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{5e-7f}, std::format("Fail log({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{2e-7f}, std::format("Fail log({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{2}, std::format("Fail log({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::log10.", []() {
        audio_bench::log("Starting Test vsl::log10.");

        audio_bench::log("Testing range 0...1");
        auto xs = Range::make<float>({1e-10f, 1, 1000000}); // 1e-10f -> -200 dB

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::log10(vsl::float32x4{x});
            const auto s = std::log10(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-6f}, std::format("Fail log10({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{3e-7f}, std::format("Fail log10({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{4}, std::format("Fail log10({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();

        audio_bench::log("Testing range 1...1000");
        xs = Range::make<float>({1, 1000, 100000});

        errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::log10(vsl::float32x4{x});
            const auto s = std::log10(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{5e-7f}, std::format("Fail log10({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{3e-7f}, std::format("Fail log10({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{3}, std::format("Fail log10({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::pow.", []() {
        audio_bench::log("Starting Test vsl::pow.");

        // Not having fma on SSE4.2 costs us a few ULPs here.
#if VSL_SSE4
        constexpr auto ulp_adj = size_t{30};
        constexpr auto rel_adj = 1.5f;
#else
        constexpr auto ulp_adj = size_t{0};
        constexpr auto rel_adj = 1.f;
#endif

        auto run = [](const char* label, float base_lo, float base_hi, float exp_lo, float exp_hi, float rel, size_t ulp, size_t n) {
            audio_bench::log(label);
            auto bases = Range::make<float>({base_lo, base_hi, n});
            auto exps = Range::make<float>({exp_lo, exp_hi, n});
            auto errors = Tracker<float>{};
            for (auto i = size_t{}; i < n; ++i) {
                const auto b = bases[i];
                const auto e = exps[i];
                const auto v = vsl::pow(vsl::float32x4{b}, vsl::float32x4{e});
                const auto s = std::pow(b, e);
                audio_bench::expect_true(vsl::all_same(v));
                audio_bench::expect_close(v[0], s, Error::Relative{rel}, std::format("Fail pow({}, {})", b, e));
                audio_bench::expect_close(v[0], s, Error::Ulp_distance{ulp}, std::format("Fail pow({}, {})", b, e));
                errors.update_with(e, v[0], s);
            }
            errors.report();
        };

        constexpr auto e = std::numbers::e_v<float>;
        // base 10, -200..+200 dB (db to lin is 10^(x/20))
        run("base 10, exp -10..10", 10, 10, -10, 10, 2e-6f * rel_adj, 9 + ulp_adj, 100000);
        run("base 2, exp -50..50", 2, 2, -50, 50, 5e-7f * rel_adj, 2 + ulp_adj, 100000);
        run("base e, exp -20..20", e, e, -20, 20, 2e-6f * rel_adj, 25 + ulp_adj, 100000);
        run("sqrt via pow, base 0..1e6", 0.001f, 1e6f, 0.5f, 0.5f, 1e-6f * rel_adj, 10 + ulp_adj, 100000);
        run("x^2.5, base 0..100", 0.001f, 100, 2.5f, 2.5f, 1e-6f * rel_adj, 12 + ulp_adj, 100000);
        run("base 0.1..0.95, exp -10", 0.1f, 0.95f, -10, -10, 2e-6f * rel_adj, 24 + ulp_adj, 100000);
        run("base 1.05..9, exp 10", 1.05f, 9, 10, 10, 2e-6f * rel_adj, 24 + ulp_adj, 100000);
    });

    Tests::add("Test vsl::sqrt.", []() {
        audio_bench::log("Starting Test vsl::sqrt.");
        auto xs = Range::make<float>({0, 100, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::sqrt(vsl::float32x4{x});
            const auto s = std::sqrt(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-6f}, std::format("Fail sqrt({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-3f}, std::format("Fail sqrt({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    // MARK: - hyperbolic trig

    Tests::add("Test vsl::cosh.", []() {
        audio_bench::log("Starting Test vsl::cosh.");
        auto xs = Range::make<float>({-5, 5, 100000});
        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::cosh(vsl::float32x4{x});
            const auto s = std::cosh(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{2e-5f}, std::format("Fail cosh({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{3e-7f}, std::format("Fail cosh({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{2}, std::format("Fail cosh({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::sinh.", []() {
        audio_bench::log("Starting Test vsl::sinh.");
        auto xs = Range::make<float>({-5, 5, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::sinh(vsl::float32x4{x});
            const auto s = std::sinh(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{2e-5f}, std::format("Fail sinh({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{2e-3f}, std::format("Fail sinh({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{18527}, std::format("Fail sinh({})", x)); // ...
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });
    
    Tests::add("Test vsl::tanh.", []() {
        audio_bench::log("Starting Test vsl::tanh.");
        auto xs = Range::make<float>({-10, 10, 100000});
        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::tanh(vsl::float32x4{x});
            const auto s = std::tanh(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{4e-3f}, std::format("Fail tanh({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{4e-3f}, std::format("Fail tanh({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{53596}, std::format("Fail tanh({})", x)); // ...
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });
}

struct Register { Register() { test_math(); } } register_{};

// MARK: - constexpr

namespace constexpr_checks {

using vsl::abs_equal;

// Primitives (bit-exact with std for these inputs).
static_assert(vsl::abs(-3.5f) == 3.5f);
static_assert(vsl::abs(2.0) == 2.0);
static_assert(vsl::trunc(2.7f) == 2.f);
static_assert(vsl::trunc(-2.7f) == -2.f);
static_assert(vsl::floor(2.7f) == 2.f);
static_assert(vsl::floor(-2.1f) == -3.f);
static_assert(vsl::ceil(2.1f) == 3.f);
static_assert(vsl::ceil(-2.7f) == -2.f);
static_assert(vsl::round(2.5f) == 3.f);
static_assert(vsl::round(-2.5f) == -3.f);
static_assert(vsl::fmod(7.5f, 2.f) == 1.5f);

// sqrt (Newton-Raphson, within tolerance of std::sqrt).
static_assert(vsl::sqrt(4.f) == 2.f);
static_assert(vsl::sqrt(0.f) == 0.f);
static_assert(abs_equal(vsl::sqrt(2.f), 1.41421356f));
static_assert(abs_equal(vsl::sqrt(1e-6f), 1e-3f));
static_assert(abs_equal(vsl::rsqrt(4.f), 0.5f));

// Exact-by-construction log/exp identities.
static_assert(vsl::log2(8.f) == 3.f);
static_assert(vsl::exp2(3.f) == 8.f);
static_assert(abs_equal(vsl::log(1.f), 0.f));
static_assert(abs_equal(vsl::exp(0.f), 1.f));
static_assert(abs_equal(vsl::log10(1000.f), 3.f, 1e-5f));
static_assert(vsl::pow(2.f, 10.f) == 1024.f);
static_assert(vsl::rel_equal(vsl::pow(10.f, 3.f), 1000.f, 1e-6f));
static_assert(vsl::rel_equal(vsl::pow(2.f, 0.5f), std::numbers::sqrt2_v<float>, 1e-6f));

// Trig / hyperbolic (same polynomial as the runtime scalar path). The minimax
// polynomials carry ~1e-7..1e-6 of error, so use a tolerance in line with the
// runtime math tests rather than abs_equal's tight default.
constexpr auto poly_tol = 1e-6f;
static_assert(abs_equal(vsl::cos<true>(0.f), 1.f, poly_tol));
static_assert(abs_equal(vsl::sin<true>(0.f), 0.f, poly_tol));
static_assert(abs_equal(vsl::tan<false>(0.f), 0.f, poly_tol));
static_assert(abs_equal(vsl::cos_2pi<true>(0.f), 1.f, poly_tol));
static_assert(abs_equal(vsl::sin_2pi<true>(0.f), 0.f, poly_tol));
static_assert(abs_equal(vsl::acos(1.f), 0.f, poly_tol));
static_assert(abs_equal(vsl::asin(0.f), 0.f, poly_tol));
static_assert(abs_equal(vsl::atan(1.f), std::numbers::pi_v<float> / 4.f, poly_tol));
static_assert(abs_equal(vsl::cosh(0.f), 1.f, poly_tol));
static_assert(abs_equal(vsl::sinh(0.f), 0.f, poly_tol));
static_assert(abs_equal(vsl::tanh(0.f), 0.f, poly_tol));

// A realistic DSP constant computed entirely at compile time:
// bilinear-transform pre-warp gain for a 1 kHz cutoff at 48 kHz, tan(pi/48).
constexpr auto g = vsl::tan<true>(std::numbers::pi_v<float> * 1000.f / 48000.f);
static_assert(abs_equal(g, 0.0655434f, 1e-5f));

} // namespace constexpr_checks

} // namespace