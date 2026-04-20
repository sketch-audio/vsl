#pragma once

#include <cmath>
#include <format>

#include <vsl/vsl.h>
#include <audio_bench/audio_bench.hpp>

template<typename T>
using Tracker = audio_bench::Error_tracker<T>; // Dumb can't do this in function scope.

inline auto test_math() -> void
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

    // MARK: - trig

    Tests::add("Test vsl::cos.", []() {
        audio_bench::log("Starting Test vsl::cos.");
        auto xs = Range::make<float>({-5, 5, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::cos(vsl::float32x4{x});
            const auto s = std::cos(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-6f}, std::format("Fail cos({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{5e-2f}, std::format("Fail cos({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::sin.", []() {
        audio_bench::log("Starting Test vsl::sin.");
        auto xs = Range::make<float>({-5, 5, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::sin(vsl::float32x4{x});
            const auto s = std::sin(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-6f}, std::format("Fail sin({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{5e-2f}, std::format("Fail sin({})", x));
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
            const auto v = vsl::tan(vsl::float32x4{x});
            const auto s = std::tan(x);
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-5f}, std::format("Fail tan({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{5e-3f}, std::format("Fail tan({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::cos_2pi.", []() {
        audio_bench::log("Starting Test vsl::cos_2pi.");
        auto xs = Range::make<float>({0, 1, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::cos_2pi(vsl::float32x4{x});
            const auto s = std::cos(2 * std::numbers::pi_v<float> * x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-4f}, std::format("Fail cos_2pi({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{5e-3f}, std::format("Fail cos_2pi({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::sin_2pi.", []() {
        audio_bench::log("Starting Test vsl::sin_2pi.");
        auto xs = Range::make<float>({0, 1, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::sin_2pi(vsl::float32x4{x});
            const auto s = std::sin(2 * std::numbers::pi_v<float> * x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-4f}, std::format("Fail sin_2pi({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1.f}, std::format("Fail sin_2pi({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    // MARK: - inverse trig

    Tests::add("Test vsl::acos.", []() {
        audio_bench::log("Starting Test vsl::acos.");
        // We're using this in butterworth filter to calculate alpha.
        auto xs = Range::make<float>({-0.706f, 0.706f, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::acos(vsl::float32x4{x});
            const auto s = std::acos(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{3e-6f}, std::format("Fail acos({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-3f}, std::format("Fail acos({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::asin.", []() {
        audio_bench::log("Starting Test vsl::asin.");
        // We're using this in butterworth filter to calculate alpha.
        auto xs = Range::make<float>({-0.706f, 0.706f, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::asin(vsl::float32x4{x});
            const auto s = std::asin(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{3e-6f}, std::format("Fail asin({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-3f}, std::format("Fail asin({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::atan.", []() {
        audio_bench::log("Starting Test vsl::atan.");
        auto xs = Range::make<float>({-2.5f, 2.5f, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::atan(vsl::float32x4{x});
            const auto s = std::atan(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-2f}, std::format("Fail atan({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-2f}, std::format("Fail atan({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    // MARK: - exp, log, pow

    Tests::add("Test vsl::exp2.", []() {
        audio_bench::log("Starting Test vsl::exp2.");
        auto xs = Range::make<float>({-10, 10, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::exp2(vsl::float32x4{x});
            const auto s = std::exp2(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{2e-4f}, std::format("Fail exp2({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-3f}, std::format("Fail exp2({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::log2.", []() {
        audio_bench::log("Starting Test vsl::log2.");

        audio_bench::log("Testing range 0...1");
        auto xs = Range::make<float>({1e-6f, 1, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::log2(vsl::float32x4{x});
            const auto s = std::log2(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-5f}, std::format("Fail log2({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{2e-2f}, std::format("Fail log2({})", x)); // Almost at 1e-2f
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
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-5f}, std::format("Fail log2({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-3f}, std::format("Fail log2({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::exp (extreme values).", []() {
        audio_bench::log("Starting Test vsl::exp (extreme values).");
        auto xs = Range::make<double>({-700, 0, 10000});

        auto errors = Tracker<double>{};
        for (auto x : xs) {
            const auto v = vsl::exp(vsl::float64x2{x});
            const auto s = std::exp(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-2}, std::format("Fail exp({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-3}, std::format("Fail exp({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::exp.", []() {
        audio_bench::log("Starting Test vsl::exp.");
        auto xs = Range::make<float>({-10, 10, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::exp(vsl::float32x4{x});
            const auto s = std::exp(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-2f}, std::format("Fail exp({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-3f}, std::format("Fail exp({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::log.", []() {
        audio_bench::log("Starting Test vsl::log.");

        audio_bench::log("Testing range 0...1");
        auto xs = Range::make<float>({1e-6f, 1, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::log(vsl::float32x4{x});
            const auto s = std::log(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-5f}, std::format("Fail log({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{2e-2f}, std::format("Fail log({})", x));
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
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-5f}, std::format("Fail log({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-3f}, std::format("Fail log({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::log10.", []() {
        audio_bench::log("Starting Test vsl::log10.");

        audio_bench::log("Testing range 0...1");
        auto xs = Range::make<float>({1e-6f, 1, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::log10(vsl::float32x4{x});
            const auto s = std::log10(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-5f}, std::format("Fail log10({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{2e-2f}, std::format("Fail log10({})", x));
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
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-5f}, std::format("Fail log10({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-3f}, std::format("Fail log10({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::pow.", []() {
        audio_bench::log("Starting Test vsl::pow.");

        audio_bench::log("Testing base 10");
        // Test base 10
        auto xs = Range::make<float>({-5, 5, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::pow(vsl::float32x4{10}, vsl::float32x4{x});
            const auto s = std::pow(10.f, x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1.f}, std::format("Fail pow(10, {})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-3f}, std::format("Fail pow(10, {})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();

        audio_bench::log("Testing exponent 0.5 (sqrt case)");
        // Test exponent 0.5 (sqrt case)
        xs = Range::make<float>({0, 100, 100000});

        errors = Tracker<float>{}; // reset
        for (auto x : xs) {
            const auto v = vsl::pow(vsl::float32x4{x}, vsl::float32x4{0.5f});
            const auto s = std::pow(x, 0.5f);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-4f}, std::format("Fail pow({}, 0.5)", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-3f}, std::format("Fail pow({}, 0.5)", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
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
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::cosh.", []() {
        audio_bench::log("Starting Test vsl::cosh.");
        auto xs = Range::make<float>({-5, 5, 100000});
        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::cosh(vsl::float32x4{x});
            const auto s = std::cosh(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{2e-5f}, std::format("Fail cosh({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-3f}, std::format("Fail cosh({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });
    
    Tests::add("Test vsl::tanh.", []() {
        audio_bench::log("Starting Test vsl::tanh.");
        auto xs = Range::make<float>({-5, 5, 100000});
        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::tanh(vsl::float32x4{x});
            const auto s = std::tanh(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{2e-4f}, std::format("Fail tanh({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-3f}, std::format("Fail tanh({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });
}