#pragma once

#include <cmath>
#include <format>

#include <vsl/vsl.h>
#include <audio_bench/audio_bench.h>

inline auto test_math() -> void
{
    // MARK: - basic

    audio_bench::add_test("Test vsl::abs.", [&]() {
        auto xs = audio_bench::make_range(-10.f, 10.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::abs(vsl::float32x4{x});
            const auto s = std::abs(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-6f}, std::format("Fail abs({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail abs({})", x), false);
        }
    });

    audio_bench::add_test("Test vsl::trunc.", [&]() {
        auto xs = audio_bench::make_range(-10.f, 10.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::trunc(vsl::float32x4{x});
            const auto s = std::trunc(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-6f}, std::format("Fail trunc({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail trunc({})", x), false);
        }
    });

    audio_bench::add_test("Test vsl::floor.", [&]() {
        auto xs = audio_bench::make_range(-10.f, 10.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::floor(vsl::float32x4{x});
            const auto s = std::floor(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-6f}, std::format("Fail floor({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail floor({})", x), false);
        }
    });

    audio_bench::add_test("Test vsl::ceil.", [&]() {
        auto xs = audio_bench::make_range(-10.f, 10.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::ceil(vsl::float32x4{x});
            const auto s = std::ceil(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-6f}, std::format("Fail ceil({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail ceil({})", x), false);
        }
    });

    audio_bench::add_test("Test vsl::round.", [&]() {
        auto xs = audio_bench::make_range(-10.f, 10.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::round(vsl::float32x4{x});
            const auto s = std::round(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-6f}, std::format("Fail round({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail round({})", x), false);
        }
    });

    // MARK: - trig

    audio_bench::add_test("Test vsl::cos.", [&]() {
        auto xs = audio_bench::make_range(-5.f, 5.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::cos(vsl::float32x4{x});
            const auto s = std::cos(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-6f}, std::format("Fail cos({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{5e-2f}, std::format("Fail cos({})", x), false);
        }
    });

    audio_bench::add_test("Test vsl::sin.", [&]() {
        auto xs = audio_bench::make_range(-5.f, 5.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::sin(vsl::float32x4{x});
            const auto s = std::sin(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-6f}, std::format("Fail sin({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{5e-2f}, std::format("Fail sin({})", x), false);
        }
    });

    audio_bench::add_test("Test vsl::tan.", [&]() {
        // Typical max values for filter calculations.
        // pi * 20000 / 44100 = ~1.424132f
        // pi * 20000 / 48000 = ~1.308997f
        auto xs = audio_bench::make_range(-1.424132f, 1.424132f, 100000);
        for (auto x : xs) {
            const auto v = vsl::tan(vsl::float32x4{x});
            const auto s = std::tan(x);
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-5f}, std::format("Fail tan({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{2e-3f}, std::format("Fail tan({})", x), false);
        }
    });

    audio_bench::add_test("Test vsl::cos_2pi.", [&]() {
        auto xs = audio_bench::make_range(0.f, 1.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::cos_2pi(vsl::float32x4{x});
            const auto s = std::cos(2 * std::numbers::pi_v<float> * x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-4f}, std::format("Fail cos_2pi({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{5e-3f}, std::format("Fail cos_2pi({})", x), false);
        }
    });

    audio_bench::add_test("Test vsl::sin_2pi.", [&]() {
        auto xs = audio_bench::make_range(0.f, 1.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::sin_2pi(vsl::float32x4{x});
            const auto s = std::sin(2 * std::numbers::pi_v<float> * x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-4f}, std::format("Fail sin_2pi({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1.f}, std::format("Fail sin_2pi({})", x), false);
        }
    });

    // MARK: - inverse trig

    audio_bench::add_test("Test vsl::acos.", [&]() {
        // We're using this in butterworth filter to calculate alpha.
        auto xs = audio_bench::make_range(-0.706f, 0.706f, 100000);
        for (auto x : xs) {
            const auto v = vsl::acos(vsl::float32x4{x});
            const auto s = std::acos(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{3e-6f}, std::format("Fail acos({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail acos({})", x), false);
        }
    });

    audio_bench::add_test("Test vsl::asin.", [&]() {
        // We're using this in butterworth filter to calculate alpha.
        auto xs = audio_bench::make_range(-0.706f, 0.706f, 100000);
        for (auto x : xs) {
            const auto v = vsl::asin(vsl::float32x4{x});
            const auto s = std::asin(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{3e-6f}, std::format("Fail asin({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail asin({})", x), false);
        }
    });

    audio_bench::add_test("Test vsl::atan.", [&]() {
        auto xs = audio_bench::make_range(-2.5f, 2.5f, 100000);
        for (auto x : xs) {
            const auto v = vsl::atan(vsl::float32x4{x});
            const auto s = std::atan(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-2f}, std::format("Fail atan({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-2f}, std::format("Fail atan({})", x), false);
        }
    });

    // MARK: - exp, log, pow

    audio_bench::add_test("Test vsl::exp2.", [&]() {
        auto xs = audio_bench::make_range(-10.f, 10.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::exp2(vsl::float32x4{x});
            const auto s = std::exp2(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{2e-4f}, std::format("Fail exp2({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail exp2({})", x), false);
        }
    });

    audio_bench::add_test("Test vsl::log2.", [&]() {
        auto xs = audio_bench::make_range(1e-6f, 1.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::log2(vsl::float32x4{x});
            const auto s = std::log2(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-5f}, std::format("Fail log2({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{2e-2f}, std::format("Fail log2({})", x), false); // Almost at 1e-2f
        }
        xs = audio_bench::make_range(1.f, 1000.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::log2(vsl::float32x4{x});
            const auto s = std::log2(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-5f}, std::format("Fail log2({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail log2({})", x), false);
        }
    });

    audio_bench::add_test(" Test vsl::exp.", [&]() {
        auto xs = audio_bench::make_range(-10.f, 10.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::exp(vsl::float32x4{x});
            const auto s = std::exp(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-2f}, std::format("Fail exp({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail exp({})", x), false);
        }
    });

    audio_bench::add_test(" Test vsl::log.", [&]() {
        auto xs = audio_bench::make_range(1e-6f, 1.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::log(vsl::float32x4{x});
            const auto s = std::log(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-5f}, std::format("Fail log({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{2e-2f}, std::format("Fail log({})", x), false);
        }
        xs = audio_bench::make_range(1.f, 1000.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::log(vsl::float32x4{x});
            const auto s = std::log(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-5f}, std::format("Fail log({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail log({})", x), false);
        }
    });

    audio_bench::add_test("Test vsl::log10.", [&]() {
        auto xs = audio_bench::make_range(1e-6f, 1.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::log10(vsl::float32x4{x});
            const auto s = std::log10(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-5f}, std::format("Fail log10({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{2e-2f}, std::format("Fail log10({})", x), false);
        }
        xs = audio_bench::make_range(1.f, 1000.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::log10(vsl::float32x4{x});
            const auto s = std::log10(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-5f}, std::format("Fail log10({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail log10({})", x), false);
        }
    });

    audio_bench::add_test("Test vsl::pow.", []() {
        // Test base 10
        auto xs = audio_bench::make_range(-5.f, 5.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::pow(vsl::float32x4{10.f}, vsl::float32x4{x});
            const auto s = std::pow(10.f, x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1.f}, std::format("Fail pow(10, {})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail pow(10, {})", x), false);
        }
        // Test exponent 0.5 (sqrt case)
        xs = audio_bench::make_range(0.f, 100.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::pow(vsl::float32x4{x}, vsl::float32x4{0.5f});
            const auto s = std::pow(x, 0.5f);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{1e-4f}, std::format("Fail pow({}, 0.5)", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail pow({}, 0.5)", x), false);
        }
    });

    // MARK: - hyperbolic trig

    audio_bench::add_test("Test vsl::sinh.", [&]() {
        auto xs = audio_bench::make_range(-5.f, 5.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::sinh(vsl::float32x4{x});
            const auto s = std::sinh(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{2e-5f}, std::format("Fail sinh({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail sinh({})", x), false);
        }
    });

    audio_bench::add_test("Test vsl::cosh.", [&]() {
        auto xs = audio_bench::make_range(-5.f, 5.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::cosh(vsl::float32x4{x});
            const auto s = std::cosh(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{2e-5f}, std::format("Fail cosh({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail cosh({})", x), false);
        }
    });
    
    audio_bench::add_test("Test vsl::tanh.", [&]() {
        auto xs = audio_bench::make_range(-5.f, 5.f, 100000);
        for (auto x : xs) {
            const auto v = vsl::tanh(vsl::float32x4{x});
            const auto s = std::tanh(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, audio_bench::Abs_err{2e-4f}, std::format("Fail tanh({})", x), false);
            audio_bench::expect_close(v[0], s, audio_bench::Rel_err{1e-3f}, std::format("Fail tanh({})", x), false);
        }
    });
}