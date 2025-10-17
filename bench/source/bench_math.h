#pragma once

#include <vsl/vsl.h>
#include <audio_bench/audio_bench.h>

// MARK: - trig

inline auto bench_cos() -> void
{
    auto xs = audio_bench::make_range(-10.f, 10.f, 100000);

    auto t_std = audio_bench::Lambda_timer("std::cos process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::cos(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Lambda_timer("vsl::cos process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::cos(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using enum audio_bench::Bench_units;

    const auto stats_std = t_std.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_std);

    const auto stats_vsl = t_vsl.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_vsl);

    audio_bench::compare_benchmarks(stats_std, stats_vsl);
}

inline auto bench_sin() -> void
{
    auto xs = audio_bench::make_range(-10.f, 10.f, 100000);

    auto t_std = audio_bench::Lambda_timer("std::sin process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::sin(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Lambda_timer("vsl::sin process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::sin(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using enum audio_bench::Bench_units;

    const auto stats_std = t_std.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_std);

    const auto stats_vsl = t_vsl.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_vsl);

    audio_bench::compare_benchmarks(stats_std, stats_vsl);
}

inline auto bench_tan() -> void
{
    auto xs = audio_bench::make_range(-10.f, 10.f, 100000);

    auto t_std = audio_bench::Lambda_timer("std::tan process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::tan(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Lambda_timer("vsl::tan process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::tan(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using enum audio_bench::Bench_units;

    const auto stats_std = t_std.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_std);

    const auto stats_vsl = t_vsl.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_vsl);

    audio_bench::compare_benchmarks(stats_std, stats_vsl);
}

inline auto bench_cos_2pi() -> void
{
    auto xs = audio_bench::make_range(0.f, 1.f, 100000);

    auto t_std = audio_bench::Lambda_timer("std::cos(2pi*x) process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::cos(2 * std::numbers::pi_v<float> * x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Lambda_timer("vsl::cos_2pi process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::cos_2pi(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using enum audio_bench::Bench_units;

    const auto stats_std = t_std.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_std);

    const auto stats_vsl = t_vsl.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_vsl);

    audio_bench::compare_benchmarks(stats_std, stats_vsl);
}

inline auto bench_sin_2pi() -> void
{
    auto xs = audio_bench::make_range(0.f, 1.f, 100000);

    auto t_std = audio_bench::Lambda_timer("std::sin(2pi*x) process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::sin(2 * std::numbers::pi_v<float> * x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Lambda_timer("vsl::sin_2pi process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::sin_2pi(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using enum audio_bench::Bench_units;

    const auto stats_std = t_std.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_std);

    const auto stats_vsl = t_vsl.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_vsl);

    audio_bench::compare_benchmarks(stats_std, stats_vsl);
}

// MARK: - exp2/log2

inline auto bench_exp2() -> void
{
    auto xs = audio_bench::make_range(-10.f, 10.f, 100000);

    auto t_std = audio_bench::Lambda_timer("std::exp2 process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::exp2(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Lambda_timer("vsl::exp2 process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::exp2(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using enum audio_bench::Bench_units;

    const auto stats_std = t_std.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_std);

    const auto stats_vsl = t_vsl.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_vsl);

    audio_bench::compare_benchmarks(stats_std, stats_vsl);
}

inline auto bench_log2() -> void
{
    auto xs = audio_bench::make_range(0.0001f, 10000.f, 100000);

    auto t_std = audio_bench::Lambda_timer("std::log2 process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::log2(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Lambda_timer("vsl::log2 process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::log2(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using enum audio_bench::Bench_units;

    const auto stats_std = t_std.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_std);

    const auto stats_vsl = t_vsl.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_vsl);

    audio_bench::compare_benchmarks(stats_std, stats_vsl);
}

// MARK: - exp

inline auto bench_exp() -> void
{
    auto xs = audio_bench::make_range(-10.f, 10.f, 100000);

    auto t_std = audio_bench::Lambda_timer("std::exp process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::exp(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Lambda_timer("vsl::exp process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::exp(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using enum audio_bench::Bench_units;

    const auto stats_std = t_std.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_std);

    const auto stats_vsl = t_vsl.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_vsl);

    audio_bench::compare_benchmarks(stats_std, stats_vsl);
}

// MARK: - pow

inline auto bench_pow() -> void
{
    auto xs = audio_bench::make_range(0.0001f, 100.f, 1000);
    auto ys = audio_bench::make_range(-10.f, 10.f, 100);

    std::vector<std::pair<float, float>> pairs;
    pairs.reserve(xs.size() * ys.size());
    for (auto x : xs) {
        for (auto y : ys) {
            pairs.emplace_back(x, y);
        }
    }

    auto t_std = audio_bench::Lambda_timer("std::pow process 100000 float pairs", [&]() {
        for (const auto& [x, y] : pairs) {
            auto z = std::pow(x, y);
            audio_bench::do_not_optimize(z);
        }
    });

    auto t_vsl = audio_bench::Lambda_timer("vsl::pow process 100000 float pairs", [&]() {
        for (const auto& [x, y] : pairs) {
            auto z = vsl::pow(x, y);
            audio_bench::do_not_optimize(z);
        }
    });

    using enum audio_bench::Bench_units;

    const auto stats_std = t_std.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_std);

    const auto stats_vsl = t_vsl.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_vsl);

    audio_bench::compare_benchmarks(stats_std, stats_vsl);
}

inline auto bench_sqrt() -> void
{
    auto xs = audio_bench::make_range(0.0001f, 10000.f, 100000);
    auto xs_vec = audio_bench::make_range(vsl::float32x4{0.0001f}, vsl::float32x4{10000.f}, 25000);

    auto t_std = audio_bench::Lambda_timer("std::sqrt process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::sqrt(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Lambda_timer("vsl::sqrt process 100000 floats", [&]() {
        for (auto x : xs_vec) {
            auto y = vsl::sqrt(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using enum audio_bench::Bench_units;

    const auto stats_std = t_std.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_std);

    const auto stats_vsl = t_vsl.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_vsl);

    audio_bench::compare_benchmarks(stats_std, stats_vsl);
}

// MARK: - sinh/cosh/tanh

inline auto bench_sinh() -> void
{
    auto xs = audio_bench::make_range(-5.f, 5.f, 100000);

    auto t_std = audio_bench::Lambda_timer("std::sinh process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::sinh(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Lambda_timer("vsl::sinh process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::sinh(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using enum audio_bench::Bench_units;

    const auto stats_std = t_std.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_std);

    const auto stats_vsl = t_vsl.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_vsl);

    audio_bench::compare_benchmarks(stats_std, stats_vsl);
}

inline auto bench_cosh() -> void
{
    auto xs = audio_bench::make_range(-5.f, 5.f, 100000);

    auto t_std = audio_bench::Lambda_timer("std::cosh process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::cosh(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Lambda_timer("vsl::cosh process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::cosh(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using enum audio_bench::Bench_units;

    const auto stats_std = t_std.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_std);

    const auto stats_vsl = t_vsl.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_vsl);

    audio_bench::compare_benchmarks(stats_std, stats_vsl);
}

inline auto bench_tanh() -> void
{
    auto xs = audio_bench::make_range(-5.f, 5.f, 100000);

    auto t_std = audio_bench::Lambda_timer("std::tanh process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::tanh(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Lambda_timer("vsl::tanh process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::tanh(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using enum audio_bench::Bench_units;

    const auto stats_std = t_std.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_std);

    const auto stats_vsl = t_vsl.run({.units = microseconds});
    audio_bench::print_bench_stats(stats_vsl);

    audio_bench::compare_benchmarks(stats_std, stats_vsl);
}