#pragma once

#include <vsl/vsl.h>
#include <audio_bench/audio_bench.h>

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

inline auto bench_pow() -> void
{
    auto xs = audio_bench::make_range(0.0001f, 100.f, 100);
    auto ys = audio_bench::make_range(-10.f, 10.f, 100);

    std::vector<std::pair<float, float>> pairs;
    pairs.reserve(xs.size() * ys.size());
    for (auto x : xs) {
        for (auto y : ys) {
            pairs.emplace_back(x, y);
        }
    }

    auto t_std = audio_bench::Lambda_timer("std::pow process 10000 float pairs", [&]() {
        for (const auto& [x, y] : pairs) {
            auto z = std::pow(x, y);
            audio_bench::do_not_optimize(z);
        }
    });

    auto t_vsl = audio_bench::Lambda_timer("vsl::pow process 10000 float pairs", [&]() {
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