#pragma once

#include <cmath>
#include <numbers>

#include <vsl/vsl.h>
#include <audio_bench/audio_bench.hpp>

// MARK: - trig

struct Bench {
    // Should we bench the vector implementation? (Not used atm.)
    static constexpr auto run_vector = false; 
    using S = float;
    using X = vsl::vector_t<Bench::S>;
    static auto to_vec(const std::vector<Bench::S>& xs) -> std::vector<X>
    {
        auto v = std::vector<X>{};
        v.reserve(xs.size());
        for (auto x : xs) {
            v.push_back(X(x));
        }
        return v;
    }
};

inline auto bench_cos() -> void
{
    using Range = audio_bench::signals::Range;
    constexpr auto pi = std::numbers::pi_v<Bench::S>;
    auto xs = Range::make<Bench::S>({-pi, pi, 100000});

    auto t_std = audio_bench::Timer("std::cos process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::cos(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::cos (no wrap) process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::cos<false>(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

inline auto bench_sin() -> void
{
    using Range = audio_bench::signals::Range;
    constexpr auto pi = std::numbers::pi_v<Bench::S>;
    auto xs = Range::make<Bench::S>({-pi, pi, 100000});

    auto t_std = audio_bench::Timer("std::sin process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::sin(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::sin (no wrap) process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::sin<false>(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

inline auto bench_tan() -> void
{
    using Range = audio_bench::signals::Range;
    constexpr auto pi_half = std::numbers::pi_v<Bench::S> / 2;
    auto xs = Range::make<Bench::S>({-pi_half, pi_half, 100000});

    auto t_std = audio_bench::Timer("std::tan process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::tan(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::tan (no wrap) process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::tan<false>(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

inline auto bench_cos_2pi() -> void
{
    using Range = audio_bench::signals::Range;
    auto xs = Range::make<Bench::S>({0.f, 1.f, 100000});

    auto t_std = audio_bench::Timer("std::cos(2pi*x) process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::cos(2 * std::numbers::pi_v<Bench::S> * x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::cos_2pi (no wrap) process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::cos_2pi<false>(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

inline auto bench_sin_2pi() -> void
{
    using Range = audio_bench::signals::Range;
    auto xs = Range::make<Bench::S>({0.f, 1.f, 100000});

    auto t_std = audio_bench::Timer("std::sin(2pi*x) process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::sin(2 * std::numbers::pi_v<Bench::S> * x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::sin_2pi (no wrap) process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::sin_2pi<false>(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

inline auto bench_acos() -> void
{
    using Range = audio_bench::signals::Range;
    auto xs = Range::make<Bench::S>({-1.f, 1.f, 100000});

    auto t_std = audio_bench::Timer("std::acos process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::acos(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::acos process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::acos(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

inline auto bench_asin()
{
    using Range = audio_bench::signals::Range;
    auto xs = Range::make<Bench::S>({-1.f, 1.f, 100000});

    auto t_std = audio_bench::Timer("std::asin process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::asin(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::asin process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::asin(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

inline auto bench_atan() -> void
{
    using Range = audio_bench::signals::Range;
    auto xs = Range::make<Bench::S>({-100.f, 100.f, 100000});

    auto t_std = audio_bench::Timer("std::atan process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::atan(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::atan process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::atan(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

// MARK: - exp2/log2

inline auto bench_exp2() -> void
{
    using Range = audio_bench::signals::Range;
    auto xs = Range::make<Bench::S>({-10.f, 10.f, 100000});

    auto t_std = audio_bench::Timer("std::exp2 process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::exp2(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::exp2 process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::exp2(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

inline auto bench_log2() -> void
{
    using Range = audio_bench::signals::Range;
    auto xs = Range::make<Bench::S>({0.0001f, 10000.f, 100000});

    auto t_std = audio_bench::Timer("std::log2 process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::log2(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::log2 process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::log2(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

// MARK: - exp

inline auto bench_exp() -> void
{
    using Range = audio_bench::signals::Range;
    auto xs = Range::make<Bench::S>({-10.f, 10.f, 100000});

    auto t_std = audio_bench::Timer("std::exp process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::exp(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::exp process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::exp(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

// MARK: - pow

inline auto bench_pow() -> void
{
    using Range = audio_bench::signals::Range;
    auto xs = Range::make<Bench::S>({0.0001f, 100.f, 1000});
    auto ys = Range::make<Bench::S>({-10.f, 10.f, 100});

    std::vector<std::pair<float, float>> pairs;
    pairs.reserve(xs.size() * ys.size());
    for (auto x : xs) {
        for (auto y : ys) {
            pairs.emplace_back(x, y);
        }
    }

    auto t_std = audio_bench::Timer("std::pow process 100000 float pairs", [&]() {
        for (const auto& [x, y] : pairs) {
            auto z = std::pow(x, y);
            audio_bench::do_not_optimize(z);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::pow process 100000 float pairs", [&]() {
        for (const auto& [x, y] : pairs) {
            auto z = vsl::pow(x, y);
            audio_bench::do_not_optimize(z);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

inline auto bench_sqrt() -> void
{
    using Range = audio_bench::signals::Range;
    auto xs = Range::make<Bench::S>({0.0001f, 10000.f, 100000});
    auto xs_vec = Bench::to_vec(xs);

    auto t_std = audio_bench::Timer("std::sqrt process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::sqrt(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::sqrt process 100000 floats", [&]() {
        for (auto x : xs_vec) {
            auto y = vsl::sqrt(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

// MARK: - sinh/cosh/tanh

inline auto bench_sinh() -> void
{
    using Range = audio_bench::signals::Range;
    auto xs = Range::make<Bench::S>({-5.f, 5.f, 100000});

    auto t_std = audio_bench::Timer("std::sinh process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::sinh(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::sinh process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::sinh(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

inline auto bench_cosh() -> void
{
    using Range = audio_bench::signals::Range;
    auto xs = Range::make<Bench::S>({-5.f, 5.f, 100000});

    auto t_std = audio_bench::Timer("std::cosh process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::cosh(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::cosh process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::cosh(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}

inline auto bench_tanh() -> void
{
    using Range = audio_bench::signals::Range;
    auto xs = Range::make<Bench::S>({-5.f, 5.f, 100000});

    auto t_std = audio_bench::Timer("std::tanh process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = std::tanh(x);
            audio_bench::do_not_optimize(y);
        }
    });

    auto t_vsl = audio_bench::Timer("vsl::tanh process 100000 floats", [&]() {
        for (auto x : xs) {
            auto y = vsl::tanh(x);
            audio_bench::do_not_optimize(y);
        }
    });

    using Benchmark = audio_bench::Benchmark;

    const auto stats_std = t_std.run({});
    const auto stats_vsl = t_vsl.run({});

    Benchmark::print(stats_std, Benchmark::Units::Microseconds);
    Benchmark::print(stats_vsl, Benchmark::Units::Microseconds);
    Benchmark::compare(stats_std, stats_vsl);
}