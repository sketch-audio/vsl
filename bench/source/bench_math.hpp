#pragma once

#include <cmath>
#include <format>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#include <vsl/vsl.hpp>
#include <audio_bench/audio_bench.hpp>

// Benchmarks every math function against std, for both scalar widths. Each
// function is run at an equal call count (N) across three variants: std scalar,
// vsl scalar, and vsl vector (broadcast). Equal call count measures per-call
// cost -- the realistic "called once per sample" case -- not best-case
// throughput, so the vector column shows what a SIMD call costs vs a std scalar
// call (the vector still does 4/2 lanes per call).
namespace bench_detail {

using audio_bench::Benchmark;
using audio_bench::Timer;
using audio_bench::do_not_optimize;
using Range = audio_bench::signals::Range;

inline constexpr auto N = size_t{100000};

inline auto us(double nanos) -> double { return nanos / 1000.0; }

inline auto report(const std::string& fn, const char* ty, const Benchmark::Stats& s, const Benchmark::Stats& sc, const Benchmark::Stats& ve) -> void
{
    std::cout << std::format(
        "{:<8} {:<3}  std {:8.1f}us | scalar {:8.1f}us {:5.2f}x | vector {:8.1f}us {:5.2f}x\n",
        fn, ty,
        us(s.mean),
        us(sc.mean), s.mean / sc.mean,
        us(ve.mean), s.mean / ve.mean);
}

template<typename S, typename VslOp, typename StdOp>
inline auto run_unary(const std::string& fn, S lo, S hi, VslOp vsl_op, StdOp std_op) -> void
{
    using V = vsl::vector_t<S>;
    const auto xs = Range::make<S>({lo, hi, N});
    auto xv = std::vector<V>{};
    xv.reserve(xs.size());
    for (auto x : xs) { xv.emplace_back(V{x}); }

    auto t_std = Timer("std", [&]() { for (auto x : xs)  { auto y = std_op(x); do_not_optimize(y); } });
    auto t_sc  = Timer("sc",  [&]() { for (auto x : xs)  { auto y = vsl_op(x); do_not_optimize(y); } });
    auto t_ve  = Timer("ve",  [&]() { for (auto& x : xv) { auto y = vsl_op(x); do_not_optimize(y); } });

    report(fn, sizeof(S) == 4 ? "f32" : "f64", t_std.run({}), t_sc.run({}), t_ve.run({}));
}

template<typename S, typename VslOp, typename StdOp>
inline auto run_binary(const std::string& fn, S alo, S ahi, S blo, S bhi, VslOp vsl_op, StdOp std_op) -> void
{
    using V = vsl::vector_t<S>;
    const auto as = Range::make<S>({alo, ahi, N});
    const auto bs = Range::make<S>({blo, bhi, N});
    auto av = std::vector<V>{};
    auto bv = std::vector<V>{};
    av.reserve(as.size());
    bv.reserve(bs.size());
    for (auto i = size_t{}; i < as.size(); ++i) { av.emplace_back(V{as[i]}); bv.emplace_back(V{bs[i]}); }

    auto t_std = Timer("std", [&]() { for (size_t i = 0; i < as.size(); ++i) { auto y = std_op(as[i], bs[i]); do_not_optimize(y); } });
    auto t_sc  = Timer("sc",  [&]() { for (size_t i = 0; i < as.size(); ++i) { auto y = vsl_op(as[i], bs[i]); do_not_optimize(y); } });
    auto t_ve  = Timer("ve",  [&]() { for (size_t i = 0; i < av.size(); ++i) { auto y = vsl_op(av[i], bv[i]); do_not_optimize(y); } });

    report(fn, sizeof(S) == 4 ? "f32" : "f64", t_std.run({}), t_sc.run({}), t_ve.run({}));
}

// Drive a unary op for both scalar widths.
template<typename VslOp, typename StdOp>
inline auto both_unary(const char* fn, double lo, double hi, VslOp vsl_op, StdOp std_op) -> void
{
    run_unary<float>(fn, static_cast<float>(lo), static_cast<float>(hi), vsl_op, std_op);
    run_unary<double>(fn, lo, hi, vsl_op, std_op);
}

template<typename VslOp, typename StdOp>
inline auto both_binary(const char* fn, double alo, double ahi, double blo, double bhi, VslOp vsl_op, StdOp std_op) -> void
{
    run_binary<float>(fn, static_cast<float>(alo), static_cast<float>(ahi), static_cast<float>(blo), static_cast<float>(bhi), vsl_op, std_op);
    run_binary<double>(fn, alo, ahi, blo, bhi, vsl_op, std_op);
}

} // namespace bench_detail

inline auto bench_rounding() -> void
{
    using namespace bench_detail;
    std::cout << "-- rounding --\n";
    both_unary("abs",   -10, 10, [](auto x) { return vsl::abs(x); },   [](auto x) { return std::abs(x); });
    both_unary("trunc", -10, 10, [](auto x) { return vsl::trunc(x); }, [](auto x) { return std::trunc(x); });
    both_unary("floor", -10, 10, [](auto x) { return vsl::floor(x); }, [](auto x) { return std::floor(x); });
    both_unary("ceil",  -10, 10, [](auto x) { return vsl::ceil(x); },  [](auto x) { return std::ceil(x); });
    both_unary("round", -10, 10, [](auto x) { return vsl::round(x); }, [](auto x) { return std::round(x); });
}

inline auto bench_roots() -> void
{
    using namespace bench_detail;
    std::cout << "-- roots --\n";
    both_unary("sqrt",  0, 1e4,    [](auto x) { return vsl::sqrt(x); },  [](auto x) { return std::sqrt(x); });
    both_unary("rsqrt", 1e-4, 1e4, [](auto x) { return vsl::rsqrt(x); }, [](auto x) { return decltype(x){1} / std::sqrt(x); });
}

inline auto bench_trig_no_wrap() -> void
{
    using namespace bench_detail;
    constexpr auto pi = std::numbers::pi_v<double>;
    std::cout << "-- trig (kernel) --\n";
    both_unary("cos", -pi, pi,        [](auto x) { return vsl::kernel::cos(x); }, [](auto x) { return std::cos(x); });
    both_unary("sin", -pi, pi,        [](auto x) { return vsl::kernel::sin(x); }, [](auto x) { return std::sin(x); });
    both_unary("tan", -1.4, 1.4,      [](auto x) { return vsl::kernel::tan(x); }, [](auto x) { return std::tan(x); });
    both_unary("cos_2pi", 0, 1, [](auto x) { return vsl::kernel::cos_2pi(x); }, [](auto x) { using T = decltype(x); return std::cos(T(2) * std::numbers::pi_v<T> * x); });
    both_unary("sin_2pi", 0, 1, [](auto x) { return vsl::kernel::sin_2pi(x); }, [](auto x) { using T = decltype(x); return std::sin(T(2) * std::numbers::pi_v<T> * x); });
    both_unary("tan_pi", -0.49, 0.49, [](auto x) { return vsl::kernel::tan_pi(x); }, [](auto x) { using T = decltype(x); return std::tan(std::numbers::pi_v<T> * x); });
}

inline auto bench_trig_wrap() -> void
{
    using namespace bench_detail;
    constexpr auto pi = std::numbers::pi_v<double>;
    std::cout << "-- trig (full range) --\n";
    both_unary("cos", -pi, pi,        [](auto x) { return vsl::cos(x); }, [](auto x) { return std::cos(x); });
    both_unary("sin", -pi, pi,        [](auto x) { return vsl::sin(x); }, [](auto x) { return std::sin(x); });
    both_unary("tan", -1.4, 1.4,      [](auto x) { return vsl::tan(x); }, [](auto x) { return std::tan(x); });
    both_unary("cos_2pi", 0, 1, [](auto x) { return vsl::cos_2pi(x); }, [](auto x) { using T = decltype(x); return std::cos(T(2) * std::numbers::pi_v<T> * x); });
    both_unary("sin_2pi", 0, 1, [](auto x) { return vsl::sin_2pi(x); }, [](auto x) { using T = decltype(x); return std::sin(T(2) * std::numbers::pi_v<T> * x); });
    both_unary("tan_pi", -0.49, 0.49, [](auto x) { return vsl::tan_pi(x); }, [](auto x) { using T = decltype(x); return std::tan(std::numbers::pi_v<T> * x); });
}

inline auto bench_invtrig() -> void
{
    using namespace bench_detail;
    std::cout << "-- inverse trig --\n";
    both_unary("acos", -1, 1,   [](auto x) { return vsl::acos(x); }, [](auto x) { return std::acos(x); });
    both_unary("asin", -1, 1,   [](auto x) { return vsl::asin(x); }, [](auto x) { return std::asin(x); });
    both_unary("atan", -10, 10, [](auto x) { return vsl::atan(x); }, [](auto x) { return std::atan(x); });
}

inline auto bench_explog() -> void
{
    using namespace bench_detail;
    std::cout << "-- exp / log --\n";
    both_unary("exp2",  -10, 10,  [](auto x) { return vsl::exp2(x); },  [](auto x) { return std::exp2(x); });
    both_unary("exp",   -10, 10,  [](auto x) { return vsl::exp(x); },   [](auto x) { return std::exp(x); });
    both_unary("log2",  1e-4, 1e4, [](auto x) { return vsl::log2(x); },  [](auto x) { return std::log2(x); });
    both_unary("log",   1e-4, 1e4, [](auto x) { return vsl::log(x); },   [](auto x) { return std::log(x); });
    both_unary("log10", 1e-4, 1e4, [](auto x) { return vsl::log10(x); }, [](auto x) { return std::log10(x); });
}

inline auto bench_power() -> void
{
    using namespace bench_detail;
    std::cout << "-- power --\n";
    both_binary("pow",  1e-4, 100, -10, 10, [](auto a, auto b) { return vsl::pow(a, b); },  [](auto a, auto b) { return std::pow(a, b); });
    both_binary("fmod", -100, 100, 0.1, 10, [](auto a, auto b) { return vsl::fmod(a, b); }, [](auto a, auto b) { return std::fmod(a, b); });
}

inline auto bench_hyperbolic() -> void
{
    using namespace bench_detail;
    std::cout << "-- hyperbolic --\n";
    both_unary("cosh", -5, 5,   [](auto x) { return vsl::cosh(x); }, [](auto x) { return std::cosh(x); });
    both_unary("sinh", -5, 5,   [](auto x) { return vsl::sinh(x); }, [](auto x) { return std::sinh(x); });
    both_unary("tanh", -10, 10, [](auto x) { return vsl::tanh(x); }, [](auto x) { return std::tanh(x); });
}

// Precision::Fast, to compare with the Standard rows above.
inline auto bench_fast() -> void
{
    using namespace bench_detail;
    using enum vsl::Precision;
    std::cout << "-- Precision::Fast --\n";
    both_unary("sin_2pi", -10, 10,     [](auto x) { return vsl::sin_2pi<Fast>(x); }, [](auto x) { using T = decltype(x); return std::sin(T(2) * std::numbers::pi_v<T> * x); });
    both_unary("tan_pi", -0.49, 0.49,  [](auto x) { return vsl::tan_pi<Fast>(x); }, [](auto x) { using T = decltype(x); return std::tan(std::numbers::pi_v<T> * x); });
    both_unary("exp2",  -10, 10,       [](auto x) { return vsl::exp2<Fast>(x); },  [](auto x) { return std::exp2(x); });
    both_unary("exp",   -10, 10,       [](auto x) { return vsl::exp<Fast>(x); },   [](auto x) { return std::exp(x); });
    both_unary("log2",  1e-4, 1e4,     [](auto x) { return vsl::log2<Fast>(x); },  [](auto x) { return std::log2(x); });
    both_unary("log",   1e-4, 1e4,     [](auto x) { return vsl::log<Fast>(x); },   [](auto x) { return std::log(x); });
    both_binary("pow",  1e-4, 100, -10, 10, [](auto a, auto b) { return vsl::pow<Fast>(a, b); }, [](auto a, auto b) { return std::pow(a, b); });
    both_unary("tanh",  -10, 10,       [](auto x) { return vsl::tanh<Fast>(x); },  [](auto x) { return std::tanh(x); });
}
