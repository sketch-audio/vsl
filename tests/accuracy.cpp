// Accuracy of every float32x4 math function over every float in its domain (or every stride-th),
// against double truth, with one bound per function and precision that every backend must meet.
//
//   accuracy            every 64th float (CI)
//   accuracy 1          every float (before a release; a few minutes on 4 threads)
//   accuracy 1 tanh     only rows whose name contains "tanh"
//
// Subnormal inputs are left out: VSL assumes flush-to-zero.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

#include <vsl/vsl.hpp>

namespace {

using V = vsl::float32x4;
using enum vsl::Precision;

auto ulp_f32(double t) -> double
{
    auto e = 0;
    std::frexp(std::abs(t), &e);
    return std::ldexp(1.0, std::max(e - 24, -149));
}

// sin(2 pi x) in double, exact at the zeros: reduce by quarter turns first.
auto sin_turns(double x) -> double
{
    const auto n = std::nearbyint(4 * x);
    const auto a = 2 * std::numbers::pi * (x - n / 4);
    switch (static_cast<long long>(n) & 3) {
        case 0: return std::sin(a);
        case 1: return std::cos(a);
        case 2: return -std::sin(a);
        default: return -std::cos(a);
    }
}

auto cos_turns(double x) -> double { return sin_turns(x + 0.25); }

enum class Metric { Ulp, Abs, Rel };

struct Stats {
    double ulp = 0, abs = 0, rel = 0;
    double ulp_at = 0, abs_at = 0, rel_at = 0;
    uint64_t nonfinite = 0;

    void add(double x, double y, double t)
    {
        if (!std::isfinite(t)) { return; }
        if (!std::isfinite(y)) { ++nonfinite; return; }
        const auto d = std::abs(y - t);
        if (const auto u = d / ulp_f32(t); u > ulp) { ulp = u; ulp_at = x; }
        if (d > abs) { abs = d; abs_at = x; }
        if (t != 0) {
            if (const auto r = d / std::abs(t); r > rel) { rel = r; rel_at = x; }
        }
    }

    void merge(const Stats& o)
    {
        if (o.ulp > ulp) { ulp = o.ulp; ulp_at = o.ulp_at; }
        if (o.abs > abs) { abs = o.abs; abs_at = o.abs_at; }
        if (o.rel > rel) { rel = o.rel; rel_at = o.rel_at; }
        nonfinite += o.nonfinite;
    }
};

// Every normal float in [lo, hi], every stride-th bit pattern, four lanes per call, four threads.
template<typename F, typename T>
auto sweep(float lo, float hi, uint32_t stride, F f, T truth) -> Stats
{
    struct Segment { uint32_t a, b; bool neg; };
    auto segs = std::vector<Segment>{};
    constexpr auto tiny = std::numeric_limits<float>::min();
    if (lo < 0) {
        segs.push_back({std::bit_cast<uint32_t>(hi < 0 ? std::max(-hi, tiny) : tiny), std::bit_cast<uint32_t>(-lo), true});
    }
    if (hi > 0) {
        segs.push_back({std::bit_cast<uint32_t>(lo > 0 ? std::max(lo, tiny) : tiny), std::bit_cast<uint32_t>(hi), false});
    }
    if (lo <= 0 && hi >= 0) { segs.push_back({0, 0, false}); } // +0

    constexpr auto num_threads = 4u;
    auto per = std::vector<Stats>(num_threads);
    auto threads = std::vector<std::thread>{};
    for (auto t = 0u; t < num_threads; ++t) {
        threads.emplace_back([&, t] {
            auto& s = per[t];
            for (const auto& seg : segs) {
                const auto count = (uint64_t{seg.b} - seg.a) / stride + 1;
                const auto begin = count * t / num_threads;
                const auto end = count * (t + 1) / num_threads;
                auto buf = std::array<float, 4>{};
                auto k = size_t{};
                auto flush = [&](size_t m) {
                    const auto y = f(V{buf[0], buf[1], buf[2], buf[3]});
                    for (auto i = size_t{}; i < m; ++i) { s.add(buf[i], y[i], truth(double{buf[i]})); }
                };
                for (auto i = begin; i < end; ++i) {
                    const auto x = std::bit_cast<float>(static_cast<uint32_t>(seg.a + i * stride));
                    buf[k++] = seg.neg ? -x : x;
                    if (k == 4) { flush(4); k = 0; }
                }
                if (k > 0) {
                    for (auto i = k; i < buf.size(); ++i) { buf[i] = buf[0]; }
                    flush(k);
                }
            }
        });
    }
    for (auto& th : threads) { th.join(); }
    auto total = Stats{};
    for (const auto& s : per) { total.merge(s); }
    return total;
}

struct Row {
    std::string name;
    float lo, hi;
    Metric metric;
    double bound;
    V (*f)(V);
    double (*truth)(double);
};

} // namespace

auto main(int argc, char** argv) -> int
{
    const auto stride = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 64u;
    const auto only = std::string{argc > 2 ? argv[2] : ""};
    constexpr auto pi = std::numbers::pi_v<float>;
    constexpr auto big = std::numeric_limits<float>::max();
    constexpr auto below_half = 0.49999997f;

    // Bounds: the worst of NEON and SSE4.2 (unfused), with a little headroom.
    const auto rows = std::vector<Row>{
        {"sin [-100, 100]", -100, 100, Metric::Ulp, 5, [](V x) { return vsl::sin(x); }, [](double x) { return std::sin(x); }},
        {"cos [-100, 100]", -100, 100, Metric::Ulp, 5.5, [](V x) { return vsl::cos(x); }, [](double x) { return std::cos(x); }},
        {"tan [-100, 100]", -100, 100, Metric::Ulp, 9.5, [](V x) { return vsl::tan(x); }, [](double x) { return std::tan(x); }},
        {"sin [100, 1e4] (abs)", 100, 1e4f, Metric::Abs, 4e-7, [](V x) { return vsl::sin(x); }, [](double x) { return std::sin(x); }},
        {"kernel::sin [-pi, pi]", -pi, pi, Metric::Ulp, 5, [](V x) { return vsl::kernel::sin(x); }, [](double x) { return std::sin(x); }},
        {"kernel::cos [-pi, pi]", -pi, pi, Metric::Ulp, 7.5, [](V x) { return vsl::kernel::cos(x); }, [](double x) { return std::cos(x); }},
        {"kernel::tan [-pi/2, pi/2]", -pi / 2, pi / 2, Metric::Ulp, 6, [](V x) { return vsl::kernel::tan(x); }, [](double x) { return std::tan(x); }},
        {"sin_2pi [-10, 10]", -10, 10, Metric::Ulp, 5, [](V x) { return vsl::sin_2pi(x); }, sin_turns},
        {"sin_2pi<Fast> [-10, 10]", -10, 10, Metric::Ulp, 5, [](V x) { return vsl::sin_2pi<Fast>(x); }, sin_turns},
        {"cos_2pi [-10, 10]", -10, 10, Metric::Ulp, 7.5, [](V x) { return vsl::cos_2pi(x); }, cos_turns},
        {"tan_pi (-1/2, 1/2)", -below_half, below_half, Metric::Ulp, 5, [](V x) { return vsl::tan_pi(x); }, [](double x) { return std::tan(std::numbers::pi * x); }},
        {"tan_pi<Fast> [0, 20 kHz / 48 kHz] (rel)", 0, 20000.f / 48000.f, Metric::Rel, 1e-4, [](V x) { return vsl::tan_pi<Fast>(x); }, [](double x) { return std::tan(std::numbers::pi * x); }},
        {"kernel::sin_2pi [0, 1]", 0, 1, Metric::Ulp, 5, [](V x) { return vsl::kernel::sin_2pi(x); }, sin_turns},
        {"kernel::sin_2pi<Fast> [0, 1]", 0, 1, Metric::Ulp, 5, [](V x) { return vsl::kernel::sin_2pi<Fast>(x); }, sin_turns},
        {"kernel::cos_2pi [0, 1]", 0, 1, Metric::Ulp, 7.5, [](V x) { return vsl::kernel::cos_2pi(x); }, cos_turns},
        {"asin [-1, 1]", -1, 1, Metric::Ulp, 5.5, [](V x) { return vsl::asin(x); }, [](double x) { return std::asin(x); }},
        {"acos [-1, 1]", -1, 1, Metric::Ulp, 5, [](V x) { return vsl::acos(x); }, [](double x) { return std::acos(x); }},
        {"atan, every float", -big, big, Metric::Ulp, 3.5, [](V x) { return vsl::atan(x); }, [](double x) { return std::atan(x); }},
        {"exp2 [-126, 128)", -126, 127.99999f, Metric::Ulp, 2, [](V x) { return vsl::exp2(x); }, [](double x) { return std::exp2(x); }},
        {"exp2<Fast> [-126, 128)", -126, 127.99999f, Metric::Ulp, 3, [](V x) { return vsl::exp2<Fast>(x); }, [](double x) { return std::exp2(x); }},
        {"exp [-87.3, 88.7]", -87.3f, 88.7f, Metric::Ulp, 2.5, [](V x) { return vsl::exp(x); }, [](double x) { return std::exp(x); }},
        {"exp<Fast> [-87.3, 88.7]", -87.3f, 88.7f, Metric::Ulp, 3.5, [](V x) { return vsl::exp<Fast>(x); }, [](double x) { return std::exp(x); }},
        {"expm1 [-17, 88.7]", -17, 88.7f, Metric::Ulp, 3.5, [](V x) { return vsl::expm1(x); }, [](double x) { return std::expm1(x); }},
        {"log2, every normal x > 0", 0, big, Metric::Ulp, 3.5, [](V x) { return vsl::log2(x); }, [](double x) { return x == 0 ? -INFINITY : std::log2(x); }},
        {"log2<Fast>, every normal x > 0", 0, big, Metric::Ulp, 22, [](V x) { return vsl::log2<Fast>(x); }, [](double x) { return x == 0 ? -INFINITY : std::log2(x); }},
        {"log, every normal x > 0", 0, big, Metric::Ulp, 3, [](V x) { return vsl::log(x); }, [](double x) { return x == 0 ? -INFINITY : std::log(x); }},
        {"log10, every normal x > 0", 0, big, Metric::Ulp, 4.5, [](V x) { return vsl::log10(x); }, [](double x) { return x == 0 ? -INFINITY : std::log10(x); }},
        {"db_to_gain [-200, 200] (rel)", -200, 200, Metric::Rel, 2e-6, [](V x) { return vsl::db_to_gain(x); }, [](double x) { return std::pow(10.0, x / 20); }},
        {"db_to_gain<Fast> [-200, 200] (rel)", -200, 200, Metric::Rel, 2e-6, [](V x) { return vsl::db_to_gain<Fast>(x); }, [](double x) { return std::pow(10.0, x / 20); }},
        {"gain_to_db [1e-10, 1e10] (abs dB)", 1e-10f, 1e10f, Metric::Abs, 3e-5, [](V x) { return vsl::gain_to_db(x); }, [](double x) { return 20 * std::log10(x); }},
        {"gain_to_db<Fast> [1e-10, 1e10] (abs dB)", 1e-10f, 1e10f, Metric::Abs, 3e-5, [](V x) { return vsl::gain_to_db<Fast>(x); }, [](double x) { return 20 * std::log10(x); }},
        {"sinh [-10, 10]", -10, 10, Metric::Ulp, 3, [](V x) { return vsl::sinh(x); }, [](double x) { return std::sinh(x); }},
        {"cosh [-10, 10]", -10, 10, Metric::Ulp, 2.5, [](V x) { return vsl::cosh(x); }, [](double x) { return std::cosh(x); }},
        {"tanh, every float", -big, big, Metric::Ulp, 17, [](V x) { return vsl::tanh(x); }, [](double x) { return std::tanh(x); }},
        {"tanh<Fast>, every float (abs)", -big, big, Metric::Abs, 8e-5, [](V x) { return vsl::tanh<Fast>(x); }, [](double x) { return std::tanh(x); }},
    };

    std::printf("stride %u%s\n", stride, stride == 1 ? " (every float)" : "");
    auto failures = 0;
    for (const auto& row : rows) {
        if (!only.empty() && row.name.find(only) == std::string::npos) { continue; }
        const auto s = sweep(row.lo, row.hi, stride, row.f, row.truth);
        const auto [value, at] = row.metric == Metric::Ulp ? std::pair{s.ulp, s.ulp_at}
                               : row.metric == Metric::Abs ? std::pair{s.abs, s.abs_at}
                                                           : std::pair{s.rel, s.rel_at};
        const auto unit = row.metric == Metric::Ulp ? "ULP" : row.metric == Metric::Abs ? "abs" : "rel";
        const auto ok = value <= row.bound && s.nonfinite == 0;
        failures += ok ? 0 : 1;
        std::printf("%-40s %s %-9.4g %s (bound %g) at %-14.8g | max abs %-9.3g max rel %-9.3g%s\n",
                    row.name.c_str(), ok ? "  " : "!!", value, unit, row.bound, at, s.abs, s.rel,
                    s.nonfinite ? "  NONFINITE" : "");
        std::fflush(stdout);
    }
    std::printf(failures ? "%d rows over their bound.\n" : "All rows within their bounds.\n", failures);
    return failures ? 1 : 0;
}
