#include <array>
#include <cmath>
#include <format>

#include <vsl/vsl.hpp>
#include <audio_bench/audio_bench.hpp>

namespace {

template<typename T>
using Tracker = audio_bench::Error_tracker<T>; // Dumb can't do this in function scope.

// Under -ffast-math on x86, clang still swaps division and 1 / sqrt for reciprocal estimates plus
// Newton steps, which VSL can't opt out of without making std::fma an errno-setting call. Worth ~1 ULP.
#if defined(__FAST_MATH__) && VSL_SSE4
constexpr auto fast_math_estimates = true;
constexpr auto slack = 2;
#else
constexpr auto fast_math_estimates = false;
constexpr auto slack = 1;
#endif

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
        auto xs = Range::make<float>({-100, 100, 1000000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::cos(vsl::float32x4{x});
            const auto s = std::cos(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{4e-7f}, std::format("Fail cos({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{4e-7f}, std::format("Fail cos({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{5}, std::format("Fail cos({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();
    });

    Tests::add("Test vsl::sin.", []() {
        audio_bench::log("Starting Test vsl::sin.");
        constexpr auto pi = std::numbers::pi_v<float>;
        auto xs = Range::make<float>({-100, 100, 1000000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::sin(vsl::float32x4{x});
            const auto s = std::sin(x);
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{3e-7f}, std::format("Fail sin({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{4e-7f}, std::format("Fail sin({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{5}, std::format("Fail sin({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();

        audio_bench::log("Testing kernel::sin on [-pi, pi]");
        xs = Range::make<float>({-pi, pi, 100000});
        for (auto x : xs) {
            const auto v = vsl::kernel::sin(vsl::float32x4{x});
            audio_bench::expect_close(v[0], std::sin(x), Error::Ulp_distance{5}, std::format("Fail kernel::sin({})", x));
        }

        audio_bench::log("Testing tiny x");
        xs = Range::make<float>({-1e-6f, 1e-6f, 100000});
        for (auto x : xs) {
            const auto v = vsl::sin(vsl::float32x4{x});
            audio_bench::expect_close(v[0], std::sin(x), Error::Ulp_distance{3}, std::format("Fail sin({})", x));
        }
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
            audio_bench::expect_close(v[0], s, Error::Absolute{2e-6f}, std::format("Fail tan({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{4e-7f}, std::format("Fail tan({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{5}, std::format("Fail tan({})", x));
            const auto k = vsl::kernel::tan(vsl::float32x4{x});
            audio_bench::expect_close(k[0], s, Error::Ulp_distance{6}, std::format("Fail kernel::tan({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();

        audio_bench::log("Testing range -100...100, poles included");
        xs = Range::make<float>({-100, 100, 1000000});
        for (auto x : xs) {
            const auto v = vsl::tan(vsl::float32x4{x});
            audio_bench::expect_close(v[0], std::tan(x), Error::Ulp_distance{9}, std::format("Fail tan({})", x));
        }
    });

    Tests::add("Test vsl::cos_2pi.", []() {
        audio_bench::log("Starting Test vsl::cos_2pi.");
        auto xs = Range::make<float>({-10, 10, 1000000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::cos_2pi(vsl::float32x4{x});
            const auto s = static_cast<float>(cos_turns(x));
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
        auto xs = Range::make<float>({-10, 10, 1000000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto v = vsl::sin_2pi(vsl::float32x4{x});
            const auto s = static_cast<float>(sin_turns(x));
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{2e-7f}, std::format("Fail sin_2pi({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{3e-7f}, std::format("Fail sin_2pi({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{5}, std::format("Fail sin_2pi({})", x));
            errors.update_with(x, v[0], s);

            const auto [sc_s, sc_c] = vsl::sin_cos_2pi(vsl::float32x4{x});
            audio_bench::expect_true(sc_s[0] == v[0] && sc_c[0] == vsl::cos_2pi(vsl::float32x4{x})[0], std::format("Fail sin_cos_2pi({})", x));
        }
        errors.report();

        audio_bench::log("Testing kernel::sin_2pi and kernel::cos_2pi on [0, 1]");
        xs = Range::make<float>({0, 1, 100000});
        for (auto x : xs) {
            const auto s = vsl::kernel::sin_2pi(vsl::float32x4{x});
            const auto c = vsl::kernel::cos_2pi(vsl::float32x4{x});
            audio_bench::expect_close(s[0], static_cast<float>(sin_turns(x)), Error::Ulp_distance{5}, std::format("Fail kernel::sin_2pi({})", x));
            audio_bench::expect_close(c[0], static_cast<float>(cos_turns(x)), Error::Ulp_distance{8}, std::format("Fail kernel::cos_2pi({})", x));
        }
    });

    Tests::add("Test vsl::tan_pi.", []() {
        audio_bench::log("Starting Test vsl::tan_pi.");
        // tan_pi(f/fs) = tan(pi * f/fs), the filter prewarp; 20000/44100 = ~0.4535f.
        auto xs = Range::make<float>({-0.4535f, 0.4535f, 100000});

        auto errors = Tracker<float>{};
        for (auto x : xs) {
            const auto shifted = x + 3.f; // a later period
            const auto v = vsl::tan_pi(vsl::float32x4{shifted});
            const auto s = static_cast<float>(std::tan(std::numbers::pi_v<double> * (static_cast<double>(shifted) - 3.0)));
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], s, Error::Absolute{2e-6f}, std::format("Fail tan_pi({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{4e-7f}, std::format("Fail tan_pi({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{5}, std::format("Fail tan_pi({})", x));
            errors.update_with(x, v[0], s);
        }
        errors.report();

        audio_bench::log("Testing kernel::tan_pi");
        xs = Range::make<float>({-0.4535f, 0.4535f, 100000});
        for (auto x : xs) {
            const auto v = vsl::kernel::tan_pi(vsl::float32x4{x});
            const auto s = static_cast<float>(std::tan(std::numbers::pi_v<double> * x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{5}, std::format("Fail kernel::tan_pi({})", x));
        }

        audio_bench::log("Testing kernel::tan_pi<Fast> on 0...20 kHz at 48 kHz");
        xs = Range::make<float>({0, 20000.f / 48000.f, 100000});
        for (auto x : xs) {
            const auto v = vsl::kernel::tan_pi<vsl::Precision::Fast>(vsl::float32x4{x});
            const auto s = static_cast<float>(std::tan(std::numbers::pi_v<double> * x));
            audio_bench::expect_close(v[0], s, Error::Relative{1e-4f}, std::format("Fail kernel::tan_pi<Fast>({})", x));
        }
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
            audio_bench::expect_close(v[0], s, Error::Absolute{3e-7f * slack}, std::format("Fail asin({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{6e-7f * slack}, std::format("Fail asin({})", x));
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

    Tests::add("Test vsl::expm1.", []() {
        audio_bench::log("Starting Test vsl::expm1.");
        auto xs = Range::make<float>({-18, 85, 1000000});
        for (auto x : xs) {
            const auto v = vsl::expm1(vsl::float32x4{x});
            audio_bench::expect_true(vsl::all_same(v));
            audio_bench::expect_close(v[0], std::expm1(x), Error::Ulp_distance{4}, std::format("Fail expm1({})", x));
            audio_bench::expect_true(vsl::expm1(x) == v[0], std::format("Fail expm1({}) scalar != vector", x));
        }

        audio_bench::log("Testing small x: one-pole coefficients at low cutoffs");
        xs = Range::make<float>({-1e-3f, 1e-3f, 100000});
        for (auto x : xs) {
            const auto v = vsl::expm1(vsl::float32x4{x});
            audio_bench::expect_close(v[0], std::expm1(x), Error::Ulp_distance{3}, std::format("Fail expm1({})", x));
        }
    });

    Tests::add("Test vsl::db_to_gain and vsl::gain_to_db.", []() {
        audio_bench::log("Starting Test vsl::db_to_gain and vsl::gain_to_db.");
        auto dbs = Range::make<float>({-200, 200, 1000000});
        for (auto db : dbs) {
            const auto g = vsl::db_to_gain(vsl::float32x4{db});
            const auto s = static_cast<float>(std::pow(10.0, db / 20.0));
            audio_bench::expect_close(g[0], s, Error::Relative{2e-6f}, std::format("Fail db_to_gain({})", db));
        }
        auto gains = Range::make<float>({1e-10f, 1e10f, 1000000});
        for (auto g : gains) {
            const auto db = vsl::gain_to_db(vsl::float32x4{g});
            const auto s = static_cast<float>(20 * std::log10(static_cast<double>(g)));
            audio_bench::expect_close(db[0], s, Error::Absolute{3e-5f}, std::format("Fail gain_to_db({})", g));
        }
        audio_bench::expect_true(vsl::db_to_gain(vsl::float32x4{0.f})[0] == 1.f, "Fail db_to_gain(0) != 1");
        audio_bench::expect_true(vsl::gain_to_db(vsl::float32x4{1.f})[0] == 0.f, "Fail gain_to_db(1) != 0");
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

    Tests::add("Test vsl::pow<Fast>.", []() {
        audio_bench::log("Starting Test vsl::pow<Fast>.");

        auto run = [](const char* label, float base_lo, float base_hi, float exp_lo, float exp_hi) {
            audio_bench::log(label);
            constexpr auto n = size_t{100000};
            auto bases = Range::make<float>({base_lo, base_hi, n});
            auto exps = Range::make<float>({exp_lo, exp_hi, n});
            auto errors = Tracker<float>{};
            for (auto i = size_t{}; i < n; ++i) {
                const auto b = bases[i];
                const auto e = exps[i];
                const auto v = vsl::pow<vsl::Precision::Fast>(vsl::float32x4{b}, vsl::float32x4{e});
                const auto s = static_cast<float>(std::pow(double{b}, double{e}));
                audio_bench::expect_close(v[0], s, Error::Relative{1e-5f}, std::format("Fail pow<Fast>({}, {})", b, e));
                errors.update_with(e, v[0], s);
            }
            errors.report();
        };

        constexpr auto e = std::numbers::e_v<float>;
        run("base 10, exp -10..10", 10, 10, -10, 10);
        run("base e, exp -20..20", e, e, -20, 20);
        run("x^2.5, base 0..100", 0.001f, 100, 2.5f, 2.5f);
        run("base 0.1..0.95, exp -10", 0.1f, 0.95f, -10, -10);
        run("base 1.05..9, exp 10", 1.05f, 9, 10, 10);
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
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{2 * slack}, std::format("Fail cosh({})", x));
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
            audio_bench::expect_close(v[0], s, Error::Relative{3e-7f}, std::format("Fail sinh({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{3}, std::format("Fail sinh({})", x));
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
            audio_bench::expect_close(v[0], s, Error::Absolute{1e-6f * slack}, std::format("Fail tanh({})", x));
            audio_bench::expect_close(v[0], s, Error::Relative{2e-6f}, std::format("Fail tanh({})", x));
            audio_bench::expect_close(v[0], s, Error::Ulp_distance{17}, std::format("Fail tanh({})", x));
            errors.update_with(x, v[0], s);

            const auto f = vsl::tanh<vsl::Precision::Fast>(vsl::float32x4{x});
            audio_bench::expect_close(f[0], s, Error::Absolute{8e-5f}, std::format("Fail tanh<Fast>({})", x));
        }
        errors.report();
    });

    // MARK: - edges

    Tests::add("Test vsl edge values.", []() {
        // Under VSL_FINITE_MATH (-ffast-math), infinities and NaN are UB: overflow saturates instead.
#if VSL_FINITE_MATH
        constexpr auto huge = std::numeric_limits<float>::max();
        constexpr auto log2_zero = -150.f;
#else
        constexpr auto huge = std::numeric_limits<float>::infinity();
        constexpr auto log2_zero = -huge;
#endif

        // Scalar and vector must agree with each other and with the expected value.
        auto expect = [](const char* name, auto f, float x, float expected) {
            const auto s = f(x);
            const auto v = f(vsl::float32x4{x});
            audio_bench::expect_true(vsl::all_same(v), std::format("Fail {}({}) lanes", name, x));
            audio_bench::expect_true(s == expected, std::format("Fail {}({}) scalar: got {}, want {}", name, x, s, expected));
            audio_bench::expect_true(v[0] == expected, std::format("Fail {}({}) vector: got {}, want {}", name, x, v[0], expected));
        };
        // Saturated overflow is FLT_MAX give or take a little under VSL_FINITE_MATH.
        auto expect_overflow = [](const char* name, auto f, float x) {
            const auto s = f(x);
            const auto v = f(vsl::float32x4{x});
            audio_bench::expect_true(vsl::all_same(v) && s == v[0], std::format("Fail {}({}) scalar != vector", name, x));
#if VSL_FINITE_MATH
            audio_bench::expect_true(s > 3.4e38f, std::format("Fail {}({}): got {}, want ~FLT_MAX", name, x, s));
#else
            audio_bench::expect_true(s == std::numeric_limits<float>::infinity(), std::format("Fail {}({}): got {}, want inf", name, x, s));
#endif
        };

        const auto exp2 = [](auto x) { return vsl::exp2(x); };
        for (auto x : {128.f, 200.f, 256.f, 1e30f}) { expect_overflow("exp2", exp2, x); }
        for (auto x : {-127.f, -200.f, -1e30f}) { expect("exp2", exp2, x, 0.f); }

        const auto exp = [](auto x) { return vsl::exp(x); };
        for (auto x : {88.8f, 100.f, 178.f, 1e30f}) { expect_overflow("exp", exp, x); }
        for (auto x : {-88.f, -100.f, -1e30f}) { expect("exp", exp, x, 0.f); }

        expect("log2", [](auto x) { return vsl::log2(x); }, 0.f, log2_zero);
        expect("log", [](auto x) { return vsl::log(x); }, 0.f, std::numbers::ln2_v<float> * log2_zero);
        expect("log10", [](auto x) { return vsl::log10(x); }, 0.f, std::numbers::ln2_v<float> / std::numbers::ln10_v<float> * log2_zero);
#if VSL_FINITE_MATH
        audio_bench::expect_true(vsl::log2(1e-40f) < -149.f, "Fail log2(subnormal)"); // -150 + u P(u)
#else
        expect("log2", [](auto x) { return vsl::log2(x); }, 1e-40f, log2_zero); // subnormal reads as 0
#endif
        expect("gain_to_db", [](auto x) { return vsl::gain_to_db(x); }, 0.f, 6.0205999132796239f * log2_zero);

        const auto expm1 = [](auto x) { return vsl::expm1(x); };
        for (auto x : {88.8f, 100.f, 1e30f}) { expect_overflow("expm1", expm1, x); }
        for (auto x : {-20.f, -100.f, -1e30f}) { expect("expm1", expm1, x, -1.f); }
        for (auto x : {1e-20f, -1e-20f, 1e-30f}) {
            audio_bench::expect_close(vsl::expm1(x), x, Error::Ulp_distance{1}, std::format("Fail expm1({})", x));
        }
        expect("db_to_gain", [](auto x) { return vsl::db_to_gain(x); }, -1e30f, 0.f);

        const auto tanh = [](auto x) { return vsl::tanh(x); };
        if constexpr (!fast_math_estimates) { // saturating at exactly 1 depends on sqrt rounding correctly
            for (auto x : {10.f, 600.f, 1e5f, 1e30f}) { expect("tanh", tanh, x, 1.f); expect("tanh", tanh, -x, -1.f); }
        }

        const auto sinh = [](auto x) { return vsl::sinh(x); };
        for (auto x : {1e-7f, -1e-7f, 1e-20f}) { expect("sinh", sinh, x, x); }

        using vsl::float32x4;
        const auto pow0 = [](float y) {
            const auto s = vsl::pow(0.f, y);
            const auto v = vsl::pow(float32x4{0.f}, float32x4{y});
            audio_bench::expect_true(s == v[0], std::format("Fail pow(0, {}) scalar != vector", y));
            return s;
        };
        audio_bench::expect_true(pow0(0.f) == 1.f, "Fail pow(0, 0)");
        audio_bench::expect_true(pow0(-1.f) == huge, "Fail pow(0, -1)");
        audio_bench::expect_true(pow0(2.f) == 0.f, "Fail pow(0, 2)");

        // Both tiers give the same results at 0, the clamps and overflow.
        const auto tier = [&]<vsl::Precision P>(std::integral_constant<vsl::Precision, P>) {
            const auto exp2 = [](auto x) { return vsl::exp2<P>(x); };
            for (auto x : {128.f, 1e30f}) { expect_overflow("exp2", exp2, x); }
            for (auto x : {-127.f, -1e30f}) { expect("exp2", exp2, x, 0.f); }
            expect("exp2", exp2, 0.f, 1.f);
            expect("exp", [](auto x) { return vsl::exp<P>(x); }, 0.f, 1.f);
            expect_overflow("exp", [](auto x) { return vsl::exp<P>(x); }, 88.8f);
            expect("db_to_gain", [](auto x) { return vsl::db_to_gain<P>(x); }, 0.f, 1.f);
            expect("log2", [](auto x) { return vsl::log2<P>(x); }, 1.f, 0.f);
            expect("log2", [](auto x) { return vsl::log2<P>(x); }, 0.f, log2_zero);
            expect("gain_to_db", [](auto x) { return vsl::gain_to_db<P>(x); }, 1.f, 0.f);
            for (auto x : {0.f, 0.5f, 1.f}) { expect("sin_2pi", [](auto x) { return vsl::sin_2pi<P>(x); }, x, 0.f); }
            expect("tan_pi", [](auto x) { return vsl::tan_pi<P>(x); }, 0.f, 0.f);
            for (auto x : {0.5f, 2.f, 1e10f}) {
                expect("pow(x, 0)", [](auto x) { return vsl::pow<P>(x, decltype(x){0.f}); }, x, 1.f);
            }
        };
        tier(std::integral_constant<vsl::Precision, vsl::Precision::Standard>{});
        tier(std::integral_constant<vsl::Precision, vsl::Precision::Fast>{});

#if !VSL_FINITE_MATH
        constexpr auto inf = std::numeric_limits<float>::infinity();
        constexpr auto nan = std::numeric_limits<float>::quiet_NaN();
        expect("exp2", exp2, inf, inf);
        expect("exp2", exp2, -inf, 0.f);
        expect("tanh", tanh, inf, 1.f);
        expect("tanh", tanh, -inf, -1.f);

        // NaN in: the bound out, on every backend.
        expect("clamp", [](auto x) { return vsl::clamp(x, 0.f, 1.f); }, nan, 0.f);
        expect("min", [](auto x) { return vsl::min(x, 1.f); }, nan, 1.f);
        expect("max", [](auto x) { return vsl::max(x, 1.f); }, nan, 1.f);

        // The exp2 clamp follows the type's exponent range.
        audio_bench::expect_true(vsl::exp2(200.0) == std::exp2(200.0), "Fail exp2(200.0) double");
        audio_bench::expect_true(vsl::exp2(vsl::float64x2{1024.0})[0] == std::numeric_limits<double>::infinity(), "Fail exp2(1024.0) double");
#endif
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
static_assert(abs_equal(vsl::cos(0.f), 1.f, poly_tol));
static_assert(abs_equal(vsl::sin(0.f), 0.f, poly_tol));
static_assert(abs_equal(vsl::kernel::tan(0.f), 0.f, poly_tol));
static_assert(abs_equal(vsl::cos_2pi(0.f), 1.f, poly_tol));
static_assert(abs_equal(vsl::sin_2pi(0.f), 0.f, poly_tol));
static_assert(abs_equal(vsl::acos(1.f), 0.f, poly_tol));
static_assert(abs_equal(vsl::asin(0.f), 0.f, poly_tol));
static_assert(abs_equal(vsl::atan(1.f), std::numbers::pi_v<float> / 4.f, poly_tol));
static_assert(abs_equal(vsl::cosh(0.f), 1.f, poly_tol));
static_assert(abs_equal(vsl::sinh(0.f), 0.f, poly_tol));
static_assert(abs_equal(vsl::tanh(0.f), 0.f, poly_tol));

// A realistic DSP constant computed entirely at compile time:
// bilinear-transform pre-warp gain for a 1 kHz cutoff at 48 kHz, tan(pi/48).
constexpr auto g = vsl::tan(std::numbers::pi_v<float> * 1000.f / 48000.f);
static_assert(abs_equal(g, 0.0655434f, 1e-5f));

} // namespace constexpr_checks

} // namespace