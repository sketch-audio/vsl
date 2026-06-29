#include <string>

#include <vsl/vsl.hpp>
#include <audio_bench/audio_bench.hpp>

namespace {

template<typename X>
auto test_complex(const std::string& name) -> void
{
    using Tests = audio_bench::Tests;
    using S = vsl::scalar_t<X>;
    using C = vsl::Complex<X>;

    // Assert a (possibly vector) Complex matches expected components in all lanes.
    auto expect_cplx = [](const C& c, S re, S im, const std::string& msg) {
        if constexpr (vsl::is_vector_v<X>) {
            audio_bench::expect_true(vsl::all(c.real == re) && vsl::all(c.imag == im), msg);
        }
        else {
            audio_bench::expect_true(c.real == re && c.imag == im, msg);
        }
    };

    Tests::add("complex - complex arithmetic (" + name + ")", [=]() {
        const auto a = C{X{1}, X{2}};
        const auto b = C{X{3}, X{4}};

        expect_cplx(a + b, 4, 6, "add");
        expect_cplx(a - b, -2, -2, "sub");
        expect_cplx(a * b, -5, 10, "mul"); // (1+2i)(3+4i) = -5+10i

        // Chosen so division is exact: (4+2i)/(1+1i) = 3 - 1i.
        const auto p = C{X{4}, X{2}};
        const auto q = C{X{1}, X{1}};
        expect_cplx(p / q, 3, -1, "div");

        // Compound assignment.
        auto acc = a;
        acc += b; expect_cplx(acc, 4, 6, "+=");
        acc -= b; expect_cplx(acc, 1, 2, "-=");
        acc *= b; expect_cplx(acc, -5, 10, "*=");
    });

    Tests::add("complex - scalar arithmetic (" + name + ")", [=]() {
        const auto a = C{X{1}, X{2}};

        expect_cplx(a + X{5}, 6, 2, "complex + scalar");
        expect_cplx(X{5} + a, 6, 2, "scalar + complex");
        expect_cplx(a - X{5}, -4, 2, "complex - scalar");
        expect_cplx(X{5} - a, 4, -2, "scalar - complex"); // 5 - (1+2i) = 4-2i
        expect_cplx(a * X{2}, 2, 4, "complex * scalar");
        expect_cplx(X{2} * a, 2, 4, "scalar * complex");

        const auto p = C{X{4}, X{2}};
        expect_cplx(p / X{2}, 2, 1, "complex / scalar");
        expect_cplx(X{2} / C{X{1}, X{1}}, 1, -1, "scalar / complex"); // 2/(1+1i) = 1-1i
    });

    Tests::add("complex - unary, conj, magnitude (" + name + ")", [=]() {
        const auto a = C{X{1}, X{2}};
        expect_cplx(-a, -1, -2, "negate");
        expect_cplx(a.conj(), 1, -2, "conj");

        // norm = re^2 + im^2, abs = sqrt(norm). (3+4i) -> 25, 5.
        const auto c = C{X{3}, X{4}};
        if constexpr (vsl::is_vector_v<X>) {
            audio_bench::expect_true(vsl::all(c.norm() == S{25}), "norm");
            audio_bench::expect_true(vsl::all(c.abs() == S{5}), "abs");
        }
        else {
            audio_bench::expect_true(c.norm() == S{25}, "norm");
            audio_bench::expect_true(c.abs() == S{5}, "abs");
        }
    });
}

// Per-lane independence: distinct lanes must stay distinct through the pipeline.
auto test_complex_lanes() -> void
{
    using audio_bench::Tests;
    using vsl::float32x4;
    using C = vsl::Complex<float32x4>;

    Tests::add("complex - per-lane independence", []() {
        const auto a = C{float32x4{1, 2, 3, 4}, float32x4{5, 6, 7, 8}};
        const auto b = C{float32x4{2, 2, 2, 2}, float32x4{0, 0, 0, 0}};
        const auto c = a * b; // multiply by real 2 -> doubles each lane

        for (size_t i = 0; i < float32x4::num_lanes; ++i) {
            audio_bench::expect_true(c.real[i] == a.real[i] * 2, "real lane scaled");
            audio_bench::expect_true(c.imag[i] == a.imag[i] * 2, "imag lane scaled");
        }
    });
}

struct Register {
    Register()
    {
        test_complex<float>("float");
        test_complex<vsl::float32x4>("float32x4");
        test_complex<double>("double");
        test_complex<vsl::float64x2>("float64x2");
        test_complex_lanes();
    }
} register_{};

} // namespace
