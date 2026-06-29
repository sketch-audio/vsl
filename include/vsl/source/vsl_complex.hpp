#pragma once

#include <concepts> // std::semiregular

#include "vsl_core.hpp" // mask_t
#include "vsl_impl.hpp" // force_inline
#include "vsl_math.hpp" // vsl::sqrt

namespace vsl {

template<typename X>
struct Complex {
    X real{};
    X imag{};

    // MARK: - Complex Arithmetic

    force_inline auto operator+(const Complex& other) const -> Complex
    {
        return {real + other.real, imag + other.imag};
    }

    force_inline auto operator+=(const Complex& other) -> Complex&
    {
        real += other.real;
        imag += other.imag;
        return *this;
    }

    force_inline auto operator-(const Complex& other) const -> Complex
    {
        return {real - other.real, imag - other.imag};
    }

    force_inline auto operator-=(const Complex& other) -> Complex&
    {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }

    force_inline auto operator*(const Complex& other) const -> Complex
    {
        return {real * other.real - imag * other.imag, real * other.imag + imag * other.real};
    }

    force_inline auto operator*=(const Complex& other) -> Complex&
    {
        const auto new_real = real * other.real - imag * other.imag;
        imag = real * other.imag + imag * other.real;
        real = new_real;
        return *this;
    }

    force_inline auto operator/(const Complex& other) const -> Complex
    {
        const auto denom = other.real * other.real + other.imag * other.imag;
        return {
            (real * other.real + imag * other.imag) / denom,
            (imag * other.real - real * other.imag) / denom
        };
    }

    force_inline auto operator/=(const Complex& other) -> Complex&
    {
        const auto denom = other.real * other.real + other.imag * other.imag;
        const auto new_real = (real * other.real + imag * other.imag) / denom;
        imag = (imag * other.real - real * other.imag) / denom;
        real = new_real;
        return *this;
    }

    // MARK: - Scalar Arithmetic

    friend force_inline auto operator+(X scalar, const Complex& complex) -> Complex
    {
        return {scalar + complex.real, complex.imag};
    }

    force_inline auto operator+(X scalar) const -> Complex
    {
        return {real + scalar, imag};
    }

    force_inline auto operator+=(X scalar) -> Complex&
    {
        real += scalar;
        return *this;
    }

    friend force_inline auto operator-(X scalar, const Complex& complex) -> Complex
    {
        return {scalar - complex.real, -complex.imag};
    }

    force_inline auto operator-(X scalar) const -> Complex
    {
        return {real - scalar, imag};
    }

    force_inline auto operator-=(X scalar) -> Complex&
    {
        real -= scalar;
        return *this;
    }

    friend force_inline auto operator*(X scalar, const Complex& complex) -> Complex
    {
        return {scalar * complex.real, scalar * complex.imag};
    }

    force_inline auto operator*(X scalar) const -> Complex
    {
        return {real * scalar, imag * scalar};
    }

    force_inline auto operator*=(X scalar) -> Complex&
    {
        real *= scalar;
        imag *= scalar;
        return *this;
    }

    friend force_inline auto operator/(X scalar, const Complex& complex) -> Complex
    {
        const auto denom = complex.real * complex.real + complex.imag * complex.imag;
        return {
            (scalar * complex.real) / denom,
            (-scalar * complex.imag) / denom
        };
    }

    force_inline auto operator/(X scalar) const -> Complex
    {
        return {real / scalar, imag / scalar};
    }

    force_inline auto operator/=(X scalar) -> Complex&
    {
        real /= scalar;
        imag /= scalar;
        return *this;
    }

    // MARK: - Unary

    force_inline auto operator-() const -> Complex
    {
        return {-real, -imag};
    }

    // MARK: - Magnitude & Conjugate

    force_inline auto conj() const -> Complex
    {
        return {real, -imag};
    }

    force_inline auto norm() const -> X
    {
        return real * real + imag * imag;
    }

    force_inline auto abs() const -> X
    {
        return vsl::sqrt(norm());
    }
};
static_assert(std::semiregular<Complex<float>>);

} // namespace vsl
