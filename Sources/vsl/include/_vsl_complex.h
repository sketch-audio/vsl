#ifndef _vsl_complex_h
#define _vsl_complex_h

namespace vsl {

template<typename X>
struct Complex {
    //
    X real{};
    X imag{};
    
    // MARK: - Complex Arithmetic
    
    Complex operator+(const Complex& other) const
    {
        return {real + other.real, imag + other.imag};
    }
    
    Complex& operator+=(const Complex& other)
    {
        real += other.real;
        imag += other.imag;
        return *this;
    }
    
    Complex operator-(const Complex& other) const
    {
        return {real - other.real, imag - other.imag};
    }
    
    Complex& operator-=(const Complex& other)
    {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }
    
    Complex operator*(const Complex& other) const
    {
        return {real * other.real - imag * other.imag, real * other.imag + imag * other.real};
    }
    
    Complex& operator*=(const Complex& other)
    {
        const auto new_real = real * other.real - imag * other.imag;
        imag = real * other.imag + imag * other.real;
        real = new_real;
        return *this;
    }
    
    Complex operator/(const Complex& other) const
    {
        const auto denom = other.real * other.real + other.imag * other.imag;
        return {
            (real * other.real + imag * other.imag) / denom,
            (imag * other.real - real * other.imag) / denom
        };
    }
    
    Complex& operator/=(const Complex& other)
    {
        const auto denom = other.real * other.real + other.imag * other.imag;
        const auto new_real = (real * other.real + imag * other.imag) / denom;
        imag = (imag * other.real - real * other.imag) / denom;
        real = new_real;
        return *this;
    }
    
    // MARK: - Scalar Arithmetic
    
    friend Complex operator+(X scalar, const Complex& complex)
    {
        return {scalar + complex.real, complex.imag};
    }
    
    Complex operator+(X scalar) const
    {
        return {real + scalar, imag};
    }
    
    Complex& operator+=(X scalar)
    {
        real += scalar;
        return *this;
    }
    
    friend Complex operator-(X scalar, const Complex& complex)
    {
        return {scalar - complex.real, -complex.imag};
    }
    
    Complex operator-(X scalar) const
    {
        return {real - scalar, imag};
    }
    
    Complex& operator-=(X scalar)
    {
        real -= scalar;
        return *this;
    }
    
    friend Complex operator*(X scalar, const Complex& complex)
    {
        return {scalar * complex.real, scalar * complex.imag};
    }
    
    Complex operator*(X scalar) const
    {
        return {real * scalar, imag * scalar};
    }
    
    Complex& operator*=(X scalar)
    {
        real *= scalar;
        imag *= scalar;
        return *this;
    }
    
    friend Complex operator/(X scalar, const Complex& complex)
    {
        const auto denom = complex.real * complex.real + complex.imag * complex.imag;
        return {
            (scalar * complex.real) / denom,
            (-scalar * complex.imag) / denom
        };
    }
    
    Complex operator/(X scalar) const
    {
        return {real / scalar, imag / scalar};
    }
    
    Complex& operator/=(X scalar)
    {
        real /= scalar;
        imag /= scalar;
        return *this;
    }
};

}

#endif /* _vsl_complex_h */
