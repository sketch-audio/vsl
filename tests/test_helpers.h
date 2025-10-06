#pragma once

#include <bit>
#include <cmath>
#include <limits>


// see: https://www.emmtrix.com/wiki/ULP_Difference_of_Float_Numbers
inline auto f32_ulp_of(float x) -> float
{
    x = std::abs(x);
    return std::nextafter(x, std::numeric_limits<float>::infinity()) - x;
}

// see: https://www.emmtrix.com/wiki/ULP_Difference_of_Float_Numbers
inline auto f32_ulp_dist(float f1, float f2) -> uint32_t
{
    if (std::signbit(f1) != std::signbit(f2)) {
        return f32_ulp_dist(std::abs(f1), 0.f) + f32_ulp_dist(std::abs(f2), 0.f) + 1;
    }

    const auto u1 = std::bit_cast<uint32_t>(f1);
    const auto u2 = std::bit_cast<uint32_t>(f2);

    return u1 > u2 ? u1 - u2 : u2 - u1;
};